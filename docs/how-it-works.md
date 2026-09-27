# How it works

Why each piece is there, in the order you would hit the problems. All of it is vLLM 0.30.0 on
an RTX 4070 Ti SUPER 16 GB (pipeline rank 0) + RTX 5090 32 GB (rank 1).

## The model shape that matters

Qwen3.8-27B is a hybrid: 64 layers, of which 48 are Gated DeltaNet (GDN, a linear-attention
layer with a fixed-size recurrent state) and 16 are full attention (4 KV heads x 256). Only the
16 attention layers grow with context, so fp8 KV costs about 2 KB per token per layer, and the
full 262,144 window needs ~8.6 GB of attention KV. The GDN state is ~3.3 MB per layer per
request, whatever the length. That shape drives everything below.

## Why pipeline parallel, and why the small card goes first

Tensor parallel splits every layer across both cards and syncs on every layer; on mismatched
cards it runs at the speed and memory of the smaller one twice over. Measured here:
**TP=2: 44.7 tok/s with 44.8K context. PP=2: 81 tok/s (FP8) with 262K.** Pipeline parallel
gives each card a contiguous slice of layers and passes one hidden state between them per
step, which is also why a chipset x1 slot is enough for the second card.

The last rank computes the output head (a 248,320 x 5,120 matrix) and runs the draft model, and the
first rank holds the input embedding and the vision encoder. The big card therefore goes
last. `VLLM_PP_LAYER_PARTITION` then balances memory, not speed: moving one more layer onto
the 5090 (20,44 instead of 21,43 for W8A16) did not change the step time (~34.5 ms either way)
and cost 344 MiB there.

## Pinning the KV pool across two different cards

By default vLLM sizes the KV pool from `--gpu-memory-utilization` on each worker and then
takes the smallest, so the 16 GB card would set the pool for the 32 GB card. The launchers
instead pass `--num-gpu-blocks-override N` (plus `--kv-cache-memory-bytes`, which only skips
profiling): vLLM uses exactly N blocks on both ranks and each rank allocates only its own
layers' share. `--gpu-memory-utilization` then does nothing except a free-VRAM check at boot.

## Patches 05, 06, 07: pipeline parallel fixes (both launchers)

- **05**: under pipeline parallel the MTP draft head never shared the target's embedding
  (an early return in `v1/worker/gpu/spec_decode/eagle/utils.py`), so the 5090 loaded a
  third 2.37 GiB copy. Without it FP8 + MTP tops out at ~150K context.
- **06**: with the V2 model runner and async scheduling, vLLM keeps `pp_size + 1` batches in
  flight; the receiving rank posts the draft broadcast one step early and both GPUs spin at
  100% forever (no error, no log lines). vLLM issue #53402. The patch caps it at `pp_size`.
  Before it, 3 of 5 long prefills hung; after it, none in every test since.
- **07**: `Qwen3_5ForConditionalGeneration` built the vision tower on every rank, but only the
  first rank ever runs it. Reusing vLLM's own "missing layer" stub on later ranks saves
  0.86 GiB on the 5090.

## Getting DFlash2 to run on pipeline parallel (W8A16 launcher)

DFlash2 is a 5-layer block-diffusion drafter that drafts 8 tokens at once from the target's
hidden states at layers 5, 19, 33, 47 and 61. Those layers live on both cards.

- **08**: stock 0.30 refuses DFlash with pipeline parallel. The patch marks the Qwen3.5 model
  as able to relay auxiliary hidden states across stages (`supports_aux_hidden_states_over_pp`),
  packs the hidden states of the drafter's layers that live on rank 0 into the tensors sent to rank 1, and routes
  `set_aux_hidden_state_layers` through the mixin that caches the slot layout. It also passes
  `quant_config` to the embedding so the INT8 `embed_tokens.weight_packed` loads.
- **09** (vLLM PR #51581): DFlash slices the drafter's fused QKV weight to precompute context
  K/V; a W4A16 drafter has packed int32 weights there and crashed. The patch dequantizes that
  slice. The W4A16 drafter is 1.2 GB instead of 3.6 GB in BF16, which is VRAM for context.

## Fitting 262,144 tokens (W8A16 launcher)

vLLM groups KV layers so that every group holds the same number of layers. With the drafter's
5 sliding-window layers next to 16 attention and 48 GDN layers, stock 0.30 picked groups of 5
and padded the 16 attention layers to 20: 25% more memory for every token. It also left the
drafter's layers at 16-token blocks while padding each block to a full page. Result on stock
0.30: **68,656 tokens** of context.

- **11** (syv-ai): picks group size 8 (no padding on attention or GDN, 3 padding layers on the
  small window group) and promotes the drafter's block size so its page is not padded.
  With it: 229,376 tokens.
- **`--block-size 1664`**: the promotion in patch 11 still silently gave up. At 7 draft tokens
  vLLM sizes the attention block at 1648 tokens (the smallest multiple of 16 whose page
  covers a GDN state), but the drafter needs 1664 to cover that page, and 1664 does not
  divide 1648. The boot log only says `Not promoting draft block sizes`. Forcing
  `--block-size 1664` (vLLM honors any value at or above its own) lets it promote:
  the same 420 blocks went from 241,448 to 294,386 tokens, and 262,144 needs only 375 blocks.
  If you change `SPEC_TOKENS`, the GDN state size changes and so does the right block size:
  grep the boot log for `Not promoting`.

## Fast follow-up turns

A chat client resends the whole conversation every turn, so prefix caching decides whether a
turn waits 1 second or re-reads 100K tokens. On this hybrid model three things interact:

1. **Patch 10** (vLLM PR #48375): with speculative decoding, vLLM drops the last cached block
   of every hit, but the GDN (Mamba) cache manager ignored that flag and could restore a
   recurrent state that had seen draft tokens later rejected. Upstream reports tool-eval
   accuracy falling from ~90% to ~50% with MTP + prefix caching (issue #43559). The patch
   honors the flag.
2. **`--prefix-cache-retention-interval <block size>`**: by default vLLM keeps only the last
   GDN state checkpoint of each prompt, which is exactly the block patch 10 refuses. So with
   patch 10 alone, **every turn re-prefilled from zero** (33K conversation: 11.1 s per turn).
   Keeping one checkpoint per block fixes it (2.2 s). Upstream discussion: #58303, #58549.
3. **`disable_eagle_block_drop: true`** in `--speculative-config` (vLLM PR #53388, already in
   0.30): the hit still lost two blocks, ~4,400 tokens, re-prefilled every turn. Turning the
   drop off keeps the whole hit: **33K follow-up in 0.9 s, 31K repeat in 1.1 s**.

Is (3) safe? The launchers default to it (`EAGLE_BLOCK_DROP=0`) after this check: a
2,600-token reply (so the decode crossed several block boundaries) followed by a cached
41K-token follow-up produced exactly the same answer as the same follow-up with prefix
caching turned off, with the drop both on and off, and draft acceptance on cache hits was
unchanged (64.8%). That is one targeted test, not an evaluation; if long sessions ever
produce empty or repetitive output, set `EAGLE_BLOCK_DROP=1`.

One more upstream bug is avoided here by construction: on 0.30, a DFlash2 cache hit indexes
the GDN state with `cache_config.block_size` (issue #58894, fix PR #55601). That is only
correct when the block size equals the GDN block, which `--block-size 1664` guarantees.
Keep it if you change anything else.

## Flags that are there on purpose

- `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`: with full CUDA graphs the
  speculative verify step was captured in a form that made MTP slower than no MTP at all on
  this model (vLLM issue #40880). With piecewise graphs MTP with 3 draft tokens went from 45 to 140-155 tok/s on a single 5090 (vLLM 0.27.1).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and a 128 MiB
  `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE`: the default FlashInfer workspace is allocated
  lazily on the first long request and ran the card out of memory at full context (measured on a single 5090).
- `--max-num-batched-tokens 1024`: 2048 gave no speedup on cached turns; 4096 ran the 5090
  out of memory on the first request.
- `--max-num-seqs 2`: the pool holds exactly one 262K request; a second request can run
  alongside whenever the first is not using the whole pool.
- `--kv-cache-dtype fp8`: halves the attention KV versus BF16. Neither checkpoint ships KV
  scales, so vLLM uses a scale of 1.0.
- `--mm-processor-kwargs max_pixels 4194304`: up to 4 MP per image (4,096 vision tokens),
  the same limits as llama.cpp's `--image-max-tokens 4096`. The encoder runs on rank 0.
- `--limit-mm-per-prompt '{"video":0}'`: video off, images unlimited.
- `--override-generation-config` temperature 0.6, top_p 0.95, top_k 20: the sampling used
  when a client sends none. Requests that set their own values win.

## Things that did not help

| tried | result |
| --- | --- |
| tensor parallel 2 on the mismatched cards | 44.7 tok/s, 44.8K context |
| W8A16 + DFlash2 on stock 0.30 (no patch 11) | 68,656 tokens of context |
| DFlash2 drafter in BF16 instead of W4A16 | +2.4 GiB on the 5090, no room for context |
| `--max-num-batched-tokens` 2048 / 4096 | no gain / out of memory |
| PP 20,44 instead of 21,43 (W8A16) | same step time, +344 MiB on the 5090 |
