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
last. `VLLM_PP_LAYER_PARTITION` then trades free memory on the big card for speed: the big
card reads its weights almost three times faster (1,792 vs 672 GB/s), so every layer moved onto
it saves about 0.4-0.5 ms per decode step and a few seconds of a long prefill, and costs that
layer's weights there (~380 MiB for W8A16, ~340 MiB for NVFP4). Measured on W8A16: 21,43 =
33.1 ms per step, 19,45 = 32.3, 18,46 = 31.7. The defaults leave about 1 GiB free on the big
card at full context for a desktop.

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

## Each rank must see its own GPU (patch 13 and the shim, all launchers)

On a mixed pair, stock vLLM 0.30 makes every pipeline rank choose and run its kernels for
**GPU 0**. Two separate places hard-code it:

- **Python**: `current_platform.get_device_capability()`, `has_device_capability()` and
  `is_device_capability_family()` default to `device_id=0`, and every kernel's `is_supported()`
  calls them with the default. Each worker does call `torch.cuda.set_device(its own GPU)`, but
  nothing reads it back. With the small card first, the 5090's worker believed it was an SM89
  card.
- **Compiled**: `get_sm_version_num()` in vLLM's extension (`_C_stable_libtorch`) asks CUDA for
  the capability of device 0. The CUTLASS FP8 and FP4 GEMMs and the NVFP4 quantization kernels
  dispatch on it.

The effect is silent. The 5090 ran the Marlin weight-only kernels meant for older cards
instead of its native FP4/FP8 ones, and the boot log hides it: the kernel-selection lines are
printed by the first rank only. Nothing crashes, it is just slower, and the NVFP4 checkpoint
ran its 4-bit layers with 16-bit activations instead of its own W4A4 recipe.

The fix has four parts and they only work together:

1. **Patch 13** (`platforms/cuda.py`): when called with the default device and CUDA is
   initialised in this process, resolve the capability of the worker's current device.
2. **`shim/smfix.so`**, loaded with `LD_PRELOAD`: redefines `get_sm_version_num()` to ask for
   the calling thread's current device. Without it, patch 13 makes rank 1 pick native kernels
   whose compiled dispatch still answers "SM 89" (`No compiled nvfp4 quantization kernel for
   SM 89`). `install.sh` builds it from `shim/smfix.cpp` (30 lines, no CUDA headers needed).
3. **`CUTE_DSL_ARCH=sm_120a`**: FlashInfer's CuTe-DSL kernels (the b12x FP4 GEMM) also compile
   for device 0's architecture unless told otherwise. Set it to the big card's architecture.
4. **`--no-enable-flashinfer-autotune`**: vLLM's kernel warmup runs the FlashInfer autotuner on
   ranks that report SM90 or newer. Once rank 1 reports the truth and rank 0 is an SM89 card,
   only rank 1 enters it, waits on a broadcast the other rank never joins, and the boot hangs
   forever at the first transfer between the cards.

What it bought, same day, same box (ms per decode step at short context, 250K-token prefill):

| launcher | before | with the fix | why |
| --- | --- | --- | --- |
| NVFP4 + DFlash2 | 27.2 ms, 126.3 s | 27.1 ms, 114.6 s | native W4A4 GEMM on the 5090: prefill -9%, decode equal at 8 rows |
| FP8 + DFlash2 | 33.2 ms, 133.1 s | 33.5 ms, 133.3 s | nothing: see below |
| W8A16 + DFlash2 | 34.5 ms | 34.5 ms, 148.6 s | nothing: INT8 group weights run Marlin on any card |

On FP8 the 5090 now selects vLLM's DeepGEMM block kernel instead of Marlin, and it does not
matter: during a prefill the two cards work as a pipeline and the 4070 is the slower stage, and
at decode the verify batch is 8 rows, where every GEMM is bound by reading the weights, not by
arithmetic. A microbench on the 5090 (weights rotated past the L2 cache, the five layer
shapes of rank 1 summed) reads Marlin 292 us at 8 rows and 4,127 at 1,024 rows, DeepGEMM and
CUTLASS 312 and 2,260. The fix is still required on every launcher, because patch 13 lives in
the shared venv and part 4 above is a boot hang, not a slowdown.

## The FP8 GEMM on an SM89 card (patch 14, FP8 launchers)

An RTX 40-series card has FP8 tensor cores but cannot run the SM90+ block-FP8 kernels, so vLLM
0.30 falls back to Marlin there, which expands the FP8 weights to 16-bit. Patch 14 (vLLM PR
#59232, with its device check changed to the worker's own device for the reason above) puts
the Triton block-FP8 kernel ahead of Marlin on SM89. That kernel needs per-GPU tuned configs:
with vLLM's defaults it cost ~9% per decode step on the 4070 Ti SUPER; with configs tuned on
that card (`configs/triton-fp8/`, produced with vLLM's `benchmarks/kernels/benchmark_w8a8_block_fp8.py`
on this model's five weight shapes) it costs ~1% per step and cuts the prefill, which the
small card dominates, by 12-18% (250K: 151.6 to 133.1 s). vLLM picks a config file by the
GPU's name, so on any other SM89 card the shipped files do nothing: either tune your own or set
`VLLM_DISABLED_KERNELS=TritonFp8BlockScaledMMKernel` to stay on Marlin. One more trap: the
file lookup also uses device 0's name, so never steer the big card to Triton.

## The NVFP4 launcher

The OrcaRouter NVFP4 checkpoint keeps attention and GDN projections in FP8 and stores the MLPs
in NVFP4 (4-bit weights and activations, a Blackwell format), 23.5 GiB. An SM89 card has no
FP4 hardware; vLLM runs those layers there through `MarlinNvFp4LinearKernel` (the scheme's
minimum capability is 7.5), which is why this checkpoint works on the mismatched pair at all.

- **Kernel on the big card**: with the fix above the 5090 has four native FP4 GEMMs to choose
  from. Per MLP at 8 rows on the 5090: FlashInfer b12x 102 us, cuDNN 149, FlashInfer CUTLASS
  150, vLLM CUTLASS 165. vLLM's order prefers FlashInfer CUTLASS, so the launcher sets
  `VLLM_DISABLED_KERNELS=FlashInferCutlassNvFp4LinearKernel`, which leaves b12x first. The
  small card is unaffected (Marlin is its only option). `--linear-backend` cannot do this: it
  applies to both ranks and the small card has no such kernel.
- **Partition 13,51**: the 4-bit layers are small, so the 5090 can hold 51 of them next to the
  output head, the drafter and its share of the KV pool. 16,48 = 25.9 ms per step,
  14,50 = 24.9, 12,52 = 24.4 (1.3 GiB left on the 5090).
- The pool arithmetic (`--block-size 1664`, 375 blocks) is the same as W8A16 below: it depends
  on the drafter and the draft length, not on the weights.

## Getting DFlash2 to run on pipeline parallel (DFlash2 launchers)

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

## Fitting 262,144 tokens (DFlash2 launchers)

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
- `"attention_backend":"TRITON_ATTN"` inside `--speculative-config` (`DRAFT_ATTN`): only the
  drafter's five sliding-window layers switch from FlashInfer to vLLM's Triton attention, about
  4% faster per step on every DFlash2 launcher (W8A16: 34.5 to 33.1 ms). Drafts never change
  the output distribution, and a teacher-forced check on 10K tokens gave the same loss to four
  digits. Do not do the same for the target: with Triton attention the decode step at 250K
  tokens went from 42 to 177 ms.
- `--no-enable-flashinfer-autotune`: part of the rank fix above; without it the boot can hang.
- `--max-num-batched-tokens 1024`: 2048 gave no speedup on cached turns; 4096 ran the 5090
  out of memory on the first request.
- `--max-num-seqs 2`: the pool holds exactly one 262K request; a second request can run
  alongside whenever the first is not using the whole pool.
- `--kv-cache-dtype fp8`: halves the attention KV versus BF16. Only the NVFP4 checkpoint ships
  calibrated KV scales; for the others vLLM uses a scale of 1.0.
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
| a different FP8 GEMM on the 5090 (DeepGEMM, CUTLASS or Marlin) | same step time and same 250K prefill: the 4070 sets the prefill, 8-row GEMMs are bandwidth-bound |
| Triton attention for the target as well as the drafter | 32.0 ms short, 177 ms per step at 250K |
| CUDA graphs for the drafter under a piecewise target | no gain over Triton drafter attention |
| vLLM PR #59448 (Triton split-KV for verify batches) | fixes the 177 ms collapse but lands 11% slower than FlashInfer at 250K |
| untuned Triton FP8 kernel on the 4070 Ti SUPER | prefill -11%, decode +9% per step (tuned: -12% and +1%) |
