# dual-gpus-vllm

Qwen3.8-27B with the **full 262,144-token context** on two *different* consumer GPUs
(an RTX 5090 32 GB plus an RTX 4070 Ti SUPER 16 GB), served by vLLM 0.30.0 with pipeline
parallelism and speculative decoding.

Four ready-to-run launchers, all at 262,144 tokens of context:

| launcher | weights | speculative decoding | decode (code / reasoning / prose) | 250K-token prefill |
| --- | --- | --- | --- | --- |
| **`launchers/nvfp4-dflash2-262k.sh`** (fastest) | NVFP4 MLPs + FP8 attention, 23.5 GB | DFlash2, 7 draft tokens | **215-259 / 233 / 110-119 tok/s** | **112 s** |
| **`launchers/w8a16-dflash2-262k.sh`** (fastest 8-bit decode) | INT8 W8A16, 28 GB | DFlash2, 7 draft tokens | **187 / 172 / 88 tok/s** | 140 s |
| `launchers/fp8-dflash2-262k.sh` (fastest 8-bit prefill) | block FP8, 28.75 GB | DFlash2, 7 draft tokens | 174-183 / 161-167 / 85 tok/s | 133 s |
| `launchers/fp8-mtp-262k.sh` (no separate drafter) | block FP8, 28.75 GB | built-in MTP head, 3 draft tokens | 99 / 94 / 72 tok/s | 137 s |

All run with vision (images up to 4 MP), prefix caching, tool calling and reasoning parsing,
behind an OpenAI-compatible API. Stock vLLM 0.30.0 cannot run any of these on two
mismatched cards, and where it does run it treats the big card as a copy of the small one;
the patches in `patches/` and the small shim in `shim/` are what make it work. The details and the
reasoning behind every flag are in [docs/how-it-works.md](docs/how-it-works.md), and all
measurements are in [docs/benchmarks.md](docs/benchmarks.md).

## What you need

- Two NVIDIA GPUs with about 48 GB combined, one bigger than the other. Tested:
  RTX 5090 32 GB + RTX 4070 Ti SUPER 16 GB. The small card sits on a chipset slot
  (nvidia-smi reports PCIe x1) and it still works, because pipeline parallelism only passes
  one small activation per step between the cards.
- Linux, a recent NVIDIA driver (tested 615.71), the CUDA toolkit (nvcc, for FlashInfer's
  JIT kernels; tested 13.4), Python 3.13, [uv](https://docs.astral.sh/uv/), `patch` and `g++`.
- About 32 GB of system RAM (tested with 31 GiB) and ~60 GB of disk for the models.
- A desktop can keep running on the big card: the configs leave 0.6-1.5 GiB free on it
  (see "peak" in [docs/benchmarks.md](docs/benchmarks.md); `PP_PARTITION` trades speed for room).
- The NVFP4 launcher needs a Blackwell (RTX 50-series) big card; the small card can be older.

## Step 1: install vLLM 0.30.0 and the patches

```bash
git clone https://github.com/ExTV/dual-gpus-vllm.git && cd dual-gpus-vllm
./install.sh
```

This creates `./venv`, installs `vllm==0.30.0` with the CUDA 13.0 torch build, applies
every patch in `patches/` (it prints `applied` for each one; re-running it is safe), copies
the tuned kernel configs in `configs/` into the venv and builds `shim/smfix.so`.

## Step 2: download the models

```bash
export MODELS=~/models        # anywhere you like; the launchers read $MODELS
mkdir -p "$MODELS"

# The DFlash2 drafter, shared by the three DFlash2 launchers
hf download syvai/Qwen3.8-27B-DFlash2-W4A16 \
    --local-dir "$MODELS/Qwen3.8-27B-DFlash2-W4A16"

# NVFP4 launcher (gated: accept the terms on its Hugging Face page first)
hf download orcarouter/Qwen3.8-27B-Uncensored-NVFP4 \
    --local-dir "$MODELS/Qwen3.8-27B-Uncensored-NVFP4"

# W8A16 launcher
hf download lued/Qwen3.8-27B-huihui-abliterated-INT8-W8A16-DFlash2 \
    --local-dir "$MODELS/Qwen3.8-27B-huihui-abliterated-INT8-W8A16-DFlash2"

# Both FP8 launchers (gated as well)
hf download orcarouter/Qwen3.8-27B-Uncensored-FP8 \
    --local-dir "$MODELS/Qwen3.8-27B-Uncensored-FP8"
```

Which one: NVFP4 is the fastest at everything and leaves the most free VRAM, at 4-bit
precision in the MLP layers (attention and GDN layers stay FP8, and it is the only one that
ships calibrated KV-cache scales). W8A16 and FP8 keep 8-bit weights everywhere; W8A16 decodes
a little faster, FP8 reads long prompts faster.

(`hf` comes with `pip install -U huggingface_hub`.) Other Qwen3.8-27B checkpoints in the same
formats should work too; point `MODEL=` at them.

## Step 3: set your GPU order

Pipeline rank 0 must be the **smaller** card and rank 1 the **bigger** one: the last rank holds
the output head and the draft model, the first holds the embeddings and the vision encoder.
Check your indexes:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

The launchers default to `CUDA_VISIBLE_DEVICES=1,0` because on the reference box the 5090 is
index 0 and the 4070 Ti SUPER is index 1. If your small card is index 0, run with
`CUDA_VISIBLE_DEVICES=0,1`.

## Step 4: start the server

```bash
./launchers/nvfp4-dflash2-262k.sh      # or any of the other three
```

The first boot compiles kernels and takes a few minutes; later boots take one to two minutes.
For the NVFP4 launcher you can run `tools/prebuild_kernels.sh` first: it builds FlashInfer's
FP4 kernels for the big card outside the server, with the compiler held to 3 jobs.
When the log shows `Application startup complete`, check it answers:

```bash
curl -s localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Name the planets of the solar system."}],
  "chat_template_kwargs": {"enable_thinking": false}}'
```

Stop it with Ctrl-C (or `pkill -f '[v]llm serve'`).

Point any OpenAI-compatible client at `http://<host>:8888/v1` with model name `qwen3.8-27b`.
Set the client's context window to at most **262,144** tokens.

## Step 5 (optional): check your numbers

```bash
venv/bin/python tools/decode_bench.py                 # decode speed on code / prose / reasoning
venv/bin/python tools/cache_check.py                  # follow-up turns should take ~1 s
venv/bin/python tools/long_prefill.py "$MODELS/Qwen3.8-27B-Uncensored-NVFP4"
                                                      # ~258K-token prompt, per-GPU VRAM peaks
venv/bin/python tools/long_decode.py "$MODELS/Qwen3.8-27B-Uncensored-NVFP4" 250000
                                                      # prefill time, then ms per decode step at 250K
```

## Settings you may want to change

All are environment variables, for example `PORT=8000 MODELS=/data ./launchers/...`.

| variable | default | what it does |
| --- | --- | --- |
| `MODELS`, `MODEL`, `DRAFTER` | `~/models/...` | where the checkpoints are |
| `CUDA_VISIBLE_DEVICES` | `1,0` | small card first, big card second (step 3) |
| `PP_PARTITION` | `13,51` (NVFP4), `19,45` (W8A16), `21,43` (FP8 DFlash2), `20,44` (FP8 MTP) | layers on each card, 64 in total; more on the big card is faster and leaves less free there |
| `MAX_LEN`, `NUM_BLOCKS` | 262,144, 375 (DFlash2) | context and KV pool; lower both to free VRAM |
| `DRAFT_ATTN` | `TRITON_ATTN` (DFlash2) | attention backend of the drafter only; empty = vLLM's choice |
| `VLLM_DISABLED_KERNELS` | see each launcher | steers the GEMM kernel choice per card ([how it works](docs/how-it-works.md)) |
| `CUTE_DSL_ARCH` | `sm_120a` | the big card's architecture (RTX 50 series); change it for another big card |
| `PORT`, `SERVED_NAME` | 8888, `qwen3.8-27b` | API port and model name |
| `TEMPLATE` | bundled Sharp v22.5.0 | chat template; a missing file falls back to the model's own |
| `EAGLE_BLOCK_DROP` | 0 | 1 restores vLLM's conservative prefix-cache behavior (slower follow-ups) |
| `MEMORY_MAX` | empty | e.g. `24G` caps the server's system RAM via a systemd user scope |
| `GPU_UTIL` | 0.80 | only a free-VRAM check at boot; it does not size anything |

For less context and more free VRAM on a DFlash2 launcher, lower `NUM_BLOCKS` and `MAX_LEN`
together; if `MAX_LEN` is too big for the blocks, the boot error prints the largest length
that fits.

## Troubleshooting

- **The boot hangs with both workers loaded and nothing in the log.** Every launcher passes
  `--no-enable-flashinfer-autotune` for this; if you built your own command line, add it
  ([why](docs/how-it-works.md#each-rank-must-see-its-own-gpu-patch-13-and-the-shim-all-launchers)).
- **`shim/smfix.so not found`**: re-run `./install.sh` (it needs `g++`).
- **An FP8 launcher decodes ~9% slower than the table on a small card that is not a 4070 Ti
  SUPER**: the Triton FP8 kernel has no tuned config for your card. Run with
  `VLLM_DISABLED_KERNELS=TritonFp8BlockScaledMMKernel`.
- **A 200K+ prompt shows no log lines for minutes.** Normal: a long prefill logs nothing.
  A 250K prompt takes about two minutes. `vllm:kv_cache_usage_perc` in `/metrics` climbs while it works.
- **`patch missing ... run ./install.sh`**: the launcher checks that the patches are in the
  venv it runs. Re-run `./install.sh`.
- **Out of memory on the big card**: the desktop, browser and other apps share it. Lower
  `NUM_BLOCKS`/`MAX_LEN`, or close GPU-heavy apps. After a CUDA out-of-memory crash or a
  `kill -9` on the card that drives your display, if new apps start crashing, check
  `journalctl -k | grep NVRM`; `NV_ERR_STATE_IN_USE` there needs a reboot.
- **Every chat turn is slow even though decode is fast**: run `tools/cache_check.py`. If the
  repeat takes as long as the cold request, prefix caching is not being reused; see
  [docs/how-it-works.md](docs/how-it-works.md#fast-follow-up-turns).
- **Requests fail with a context-length error**: the client's context window (plus its
  `max_tokens`) must fit inside `MAX_LEN`.

## Patches

Applied in order by `install.sh`, all against vLLM 0.30.0. The numbers match the
`LOCAL PATCH (NN)` markers they leave in the code.

| patch | needed by | what it fixes | source |
| --- | --- | --- | --- |
| 02 | all (optional) | streamed tool-call arguments cut off mid-value come back closed, not unparseable | vLLM PR #53739 |
| 03 | all (optional) | a tool argument that itself contains `<parameter=...>` text (editing a chat template) is no longer split into a phantom argument | local |
| 05 | FP8 MTP | the MTP head reuses the target's embedding under pipeline parallel instead of loading a third 2.37 GiB copy | local |
| 06 | all | pipeline-parallel deadlock with the V2 model runner + async scheduling + speculative decoding (vLLM issue #53402) | local |
| 07 | all | the vision tower is built only on the first rank (-0.86 GiB on the big card) | local |
| 08 | DFlash2 | DFlash reads hidden states from layers on both cards: relays them across pipeline stages; quantized embedding loads | local + syv-ai |
| 09 | DFlash2 | a quantized (W4A16) DFlash drafter loads instead of crashing | vLLM PR #51581 |
| 10 | all | prefix-cache correctness fix for hybrid models with speculative decoding | vLLM PR #48375 |
| 11 | DFlash2 | KV-cache grouping with a sliding-window drafter: removes 25% padding and the drafter's page blow-up | syv-ai |
| 12 | optional | quantized embedding in the MTP module loads (for MTP on INT8-embedding checkpoints) | syv-ai |
| 13 | all | each pipeline rank selects kernels for its own GPU instead of GPU 0; needs `shim/smfix.so` for the compiled side | local |
| 14 | FP8 | an SM89 card runs the Triton block-FP8 kernel (native FP8 math) ahead of Marlin; fast only with the tuned configs in `configs/triton-fp8/` | vLLM PR #59232, device check made per-rank |

syv-ai's patches come from [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen)
(Apache-2.0), which serves the same model on 24 GB cards.

## Credits

- Models: [lued](https://huggingface.co/lued/Qwen3.8-27B-huihui-abliterated-INT8-W8A16-DFlash2)
  (W8A16 of [huihui-ai](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated)),
  [syvai](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16) (W4A16 of
  [incoai's DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)),
  orcarouter ([FP8](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8),
  [NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4)).
- Chat template: Sharp v22.5.0 from
  [peculiar-ragdoll/Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates)
  (Apache-2.0), bundled unmodified in `chat-templates/`.
- [vLLM](https://github.com/vllm-project/vllm) and the authors of the upstream PRs above.

## License

Apache-2.0, see [LICENSE](LICENSE). The patches modify vLLM, which is also Apache-2.0.
