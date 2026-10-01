# Benchmarks

Reference box: RTX 5090 32 GB (PCIe 5.0 x16, also drives a KDE desktop that holds 0.4-1.8 GiB)
+ RTX 4070 Ti SUPER 16 GB (chipset slot, nvidia-smi reports PCIe x1), Intel i7-14700K,
31 GiB RAM, Arch Linux, NVIDIA driver 615.71, CUDA 13.4, vLLM 0.30.0 with the patches in this
repo. One request at a time unless noted. The tables below were measured on 2026-10-01 with
the launchers in this repo at their defaults; the follow-up, vision and history sections are
from September 2026.

How to reproduce: `tools/decode_bench.py` (decode), `tools/long_decode.py <model> 32000` and
`... 250000` (prefill time, then decode at that depth), `tools/long_prefill.py` (VRAM peaks),
`tools/cache_check.py` (follow-up turns).

## Decode speed

Natural prompts, thinking off, temperature 0, 700 output tokens (`tools/decode_bench.py`).

| | NVFP4 + DFlash2 | W8A16 + DFlash2 | FP8 + DFlash2 | FP8 + MTP (3 draft tokens) |
| --- | --- | --- | --- | --- |
| code | **215-259 tok/s** (78% acceptance) | 187 tok/s (72%) | 174-183 tok/s (69-73%) | 99 tok/s (87%) |
| reasoning | **233 tok/s** (69%) | 172 tok/s (65%) | 161-167 tok/s (63-66%) | 94 tok/s (79%) |
| prose | **110-119 tok/s** (25%) | 88 tok/s (26%) | 85 tok/s (26%) | 72 tok/s (53%) |
| time per decode step, short context | **24.8 ms** | 32.3 ms | 33.5 ms | 36.8 ms |
| time per decode step at 32K tokens | **25.7 ms** | 33.1 ms | 34.7 ms | 36.8 ms |
| time per decode step at 250K tokens | **31.7 ms** | 39.6 ms | 42.2 ms | 44.7 ms |
| without speculative decoding | | | | 36 tok/s |

Speculative decoding gains depend on how predictable the text is: DFlash2 drafts 8 tokens per
step, which pays off hugely on code and structured reasoning and much less on free prose.
Tokens per second moves with acceptance, and acceptance moves a few points between boots of
the same config (the ranges above are two boots), so **compare configs by time per step**.

## Context and memory

| | NVFP4 + DFlash2 | W8A16 + DFlash2 | FP8 + DFlash2 | FP8 + MTP |
| --- | --- | --- | --- | --- |
| max context | 262,144 | 262,144 | 262,144 | 262,144 |
| KV pool | 262,844 tokens (375 blocks of 1,664) | same | same | 263,608 tokens (180 blocks of 1,600) |
| layers per card (4070 / 5090) | 13 / 51 | 19 / 45 | 21 / 43 | 20 / 44 |
| idle VRAM after boot (5090 / 4070) | 30,873 / 10,385 MiB | 31,291 / 13,709 MiB | 31,753 / 15,489 MiB | 30,737 / 14,239 MiB |
| peak through a 250K prompt (5090 / 4070) | **31,055 / 10,487 MiB** | 31,473 / 13,973 MiB | 31,975 / 15,731 MiB | 30,897 / 14,439 MiB |

Peaks include the desktop's share of the 5090 (about 0.4 GiB during these runs). The 5090 has
32,607 MiB, so the FP8 + DFlash2 launcher leaves about 0.6 GiB and the others 1.1-1.7 GiB.

## Prefill (time to first token)

| prompt | NVFP4 + DFlash2 | W8A16 + DFlash2 | FP8 + DFlash2 | FP8 + MTP |
| --- | --- | --- | --- | --- |
| 32K tokens | **6.1 s** | 10.0 s | 8.7 s | 8.5 s |
| 250K tokens | **111.7 s** | 140.0 s | 133.9 s | 136.8 s |

The needle in the middle of the prompt was recalled on every run. The small card sets the
prefill time: the two cards work as a pipeline and the 4070 is the slower stage, which is why
fewer layers on it (NVFP4, W8A16) and a native FP8 GEMM on it (both FP8 launchers) help, and
why nothing done to the 5090's kernels moves these numbers.

## What the October retune changed

"Before" is the same launcher on the same day without the change named in the last column,
except the FP8 + MTP row, whose "before" is the September measurement.

| launcher | before | now | what did it |
| --- | --- | --- | --- |
| W8A16 + DFlash2, ms per step | 34.5 | 32.3 | drafter on Triton attention (33.1), then 19,45 instead of 21,43 |
| W8A16 + DFlash2, 250K prefill | 148.6 s | 140.0 s | two fewer layers on the 4070 |
| FP8 + MTP, prefill | 10.1 s at 32K, 162.4 s at 258K | 8.5 s at 32K, 136.8 s at 250K | tuned Triton FP8 kernel on the 4070 (patch 14) |
| NVFP4 + DFlash2, ms per step | 27.2 | 24.8 | drafter attention, 13,51 instead of 16,48 |
| NVFP4 + DFlash2, 250K prefill | 126.3 s | 111.7 s | native FP4 GEMM on the 5090 (patch 13 + shim), partition |

W8A16 partitions measured with the Triton drafter (ms per step short / at 250K, 250K prefill,
5090 peak): 21,43 = 33.1 / 41.2, 148.2 s, 30,751 MiB; 19,45 = 32.3 / 39.6, 140.0 s, 31,473;
18,46 = 31.7 / 39.0, 139.0 s, 31,859. NVFP4: 16,48 = 25.9 / 33.1, 114.7 s, 29,089;
14,50 = 24.9 / 32.0, 111.7 s, 30,829; 12,52 = 24.4 / 31.3, 111.4 s, 31,337.

## Follow-up turns (prefix cache)

`tools/cache_check.py`: a ~33K-token conversation, then the same prompt again, then two
follow-ups. What a chat client feels on every turn. Measured in September at the 21,43 partition.

| | cold | repeat | +400 tokens | +3K tokens |
| --- | --- | --- | --- | --- |
| W8A16, final launcher (`EAGLE_BLOCK_DROP=0`) | 11.2 s | **0.8 s** | **0.3-0.9 s** | **1.6 s** |
| W8A16, `EAGLE_BLOCK_DROP=1` (conservative) | 11.3 s | 2.1 s | 2.2 s | 2.8 s |
| W8A16, patch 10 without the retention flag | 11.1 s | 11.1 s | 11.3 s | 2.8 s |
| FP8, final launcher (`EAGLE_BLOCK_DROP=0`) | 10.6 s | **0.7 s** | **0.8 s** | **1.4 s** |

At 165K tokens with `EAGLE_BLOCK_DROP=1` a follow-up took 3.5 s instead of ~84 s cold.

## Vision

Two 2048 x 2048 images (4.2 MP each, 8,374 prompt tokens with the text) read correctly
(background colors and the exact text on each) on the W8A16 launcher. The FP8 launcher read
7 of 7 vision probes.

## History: how the W8A16 context grew

| step | max context |
| --- | --- |
| stock vLLM 0.30 + DFlash2 on pipeline parallel | does not start |
| + patch 08 (hidden states across stages), BF16 drafter | 68,656 at best |
| + patches 09 and 11 (W4A16 drafter, KV grouping) | 229,376 |
| + `--block-size 1664` | **262,144**, with ~800 MiB more free on the 5090 than 229K had |
