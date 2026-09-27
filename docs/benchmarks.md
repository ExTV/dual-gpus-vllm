# Benchmarks

Reference box: RTX 5090 32 GB (PCIe 5.0 x16, also drives a KDE desktop that holds 0.4-1.8 GiB)
+ RTX 4070 Ti SUPER 16 GB (chipset slot, nvidia-smi reports PCIe x1), Intel i7-14700K,
31 GiB RAM, Arch Linux, NVIDIA driver 615.71, CUDA 13.4, vLLM 0.30.0 with the patches in this
repo. One request at a time unless noted. Measured September 2026.

How to reproduce: `tools/decode_bench.py` (decode), `tools/long_prefill.py` (long prompt and
VRAM peaks), `tools/cache_check.py` (follow-up turns), and for random-token throughput
`vllm bench serve --dataset-name random --random-input-len 1024 --random-output-len 512
--num-prompts 5 --max-concurrency 1 --ignore-eos --temperature 0`.

## Decode speed

Natural prompts, thinking off, temperature 0, 700 output tokens (`tools/decode_bench.py`).
Both launchers measured the same day, and again from a clean `./install.sh` with the repo
launchers (same results within 1%).

| | W8A16 + DFlash2 (7 draft tokens) | FP8 + MTP (3 draft tokens) |
| --- | --- | --- |
| code | **180 tok/s** (74% acceptance) | 99 tok/s (88%) |
| reasoning | **159 tok/s** (65%) | 93 tok/s (80%) |
| prose | **86 tok/s** (28%) | 71 tok/s (53%) |
| random tokens 1K in / 512 out (`vllm bench serve`) | **122.0 tok/s** | 81.3 tok/s (ITL 36.7 ms) |
| time per decode step | ~34.5 ms | ~36.7 ms |
| without speculative decoding | | 36 tok/s |

Speculative decoding gains depend on how predictable the text is: DFlash2 drafts 8 tokens per
step, which pays off hugely on code and structured reasoning and much less on free prose.
Random-token benchmarks overstate acceptance; judge by the natural prompts.

## Context and memory

| | W8A16 + DFlash2 | FP8 + MTP |
| --- | --- | --- |
| max context | 262,144 | 262,144 |
| KV pool | 262,844 tokens (375 blocks of 1,664) | 263,608 tokens (180 blocks of 1,600) |
| layers per card (4070 / 5090) | 21 / 43 | 20 / 44 |
| weights loaded (4070 / 5090) | 9.76 / 20.54 GiB | 10.56 / 21.17 GiB |
| idle VRAM after boot (5090 / 4070) | 30,817 / 14,433 MiB | 30,885 / 14,441 MiB |
| peak during a ~258-260K prompt (5090 / 4070) | **31,059 / 14,675 MiB** | 31,113 / 14,605 MiB |
| peak with 2 x 4 MP images (5090 / 4070) | 31,067 / 15,001 MiB | 4070 14,905 MiB |

Peaks include the desktop's share of the 5090.

## Prefill (time to first token)

| prompt | W8A16 + DFlash2 | FP8 + MTP |
| --- | --- | --- |
| 32K tokens | 10.1 s | 10.1 s (3,160 tok/s) |
| 145K tokens | | 68.5-68.8 s |
| ~258-260K tokens | **161-163 s**, needle found, 0 preemptions | 162.4 s, needle found, 0 preemptions |

## Follow-up turns (prefix cache)

`tools/cache_check.py`: a ~33K-token conversation, then the same prompt again, then two
follow-ups. What a chat client feels on every turn.

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
