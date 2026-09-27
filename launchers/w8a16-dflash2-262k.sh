#!/usr/bin/env bash
# Qwen3.8-27B (Huihui abliterated) INT8 W8A16 + DFlash2 speculative decoding, full 262,144 context,
# pipeline parallel over two different GPUs. The fastest of the two setups; see README.md.
#
# Every default below was measured on an RTX 4070 Ti SUPER 16 GB (rank 0) + RTX 5090 32 GB (rank 1).
# Override any of them from the environment, for example:
#   MODELS=/data/models PORT=8000 ./launchers/w8a16-dflash2-262k.sh
# Extra arguments are passed straight to `vllm serve`.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)

# GPU order: pipeline rank 0 must be the SMALLER card (embeddings, vision encoder, first layers),
# rank 1 the BIGGER card (remaining layers, lm_head, the drafter). CUDA_DEVICE_ORDER=PCI_BUS_ID
# makes these indexes the same as in nvidia-smi. On the reference box nvidia-smi shows the 5090
# as 0 and the 4070 Ti SUPER as 1, hence "1,0".
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,0}
# Layers per rank (64 total). 21 on the 16 GB card, 43 on the 32 GB card.
export VLLM_PP_LAYER_PARTITION=${PP_PARTITION:-21,43}

export CUDA_HOME=${CUDA_HOME:-$([ -d /opt/cuda ] && echo /opt/cuda || echo /usr/local/cuda)}
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728
export MAX_JOBS=${MAX_JOBS:-3} TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-3}

MODELS=${MODELS:-$HOME/models}
MODEL=${MODEL:-$MODELS/Qwen3.8-27B-huihui-abliterated-INT8-W8A16-DFlash2}
DRAFTER=${DRAFTER:-$MODELS/Qwen3.8-27B-DFlash2-W4A16}
VLLM_BIN=${VLLM_BIN:-$HERE/venv/bin/vllm}
TEMPLATE=${TEMPLATE:-$HERE/chat-templates/sharp-v22.5.0.jinja}
PORT=${PORT:-8888}
SERVED_NAME=${SERVED_NAME:-qwen3.8-27b}

MAX_LEN=${MAX_LEN:-262144}
# The KV pool is pinned in blocks. 375 is the smallest count the boot check accepts for 262,144.
# To give VRAM back, lower NUM_BLOCKS and MAX_LEN together; the boot error prints the max length
# a given block count can hold.
NUM_BLOCKS=${NUM_BLOCKS:-375}
SPEC_TOKENS=${SPEC_TOKENS:-7}
# 1664 is what makes 262,144 fit, see docs/how-it-works.md. It is tied to SPEC_TOKENS=7.
BLOCK_SIZE=${BLOCK_SIZE:-1664}
# 0 = keep the whole prefix-cache hit (fast follow-up turns). 1 = vLLM's conservative default,
# which re-prefills ~4,400 extra tokens per turn. See docs/how-it-works.md.
EAGLE_BLOCK_DROP=${EAGLE_BLOCK_DROP:-0}
DROP_FLAG=$([ "$EAGLE_BLOCK_DROP" = 1 ] && echo false || echo true)
# Only a boot-time free-VRAM check once the pool is pinned; it sizes nothing.
GPU_UTIL=${GPU_UTIL:-0.80}
# Optional RAM cap for the whole server (systemd user scope), e.g. MEMORY_MAX=24G. Empty = none.
MEMORY_MAX=${MEMORY_MAX:-}

[ -x "$VLLM_BIN" ] || { echo "vllm not found at $VLLM_BIN (run ./install.sh first)" >&2; exit 1; }
for d in "$MODEL" "$DRAFTER"; do
    [ -f "$d/config.json" ] || { echo "model not found: $d (see README step 2)" >&2; exit 1; }
done

SITE=$(dirname "$VLLM_BIN")/../lib/python3.*/site-packages/vllm
check() { grep -q "$2" $SITE/$1 || { echo "patch missing in $1 ($3); run ./install.sh" >&2; exit 3; }; }
check v1/worker/gpu/spec_decode/eagle/utils.py 'LOCAL PATCH' 05
check config/vllm.py 'LOCAL PATCH' 06
check model_executor/models/interfaces.py 'LOCAL PATCH' 07
check model_executor/models/qwen3_5.py 'LOCAL PATCH (08)' 08
check model_executor/models/qwen3_dflash.py '_dense_kv_rows' 09
check v1/core/single_type_kv_cache_manager.py 'vllm#48375' 10
check v1/core/kv_cache_utils.py '_prefer_padding_sliding_window_buckets' 11

if ss -ltn | awk '{print $4}' | grep -q ":$PORT\$"; then
    echo "port $PORT is already in use" >&2
    exit 2
fi

TEMPLATE_ARGS=()
if [ -f "$TEMPLATE" ]; then
    TEMPLATE_ARGS=(--chat-template "$TEMPLATE")
else
    echo "note: $TEMPLATE not found, using the checkpoint's own chat template" >&2
fi
RUN=()
[ -n "$MEMORY_MAX" ] && RUN=(systemd-run --user --scope -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=8G)

exec "${RUN[@]}" "$VLLM_BIN" serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend mp \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs 2 \
    --max-num-batched-tokens 1024 \
    --block-size "$BLOCK_SIZE" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-memory-bytes 9000000000 \
    --num-gpu-blocks-override "$NUM_BLOCKS" \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --prefix-cache-retention-interval "$BLOCK_SIZE" \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$SPEC_TOKENS,\"disable_eagle_block_drop\":$DROP_FLAG}" \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --mm-processor-kwargs '{"max_pixels":4194304,"min_pixels":1048576}' \
    --limit-mm-per-prompt '{"video":0}' \
    "${TEMPLATE_ARGS[@]}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0}' \
    "$@"
