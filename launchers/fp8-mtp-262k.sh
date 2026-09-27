#!/usr/bin/env bash
# Qwen3.8-27B (OrcaRouter uncensored) block-FP8 + the checkpoint's own MTP head, full 262,144
# context, pipeline parallel over two different GPUs. See README.md.
#
# Every default below was measured on an RTX 4070 Ti SUPER 16 GB (rank 0) + RTX 5090 32 GB (rank 1).
# Override any of them from the environment, for example:
#   MODELS=/data/models PORT=8000 ./launchers/fp8-mtp-262k.sh
# Extra arguments are passed straight to `vllm serve`.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)

# GPU order: pipeline rank 0 must be the SMALLER card (embeddings, vision encoder, first layers),
# rank 1 the BIGGER card (remaining layers, lm_head, the MTP head). CUDA_DEVICE_ORDER=PCI_BUS_ID
# makes these indexes the same as in nvidia-smi. On the reference box nvidia-smi shows the 5090
# as 0 and the 4070 Ti SUPER as 1, hence "1,0".
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,0}
# Layers per rank (64 total). The FP8 weights are bigger than W8A16's, so the 16 GB card takes 20.
export VLLM_PP_LAYER_PARTITION=${PP_PARTITION:-20,44}

export CUDA_HOME=${CUDA_HOME:-$([ -d /opt/cuda ] && echo /opt/cuda || echo /usr/local/cuda)}
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728
export MAX_JOBS=${MAX_JOBS:-3} TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-3}

MODELS=${MODELS:-$HOME/models}
MODEL=${MODEL:-$MODELS/Qwen3.8-27B-Uncensored-FP8}
VLLM_BIN=${VLLM_BIN:-$HERE/venv/bin/vllm}
TEMPLATE=${TEMPLATE:-$HERE/chat-templates/sharp-v22.5.0.jinja}
PORT=${PORT:-8888}
SERVED_NAME=${SERVED_NAME:-qwen3.8-27b}

MAX_LEN=${MAX_LEN:-262144}
MTP_TOKENS=${MTP_TOKENS:-3}
# See docs/how-it-works.md for the same two knobs on the W8A16 launcher.
EAGLE_BLOCK_DROP=${EAGLE_BLOCK_DROP:-0}
DROP_FLAG=$([ "$EAGLE_BLOCK_DROP" = 1 ] && echo false || echo true)
GPU_UTIL=${GPU_UTIL:-0.80}
MEMORY_MAX=${MEMORY_MAX:-}

# Block size = smallest multiple of 16 whose fp8 attention page (2,048 B/token) covers the
# GDN state (float32 SSM 3,145,728 B + conv 10,240 x (3 + K) x 2 B): 1600 at K=3, 1568 at 0.
# One request needs its attention blocks + 3 GDN groups x (2 + K) state blocks; +1 spare.
BLOCK=$(( ( (3145728 + 20480 * (3 + MTP_TOKENS)) / 2048 + 15) / 16 * 16 ))
NUM_BLOCKS=${NUM_BLOCKS:-$(( (MAX_LEN + BLOCK - 1) / BLOCK + 3 * (2 + MTP_TOKENS) + 1 ))}

SPEC_ARGS=()
((MTP_TOKENS > 0)) && SPEC_ARGS=(
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS,\"disable_eagle_block_drop\":$DROP_FLAG}"
    --prefix-cache-retention-interval "$BLOCK")

[ -x "$VLLM_BIN" ] || { echo "vllm not found at $VLLM_BIN (run ./install.sh first)" >&2; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "model not found: $MODEL (see README step 2)" >&2; exit 1; }

SITE=$(dirname "$VLLM_BIN")/../lib/python3.*/site-packages/vllm
check() { grep -q "$2" $SITE/$1 || { echo "patch missing in $1 ($3); run ./install.sh" >&2; exit 3; }; }
check v1/worker/gpu/spec_decode/eagle/utils.py 'LOCAL PATCH' 05
check config/vllm.py 'LOCAL PATCH' 06
check model_executor/models/interfaces.py 'LOCAL PATCH' 07

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
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-memory-bytes 9000000000 \
    --num-gpu-blocks-override "$NUM_BLOCKS" \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    "${SPEC_ARGS[@]}" \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --mm-processor-kwargs '{"max_pixels":4194304,"min_pixels":1048576}' \
    --limit-mm-per-prompt '{"video":0}' \
    "${TEMPLATE_ARGS[@]}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0}' \
    "$@"
