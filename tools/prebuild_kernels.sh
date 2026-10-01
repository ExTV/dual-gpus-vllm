#!/usr/bin/env bash
# Optional, NVFP4 launcher only: build FlashInfer's FP4 kernels for the big card before the first
# boot, with the compiler clamped to 3 jobs (about 4 GiB of RAM each at the peak). FlashInfer
# builds a kernel the first time it is used; doing it here keeps those minutes out of the boot
# and shows the compiler output if something is wrong with the CUDA toolkit.
#   tools/prebuild_kernels.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,0}
export CUDA_HOME=${CUDA_HOME:-$([ -d /opt/cuda ] && echo /opt/cuda || echo /usr/local/cuda)}
export PATH=$CUDA_HOME/bin:$PATH
export MAX_JOBS=${MAX_JOBS:-3}
export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_120a}
SHIM=${SHIM:-$HERE/shim/smfix.so}
[ -f "$SHIM" ] || { echo "$SHIM not found (run ./install.sh)" >&2; exit 1; }
export LD_PRELOAD=$SHIM
exec "${VLLM_PYTHON:-$HERE/venv/bin/python}" "$HERE/tools/prebuild_kernels.py" "$@"
