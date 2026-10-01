#!/usr/bin/env bash
# Creates ./venv with vLLM 0.30.0 (torch for CUDA 13.0), applies every patch in patches/,
# installs the tuned Triton FP8 kernel configs and builds the small LD_PRELOAD shim in shim/.
# Safe to re-run: patches that are already applied are skipped.
set -euo pipefail
cd "$(dirname "$0")"

VENV=${VENV:-venv}
PYTHON=${PYTHON:-3.13}

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v patch >/dev/null || { echo "the 'patch' tool is required" >&2; exit 1; }
command -v g++ >/dev/null || { echo "g++ is required (builds shim/smfix.so)" >&2; exit 1; }

if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python "$PYTHON" "$VENV"
fi
# --torch-backend cu130, never auto: auto picks cu132, which has no torchaudio build and
# breaks the vLLM import without a clear error.
uv pip install --python "$VENV/bin/python" vllm==0.30.0 --torch-backend cu130

SITE=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
installed=$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("vllm"))')
[ "$installed" = "0.30.0" ] || { echo "expected vllm 0.30.0, found $installed" >&2; exit 1; }

for p in patches/*.patch; do
    if patch -p1 -d "$SITE" -R --dry-run --force --quiet < "$p" >/dev/null 2>&1; then
        echo "already applied  $(basename "$p")"
    elif patch -p1 -d "$SITE" --dry-run --forward --quiet < "$p" >/dev/null 2>&1; then
        patch -p1 -d "$SITE" --forward --quiet -b -z .orig-0.30.0 < "$p"
        echo "applied          $(basename "$p")"
    else
        echo "FAILED           $(basename "$p") (is this a clean vllm 0.30.0?)" >&2
        exit 1
    fi
done

# Tuned Triton block-FP8 kernel configs (one file per weight shape and GPU name). vLLM picks a
# file by the GPU's name, so these only take effect on the card they were tuned on.
cp configs/triton-fp8/*.json "$SITE/vllm/model_executor/layers/quantization/utils/configs/"
echo "installed        $(ls configs/triton-fp8/*.json | wc -l) Triton FP8 kernel configs"

# The shim makes vLLM's compiled kernels dispatch on each worker's own GPU (see docs/how-it-works.md).
g++ -shared -fPIC -O2 -o shim/smfix.so shim/smfix.cpp -ldl
echo "built            shim/smfix.so"

echo
echo "Done. vLLM binary: $(cd "$VENV" && pwd)/bin/vllm"
echo "Next: download the models (README step 2), then run a launcher from launchers/."
