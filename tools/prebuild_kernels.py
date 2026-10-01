"""Build the FlashInfer JIT kernels the big card needs for NVFP4 (FP4 quantization, CUTLASS and
b12x FP4 GEMM) before the first server boot, outside vLLM. Run it through prebuild_kernels.sh,
which sets the same environment as the launchers.

Never run FlashInfer's JIT without MAX_JOBS: an unclamped build starts one compiler per core at
about 1.5 GiB each and can take the whole machine down.
"""
import os
import sys

import torch
from torch import nn

assert os.environ.get("MAX_JOBS"), "run this through tools/prebuild_kernels.sh (MAX_JOBS must be set)"
dev = torch.cuda.device_count() - 1  # the last pipeline rank = the big card
torch.cuda.set_device(dev)
print("device", torch.cuda.get_device_name(dev), flush=True)

import vllm.model_executor.kernels.linear as L  # noqa: E402


def make_layer(n, k):
    layer = nn.Module()
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.params_dtype = torch.bfloat16
    layer.weight = nn.Parameter(torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device="cuda"), requires_grad=False)
    layer.weight_scale = nn.Parameter((torch.rand(n, k // 16, device="cuda") * 0.5 + 0.5).to(torch.float8_e4m3fn), requires_grad=False)
    wgs = torch.tensor(1.0 / 400.0, dtype=torch.float32, device="cuda")
    igs = torch.tensor(1.0 / 6.0, dtype=torch.float32, device="cuda")
    layer.weight_global_scale = nn.Parameter(1.0 / wgs, requires_grad=False)
    layer.input_global_scale = nn.Parameter(1.0 / igs, requires_grad=False)
    layer.input_global_scale_inv = nn.Parameter(igs, requires_grad=False)
    layer.alpha = nn.Parameter(layer.input_global_scale * layer.weight_global_scale, requires_grad=False)
    return layer


names = sys.argv[1:] or ["FlashInferB12xNvFp4LinearKernel", "FlashInferCutlassNvFp4LinearKernel"]
for name in names:
    kernel = getattr(L, name)(L.NvFp4LinearLayerConfig())
    layer = make_layer(5120, 5120)
    kernel.process_weights_after_loading(layer)
    for m in (8, 1024):
        kernel.apply_weights(layer, torch.randn(m, 5120, dtype=torch.bfloat16, device="cuda"), None)
    torch.cuda.synchronize()
    print(name, "built and ran OK", flush=True)
