// LD_PRELOAD shim for vLLM 0.30's compiled extension.
//
// csrc/libtorch_stable/cutlass_extensions/common.cpp defines
//     int32_t get_sm_version_num()
// as the compute capability of CUDA device 0, and every CUTLASS entry point
// (cutlass_scaled_mm, cutlass_scaled_fp4_mm, the nvfp4 quant guards, MoE mm)
// dispatches on it. Under pipeline parallel with a mixed pair the rank on the
// second GPU therefore dispatches as if it were the first GPU (here: the 5090
// rank sees the 4070's SM 89 and either refuses the op or runs Ada kernels).
//
// This shim redefines the symbol to use the calling thread's current CUDA
// device, which every vLLM worker sets to its own GPU. The symbol is a global
// default-visibility function called through the PLT, so LD_PRELOAD wins.
// cudaGetDevice / cudaDeviceGetAttribute are resolved at call time from the
// libcudart already loaded by torch, so no CUDA library is linked here.
//
// Build:  g++ -shared -fPIC -O2 -o smfix.so smfix.cpp -ldl
// Use:    LD_PRELOAD=/path/smfix.so vllm serve ...

#include <dlfcn.h>
#include <stdint.h>

typedef int (*cudaGetDevice_t)(int*);
typedef int (*cudaDeviceGetAttribute_t)(int*, int, int);

// cudaDevAttrComputeCapabilityMajor = 75, cudaDevAttrComputeCapabilityMinor = 76
static const int kAttrMajor = 75;
static const int kAttrMinor = 76;

int32_t get_sm_version_num() {
  static cudaGetDevice_t getDevice = nullptr;
  static cudaDeviceGetAttribute_t getAttr = nullptr;
  if (!getDevice) getDevice = (cudaGetDevice_t)dlsym(RTLD_DEFAULT, "cudaGetDevice");
  if (!getAttr) getAttr = (cudaDeviceGetAttribute_t)dlsym(RTLD_DEFAULT, "cudaDeviceGetAttribute");
  int device = 0;
  if (getDevice) getDevice(&device);
  int major = 0, minor = 0;
  if (getAttr) {
    getAttr(&major, kAttrMajor, device);
    getAttr(&minor, kAttrMinor, device);
  }
  return major * 10 + minor;
}
