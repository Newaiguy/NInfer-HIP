#pragma once
#include <msvc_host_compat.h>
// NInfer -> HIP port shim: cuda_fp8.h
#include <hip/hip_fp8.h>

using __nv_fp8_e4m3  = __hip_fp8_e4m3;
using __nv_fp8_e5m2  = __hip_fp8_e5m2;
using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3;
using __nv_fp8x2_e5m2 = __hip_fp8x2_e5m2;
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3;
using __nv_fp8x4_e5m2 = __hip_fp8x4_e5m2;
using __nv_fp8_storage_t   = __hip_fp8_storage_t;
using __nv_fp8x2_storage_t = __hip_fp8x2_storage_t;
using __nv_fp8x4_storage_t = __hip_fp8x4_storage_t;

// cvt helpers (CUDA signature: (float, satflag, interp))
inline __device__ __nv_fp8_e4m3 __nv_cvt_float_to_fp8(float x, const int, const int) {
    return __nv_fp8_e4m3(x);
}
inline __device__ __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(float2 f, const int, const int) {
    return __hip_fp8x2_e4m3(f).__x;
}
