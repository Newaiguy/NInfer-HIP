#pragma once
#include <msvc_host_compat.h>
// NInfer -> HIP port shim: cuda_bf16.h wrapper
// Bypass hip/hip_bf16.h (its nvidia_detail branch expects a real CUDA SDK
// install) and bind directly to the AMD implementation header.
//
// Pure-host MSVC .cpp builds only pass bf16 types through (pointer casts into
// kernel launches); amd_hip_bf16.h cannot be parsed by MSVC (clang builtins
// __bf16/__bf16_2, device math overloads), so provide minimal 2/4-byte types
// with identical layout. Device compiles (clang) keep the real header.
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__HIPCC__)
struct __shim_nv_bf16 {
    unsigned short __x;
    constexpr __shim_nv_bf16() = default;
};
struct __shim_nv_bf162 {
    __shim_nv_bf16 x, y;
};
using __nv_bfloat16  = __shim_nv_bf16;
using __nv_bfloat162 = __shim_nv_bf162;
#else
#include <hip/amd_detail/amd_hip_bf16.h>

using __nv_bfloat16  = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;

// Intrinsic aliases (CUDA name -> HIP name; _rn is HIP's default rounding)
inline __device__ __hip_bfloat16 __float2bfloat16_rn(float x) { return __float2bfloat16(x); }
inline __device__ __hip_bfloat162 __floats2bfloat162_rn(float x, float y) { return __float22bfloat162_rn(make_float2(x, y)); }
#endif
