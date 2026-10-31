#pragma once
#include <msvc_host_compat.h>
// NInfer -> HIP port shim: cuda_fp4.h (no direct HIP equivalent)
// e2m1 (NVFP4) software decode/encode. Used by the nvfp4 codec + KV cache
// lanes; on RDNA there is no fp4 hardware MMA, so this is data-path only.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// e2m1 lookup: 3-bit magnitude code -> value {0, 0.5, 1, 1.5, 2, 3, 4, 6}
__device__ __forceinline__ float __e2m1_lut(unsigned code) {
    constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return lut[code & 7];
}

typedef unsigned char __nv_fp4x2_storage_t;

struct __nv_fp4x2_e2m1 {
    __nv_fp4x2_storage_t __x;
    __nv_fp4x2_e2m1() = default;

    // two e2m1 values packed in one byte: low nibble = first, high = second
    __device__ __forceinline__ operator float2() const {
        const unsigned lo = __x & 0xFu;
        const unsigned hi = (__x >> 4) & 0xFu;
        float2 out;
        out.x = ((lo & 8u) ? -1.0f : 1.0f) * __e2m1_lut(lo & 7u);
        out.y = ((hi & 8u) ? -1.0f : 1.0f) * __e2m1_lut(hi & 7u);
        return out;
    }
    __device__ __forceinline__ operator __half2() const {
        const float2 f = static_cast<float2>(*this);
        return __floats2half2_rn(f.x, f.y);
    }
};

// round-to-nearest-even encode of two floats into e2m1x2
__device__ __forceinline__ unsigned char __e2m1_encode1(float v) {
    const unsigned char s = v < 0.0f ? 8u : 0u;
    const float a = v < 0.0f ? -v : v;
    unsigned char m;
    if (a < 0.25f)      m = 0;  // 0
    else if (a < 0.75f) m = 1;  // 0.5
    else if (a < 1.25f) m = 2;  // 1
    else if (a < 1.75f) m = 3;  // 1.5
    else if (a < 2.5f)  m = 4;  // 2
    else if (a < 3.5f)  m = 5;  // 3
    else if (a < 5.0f)  m = 6;  // 4
    else                m = 7;  // 6
    return s | m;
}

__device__ __forceinline__ __nv_fp4x2_storage_t __nv_cvt_float2_to_fp4x2(float2 f, const int = 0,
                                                                          const int = 0) {
    return static_cast<__nv_fp4x2_storage_t>(__e2m1_encode1(f.x) | (__e2m1_encode1(f.y) << 4));
}
