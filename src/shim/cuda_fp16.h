#pragma once
#include <msvc_host_compat.h>
// NInfer -> HIP port shim: cuda_fp16.h wrapper
#include <hip/hip_fp16.h>

// AMD port: ROCm's amd_hip_fp16.h only defines the fp16 bitcast intrinsics under
// #if defined(__clang__) && defined(__HIP__). The MSVC host pass (cl.exe, ternary
// storage headers included from host TUs) therefore lacks __ushort_as_half. The shim
// typedefs _Float16 as float, so __half carries a float value; produce the exact
// half->float value by hand and route it through __float2half.
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__CUDACC__)
#include <cstring>

inline float __shim_half_bits_to_float(unsigned short h) {
    const unsigned sign = static_cast<unsigned>(h & 0x8000u) << 16;
    const unsigned exp  = static_cast<unsigned>(h & 0x7c00u) >> 10;
    const unsigned man  = static_cast<unsigned>(h & 0x03ffu);
    unsigned bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            unsigned m = man;
            do {
                m <<= 1;
                ++e;
            } while ((m & 0x400u) == 0);
            bits = sign | (static_cast<unsigned>(127 - 15 - e) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

inline __half __ushort_as_half(unsigned short x) {
    return __float2half(__shim_half_bits_to_float(x));
}
#endif
