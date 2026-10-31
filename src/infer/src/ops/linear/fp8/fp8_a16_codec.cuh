#pragma once

// Exact E4M3-to-BF16 operand widening shared by the small-T and GEMM A16 routes.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

union Fp8A16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned fp8_e4m3x2_to_bf16x2_bits(unsigned packed) {
    __nv_fp8x2_e4m3 fp8;
    fp8.__x                 = static_cast<std::uint16_t>(packed);
    const __half2 half_pair = static_cast<__half2>(fp8);
    Fp8A16PairBits result;
    // AMD port: half->bf16 via float (HIP ctor ambiguity)
    result.pair = __halves2bfloat162(
        __float2bfloat16(__low2float(half_pair)), __float2bfloat16(__high2float(half_pair)));
    return result.bits;
}

__device__ __forceinline__ int fp8_a16_shared_col_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

} // namespace ninfer::ops::detail
