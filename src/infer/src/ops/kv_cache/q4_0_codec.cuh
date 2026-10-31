#pragma once

// ggml q4_0 KV codec (llama.cpp-compatible block layout):
//   block = 32 values: fp16 scale d + 16 bytes 4-bit codes (byte b = {value b low
//   nibble, value b+16 high nibble}). Quant: d = smax / -8 where smax is the signed
//   value with max |.|; code = min(15, (int)(x*id + 8.5f)); dequant x = (q - 8) * d.
//   Per 256-d head row: 128 code bytes + 8 scales. Page-major plane indexing mirrors
//   paged_kv_element_offset.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kQ4_0Block = 32;
inline constexpr int kQ4_0CodesPerBlock = 16;
inline constexpr int kQ4_0CodesPerRow = 128;  // 256 d / 2

template <int HeadExtent>
__device__ __forceinline__ std::int64_t q4_0_code_offset(std::int32_t physical_page,
                                                         std::int32_t head,
                                                         std::int32_t page_offset,
                                                         std::int32_t d) {
    return static_cast<std::int64_t>(d >> 1) +
           static_cast<std::int64_t>(kQ4_0CodesPerRow) *
               (static_cast<std::int64_t>(page_offset) +
                static_cast<std::int64_t>(kPagedKVPageSize) *
                    (static_cast<std::int64_t>(head) +
                     static_cast<std::int64_t>(HeadExtent) * physical_page));
}

template <int HeadExtent>
__device__ __forceinline__ std::int64_t q4_0_scale_offset(std::int32_t physical_page,
                                                          std::int32_t head,
                                                          std::int32_t page_offset,
                                                          std::int32_t d) {
    return static_cast<std::int64_t>(d >> 5) +
           static_cast<std::int64_t>(kQ4_0CodesPerRow / kQ4_0CodesPerBlock) *
               (static_cast<std::int64_t>(page_offset) +
                static_cast<std::int64_t>(kPagedKVPageSize) *
                    (static_cast<std::int64_t>(head) +
                     static_cast<std::int64_t>(HeadExtent) * physical_page));
}

// Dequantize 8 consecutive values (one aligned window: d0 % 32 in {0,8,16,24}, so
// the 8 nibbles sit in 8 contiguous code bytes starting at byte d0&15). Matches
// dequantize_row_q4_0: byte b = {value b low, value b+16 high}; x = (q - 8) * d.
__device__ __forceinline__ void q4_0_dequant8(float out[8], std::uint64_t codes8,
                                              bool low_nibble, __half scale) {
    const float d = __half2float(scale);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int byte = static_cast<int>((codes8 >> (8 * i)) & 0xFFu);
        const int nib  = low_nibble ? (byte & 0xF) : (byte >> 4);
        out[i]         = static_cast<float>(nib - 8) * d;
    }
}

// Quantize one 32-value block with one warp (lane = value j, llama layout: byte
// b = {value b low, value b+16 high}). ALL 32 lanes execute the shfl (converged;
// FullMask requires every lane), lanes 0..15 write codes[b] = low nibble of lane b
// | high nibble of lane b+16. scale is stored as fp16 d = smax / -8.
__device__ __forceinline__ void q4_0_block_encode(float val, int lane,
                                                  std::uint8_t* __restrict__ codes,
                                                  __half& scale) {
    constexpr unsigned FullMask = 0xffffffffu;
    const float av              = fabsf(val);
    const float amax            = warp_max(av, FullMask);
    // smax = the SIGNED value whose |.| == amax. Match on |val|, not val: if the
    // block max is negative, val == amax is never true and smax would collapse to
    // -INF (scale -> +INF -> NaN dequant).
    const float smax = warp_max(fabsf(val) == amax ? val : -CUDART_INF_F, FullMask);
    const float d    = smax / -8.0f;
    const float id   = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0) { scale = __float2half_rn(d); }
    const int q        = static_cast<int>(val * id + 8.5f);
    const std::uint8_t nib = static_cast<std::uint8_t>(q > 15 ? 15 : (q < 0 ? 0 : q));
    // all lanes participate in the shuffle; only lanes 0..15 store
    const int hi_val = __shfl_sync(FullMask, static_cast<int>(nib), (lane + 16) & 31);
    if (lane < 16) {
        const std::uint8_t hi = static_cast<std::uint8_t>(hi_val & 0xF);
        codes[lane]           = static_cast<std::uint8_t>(nib | static_cast<std::uint8_t>(hi << 4));
    }
}

} // namespace ninfer::ops
