// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#pragma once

// Decode-shaped ternary GEMV (T == 1) for PQ2_0: one warp owns one output row.
//
// Why: the reference kernel in ternary_rowsplit_gemm.cuh gives each output row a full 128-thread
// CTA and seven block-wide barriers. That is fine for correctness but leaves decode far from the
// card's measured bandwidth. Here each lane covers four consecutive weights of every 128-group, so
// a warp reads exactly one 32-byte code span plus one 2-byte scale per group and reduces through
// shuffles with no __syncthreads at all.
//
// PQ2_0 packing makes the mapping exact rather than approximate: a group is 32 bytes holding four
// two-bit codes each, so for lane l the four columns 4l..4l+3 live entirely in byte l. Lane l
// loads one byte, decodes four weights, and consumes four bf16 activations.
//
// Occupancy, not instruction count, is what this kernel is tuned for: a GEMV needs many warps in
// flight to cover DRAM latency. A two-rows-per-warp variant (which reuses the activation loads)
// measured SLOWER (32.8 vs 46.4 t/s) because the extra accumulators and row bases cost registers
// and dropped the resident warp count. So: one row per warp, and only the micro-optimisations that
// remove instructions without adding state -- one 16-bit scale load, paired bf16 activation
// loads, and the per-group scale multiply hoisted out of the four FMAs.
//
// LAYOUT: ninfer/ggml put ne[0] on the contiguous axis, so a [k, 1] activation keeps element
// (column, 0) at column, and a one-token output row is simply out[row].

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Warps per block for the GEMV below.
inline constexpr int kGemvWarpsPerBlock = 8;

inline constexpr int kGemvCodeBytesPerGroup  = 32;
inline constexpr int kGemvScaleBytesPerGroup = 2;
inline constexpr int kGemvGroupK             = 128;

__device__ __forceinline__ float gemv_scale(const std::uint8_t* scale_ptr) {
    // One 16-bit load instead of two byte loads plus a shift/or; the scale plane is 2-byte aligned.
    const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(scale_ptr);
    return __half2float(__ushort_as_half(bits));
}

// iamwavecut v3 T2 codec: field bit0 selects the value (0 -> off, 1 -> on), bit1 flips the sign.
//   field 0 (00) -> 0, 1 (01) -> +1, 2 (10) -> 0, 3 (11) -> -1
// Arithmetic form (no branches): magnitude = field & 1, sign = 1 - 2*((field>>1)&1).
__device__ __forceinline__ float t2_field(std::uint32_t field) {
    return static_cast<float>(static_cast<int>(field & 0x1u) *
                              (1 - 2 * static_cast<int>((field >> 1) & 0x1u)));
}

__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                             const std::uint8_t* __restrict__ codes,
                             const std::uint8_t* __restrict__ scales,
                             __nv_bfloat16* __restrict__ out, std::int32_t rows,
                             std::int32_t groups_per_row, std::int32_t t2_codec) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator = 0.0f;

    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low        = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base));
        const float2 high = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(x + base + 2));

        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float dot = t2_codec
            ? fmaf(t2_field(raw & 3u), low.x,
                   fmaf(t2_field((raw >> 2) & 3u), low.y,
                        fmaf(t2_field((raw >> 4) & 3u), high.x,
                             t2_field((raw >> 6) & 3u) * high.y)))
            : fmaf(static_cast<float>(static_cast<int>(raw & 3u) - 1), low.x,
                   fmaf(static_cast<float>(static_cast<int>((raw >> 2) & 3u) - 1),
                        low.y,
                        fmaf(static_cast<float>(static_cast<int>((raw >> 4) & 3u) - 1),
                             high.x,
                             static_cast<float>(static_cast<int>((raw >> 6) & 3u) - 1) *
                                 high.y)));
        accumulator = fmaf(gemv_scale(scale_row + group * kGemvScaleBytesPerGroup), dot,
                           accumulator);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
    }
    if (lane == 0) { out[warp] = __float2bfloat16_rn(accumulator); }
}

// Small-token-tile variant, for the speculative VERIFY pass (T = draft + 1, i.e. 2..4).
//
// A per-token GEMV would re-read every weight for every token, and weights are exactly what the
// decode path is bound by, so a verify round would cost T full decode passes and speculation could
// never pay for itself (measured: 15.7 t/s with MTP N=2 against 62.1 t/s without). This kernel
// keeps the weights single-read: the code byte and scale are loaded once per group and reused for
// all kT tokens, while each token contributes its own four activations.
template <int kT>
__global__ __launch_bounds__(kGemvWarpsPerBlock * 32)
void ternary_pq2_gemv_tile_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t groups_per_row, std::int32_t tokens,
                                  std::int32_t out_row_stride, std::int32_t t2_codec) {
    static_assert(kT >= 1 && kT <= 8, "tile size must be small enough to keep accumulators in registers");
    // Byte -> 4 decoded weights, built once per CTA before any early return so a
    // partially-populated block cannot miss the barrier. T2/PQ2 selected at runtime.
    __shared__ __align__(16) float4 t2_lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += kGemvWarpsPerBlock * 32) {
        float w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int field = (i >> (2 * j)) & 3;
            w[j] = t2_codec ? t2_field(static_cast<std::uint32_t>(field))
                            : static_cast<float>(field - 1);
        }
        t2_lut[i] = make_float4(w[0], w[1], w[2], w[3]);
    }
    __syncthreads();
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kGemvWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (warp >= rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(warp) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(warp) * groups_per_row * kGemvScaleBytesPerGroup;

    float accumulator[kT];
#pragma unroll
    for (int t = 0; t < kT; ++t) { accumulator[t] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        // Weights: one byte of codes + one 16-bit scale, reused across every token in the tile.
        const std::uint8_t raw = code_row[group * kGemvCodeBytesPerGroup + lane];
        const float scale = gemv_scale(scale_row + group * kGemvScaleBytesPerGroup);
        const float4 wv  = t2_lut[raw];
        const float weight0 = wv.x;
        const float weight1 = wv.y;
        const float weight2 = wv.z;
        const float weight3 = wv.w;

        const std::int32_t base = group * kGemvGroupK + lane * 4;
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t < tokens) {
                const __nv_bfloat16* x_token =
                    x + static_cast<std::int64_t>(t) * groups_per_row * kGemvGroupK;
                const float2 low = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base));
                const float2 high = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(x_token + base + 2));
                const float dot = fmaf(weight0, low.x,
                                       fmaf(weight1, low.y, fmaf(weight2, high.x, weight3 * high.y)));
                accumulator[t] = fmaf(scale, dot, accumulator[t]);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float value = accumulator[t];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0 && t < tokens) {
            // Token-major output: element (row, token) lives at token * out_row_stride + row.
            out[static_cast<std::int64_t>(t) * out_row_stride + warp] =
                __float2bfloat16_rn(value);
        }
    }
}


// ---------------------------------------------------------------------------
// Shared-memory-activation GEMV (AMD port, T == 1): the one-row-per-warp kernel
// above re-reads the whole activation from L2 once PER ROW (N rows x K bytes),
// which at N=5120, K=6144 is ~61 MB of L2 traffic against 8.5 MB of weights and
// caps the kernel at ~45% of DRAM bandwidth. Here a 256-thread CTA stages the
// activation into shared memory ONCE and computes 16 output rows (2 per warp):
// DRAM traffic collapses to the weight stream plus one activation read per CTA.
//
// Dynamic shared memory = K * 2 bytes (34 KB at K = 17408, under the 48 KB cap).
// ---------------------------------------------------------------------------
inline constexpr int kGemvSmemRowsPerCta  = 16;
inline constexpr int kGemvSmemThreads     = 256;

template <bool T2Codec>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t groups_per_row) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[];  // [K]
    __shared__ __align__(16) float4 t2_lut[256];  // byte -> 4 decoded weights (T2/PQ2 by template)

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int k    = groups_per_row * kGemvGroupK;

    // Build the 256-entry decode LUT once per CTA: byte holds four 2-bit fields.
    // T2Codec selects the iamwavecut bit0/bit1 table; otherwise the GGUF (c-1) table.
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += kGemvSmemThreads) {
        float w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int field = (i >> (2 * j)) & 3;
            w[j] = T2Codec ? t2_field(static_cast<std::uint32_t>(field))
                           : static_cast<float>(field - 1);
        }
        t2_lut[i] = make_float4(w[0], w[1], w[2], w[3]);
    }

    // Stage the activation once per CTA (16B chunks).
    for (int chunk = static_cast<int>(threadIdx.x); chunk < k / 8;
         chunk += kGemvSmemThreads) {
        *reinterpret_cast<uint4*>(&act_sh[chunk * 8]) =
            *reinterpret_cast<const uint4*>(&x[chunk * 8]);
    }
    __syncthreads();

    // Two rows per warp; rows are strided so consecutive warps cover consecutive pairs.
    const int row_pairs = rows / 2;
    const int pair0 =
        static_cast<int>(blockIdx.x) * (kGemvSmemThreads / 32) + warp;
    if (pair0 >= row_pairs) { return; }
    const int row_a = pair0 * 2;
    const int row_b = row_a + 1;

    const std::uint8_t* code_row_a =
        codes + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* code_row_b =
        codes + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row_a =
        scales + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::uint8_t* scale_row_b =
        scales + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvScaleBytesPerGroup;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
        const float2 high = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));

        const std::uint8_t raw_a = code_row_a[group * kGemvCodeBytesPerGroup + lane];
        const std::uint8_t raw_b = code_row_b[group * kGemvCodeBytesPerGroup + lane];
        const float4 wa = t2_lut[raw_a];
        const float4 wb = t2_lut[raw_b];
        const float dot_a = fmaf(wa.x, low.x,
                                 fmaf(wa.y, low.y, fmaf(wa.z, high.x, wa.w * high.y)));
        const float dot_b = fmaf(wb.x, low.x,
                                 fmaf(wb.y, low.y, fmaf(wb.z, high.x, wb.w * high.y)));
        acc_a = fmaf(gemv_scale(scale_row_a + group * kGemvScaleBytesPerGroup), dot_a, acc_a);
        acc_b = fmaf(gemv_scale(scale_row_b + group * kGemvScaleBytesPerGroup), dot_b, acc_b);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xffffffffu, acc_a, offset);
        acc_b += __shfl_down_sync(0xffffffffu, acc_b, offset);
    }
    if (lane == 0) {
        out[row_a] = __float2bfloat16_rn(acc_a);
        out[row_b] = __float2bfloat16_rn(acc_b);
    }
}

// Four-groups-per-iteration variant with 4-byte-per-lane COALESCED weight reads:
// per iteration the warp reads one 128B block (4 groups) as 32 x uint32, so the
// DRAM/L2 line utilization jumps from 32B/warp (1B/lane) to a full 128B/warp.
// Each lane's 4 code bytes lie in the group (lane>>5) of the 4-group block and
// map to 16 consecutive weights; the act column is read from smem accordingly.
template <bool T2Codec>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_v4x4_kernel(const __nv_bfloat16* __restrict__ x,
                                       const std::uint8_t* __restrict__ codes,
                                       const std::uint8_t* __restrict__ scales,
                                       __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                       std::int32_t groups_per_row) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[];  // [K]
    __shared__ __align__(16) float4 t2_lut[256];  // byte -> 4 decoded weights

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int k    = groups_per_row * kGemvGroupK;

    for (int i = static_cast<int>(threadIdx.x); i < 256; i += kGemvSmemThreads) {
        float w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int field = (i >> (2 * j)) & 3;
            w[j] = T2Codec ? t2_field(static_cast<std::uint32_t>(field))
                           : static_cast<float>(field - 1);
        }
        t2_lut[i] = make_float4(w[0], w[1], w[2], w[3]);
    }
    for (int chunk = static_cast<int>(threadIdx.x); chunk < k / 8;
         chunk += kGemvSmemThreads) {
        *reinterpret_cast<uint4*>(&act_sh[chunk * 8]) =
            *reinterpret_cast<const uint4*>(&x[chunk * 8]);
    }
    __syncthreads();

    const int row_pairs = rows / 2;
    const int pair0 =
        static_cast<int>(blockIdx.x) * (kGemvSmemThreads / 32) + warp;
    if (pair0 >= row_pairs) { return; }
    const int row_a = pair0 * 2;
    const int row_b = row_a + 1;

    const std::uint8_t* code_row_a =
        codes + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* code_row_b =
        codes + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row_a =
        scales + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::uint8_t* scale_row_b =
        scales + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvScaleBytesPerGroup;

    // Per 4-group block: lane l reads bytes [4l, 4l+4) of the 128B block; 8 lanes
    // cover one group's 32 code bytes, so the block-internal group is lane/8 and
    // the within-group byte column is (lane%8)*4.
    const int gg     = lane >> 3;      // group index within the 4-block (0..3)
    const int within = (lane & 7) * 4; // byte offset inside the group
    const int act_col = within + lane * 12;  // unused placeholder (kept minimal)

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    const int even = groups_per_row & ~3;
    for (int g4 = 0; g4 < even; g4 += 4) {
        const std::uint32_t wa = *reinterpret_cast<const std::uint32_t*>(
            code_row_a + g4 * kGemvCodeBytesPerGroup + lane * 4);
        const std::uint32_t wb = *reinterpret_cast<const std::uint32_t*>(
            code_row_b + g4 * kGemvCodeBytesPerGroup + lane * 4);
        // 16 activation values for this lane's 16 weights (group >= g4+gg).
        const int base = (g4 + gg) * kGemvGroupK + within * 4;
        float2 ax[8];
#pragma unroll
        for (int u = 0; u < 8; ++u)
            ax[u] = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2 * u]));
        const float sa = gemv_scale(scale_row_a + (g4 + gg) * kGemvScaleBytesPerGroup);
        const float sb = gemv_scale(scale_row_b + (g4 + gg) * kGemvScaleBytesPerGroup);
        float da = 0.0f, db = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 qa = t2_lut[(wa >> (8 * j)) & 0xffu];
            const float4 qb = t2_lut[(wb >> (8 * j)) & 0xffu];
            // Byte j of the uint32 holds weights [4j,4j+4) -> act offset 4j.
            const float* a4 = &ax[2 * j].x;
            da += qa.x * a4[0] + qa.y * a4[1] + qa.z * a4[2] + qa.w * a4[3];
            db += qb.x * a4[0] + qb.y * a4[1] + qb.z * a4[2] + qb.w * a4[3];
        }
        acc_a = fmaf(sa, da, acc_a);
        acc_b = fmaf(sb, db, acc_b);
    }
    if (even != groups_per_row) {  // tail: single groups, 1-byte reads
        for (int group = even; group < groups_per_row; ++group) {
            const std::int32_t base = group * kGemvGroupK + lane * 4;
            const float2 low = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
            const float2 high = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
            const std::uint8_t raw_a = code_row_a[group * kGemvCodeBytesPerGroup + lane];
            const std::uint8_t raw_b = code_row_b[group * kGemvCodeBytesPerGroup + lane];
            const float4 wa = t2_lut[raw_a];
            const float4 wb = t2_lut[raw_b];
            acc_a = fmaf(gemv_scale(scale_row_a + group * kGemvScaleBytesPerGroup),
                         fmaf(wa.x, low.x, fmaf(wa.y, low.y, fmaf(wa.z, high.x, wa.w * high.y))),
                         acc_a);
            acc_b = fmaf(gemv_scale(scale_row_b + group * kGemvScaleBytesPerGroup),
                         fmaf(wb.x, low.x, fmaf(wb.y, low.y, fmaf(wb.z, high.x, wb.w * high.y))),
                         acc_b);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xffffffffu, acc_a, offset);
        acc_b += __shfl_down_sync(0xffffffffu, acc_b, offset);
    }
    if (lane == 0) {
        out[row_a] = __float2bfloat16_rn(acc_a);
        out[row_b] = __float2bfloat16_rn(acc_b);
    }
}

// Two-groups-per-iteration variant of the shared-memory GEMV above: every loop
// body serves two consecutive 128-groups (8 weights per row) to halve the index
// arithmetic and give the LUT lookups twice the independent work to overlap.
// A/B switch: NINFER_GEMV_DUAL=1 in launch_ternary_gemm_t1. Identical numerics
// (same FMA order per group, just two groups in flight).
template <bool T2Codec>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_dual_kernel(const __nv_bfloat16* __restrict__ x,
                                       const std::uint8_t* __restrict__ codes,
                                       const std::uint8_t* __restrict__ scales,
                                       __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                       std::int32_t groups_per_row) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[];  // [K]
    __shared__ __align__(16) float4 t2_lut[256];  // byte -> 4 decoded weights (T2/PQ2 by template)

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int k    = groups_per_row * kGemvGroupK;

    for (int i = static_cast<int>(threadIdx.x); i < 256; i += kGemvSmemThreads) {
        float w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int field = (i >> (2 * j)) & 3;
            w[j] = T2Codec ? t2_field(static_cast<std::uint32_t>(field))
                           : static_cast<float>(field - 1);
        }
        t2_lut[i] = make_float4(w[0], w[1], w[2], w[3]);
    }

    for (int chunk = static_cast<int>(threadIdx.x); chunk < k / 8;
         chunk += kGemvSmemThreads) {
        *reinterpret_cast<uint4*>(&act_sh[chunk * 8]) =
            *reinterpret_cast<const uint4*>(&x[chunk * 8]);
    }
    __syncthreads();

    const int row_pairs = rows / 2;
    const int pair0 =
        static_cast<int>(blockIdx.x) * (kGemvSmemThreads / 32) + warp;
    if (pair0 >= row_pairs) { return; }
    const int row_a = pair0 * 2;
    const int row_b = row_a + 1;

    const std::uint8_t* code_row_a =
        codes + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* code_row_b =
        codes + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row_a =
        scales + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::uint8_t* scale_row_b =
        scales + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvScaleBytesPerGroup;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    const int even = groups_per_row & ~1;
    for (int group = 0; group < even; group += 2) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
        const float2 high = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
        const float2 low2  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + kGemvGroupK]));
        const float2 high2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + kGemvGroupK + 2]));

        const std::uint8_t raw_a0 = code_row_a[group * kGemvCodeBytesPerGroup + lane];
        const std::uint8_t raw_b0 = code_row_b[group * kGemvCodeBytesPerGroup + lane];
        const std::uint8_t raw_a1 = code_row_a[(group + 1) * kGemvCodeBytesPerGroup + lane];
        const std::uint8_t raw_b1 = code_row_b[(group + 1) * kGemvCodeBytesPerGroup + lane];
        const float4 wa0 = t2_lut[raw_a0];
        const float4 wb0 = t2_lut[raw_b0];
        const float4 wa1 = t2_lut[raw_a1];
        const float4 wb1 = t2_lut[raw_b1];

        acc_a = fmaf(gemv_scale(scale_row_a + group * kGemvScaleBytesPerGroup),
                     fmaf(wa0.x, low.x, fmaf(wa0.y, low.y, fmaf(wa0.z, high.x, wa0.w * high.y))),
                     acc_a);
        acc_b = fmaf(gemv_scale(scale_row_b + group * kGemvScaleBytesPerGroup),
                     fmaf(wb0.x, low.x, fmaf(wb0.y, low.y, fmaf(wb0.z, high.x, wb0.w * high.y))),
                     acc_b);
        acc_a = fmaf(gemv_scale(scale_row_a + (group + 1) * kGemvScaleBytesPerGroup),
                     fmaf(wa1.x, low2.x, fmaf(wa1.y, low2.y, fmaf(wa1.z, high2.x, wa1.w * high2.y))),
                     acc_a);
        acc_b = fmaf(gemv_scale(scale_row_b + (group + 1) * kGemvScaleBytesPerGroup),
                     fmaf(wb1.x, low2.x, fmaf(wb1.y, low2.y, fmaf(wb1.z, high2.x, wb1.w * high2.y))),
                     acc_b);
    }

    if (even != groups_per_row) {  // odd tail: one single group
        const std::int32_t base = even * kGemvGroupK + lane * 4;
        const float2 low  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
        const float2 high = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
        const std::uint8_t raw_a = code_row_a[even * kGemvCodeBytesPerGroup + lane];
        const std::uint8_t raw_b = code_row_b[even * kGemvCodeBytesPerGroup + lane];
        const float4 wa = t2_lut[raw_a];
        const float4 wb = t2_lut[raw_b];
        const float dot_a = fmaf(wa.x, low.x, fmaf(wa.y, low.y, fmaf(wa.z, high.x, wa.w * high.y)));
        const float dot_b = fmaf(wb.x, low.x, fmaf(wb.y, low.y, fmaf(wb.z, high.x, wb.w * high.y)));
        acc_a = fmaf(gemv_scale(scale_row_a + even * kGemvScaleBytesPerGroup), dot_a, acc_a);
        acc_b = fmaf(gemv_scale(scale_row_b + even * kGemvScaleBytesPerGroup), dot_b, acc_b);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xffffffffu, acc_a, offset);
        acc_b += __shfl_down_sync(0xffffffffu, acc_b, offset);
    }
    if (lane == 0) {
        out[row_a] = __float2bfloat16_rn(acc_a);
        out[row_b] = __float2bfloat16_rn(acc_b);
    }
}

// Four-groups-per-iteration variant of the shared-memory GEMV (A/B: NINFER_GEMV_V4=1).
// MEASURED standalone (probe/bench_gemv_variants.cu, gfx1201, isolated kernels):
//   gate_up 34816x5120  0.092 -> 0.084 ms (518 -> 563 GB/s)
//   down    5120x17408  0.063 -> 0.052 ms (379 -> 460 GB/s)
// The unrolled body gives the long-K loops (136 groups on down) four independent
// weight/scale streams, and reuses the LUT lookups' ILP. Identical FMA order per
// group, so numerics match the one-group kernel group-for-group.
template <bool T2Codec>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_v4_kernel(const __nv_bfloat16* __restrict__ x,
                                     const std::uint8_t* __restrict__ codes,
                                     const std::uint8_t* __restrict__ scales,
                                     __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                     std::int32_t groups_per_row) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[];  // [K]
    __shared__ __align__(16) float4 t2_lut[256];  // byte -> 4 decoded weights

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int k    = groups_per_row * kGemvGroupK;

    for (int i = static_cast<int>(threadIdx.x); i < 256; i += kGemvSmemThreads) {
        float w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int field = (i >> (2 * j)) & 3;
            w[j] = T2Codec ? t2_field(static_cast<std::uint32_t>(field))
                           : static_cast<float>(field - 1);
        }
        t2_lut[i] = make_float4(w[0], w[1], w[2], w[3]);
    }

    for (int chunk = static_cast<int>(threadIdx.x); chunk < k / 8;
         chunk += kGemvSmemThreads) {
        *reinterpret_cast<uint4*>(&act_sh[chunk * 8]) =
            *reinterpret_cast<const uint4*>(&x[chunk * 8]);
    }
    __syncthreads();

    const int row_pairs = rows / 2;
    const int pair0 =
        static_cast<int>(blockIdx.x) * (kGemvSmemThreads / 32) + warp;
    if (pair0 >= row_pairs) { return; }
    const int row_a = pair0 * 2;
    const int row_b = row_a + 1;

    const std::uint8_t* code_row_a =
        codes + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* code_row_b =
        codes + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row_a =
        scales + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::uint8_t* scale_row_b =
        scales + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvScaleBytesPerGroup;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    const int even = groups_per_row & ~3;
    for (int group = 0; group < even; group += 4) {
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const std::int32_t g = group + sub;
            const std::int32_t base = g * kGemvGroupK + lane * 4;
            const float2 low = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
            const float2 high = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
            const std::uint8_t raw_a = code_row_a[g * kGemvCodeBytesPerGroup + lane];
            const std::uint8_t raw_b = code_row_b[g * kGemvCodeBytesPerGroup + lane];
            const float4 wa = t2_lut[raw_a];
            const float4 wb = t2_lut[raw_b];
            acc_a = fmaf(gemv_scale(scale_row_a + g * kGemvScaleBytesPerGroup),
                         fmaf(wa.x, low.x, fmaf(wa.y, low.y, fmaf(wa.z, high.x, wa.w * high.y))),
                         acc_a);
            acc_b = fmaf(gemv_scale(scale_row_b + g * kGemvScaleBytesPerGroup),
                         fmaf(wb.x, low.x, fmaf(wb.y, low.y, fmaf(wb.z, high.x, wb.w * high.y))),
                         acc_b);
        }
    }
    if (even != groups_per_row) {  // tail: 1..3 single groups
        for (int group = even; group < groups_per_row; ++group) {
            const std::int32_t base = group * kGemvGroupK + lane * 4;
            const float2 low = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
            const float2 high = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
            const std::uint8_t raw_a = code_row_a[group * kGemvCodeBytesPerGroup + lane];
            const std::uint8_t raw_b = code_row_b[group * kGemvCodeBytesPerGroup + lane];
            const float4 wa = t2_lut[raw_a];
            const float4 wb = t2_lut[raw_b];
            acc_a = fmaf(gemv_scale(scale_row_a + group * kGemvScaleBytesPerGroup),
                         fmaf(wa.x, low.x, fmaf(wa.y, low.y, fmaf(wa.z, high.x, wa.w * high.y))),
                         acc_a);
            acc_b = fmaf(gemv_scale(scale_row_b + group * kGemvScaleBytesPerGroup),
                         fmaf(wb.x, low.x, fmaf(wb.y, low.y, fmaf(wb.z, high.x, wb.w * high.y))),
                         acc_b);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xffffffffu, acc_a, offset);
        acc_b += __shfl_down_sync(0xffffffffu, acc_b, offset);
    }
    if (lane == 0) {
        out[row_a] = __float2bfloat16_rn(acc_a);
        out[row_b] = __float2bfloat16_rn(acc_b);
    }
}

// v8: rows-per-warp template. Every prior SIMT variant reads the staged
// activation from smem once per group per warp; with R rows/warp that smem read
// is amortized R-fold (v4 = 2x, this = 4x/8x). Weight lines are still walked
// sequentially so the 128B line is reused across 4 consecutive groups from L1.
// Register decode (no smem LUT) keeps the inner loop DRAM/LDS-only. A/B: NINFER_GEMV_V8=1.
template <bool T2Codec, int R>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_v8_kernel(const __nv_bfloat16* __restrict__ x,
                                     const std::uint8_t* __restrict__ codes,
                                     const std::uint8_t* __restrict__ scales,
                                     __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                     std::int32_t groups_per_row) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[];  // [K]

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int k    = groups_per_row * kGemvGroupK;

    for (int chunk = static_cast<int>(threadIdx.x); chunk < k / 8;
         chunk += kGemvSmemThreads) {
        *reinterpret_cast<uint4*>(&act_sh[chunk * 8]) =
            *reinterpret_cast<const uint4*>(&x[chunk * 8]);
    }
    __syncthreads();

    // R rows per warp; CTA covers kWarps*R rows.
    const int rows_per_cta = (kGemvSmemThreads / 32) * R;
    const int row_base     = blockIdx.x * rows_per_cta + warp * R;
    if (row_base >= rows) { return; }

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    const std::int64_t code_stride = static_cast<std::int64_t>(groups_per_row) * kGemvCodeBytesPerGroup;
    const std::int64_t scale_stride = static_cast<std::int64_t>(groups_per_row) * kGemvScaleBytesPerGroup;

    const int even = groups_per_row & ~3;
    for (int g4 = 0; g4 < even; g4 += 4) {
        const std::int32_t base = g4 * kGemvGroupK + lane * 4;
        const float2 low = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
        const float2 high = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const std::int32_t g = g4 + sub;
            const std::int32_t abase = base + sub * kGemvGroupK;
            (void)abase;
            const float2 alow = (sub == 0) ? low : __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[g * kGemvGroupK + lane * 4]));
            const float2 ahigh = (sub == 0) ? high : __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[g * kGemvGroupK + lane * 4 + 2]));
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row_base + r;
                const bool ok = row < rows;
                const int rs  = ok ? row : row_base;
                const std::uint8_t raw = codes[static_cast<std::int64_t>(rs) * code_stride +
                                               g * kGemvCodeBytesPerGroup + lane];
                float w[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const std::uint32_t f = (raw >> (2 * j)) & 3u;
                    w[j] = T2Codec ? t2_field(f) : static_cast<float>(static_cast<int>(f) - 1);
                }
                const float d = fmaf(w[0], alow.x, fmaf(w[1], alow.y, fmaf(w[2], ahigh.x, w[3] * ahigh.y)));
                const float s = ok ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                    &scales[static_cast<std::int64_t>(rs) * scale_stride + g * 2]))) : 0.0f;
                acc[r] = fmaf(s, d, acc[r]);
            }
        }
    }
    if (even != groups_per_row) {
        for (int group = even; group < groups_per_row; ++group) {
            const std::int32_t base = group * kGemvGroupK + lane * 4;
            const float2 low = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
            const float2 high = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row_base + r;
                const bool ok = row < rows;
                const int rs  = ok ? row : row_base;
                const std::uint8_t raw = codes[static_cast<std::int64_t>(rs) * code_stride +
                                               group * kGemvCodeBytesPerGroup + lane];
                float w[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const std::uint32_t f = (raw >> (2 * j)) & 3u;
                    w[j] = T2Codec ? t2_field(f) : static_cast<float>(static_cast<int>(f) - 1);
                }
                const float d = fmaf(w[0], low.x, fmaf(w[1], low.y, fmaf(w[2], high.x, w[3] * high.y)));
                const float s = ok ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                    &scales[static_cast<std::int64_t>(rs) * scale_stride + group * 2]))) : 0.0f;
                acc[r] = fmaf(s, d, acc[r]);
            }
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += __shfl_down_sync(0xffffffffu, acc[r], offset);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row_base + r;
        if (lane == 0 && row < rows) out[row] = __float2bfloat16_rn(acc[r]);
    }
}

// v9: 16B-per-lane (512B/warp/instruction) weight reads -- the access width that
// the DRAM+FMA probes prove this die can stream at 600+ GB/s -- combined with the
// v4 decode semantics. Each lane's uint4 covers 4 consecutive code bytes of a row
// (16 weights, 16 FMAs), so a warp's 32 x uint4 = 128B = one full cache line per
// 4-group block, issued as a single wide transaction. A/B: NINFER_GEMV_V9=1.
template <bool T2Codec>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_v9_kernel(const __nv_bfloat16* __restrict__ x,
                                     const std::uint8_t* __restrict__ codes,
                                     const std::uint8_t* __restrict__ scales,
                                     __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                     std::int32_t groups_per_row) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[];  // [K]

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int k    = groups_per_row * kGemvGroupK;

    for (int chunk = static_cast<int>(threadIdx.x); chunk < k / 8;
         chunk += kGemvSmemThreads) {
        *reinterpret_cast<uint4*>(&act_sh[chunk * 8]) =
            *reinterpret_cast<const uint4*>(&x[chunk * 8]);
    }
    __syncthreads();

    const int row_pairs = rows / 2;
    const int pair0 =
        static_cast<int>(blockIdx.x) * (kGemvSmemThreads / 32) + warp;
    if (pair0 >= row_pairs) { return; }
    const int row_a = pair0 * 2;
    const int row_b = row_a + 1;

    const std::uint8_t* code_row_a =
        codes + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* code_row_b =
        codes + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row_a =
        scales + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::uint8_t* scale_row_b =
        scales + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvScaleBytesPerGroup;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    // 16B/lane, fully coalesced 128B/warp. With 4 groups = 128 code bytes in
    // flight, lane l reads bytes [4l, 4l+4) -> group g4 + (l>>3), 4 consecutive
    // bytes at (l&7)*4 => 16 weights at dim 16*(l&7)..+15 of that group.
    const int gg = lane >> 3;              // group within the 4-block (0..3)
    const int dim0 = 16 * (lane & 7);      // start dim in the group
    const int even_groups = groups_per_row & ~3;
    for (int g4 = 0; g4 < even_groups; g4 += 4) {
        const std::uint32_t wa = *reinterpret_cast<const std::uint32_t*>(
            code_row_a + g4 * kGemvCodeBytesPerGroup + lane * 4);
        const std::uint32_t wb = *reinterpret_cast<const std::uint32_t*>(
            code_row_b + g4 * kGemvCodeBytesPerGroup + lane * 4);
        const int grp = g4 + gg;
        const int abase = grp * kGemvGroupK + dim0;
        float a16[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) a16[j] = __bfloat162float(act_sh[abase + j]);
        float da = 0.0f, db = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int byte_idx = j >> 2;              // 0..3
            const int w_off = j & 3;                  // 0..3 within the byte
            const unsigned f_a = (wa >> (8 * byte_idx + 2 * w_off)) & 3u;
            const unsigned f_b = (wb >> (8 * byte_idx + 2 * w_off)) & 3u;
            // Group scale is constant across the 16 weights (per-128 group).
            // We apply it once per group outside the loop; here just decode+FMA.
            const float wva = T2Codec ? t2_field(f_a) : static_cast<float>(static_cast<int>(f_a) - 1);
            const float wvb = T2Codec ? t2_field(f_b) : static_cast<float>(static_cast<int>(f_b) - 1);
            da = fmaf(wva, a16[j], da);
            db = fmaf(wvb, a16[j], db);
        }
        acc_a = fmaf(gemv_scale(scale_row_a + grp * kGemvScaleBytesPerGroup), da, acc_a);
        acc_b = fmaf(gemv_scale(scale_row_b + grp * kGemvScaleBytesPerGroup), db, acc_b);
    }
    if (even_groups != groups_per_row) {  // tail: 1..3 single groups, byte reads
        for (int group = even_groups; group < groups_per_row; ++group) {
            const std::int32_t base = group * kGemvGroupK + lane * 4;
            const float2 low = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
            const float2 high = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
            const std::uint8_t raw_a = code_row_a[group * kGemvCodeBytesPerGroup + lane];
            const std::uint8_t raw_b = code_row_b[group * kGemvCodeBytesPerGroup + lane];
            float wa[4], wb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t fa = (raw_a >> (2 * j)) & 3u;
                const std::uint32_t fb = (raw_b >> (2 * j)) & 3u;
                wa[j] = T2Codec ? t2_field(fa) : static_cast<float>(static_cast<int>(fa) - 1);
                wb[j] = T2Codec ? t2_field(fb) : static_cast<float>(static_cast<int>(fb) - 1);
            }
            const float da = fmaf(wa[0], low.x, fmaf(wa[1], low.y, fmaf(wa[2], high.x, wa[3] * high.y)));
            const float db = fmaf(wb[0], low.x, fmaf(wb[1], low.y, fmaf(wb[2], high.x, wb[3] * high.y)));
            acc_a = fmaf(gemv_scale(scale_row_a + group * kGemvScaleBytesPerGroup), da, acc_a);
            acc_b = fmaf(gemv_scale(scale_row_b + group * kGemvScaleBytesPerGroup), db, acc_b);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xffffffffu, acc_a, offset);
        acc_b += __shfl_down_sync(0xffffffffu, acc_b, offset);
    }
    if (lane == 0) {
        out[row_a] = __float2bfloat16_rn(acc_a);
        out[row_b] = __float2bfloat16_rn(acc_b);
    }
}

// v9-pipe (B22): software-pipelined v9. bench_gemv_wall proved the wall is the
// LDS-read + decode serialization (LDS-only 1870 GB/s, DRAM+ALU-only 709, combined
// 432). This variant prefetches the NEXT iteration's weight words (global) and act
// values (LDS) BEFORE the current iteration's decode+FMA chain, so the two latency
// sources overlap with the ALU work.
template <bool T2Codec>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_v9_pipe_kernel(const __nv_bfloat16* __restrict__ x,
                                          const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales,
                                          __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                          std::int32_t groups_per_row) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[];  // [K]

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int k    = groups_per_row * kGemvGroupK;

    for (int chunk = static_cast<int>(threadIdx.x); chunk < k / 8;
         chunk += kGemvSmemThreads) {
        *reinterpret_cast<uint4*>(&act_sh[chunk * 8]) =
            *reinterpret_cast<const uint4*>(&x[chunk * 8]);
    }
    __syncthreads();

    const int row_pairs = rows / 2;
    const int pair0 =
        static_cast<int>(blockIdx.x) * (kGemvSmemThreads / 32) + warp;
    if (pair0 >= row_pairs) { return; }
    const int row_a = pair0 * 2;
    const int row_b = row_a + 1;

    const std::uint8_t* code_row_a =
        codes + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* code_row_b =
        codes + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row_a =
        scales + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::uint8_t* scale_row_b =
        scales + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvScaleBytesPerGroup;

    const int gg   = lane >> 3;
    const int dim0 = 16 * (lane & 7);
    const int even_groups = groups_per_row & ~3;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    // ---- prologue: load iteration 0's codes + act into the "current" registers --
    std::uint32_t wa_c = *reinterpret_cast<const std::uint32_t*>(
        code_row_a + lane * 4);
    std::uint32_t wb_c = *reinterpret_cast<const std::uint32_t*>(
        code_row_b + lane * 4);
    float a16_c[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) a16_c[j] = __bfloat162float(act_sh[gg * 128 + dim0 + j]);

    for (int g4 = 0; g4 < even_groups; g4 += 4) {
        const int grp = g4 + gg;
        // ---- prefetch NEXT iteration (global + LDS) before the FMA chain ----
        std::uint32_t wa_n = 0, wb_n = 0;
        float a16_n[16];
        if (g4 + 4 < even_groups) {
            wa_n = *reinterpret_cast<const std::uint32_t*>(
                code_row_a + (g4 + 4) * kGemvCodeBytesPerGroup + lane * 4);
            wb_n = *reinterpret_cast<const std::uint32_t*>(
                code_row_b + (g4 + 4) * kGemvCodeBytesPerGroup + lane * 4);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                a16_n[j] = __bfloat162float(act_sh[(g4 + 4 + gg) * 128 + dim0 + j]);
            }
        }
        // ---- decode + FMA on the CURRENT iteration ----
        float da = 0.0f, db = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const unsigned f_a = (wa_c >> (8 * (j >> 2) + 2 * (j & 3))) & 3u;
            const unsigned f_b = (wb_c >> (8 * (j >> 2) + 2 * (j & 3))) & 3u;
            da = fmaf(T2Codec ? t2_field(f_a) : static_cast<float>(static_cast<int>(f_a) - 1),
                      a16_c[j], da);
            db = fmaf(T2Codec ? t2_field(f_b) : static_cast<float>(static_cast<int>(f_b) - 1),
                      a16_c[j], db);
        }
        acc_a = fmaf(gemv_scale(scale_row_a + grp * kGemvScaleBytesPerGroup), da, acc_a);
        acc_b = fmaf(gemv_scale(scale_row_b + grp * kGemvScaleBytesPerGroup), db, acc_b);
        // ---- rotate ----
        wa_c = wa_n;
        wb_c = wb_n;
#pragma unroll
        for (int j = 0; j < 16; ++j) a16_c[j] = a16_n[j];
    }
    if (even_groups != groups_per_row) {  // tail: 1..3 single groups, byte reads
        for (int group = even_groups; group < groups_per_row; ++group) {
            const std::int32_t base = group * kGemvGroupK + lane * 4;
            const float2 low = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
            const float2 high = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
            const std::uint8_t raw_a = code_row_a[group * kGemvCodeBytesPerGroup + lane];
            const std::uint8_t raw_b = code_row_b[group * kGemvCodeBytesPerGroup + lane];
            float wa[4], wb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t fa = (raw_a >> (2 * j)) & 3u;
                const std::uint32_t fb = (raw_b >> (2 * j)) & 3u;
                wa[j] = T2Codec ? t2_field(fa) : static_cast<float>(static_cast<int>(fa) - 1);
                wb[j] = T2Codec ? t2_field(fb) : static_cast<float>(static_cast<int>(fb) - 1);
            }
            const float da =
                fmaf(wa[0], low.x, fmaf(wa[1], low.y, fmaf(wa[2], high.x, wa[3] * high.y)));
            const float db =
                fmaf(wb[0], low.x, fmaf(wb[1], low.y, fmaf(wb[2], high.x, wb[3] * high.y)));
            acc_a = fmaf(gemv_scale(scale_row_a + group * kGemvScaleBytesPerGroup), da, acc_a);
            acc_b = fmaf(gemv_scale(scale_row_b + group * kGemvScaleBytesPerGroup), db, acc_b);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xffffffffu, acc_a, offset);
        acc_b += __shfl_down_sync(0xffffffffu, acc_b, offset);
    }
    if (lane == 0) {
        out[row_a] = __float2bfloat16_rn(acc_a);
        out[row_b] = __float2bfloat16_rn(acc_b);
    }
}

// v9-ksplit (B22): split-K variant of v9 for the small-n decode shapes (the down
// projections, n=5120). Plain v9 launches only (n/16) CTAs -- 320 CTAs = ~15
// warps/SM = 23% of the card -- and stages the FULL activation row in smem
// (k=17408 -> 34 KiB/CTA caps residency at 3 CTAs/SM). Slicing the group range
// across `ks` grid.y restores occupancy. Deterministic: each (row, split) partial
// is written exactly once (no atomics); the reduce kernel sums in fixed order.
template <bool T2Codec>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_v9_ksplit_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float* __restrict__ partial,
    std::int32_t rows, std::int32_t groups_per_row, std::int32_t g_base,
    std::int32_t k_splits) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[];
    const int g_begin = static_cast<int>(blockIdx.y) * g_base;
    const int g_end   = (g_begin + g_base < groups_per_row) ? (g_begin + g_base)
                                                            : groups_per_row;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int slice_groups = g_end - g_begin;
    for (int chunk = static_cast<int>(threadIdx.x);
         chunk < slice_groups * (kGemvGroupK / 8); chunk += kGemvSmemThreads) {
        *reinterpret_cast<uint4*>(&act_sh[chunk * 8]) =
            *reinterpret_cast<const uint4*>(&x[g_begin * kGemvGroupK + chunk * 8]);
    }
    __syncthreads();

    const int row_pairs = rows / 2;
    const int pair0 =
        static_cast<int>(blockIdx.x) * (kGemvSmemThreads / 32) + warp;
    float acc_a = 0.0f;
    float acc_b = 0.0f;
    if (pair0 < row_pairs && slice_groups > 0) {
        const int row_a = pair0 * 2;
        const int row_b = row_a + 1;
        const std::uint8_t* code_row_a =
            codes + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvCodeBytesPerGroup;
        const std::uint8_t* code_row_b =
            codes + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvCodeBytesPerGroup;
        const std::uint8_t* scale_row_a =
            scales + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvScaleBytesPerGroup;
        const std::uint8_t* scale_row_b =
            scales + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvScaleBytesPerGroup;

        const int gg   = lane >> 3;
        const int dim0 = 16 * (lane & 7);
        const int even_groups = slice_groups & ~3;  // aligned 4-group blocks in-slice
        for (int g4 = 0; g4 < even_groups; g4 += 4) {
            const int grp = g_begin + g4 + gg;
            // code block base = (g_begin+g4)*32B; lane*4 spans the whole 128B block
            // continuously (group = lane>>3 = gg). Using grp*32 here would double-
            // count gg (grp already carries it) and read the wrong bytes.
            const std::uint32_t wa = *reinterpret_cast<const std::uint32_t*>(
                code_row_a + (g_begin + g4) * kGemvCodeBytesPerGroup + lane * 4);
            const std::uint32_t wb = *reinterpret_cast<const std::uint32_t*>(
                code_row_b + (g_begin + g4) * kGemvCodeBytesPerGroup + lane * 4);
            const int abase = (g4 + gg) * kGemvGroupK + dim0;
            float a16[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) a16[j] = __bfloat162float(act_sh[abase + j]);
            float da = 0.0f, db = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int byte_idx = j >> 2;
                const int w_off    = j & 3;
                const unsigned f_a = (wa >> (8 * byte_idx + 2 * w_off)) & 3u;
                const unsigned f_b = (wb >> (8 * byte_idx + 2 * w_off)) & 3u;
                const float wva =
                    T2Codec ? t2_field(f_a) : static_cast<float>(static_cast<int>(f_a) - 1);
                const float wvb =
                    T2Codec ? t2_field(f_b) : static_cast<float>(static_cast<int>(f_b) - 1);
                da = fmaf(wva, a16[j], da);
                db = fmaf(wvb, a16[j], db);
            }
            acc_a = fmaf(gemv_scale(scale_row_a + grp * kGemvScaleBytesPerGroup), da, acc_a);
            acc_b = fmaf(gemv_scale(scale_row_b + grp * kGemvScaleBytesPerGroup), db, acc_b);
        }
        for (int group = even_groups; group < slice_groups; ++group) {  // in-slice tail
            const int grp = g_begin + group;
            const std::int32_t base = group * kGemvGroupK + lane * 4;
            const float2 low = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
            const float2 high = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
            const std::uint8_t raw_a = code_row_a[grp * kGemvCodeBytesPerGroup + lane];
            const std::uint8_t raw_b = code_row_b[grp * kGemvCodeBytesPerGroup + lane];
            float wa[4], wb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t fa = (raw_a >> (2 * j)) & 3u;
                const std::uint32_t fb = (raw_b >> (2 * j)) & 3u;
                wa[j] = T2Codec ? t2_field(fa) : static_cast<float>(static_cast<int>(fa) - 1);
                wb[j] = T2Codec ? t2_field(fb) : static_cast<float>(static_cast<int>(fb) - 1);
            }
            const float da =
                fmaf(wa[0], low.x, fmaf(wa[1], low.y, fmaf(wa[2], high.x, wa[3] * high.y)));
            const float db =
                fmaf(wb[0], low.x, fmaf(wb[1], low.y, fmaf(wb[2], high.x, wb[3] * high.y)));
            acc_a = fmaf(gemv_scale(scale_row_a + grp * kGemvScaleBytesPerGroup), da, acc_a);
            acc_b = fmaf(gemv_scale(scale_row_b + grp * kGemvScaleBytesPerGroup), db, acc_b);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xffffffffu, acc_a, offset);
        acc_b += __shfl_down_sync(0xffffffffu, acc_b, offset);
    }
    if (lane == 0) {
        partial[static_cast<std::int64_t>(pair0 * 2) * k_splits + blockIdx.y]     = acc_a;
        partial[static_cast<std::int64_t>(pair0 * 2 + 1) * k_splits + blockIdx.y] = acc_b;
    }
}

__global__ void ternary_gemv_ksplit_reduce_kernel(const float* __restrict__ partial,
                                                  __nv_bfloat16* __restrict__ out,
                                                  std::int32_t rows, std::int32_t k_splits) {
    const int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (row >= rows) { return; }
    float s = 0.0f;
    for (int ks = 0; ks < k_splits; ++ks) {
        s += partial[static_cast<std::int64_t>(row) * k_splits + ks];
    }
    out[row] = __float2bfloat16_rn(s);
}

// v5: v4 structure with the smem LUT removed - each code byte is decoded to its
// four ternary values DIRECTLY in registers via t2_field. The v4 LUT costs a 16B
// smem float4 read per weight byte (16x traffic amplification on the smem path);
// this variant trades that for 4 ALU decodes. A/B: NINFER_GEMV_V5=1.
template <bool T2Codec>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_v5_kernel(const __nv_bfloat16* __restrict__ x,
                                     const std::uint8_t* __restrict__ codes,
                                     const std::uint8_t* __restrict__ scales,
                                     __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                     std::int32_t groups_per_row) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[];  // [K]

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int k    = groups_per_row * kGemvGroupK;

    for (int chunk = static_cast<int>(threadIdx.x); chunk < k / 8;
         chunk += kGemvSmemThreads) {
        *reinterpret_cast<uint4*>(&act_sh[chunk * 8]) =
            *reinterpret_cast<const uint4*>(&x[chunk * 8]);
    }
    __syncthreads();

    const int row_pairs = rows / 2;
    const int pair0 =
        static_cast<int>(blockIdx.x) * (kGemvSmemThreads / 32) + warp;
    if (pair0 >= row_pairs) { return; }
    const int row_a = pair0 * 2;
    const int row_b = row_a + 1;

    const std::uint8_t* code_row_a =
        codes + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* code_row_b =
        codes + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row_a =
        scales + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::uint8_t* scale_row_b =
        scales + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvScaleBytesPerGroup;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    const int even = groups_per_row & ~3;
    for (int group = 0; group < even; group += 4) {
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const std::int32_t g = group + sub;
            const std::int32_t base = g * kGemvGroupK + lane * 4;
            const float2 low = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
            const float2 high = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
            const std::uint8_t raw_a = code_row_a[g * kGemvCodeBytesPerGroup + lane];
            const std::uint8_t raw_b = code_row_b[g * kGemvCodeBytesPerGroup + lane];
            float wa[4], wb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t fa = (raw_a >> (2 * j)) & 3u;
                const std::uint32_t fb = (raw_b >> (2 * j)) & 3u;
                wa[j] = T2Codec ? t2_field(fa) : static_cast<float>(static_cast<int>(fa) - 1);
                wb[j] = T2Codec ? t2_field(fb) : static_cast<float>(static_cast<int>(fb) - 1);
            }
            const float da = fmaf(wa[0], low.x, fmaf(wa[1], low.y, fmaf(wa[2], high.x, wa[3] * high.y)));
            const float db = fmaf(wb[0], low.x, fmaf(wb[1], low.y, fmaf(wb[2], high.x, wb[3] * high.y)));
            acc_a = fmaf(gemv_scale(scale_row_a + g * kGemvScaleBytesPerGroup), da, acc_a);
            acc_b = fmaf(gemv_scale(scale_row_b + g * kGemvScaleBytesPerGroup), db, acc_b);
        }
    }
    if (even != groups_per_row) {  // tail: 1..3 single groups
        for (int group = even; group < groups_per_row; ++group) {
            const std::int32_t base = group * kGemvGroupK + lane * 4;
            const float2 low = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
            const float2 high = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
            const std::uint8_t raw_a = code_row_a[group * kGemvCodeBytesPerGroup + lane];
            const std::uint8_t raw_b = code_row_b[group * kGemvCodeBytesPerGroup + lane];
            float wa[4], wb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t fa = (raw_a >> (2 * j)) & 3u;
                const std::uint32_t fb = (raw_b >> (2 * j)) & 3u;
                wa[j] = T2Codec ? t2_field(fa) : static_cast<float>(static_cast<int>(fa) - 1);
                wb[j] = T2Codec ? t2_field(fb) : static_cast<float>(static_cast<int>(fb) - 1);
            }
            const float da = fmaf(wa[0], low.x, fmaf(wa[1], low.y, fmaf(wa[2], high.x, wa[3] * high.y)));
            const float db = fmaf(wb[0], low.x, fmaf(wb[1], low.y, fmaf(wb[2], high.x, wb[3] * high.y)));
            acc_a = fmaf(gemv_scale(scale_row_a + group * kGemvScaleBytesPerGroup), da, acc_a);
            acc_b = fmaf(gemv_scale(scale_row_b + group * kGemvScaleBytesPerGroup), db, acc_b);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xffffffffu, acc_a, offset);
        acc_b += __shfl_down_sync(0xffffffffu, acc_b, offset);
    }
    if (lane == 0) {
        out[row_a] = __float2bfloat16_rn(acc_a);
        out[row_b] = __float2bfloat16_rn(acc_b);
    }
}

// Sixteen-groups-per-iteration variant. MEASURED standalone (bench_gemv_variants.cu):
//   gate_up 34816x5120  0.086 -> 0.077 ms (550 -> 612 GB/s)  -- wins on K<=8K shapes
//   down    5120x17408  0.052 -> 0.077 ms (458 -> 307 GB/s)  -- register pressure LOSES
// So the launcher picks v16 only for groups_per_row <= 64 (K <= 8192) and keeps v4
// for the long-K down projection. Identical per-group FMA order => same numerics.
template <bool T2Codec>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_v16_kernel(const __nv_bfloat16* __restrict__ x,
                                      const std::uint8_t* __restrict__ codes,
                                      const std::uint8_t* __restrict__ scales,
                                      __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                      std::int32_t groups_per_row) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[];  // [K]
    __shared__ __align__(16) float4 t2_lut[256];  // byte -> 4 decoded weights

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int k    = groups_per_row * kGemvGroupK;

    for (int i = static_cast<int>(threadIdx.x); i < 256; i += kGemvSmemThreads) {
        float w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int field = (i >> (2 * j)) & 3;
            w[j] = T2Codec ? t2_field(static_cast<std::uint32_t>(field))
                           : static_cast<float>(field - 1);
        }
        t2_lut[i] = make_float4(w[0], w[1], w[2], w[3]);
    }

    for (int chunk = static_cast<int>(threadIdx.x); chunk < k / 8;
         chunk += kGemvSmemThreads) {
        *reinterpret_cast<uint4*>(&act_sh[chunk * 8]) =
            *reinterpret_cast<const uint4*>(&x[chunk * 8]);
    }
    __syncthreads();

    const int row_pairs = rows / 2;
    const int pair0 =
        static_cast<int>(blockIdx.x) * (kGemvSmemThreads / 32) + warp;
    if (pair0 >= row_pairs) { return; }
    const int row_a = pair0 * 2;
    const int row_b = row_a + 1;

    const std::uint8_t* code_row_a =
        codes + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* code_row_b =
        codes + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row_a =
        scales + static_cast<std::int64_t>(row_a) * groups_per_row * kGemvScaleBytesPerGroup;
    const std::uint8_t* scale_row_b =
        scales + static_cast<std::int64_t>(row_b) * groups_per_row * kGemvScaleBytesPerGroup;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    constexpr int kGroups = 16;
    const int even = groups_per_row & ~(kGroups - 1);
    for (int group = 0; group < even; group += kGroups) {
#pragma unroll
        for (int sub = 0; sub < kGroups; ++sub) {
            const std::int32_t g = group + sub;
            const std::int32_t base = g * kGemvGroupK + lane * 4;
            const float2 low = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
            const float2 high = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
            const std::uint8_t raw_a = code_row_a[g * kGemvCodeBytesPerGroup + lane];
            const std::uint8_t raw_b = code_row_b[g * kGemvCodeBytesPerGroup + lane];
            const float4 wa = t2_lut[raw_a];
            const float4 wb = t2_lut[raw_b];
            acc_a = fmaf(gemv_scale(scale_row_a + g * kGemvScaleBytesPerGroup),
                         fmaf(wa.x, low.x, fmaf(wa.y, low.y, fmaf(wa.z, high.x, wa.w * high.y))),
                         acc_a);
            acc_b = fmaf(gemv_scale(scale_row_b + g * kGemvScaleBytesPerGroup),
                         fmaf(wb.x, low.x, fmaf(wb.y, low.y, fmaf(wb.z, high.x, wb.w * high.y))),
                         acc_b);
        }
    }
    if (even != groups_per_row) {  // tail: 1..15 single groups
        for (int group = even; group < groups_per_row; ++group) {
            const std::int32_t base = group * kGemvGroupK + lane * 4;
            const float2 low = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
            const float2 high = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
            const std::uint8_t raw_a = code_row_a[group * kGemvCodeBytesPerGroup + lane];
            const std::uint8_t raw_b = code_row_b[group * kGemvCodeBytesPerGroup + lane];
            const float4 wa = t2_lut[raw_a];
            const float4 wb = t2_lut[raw_b];
            acc_a = fmaf(gemv_scale(scale_row_a + group * kGemvScaleBytesPerGroup),
                         fmaf(wa.x, low.x, fmaf(wa.y, low.y, fmaf(wa.z, high.x, wa.w * high.y))),
                         acc_a);
            acc_b = fmaf(gemv_scale(scale_row_b + group * kGemvScaleBytesPerGroup),
                         fmaf(wb.x, low.x, fmaf(wb.y, low.y, fmaf(wb.z, high.x, wb.w * high.y))),
                         acc_b);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xffffffffu, acc_a, offset);
        acc_b += __shfl_down_sync(0xffffffffu, acc_b, offset);
    }
    if (lane == 0) {
        out[row_a] = __float2bfloat16_rn(acc_a);
        out[row_b] = __float2bfloat16_rn(acc_b);
    }
}

} // namespace ninfer::ops::detail

// ---------------------------------------------------------------------------
// MULTI-TOKEN shared-memory GEMV (AMD port) -- the T = 2..16 regime, i.e. the
// speculative VERIFY pass.
//
// MEASURED, and this is why the kernel exists (probe/bench_small_t.cu, gfx1201,
// gate_up N=34816 K=5120, weight traffic 47.3 MB):
//
//   T=1  gemv_smem  (the existing decode kernel) 0.109 ms   435 GB/s
//   T=2  wmma2 prefill  BM64 x BN32              0.217 ms   218 GB/s
//   T=2  gemv_tile<4>                            0.227 ms   209 GB/s
//   T=2  mma 16-token block                      5.826 ms     8 GB/s   <- the old path
//
// The WMMA kernel is not bandwidth bound at small T, it is COMPUTE bound: with
// BN = 32 it evaluates 16x more mma work than the token count needs, which is why
// it tops out at a third of the GEMV's bandwidth. The SIMT tile GEMV has the
// opposite problem: no waste, but it re-reads the activation from L2 once per
// output row (N*K*2 bytes against N*K/4 bytes of weights).
//
// This kernel is the T=1 smem GEMV with each weight byte made to serve kT tokens:
//   * one 256-thread CTA owns kWarpRows * 8 rows, so the weight stream spreads over
//     thousands of CTAs and every SM stays busy;
//   * the kT activation vectors are staged into shared memory in K-chunks, so DRAM
//     traffic is the weight stream plus one activation read per CTA;
//   * kT * kWarpRows accumulators per warp live in registers, so the token loop
//     costs FMA and shared-load issue but no extra weight traffic at all.
// kWarpRows matters more than it looks: the activation loads are shared by every
// row in the warp, so doubling the rows per warp halves the shared-memory traffic
// per weight byte -- which is what limits the T=1 kernel to 435 GB/s, not DRAM.
// The chunk width is a runtime argument (dynamic shared memory) so that the same
// kernel serves K=5120 in one chunk and K=17408 in several, at kT up to 16.
// ---------------------------------------------------------------------------
namespace ninfer::ops::detail {

// Shared-memory budget per CTA. The T=1 kernel uses K*2 bytes (10 KB at K=5120) and
// wins because it keeps ~6 CTAs per SM resident; a 24 KB staging request would drop
// that to 2 and cost more in latency hiding than the extra token reuse is worth, so
// the budget is deliberately small and K is walked in chunks instead.
inline constexpr int kGemvMultiSmemBudget = 48 * 1024;
inline constexpr int kGemvMultiWarps      = kGemvSmemThreads / 32;

template <int kT, int kRows>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_multi_kernel(const __nv_bfloat16* __restrict__ x,
                                        const std::uint8_t* __restrict__ codes,
                                        const std::uint8_t* __restrict__ scales,
                                        __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                        std::int32_t groups_per_row, std::int32_t tokens,
                                        std::int32_t out_row_stride, std::int32_t chunk_groups,
                                        std::int32_t t2_codec) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[]; // [kT][chunk_groups * 128]
    __shared__ __align__(16) float4 t2_lut[256];  // byte -> 4 decoded weights (T2/PQ2 by runtime codec)

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int k    = groups_per_row * kGemvGroupK;
    const int ck   = chunk_groups * kGemvGroupK;

    // Build the decode LUT once per CTA; every warp reaches the barrier below
    // (there is no early return before it), so the block cannot deadlock.
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += kGemvSmemThreads) {
        float w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int field = (i >> (2 * j)) & 3;
            w[j] = t2_codec ? t2_field(static_cast<std::uint32_t>(field))
                            : static_cast<float>(field - 1);
        }
        t2_lut[i] = make_float4(w[0], w[1], w[2], w[3]);
    }
    __syncthreads();

    const int row_base = (static_cast<int>(blockIdx.x) * kGemvMultiWarps + warp) * kRows;
    // Full-tile CTAs only; the launcher checks divisibility and falls back otherwise.
    // This is NOT an early return before the chunk loop: the loop contains
    // __syncthreads(), and a returning warp would hang the barrier.
    const bool active = row_base + kRows <= rows;

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row_base) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row_base) * groups_per_row * kGemvScaleBytesPerGroup;

    float acc[kT][kRows];
#pragma unroll
    for (int t = 0; t < kT; ++t) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) { acc[t][r] = 0.0f; }
    }

    for (int base_k = 0; base_k < k; base_k += ck) {
        const int here = (k - base_k) < ck ? (k - base_k) : ck;
        {
            // chunk_groups is a whole number of 128-groups, so `here` is a multiple of 8
            // and the 16-byte copies are exact. Tokens past `tokens` are zero-filled instead
            // of read: x holds exactly tokens*K elements, so reading the padded slots would
            // walk off the end of the activation buffer.
            const int pairs = here / 8;
            for (int idx = static_cast<int>(threadIdx.x); idx < kT * pairs;
                 idx += kGemvSmemThreads) {
                const int t = idx / pairs;
                const int j = (idx - t * pairs) * 8;
                if (t < tokens) {
                    *reinterpret_cast<uint4*>(&act_sh[t * ck + j]) =
                        *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(t) * k +
                                                        base_k + j);
                } else {
                    *reinterpret_cast<uint4*>(&act_sh[t * ck + j]) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();

        if (active) {
            const int groups_here = here / kGemvGroupK;
            const int group0      = base_k / kGemvGroupK;
            for (int g = 0; g < groups_here; ++g) {
                const std::int32_t off = g * kGemvGroupK + lane * 4;
                float w[kRows][4];
                float sc[kRows];
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    const std::int64_t at =
                        (static_cast<std::int64_t>(r) * groups_per_row + group0 + g);
                    const std::uint8_t raw = code_row[at * kGemvCodeBytesPerGroup + lane];
                    const float4 wv = t2_lut[raw];
                    w[r][0] = wv.x;
                    w[r][1] = wv.y;
                    w[r][2] = wv.z;
                    w[r][3] = wv.w;
                    sc[r]   = gemv_scale(scale_row + at * kGemvScaleBytesPerGroup);
                }
#pragma unroll
                for (int t = 0; t < kT; ++t) {
                    if (t < tokens) {
                        const float2 low = __bfloat1622float2(
                            *reinterpret_cast<const __nv_bfloat162*>(&act_sh[t * ck + off]));
                        const float2 high = __bfloat1622float2(
                            *reinterpret_cast<const __nv_bfloat162*>(&act_sh[t * ck + off + 2]));
#pragma unroll
                        for (int r = 0; r < kRows; ++r) {
                            const float dot =
                                fmaf(w[r][0], low.x,
                                     fmaf(w[r][1], low.y,
                                          fmaf(w[r][2], high.x, w[r][3] * high.y)));
                            acc[t][r] = fmaf(sc[r], dot, acc[t][r]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!active) { return; }
#pragma unroll
    for (int t = 0; t < kT; ++t) {
        if (t < tokens) {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                float v = acc[t][r];
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    v += __shfl_down_sync(0xffffffffu, v, offset);
                }
                if (lane == 0) {
                    out[static_cast<std::int64_t>(t) * out_row_stride + row_base + r] =
                        __float2bfloat16_rn(v);
                }
            }
        }
    }
}

// Rows per warp that keeps kT * kRows inside a sane accumulator budget.
template <int kT>
// MEASURED (probe/bench_small_t.cu, gate_up N=34816 K=5120, T=4):
//   rows/warp 2 -> 0.212 ms | 4 -> 0.201 | 8 -> 0.187 ms  (r8 wins on the wide shape)
//   rows/warp 4 -> 0.264    | 8 -> 0.230 ms               (r4 wins on down N=5120 K=17408)
//   budget  8k/16k/32k: differences within noise (the chunk loop is not the limiter)
// A rows-per-warp of 8 shares each activation load across 8 rows, which halves the
// shared-memory traffic per weight byte again over r4. r8 costs kT*8 accumulators, so it
// is only used while kT <= 8; wider token tiles fall back to r2.
struct GemvMultiRows {
    static constexpr int value = kT <= 8 ? 8 : 2;
};

template <int kT, int kRows>
inline bool launch_gemv_multi_pair(const __nv_bfloat16* x, const std::uint8_t* codes,
                                   const std::uint8_t* scales, __nv_bfloat16* out,
                                   std::int32_t rows, std::int32_t groups_per_row,
                                   std::int32_t tokens, std::int32_t out_row_stride,
                                   cudaStream_t stream, std::int32_t t2_codec,
                                   std::int32_t smem_budget = kGemvMultiSmemBudget) {
    if ((rows % (kRows * kGemvMultiWarps)) != 0) { return false; }
    std::int32_t chunk_groups = smem_budget / (kT * kGemvGroupK * 2);
    if (chunk_groups < 1) { chunk_groups = 1; }
    if (chunk_groups > groups_per_row) { chunk_groups = groups_per_row; }
    const std::int32_t smem = kT * chunk_groups * kGemvGroupK * 2;
    const unsigned grid =
        static_cast<unsigned>(rows / (kRows * kGemvMultiWarps));
    ternary_pq2_gemv_smem_multi_kernel<kT, kRows><<<grid, kGemvSmemThreads, smem, stream>>>(
        x, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, chunk_groups,
        t2_codec);
    return true;
}

template <int kT>
inline bool launch_gemv_multi_t(const __nv_bfloat16* x, const std::uint8_t* codes,
                                const std::uint8_t* scales, __nv_bfloat16* out,
                                std::int32_t rows, std::int32_t groups_per_row,
                                std::int32_t tokens, std::int32_t out_row_stride,
                                cudaStream_t stream, std::int32_t t2_codec) {
    return launch_gemv_multi_pair<kT, GemvMultiRows<kT>::value>(
        x, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, stream, t2_codec);
}

// Returns false when the shape or token count has no instantiation, so the caller can
// fall back to the WMMA path instead of throwing.
inline bool launch_pq2_gemv_smem_multi(const __nv_bfloat16* x, const std::uint8_t* codes,
                                       const std::uint8_t* scales, __nv_bfloat16* out,
                                       std::int32_t rows, std::int32_t groups_per_row,
                                       std::int32_t tokens, std::int32_t out_row_stride,
                                       cudaStream_t stream, std::int32_t t2_codec) {
    switch (tokens) {
    case 2: return launch_gemv_multi_t<2>(x, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, stream, t2_codec);
    case 3: return launch_gemv_multi_t<3>(x, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, stream, t2_codec);
    case 4: return launch_gemv_multi_t<4>(x, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, stream, t2_codec);
    case 5: return launch_gemv_multi_t<5>(x, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, stream, t2_codec);
    case 6: return launch_gemv_multi_t<6>(x, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, stream, t2_codec);
    case 8: return launch_gemv_multi_t<8>(x, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, stream, t2_codec);
    case 12: return launch_gemv_multi_t<12>(x, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, stream, t2_codec);
    case 16: return launch_gemv_multi_t<16>(x, codes, scales, out, rows, groups_per_row, tokens, out_row_stride, stream, t2_codec);
    default: return false;
    }
}

// ---------------------------------------------------------------------------
// Row-count variant of the shared-memory GEMV above (AMD port).
//
// The two-rows-per-warp form is tuned for the WIDE projections: at N=34816 it
// launches 2177 CTAs and streams weights at ~435 GB/s. The MLP DOWN projection is
// the mirror image -- N=5120, K=17408 -- and there the same kernel launches only
// 320 CTAs for 64 CUs, measuring 248 GB/s. Fewer rows per CTA widens the grid and
// costs nothing but a second copy of the activation staging per row.
// ---------------------------------------------------------------------------
template <int kRowsPerWarp>
__global__ __launch_bounds__(kGemvSmemThreads)
void ternary_pq2_gemv_smem_rows_kernel(const __nv_bfloat16* __restrict__ x,
                                       const std::uint8_t* __restrict__ codes,
                                       const std::uint8_t* __restrict__ scales,
                                       __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                       std::int32_t groups_per_row) {
    extern __shared__ __align__(16) __nv_bfloat16 act_sh[];  // [K]
    __shared__ __align__(16) float4 t2_lut[256];  // PQ2 (c-1) decode table

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int k    = groups_per_row * kGemvGroupK;

    for (int i = static_cast<int>(threadIdx.x); i < 256; i += kGemvSmemThreads) {
        float w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int field = (i >> (2 * j)) & 3;
            w[j] = static_cast<float>(field - 1);
        }
        t2_lut[i] = make_float4(w[0], w[1], w[2], w[3]);
    }

    for (int chunk = static_cast<int>(threadIdx.x); chunk < k / 8; chunk += kGemvSmemThreads) {
        *reinterpret_cast<uint4*>(&act_sh[chunk * 8]) =
            *reinterpret_cast<const uint4*>(&x[chunk * 8]);
    }
    __syncthreads();

    const int row0 = (static_cast<int>(blockIdx.x) * (kGemvSmemThreads / 32) + warp) * kRowsPerWarp;
    if (row0 + kRowsPerWarp > rows) { return; }

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row0) * groups_per_row * kGemvCodeBytesPerGroup;
    const std::uint8_t* scale_row =
        scales + static_cast<std::int64_t>(row0) * groups_per_row * kGemvScaleBytesPerGroup;

    float acc[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) { acc[r] = 0.0f; }

    for (int group = 0; group < groups_per_row; ++group) {
        const std::int32_t base = group * kGemvGroupK + lane * 4;
        const float2 low = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&act_sh[base]));
        const float2 high =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&act_sh[base + 2]));
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const std::int64_t at = static_cast<std::int64_t>(r) * groups_per_row + group;
            const std::uint8_t raw = code_row[at * kGemvCodeBytesPerGroup + lane];
            const float4 wv = t2_lut[raw];
            const float dot = fmaf(wv.x, low.x, fmaf(wv.y, low.y, fmaf(wv.z, high.x, wv.w * high.y)));
            acc[r] = fmaf(gemv_scale(scale_row + at * kGemvScaleBytesPerGroup), dot, acc[r]);
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        float v = acc[r];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) { out[row0 + r] = __float2bfloat16_rn(v); }
    }
}

} // namespace ninfer::ops::detail

