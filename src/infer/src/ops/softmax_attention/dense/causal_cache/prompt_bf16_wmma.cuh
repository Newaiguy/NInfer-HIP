#pragma once

// AMD port (gfx1201): zero-shuffle hardware-WMMA rewrite of the BF16 causal prompt
// (prefill) attention kernel. The upstream causal_attention_prompt_bf16_kernel is a
// faithful FlashAttention-2 (same CTA/warp decomposition, same online softmax, same
// bottom-right causal alignment and paged-KV walk) but it needs 96 KiB of dynamic
// shared memory -- above the gfx12 64 KiB block cap, so it could never launch here --
// and its QK^T / PV contraction runs on the SOFTWARE mma emulation, which this card's
// shuffle throughput caps at ~0.95 TFLOPS.
//
// This port keeps the upstream skeleton (grid (q-tile, q-head), Br=Bc=64, 4 warps x
// 16 query rows, per-tile row max + alpha rescale + running m/l, P rounded once at the
// PV operand boundary) and swaps the contraction engine:
//   * QK^T: A = Q fragment built DIRECTLY from global (the HW A layout is lane L =
//     q-row, u32[i] = {Q(L, 2i), Q(L, 2i+1)} -- Q is token-major per head, so one 16B
//     load per lane per 16-d window); B = K^T fragment also DIRECTLY from global (the
//     HW B layout is lane L = key column, u32[i] = {K(2i, L), K(2i+1, L)} -- K cache is
//     [page][head][token][d], d contiguous). Zero staging, zero shuffles.
//   * V: staged once per Bc tile into smem TRANSPOSED (Vt[d][t], fp16 -> bf16), so the
//     PV B fragment is one conflict-free LDS.128. Vt + the P staging tile fit in 40 KiB
//     -- under the 48 KiB default dynamic-smem ceiling, no attribute opt-in needed.
//   * P: the HW C layout of the QK mma (lane = key column, 8 row slots) is transposed
//     through an 8 KiB smem tile, and the PV A fragment reads back token-major with one
//     LDS.128. Probabilities are rounded to bf16 (P in [0,1]; the 2^-9 relative error is
//     far below the bf16 output rounding).
//   * Scale: folded into exp2 exactly like upstream (scale_l2), softmax in fp32.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/q4_0_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh" // AMD port: G64 int8 codec + D256 Hadamard rotation

#include <math_constants.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>

namespace ninfer::ops {

namespace attn_wmma_detail {

typedef float v4f __attribute__((vector_size(16)));
typedef float v8f __attribute__((vector_size(32)));

// D(hw) = A(hw) * B(hw) + C(hw); validated bit-exact by probe/mma_prim_probe.cu and
// the engine layer dumps of the ternary prefill port. out and cw may alias at the
// call site, so the result is staged through a local temporary.
__device__ __forceinline__ void wmma16(float (&out)[8], const float (&aw)[4],
                                       const float (&bw)[4], const float (&cw)[8]) {
    v4f a;
    v4f b;
    v8f c;
#pragma unroll
    for (int i = 0; i < 4; ++i) { a[i] = aw[i]; b[i] = bw[i]; }
#pragma unroll
    for (int i = 0; i < 8; ++i) { c[i] = cw[i]; }
    const v8f r = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    float tmp[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { tmp[i] = r[i]; }
#pragma unroll
    for (int i = 0; i < 8; ++i) { out[i] = tmp[i]; }
}

// Fp16-KV twin (Fp16 paged-KV storage path): identical 16x16x16 fragment layout with
// fp16 operands -> f32 accumulator.
__device__ __forceinline__ void wmma16_f16(float (&out)[8], const float (&aw)[4],
                                           const float (&bw)[4], const float (&cw)[8]) {
    v4f a;
    v4f b;
    v8f c;
#pragma unroll
    for (int i = 0; i < 4; ++i) { a[i] = aw[i]; b[i] = bw[i]; }
#pragma unroll
    for (int i = 0; i < 8; ++i) { c[i] = cw[i]; }
    const v8f r = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    float tmp[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { tmp[i] = r[i]; }
#pragma unroll
    for (int i = 0; i < 8; ++i) { out[i] = tmp[i]; }
}

__device__ __forceinline__ __nv_bfloat16 __ushort_as_bfloat16_persist(unsigned bits) {
    __nv_bfloat16 v;
    *reinterpret_cast<unsigned short*>(&v) = static_cast<unsigned short>(bits);
    return v;
}
} // namespace attn_wmma_detail

using attn_wmma_detail::__ushort_as_bfloat16_persist;

// ---------------------------------------------------------------------------
// Shared-memory plan (dynamic):
//   Vt: [D=256][Bc=64] bf16, d-major (PV B operand), XOR-swizzled over 8-key groups
//   Ps: [Br=64][Bc=64] bf16, token-major (PV A operand), XOR-swizzled over 8-key groups
// ---------------------------------------------------------------------------
inline constexpr int kCausalPromptWmmaVtBytes = kCausalPromptHeadDim * kCausalPromptBc *
                                                static_cast<int>(sizeof(__nv_bfloat16));
inline constexpr int kCausalPromptWmmaPsBytes = kCausalPromptBr * kCausalPromptBc *
                                                static_cast<int>(sizeof(__nv_bfloat16));
inline constexpr int kCausalPromptWmmaSmemBytes =
    kCausalPromptWmmaVtBytes + kCausalPromptWmmaPsBytes;
static_assert(kCausalPromptWmmaSmemBytes <= 48 * 1024,
              "the WMMA prompt kernel must fit the default dynamic-smem ceiling");

// KInt8 variant (Int8Group64 paged-KV storage): the cached K is D256-Hadamard-rotated
// and G64-quantized, so Q must be staged rotated too. Q is staged as rotated int8
// codes + one FP16 G64 scale per row (16 KiB + 512 B) instead of a bf16 tile (32 KiB)
// so the whole kernel stays under the gfx12 64 KiB per-block dynamic-smem cap
// (Vt 32 KiB + Ps 8 KiB + Q codes 16 KiB + Q scales 512 B = 57856 B).
inline constexpr int kCausalPromptWmmaInt8QBytes =
    kCausalPromptBr * kCausalPromptHeadDim;  // 16 KiB rotated int8 codes
inline constexpr int kCausalPromptWmmaInt8QScaleBytes =
    kCausalPromptBr * kKVCacheInt8Groups * static_cast<int>(sizeof(__half));  // 512 B
inline constexpr int kCausalPromptWmmaInt8SmemBytes =
    kCausalPromptWmmaSmemBytes + kCausalPromptWmmaInt8QBytes +
    kCausalPromptWmmaInt8QScaleBytes;
static_assert(kCausalPromptWmmaInt8SmemBytes <= 64 * 1024,
              "the KInt8 WMMA prompt kernel must fit the gfx12 64 KiB per-block cap");

// d-major Vt address with the 8-key group XOR-swizzled by the d row (the PV B fragment
// reads one 16B group per lane across 16 d rows; the swizzle spreads the groups).
__device__ __forceinline__ int causal_prompt_wmma_vt(int d, int t) {
    return d * kCausalPromptBc + (((t >> 3) ^ (d & 7)) << 3) | (t & 7);
}

// Token-major Ps address, same swizzle family (the PV A fragment reads one 16B group
// per lane across 16 q rows).
__device__ __forceinline__ int causal_prompt_wmma_ps(int q, int t) {
    return q * kCausalPromptBc + (((t >> 3) ^ (q & 7)) << 3) | (t & 7);
}

// Stage V[k0 .. k0+Bc) of one kv head from the paged cache into the TRANSPOSED
// Vt tile. KFp16: the Vt tile is fp16 too (raw copy, no fp16->f32->bf16 round-trip);
// otherwise it is bf16 (the bf16 WMMA PV operand). KQ4: V plane is ggml q4_0,
// dequantized to fp16. Keys beyond max_query_abs are zeroed (the causal mask drops
// them, but the padded cache tail must never feed NaNs).
template <typename Geometry, bool KFp16, bool KQ4, bool KInt8, bool KInt4 = false>
__device__ __forceinline__ void causal_prompt_wmma_stage_vt(
    __nv_bfloat16* vt, const __half* __restrict__ cache_v,
    const __half* __restrict__ cache_v_scale, int kv_head, int k0,
    int max_query_abs, int physical_page, int tid, int threads) {
    constexpr int D  = kCausalPromptHeadDim;
    constexpr int Bc = kCausalPromptBc;
    constexpr int VecPerRow = D / 8;  // 16B = 8 fp16
    const __half* cache_block =
        cache_v + paged_kv_element_offset<D, Geometry::KVHeads>(physical_page, kv_head,
                                                                k0 & kPagedKVPageMask, 0);
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += threads) {
        const int key_l = chunk >> 5;        // / VecPerRow (32)
        const int d     = (chunk & 31) << 3; // (chunk % 32) * 8
        if constexpr (KInt8) {
            // int8 G64 V: 8 code bytes + one G64 scale -> bf16 Vt tile.
            const int key = k0 + key_l;
            if (key <= max_query_abs) {
                const int key_off = key & kPagedKVPageMask;
                const std::int64_t code_off = kv_cache_int8_quant_code_index<Geometry>(
                    physical_page, kv_head, d, key_off);
                const float vsf = __half2float(cache_v_scale[kv_cache_int8_quant_scale_index<
                        Geometry>(physical_page, kv_head, d >> 6, key_off)]);
                if (isnan(vsf) || isinf(vsf)) {
                    printf("[prompt KInt8 Vt] BAD V scale key=%d head=%d d=%d page=%d vsf=%f\n",
                           key, kv_head, d, physical_page, vsf);
                }
                const int4 vv = kv_cache_int8_dequant_i8x8_from(
                    reinterpret_cast<const std::int8_t*>(cache_v) + code_off, vsf);
                const auto lo = [](unsigned bits) {
                    return __ushort_as_bfloat16_persist(bits & 0xffffu);
                };
                const auto hi = [](unsigned bits) {
                    return __ushort_as_bfloat16_persist((bits >> 16) & 0xffffu);
                };
                vt[causal_prompt_wmma_vt(d + 0, key_l)] = lo(static_cast<unsigned>(vv.x));
                vt[causal_prompt_wmma_vt(d + 1, key_l)] = hi(static_cast<unsigned>(vv.x));
                vt[causal_prompt_wmma_vt(d + 2, key_l)] = lo(static_cast<unsigned>(vv.y));
                vt[causal_prompt_wmma_vt(d + 3, key_l)] = hi(static_cast<unsigned>(vv.y));
                vt[causal_prompt_wmma_vt(d + 4, key_l)] = lo(static_cast<unsigned>(vv.z));
                vt[causal_prompt_wmma_vt(d + 5, key_l)] = hi(static_cast<unsigned>(vv.z));
                vt[causal_prompt_wmma_vt(d + 6, key_l)] = lo(static_cast<unsigned>(vv.w));
                vt[causal_prompt_wmma_vt(d + 7, key_l)] = hi(static_cast<unsigned>(vv.w));
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    vt[causal_prompt_wmma_vt(d + i, key_l)] = __float2bfloat16_rn(0.0f);
                }
            }
        } else if constexpr (KInt4) {
            // RotatedInt4KeyInt4ValueE8 V plane: packed signed int4 + G32 scales.
            // 8 values = 4 code bytes; byte (d+i)>>1 holds dim (d+i) in the low nibble
            // when even, high when odd. Dequant to the bf16 Vt tile (this storage rides
            // the bf16 PV mma, exactly like the Int8Group64 variant).
            const int key = k0 + key_l;
            const __half* vt16 = reinterpret_cast<__half*>(vt);
            if (key <= max_query_abs) {
                const int key_off           = key & kPagedKVPageMask;
                const std::int64_t code_off = kv_cache_int4_value_code_index<Geometry>(
                    physical_page, kv_head, d >> 1, key_off);
                const std::uint32_t codes4 = *reinterpret_cast<const std::uint32_t*>(
                    reinterpret_cast<const std::uint8_t*>(cache_v) + code_off);
                const float vsf = __half2float(cache_v_scale[kv_cache_int4_value_scale_index<
                        Geometry>(physical_page, kv_head, d >> 5, key_off)]);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int nib = (codes4 >> (8 * (i >> 1) + 4 * ((d + i) & 1))) & 0x0fu;
                    vt[causal_prompt_wmma_vt(d + i, key_l)] =
                        __float2bfloat16_rn(static_cast<float>((nib ^ 8) - 8) * vsf);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    vt[causal_prompt_wmma_vt(d + i, key_l)] = __float2bfloat16_rn(0.0f);
                }
            }
            (void)vt16;
        } else if constexpr (KQ4) {
            // q4_0 V: 8 values = 4 code bytes + 1 scale (d%32 in {0,8,16,24}).
            __half* vt16 = reinterpret_cast<__half*>(vt);
            if (k0 + key_l <= max_query_abs) {
                const std::int64_t code_off = q4_0_code_offset<Geometry::KVHeads>(
                    physical_page, kv_head, (k0 + key_l) & kPagedKVPageMask,
                    (d / kQ4_0Block) * kQ4_0Block) + (d & 15);
                const std::int64_t scale_off = q4_0_scale_offset<Geometry::KVHeads>(
                    physical_page, kv_head, (k0 + key_l) & kPagedKVPageMask, d);
                const std::uint64_t codes8 = *reinterpret_cast<const std::uint64_t*>(
                    reinterpret_cast<const std::uint8_t*>(cache_v) + code_off);
                const __half sc = cache_v_scale[scale_off];
                float fv[8];
                q4_0_dequant8(fv, codes8, (d & 16) == 0, sc);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    vt16[causal_prompt_wmma_vt(d + i, key_l)] = __float2half_rn(fv[i]);
                }
            } else {
                // AMD fix (gfx1201): keys beyond max_query_abs get a causal-mask score
                // but their Vt tile must not feed NaN into the WMMA PV product. The
                // KFp16 branch zeroes these slots explicitly; the q4_0 branch left them
                // uninitialized, so on single-page prefill (chunk 1 = 45 tokens, tile
                // covers key 45..63 which are not yet written, scale = +Inf sentinel)
                // stale/NaN values leaked into the PV product and the whole attention
                // output for layers 1.. turned NaN -> "!".
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    vt16[causal_prompt_wmma_vt(d + i, key_l)] = __float2half_rn(0.0f);
                }
            }
        } else if constexpr (KFp16) {
            int4 raw = make_int4(0, 0, 0, 0);
            if (k0 + key_l <= max_query_abs) {
                raw = load_vec<int4>(&cache_block[key_l * D + d]);
            }
            __half* vt16 = reinterpret_cast<__half*>(vt);
            const auto half_from = [](unsigned b, int s) {
                return __ushort_as_half(static_cast<unsigned short>((b >> s) & 0xffffu));
            };
            const unsigned x = static_cast<unsigned>(raw.x);
            const unsigned y = static_cast<unsigned>(raw.y);
            const unsigned z = static_cast<unsigned>(raw.z);
            const unsigned w = static_cast<unsigned>(raw.w);
            vt16[causal_prompt_wmma_vt(d + 0, key_l)] = half_from(x, 0);
            vt16[causal_prompt_wmma_vt(d + 1, key_l)] = half_from(x, 16);
            vt16[causal_prompt_wmma_vt(d + 2, key_l)] = half_from(y, 0);
            vt16[causal_prompt_wmma_vt(d + 3, key_l)] = half_from(y, 16);
            vt16[causal_prompt_wmma_vt(d + 4, key_l)] = half_from(z, 0);
            vt16[causal_prompt_wmma_vt(d + 5, key_l)] = half_from(z, 16);
            vt16[causal_prompt_wmma_vt(d + 6, key_l)] = half_from(w, 0);
            vt16[causal_prompt_wmma_vt(d + 7, key_l)] = half_from(w, 16);
        } else {
            int4 raw = make_int4(0, 0, 0, 0);
            if (k0 + key_l <= max_query_abs) {
                raw = load_vec<int4>(&cache_block[key_l * D + d]);
            }
            const __half2 h0 = half2_from_bits(static_cast<std::uint32_t>(raw.x));
            const __half2 h1 = half2_from_bits(static_cast<std::uint32_t>(raw.y));
            const __half2 h2 = half2_from_bits(static_cast<std::uint32_t>(raw.z));
            const __half2 h3 = half2_from_bits(static_cast<std::uint32_t>(raw.w));
            const float2 f0 = __half22float2(h0);
            const float2 f1 = __half22float2(h1);
            const float2 f2 = __half22float2(h2);
            const float2 f3 = __half22float2(h3);
            vt[causal_prompt_wmma_vt(d + 0, key_l)]     = __float2bfloat16_rn(f0.x);
            vt[causal_prompt_wmma_vt(d + 1, key_l)]     = __float2bfloat16_rn(f0.y);
            vt[causal_prompt_wmma_vt(d + 2, key_l)]     = __float2bfloat16_rn(f1.x);
            vt[causal_prompt_wmma_vt(d + 3, key_l)]     = __float2bfloat16_rn(f1.y);
            vt[causal_prompt_wmma_vt(d + 4, key_l)]     = __float2bfloat16_rn(f2.x);
            vt[causal_prompt_wmma_vt(d + 5, key_l)]     = __float2bfloat16_rn(f2.y);
            vt[causal_prompt_wmma_vt(d + 6, key_l)]     = __float2bfloat16_rn(f3.x);
            vt[causal_prompt_wmma_vt(d + 7, key_l)]     = __float2bfloat16_rn(f3.y);
        }
    }
}

// FlashAttention-2 prefill, one CTA per (query 64-row block, query head). Same
// decomposition, masking and online-softmax contract as
// causal_attention_prompt_bf16_kernel; the contraction engine is the gfx12 hardware
// WMMA with zero-shuffle fragments.
template <typename Geometry, typename Metadata, bool KFp16 = false, bool KQ4 = false,
          bool KInt8 = false, bool KInt4 = false, bool KInt8Bf16 = false>
__launch_bounds__(kCausalPromptThreads, 1) __global__
    void causal_attention_prompt_bf16_wmma_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ cache_k,
        const __half* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
        const __half* __restrict__ cache_v_scale, Metadata metadata,
        const std::int32_t* __restrict__ positions, float scale,
        __nv_bfloat16* __restrict__ out, std::int32_t width) {
    using attn_wmma_detail::wmma16;
    using attn_wmma_detail::wmma16_f16;
    constexpr int D       = kCausalPromptHeadDim;  // 256
    constexpr int Br      = kCausalPromptBr;       // 64
    constexpr int Bc      = kCausalPromptBc;       // 64
    constexpr int Threads = kCausalPromptThreads;  // 128
    constexpr int QKKs    = D / 16;                // 16 k-steps over head_dim
    constexpr int QKNt    = Bc / 16;               // 4  score tiles over the key tile
    constexpr int PVKs    = Bc / 16;               // 4  PV contraction steps over keys
    constexpr int PVNt    = D / 16;                // 16 PV output n-tiles
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned LowerMask = 0x0000ffffu;
    constexpr unsigned UpperMask = 0xffff0000u;
    constexpr unsigned FullMask  = 0xffffffffu;

    extern __shared__ __align__(16) __nv_bfloat16 causal_wmma_smem[];
    __nv_bfloat16* vt = causal_wmma_smem;                            // [D, Bc] d-major
    __nv_bfloat16* ps = vt + D * Bc;                                 // [Br, Bc] token-major
    // KInt8 only: staged rotated Q as int8 codes (b16-column swizzled, see
    // causal_prompt_store_byte_swizzled) + one fp16 G64 scale per row.
    std::int8_t* q_i8 = reinterpret_cast<std::int8_t*>(ps + Br * Bc);
    __half* q_scale   = reinterpret_cast<__half*>(q_i8 + Br * D);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                 Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    // HW fragment indices. QK: lane L<16 = q-row L / key col L, lanes 16-31 = the +8
    // d-contract half. PV: A lane L<16 = q-row L, B lane L<16 = d col L.
    const int q_row        = warp * 16 + (lane & 15);           // warp-local q row (A row)
    const int q_global     = q0 + q_row;                        // absolute query token
    const bool q_ok        = q_global < tokens;
    const int q_global_safe = q_ok ? q_global : q0;             // bounded load; masked later
    const int d_half       = (lane >> 4) * 8;                   // 0 / 8 within a 16-d window
    const int q_abs        = q_ok ? base_pos + q_global : -1;   // causal bound for this lane's rows? no --
    (void)q_abs;
    // NOTE: the causal bound is per C-slot row, not per A lane; see the softmax step.

    float acc[PVNt][8];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[n][i] = 0.0f; }
    }
    // Running softmax state per C slot: slot i covers q row (warp*16 + i + 8*(lane>=16)).
    float m_run[8];
    float l_run[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { m_run[i] = -CUDART_INF_F; l_run[i] = 0.0f; }

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int n_block_max   = (max_query_abs / Bc) + 1;  // n_block_min == 0
    const float scale_l2    = scale * Log2E;

    const unsigned my_mask = (lane < 16) ? LowerMask : UpperMask;
    // C-slot row mapping: slot i covers q row (warp*16 + i + 8*(lane>=16)), i.e. an
    // affine ramp off one base. Keeping the three per-slot arrays this replaced cost
    // 24 VGPRs live across the whole kernel, which is what pushed the kernel into
    // VGPR spill (256 VGPRs / 33 spilled / 136 B scratch per lane).
    const int ps_row_base  = warp * 16 + ((lane >= 16) ? 8 : 0);  // tile-relative row of slot 0
    const int slot_qrow0   = q0 + ps_row_base;                    // absolute token of slot 0
    const int slot_qabs0   = base_pos + slot_qrow0;               // its causal bound

    int physical_page = block_table[0];

    // ---- KInt8: stage the D256-Hadamard-rotated Q as int8 codes + G64 scales ----
    // The Int8Group64 K is stored rotated (the append kernel applies the same fixed
    // normalized_hadamard_d256_inplace), so QK must contract rotated Q with the stored
    // K. One warp owns one row: lane l holds dims l + 32u, rotates in place exactly
    // like the SIMT/decode paths, and stores the G64 codes. Beyond-tile rows stage
    // zeros (masked to -inf in softmax). Dequant happens per QK fragment below.
    if constexpr (KInt8 || KInt4 || KInt8Bf16) {
        constexpr int Groups = kKVCacheInt8Groups;
        for (int row = warp; row < Br; row += 4) {
            float qv[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int d = lane + 32 * u;
                qv[u] = (row < tile_rows)
                            ? __bfloat162float(
                                  q[causal_prompt_q_index<Geometry>(q_head, d, q0 + row)])
                            : 0.0f;
            }
            normalized_hadamard_d256_inplace(qv, lane);
#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                const int d0    = grp * kKVCacheInt8Group + lane;
                const int d1    = d0 + 32;
                const float x0  = qv[2 * grp];
                const float x1  = qv[2 * grp + 1];
                float absmax    = fmaxf(fabsf(x0), fabsf(x1));
                absmax          = warp_max(absmax, FullMask);
                const auto qp   = kv_cache_int8_quant_params(absmax);
                if constexpr (KInt4 || KInt8Bf16) {
                    // The int4 QK rides the bf16 mma, so the staged rotated Q is read back
                    // as plain linear codes (row-major D). The int8 b16-column swizzle only
                    // exists to feed the native i8 mma's A-fragment layout.
                    q_i8[row * D + d0] = kv_cache_int8_quant_code(x0, qp.inverse_scale);
                    q_i8[row * D + d1] = kv_cache_int8_quant_code(x1, qp.inverse_scale);
                } else {
                    causal_prompt_store_byte_swizzled(
                        q_i8, row, d0, kv_cache_int8_quant_code(x0, qp.inverse_scale));
                    causal_prompt_store_byte_swizzled(
                        q_i8, row, d1, kv_cache_int8_quant_code(x1, qp.inverse_scale));
                }
                if (lane == 0) { q_scale[row * Groups + grp] = qp.scale; }
            }
        }
        __syncthreads();
    }

    for (int kb = 0; kb < n_block_max; ++kb) {
        const int k0 = kb * Bc;
        physical_page = block_table[kb];

        // ---- stage V(kb) transposed (cache -> Vt tile) -----------------------------
        __syncthreads();  // previous tile's Vt/Ps consumers are done
        causal_prompt_wmma_stage_vt<Geometry, KFp16, KQ4, KInt8, KInt4>(vt, cache_v, cache_v_scale,
                                                                  kv_head, k0, max_query_abs,
                                                                  physical_page, tid, Threads);
        __syncthreads();

        // ---- S = Q K^T for this warp's 16 rows over all Bc keys ------------------
        // Score tiles in HW C layout: score[tile][slot] with slot i at
        // (q row warp*16 + i + 8*(lane>=16), key col tile*16 + lane%16).
        float score[QKNt][8];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { score[nt][i] = 0.0f; }
        }
#pragma unroll 2
        for (int ki = 0; ki < QKKs; ++ki) {
            // A = Q: one 16B global load (token-major, d contiguous). For the fp16
            // WMMA the 8 bf16 halves are converted to fp16 in registers. KInt8 uses
            // the staged rotated-Q int8 codes raw (no dequant) and contracts them on
            // the native int8 WMMA (mma_s8_hw_direct); the G64 scale is applied to
            // the s32 accumulator afterwards.
            float aw[4];
            unsigned a_code[2] = {0, 0};  // gfx12-layout raw int8 carriers (KInt8)
            if constexpr (KInt8 && !KInt8Bf16) {
#ifdef NINFER_HW_WMMMA
                const int d0 = ki * 16 + d_half;
                // b16-column swizzled staging: the 8 codes of dims [d0, d0+8) are the
                // contiguous 8 bytes at (row*128 + ((ki ^ (row&7))<<3) + ((d0>>1)&7))*2.
                const int byte_off =
                    (q_row * (D / 2) + ((ki ^ (q_row & 7)) << 3) + ((d0 >> 1) & 7)) * 2;
                // gfx12 i8 A layout: lane L<16 = row L, k0..7 (i8 dims d0..d0+7);
                // lane L>=16 = row L-16, k8..15 (dims d0+8..d0+15). The two u32s are
                // the 8 codes in dimension order.
                a_code[0] = *reinterpret_cast<const unsigned*>(&q_i8[byte_off]);
                a_code[1] = *reinterpret_cast<const unsigned*>(&q_i8[byte_off + 4]);
#else
                const int d0 = ki * 16 + d_half;
                const int byte_off =
                    (q_row * (D / 2) + ((ki ^ (q_row & 7)) << 3) + ((d0 >> 1) & 7)) * 2;
                const float qsf = __half2float(q_scale[q_row * kKVCacheInt8Groups + (d0 >> 6)]);
                const int4 qi  = kv_cache_int8_dequant_i8x8_from(&q_i8[byte_off], qsf);
                aw[0] = __uint_as_float(static_cast<unsigned>(qi.x));
                aw[1] = __uint_as_float(static_cast<unsigned>(qi.y));
                aw[2] = __uint_as_float(static_cast<unsigned>(qi.z));
                aw[3] = __uint_as_float(static_cast<unsigned>(qi.w));
#endif
            } else if constexpr (KInt8Bf16) {
                // AMD port (gfx1201): KInt8Bf16 = int8 G64 codes dequantized to bf16 and
                // contracted with the plain bf16 WMMA. Identical math to the native s8
                // path (same rotated Q codes, same G64 scales) but it never touches
                // mma_s8_hw_direct, which corrupts long-context prefill on this port.
                // The staged Q is read back linearly (see the store side above).
                const int d0       = ki * 16 + d_half;
                const int byte_off = q_row * D + d0;
                const float qsf    = __half2float(q_scale[q_row * kKVCacheInt8Groups + (d0 >> 6)]);
                const int4 qi      = kv_cache_int8_dequant_i8x8_from(&q_i8[byte_off], qsf);
                aw[0] = __uint_as_float(static_cast<unsigned>(qi.x));
                aw[1] = __uint_as_float(static_cast<unsigned>(qi.y));
                aw[2] = __uint_as_float(static_cast<unsigned>(qi.z));
                aw[3] = __uint_as_float(static_cast<unsigned>(qi.w));
            } else if constexpr (KInt4) {
                // int4 K is stored D256-Hadamard-rotated, so contract the rotated Q.
                // The staged int8 codes are dequantized back to bf16 here (the int4
                // QK rides the bf16 mma; see the dispatch below).
                const int d0 = ki * 16 + d_half;
                // Linear staging (see the store side): 8 consecutive codes at row*D + d0.
                const int byte_off = q_row * D + d0;
                const float qsf = __half2float(q_scale[q_row * kKVCacheInt8Groups + (d0 >> 6)]);
                const int4 qi  = kv_cache_int8_dequant_i8x8_from(&q_i8[byte_off], qsf);
                aw[0] = __uint_as_float(static_cast<unsigned>(qi.x));
                aw[1] = __uint_as_float(static_cast<unsigned>(qi.y));
                aw[2] = __uint_as_float(static_cast<unsigned>(qi.z));
                aw[3] = __uint_as_float(static_cast<unsigned>(qi.w));
            } else {
                const __nv_bfloat16* q_ptr =
                    q + causal_prompt_q_index<Geometry>(q_head, ki * 16 + d_half, q_global_safe);
                const uint4 qv = *reinterpret_cast<const uint4*>(q_ptr);
                if constexpr (KFp16 || KQ4) {
                    const int4 f16 = bf16x8_bits_to_f16x8_bits(
                        make_int4(static_cast<int>(qv.x), static_cast<int>(qv.y),
                                  static_cast<int>(qv.z), static_cast<int>(qv.w)));
                    aw[0] = __uint_as_float(static_cast<unsigned>(f16.x));
                    aw[1] = __uint_as_float(static_cast<unsigned>(f16.y));
                    aw[2] = __uint_as_float(static_cast<unsigned>(f16.z));
                    aw[3] = __uint_as_float(static_cast<unsigned>(f16.w));
                } else {
                    aw[0] = __uint_as_float(qv.x);
                    aw[1] = __uint_as_float(qv.y);
                    aw[2] = __uint_as_float(qv.z);
                    aw[3] = __uint_as_float(qv.w);
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                // B = K^T: one 16B global load (key token-major, d contiguous). The
                // 16 keys of tile nt map to lanes 0-15; lanes 16-31 take d+8.
                const int key_l    = k0 + nt * 16 + (lane & 15);
                const int page_off = key_l & kPagedKVPageMask;
                const int key_page = block_table[key_l >> kPagedKVPageShift];
                float bw[4];
                unsigned b_code[2] = {0, 0};  // gfx12-layout raw int8 carriers (KInt8)
                if constexpr (KInt8 && !KInt8Bf16) {
#ifdef NINFER_HW_WMMMA
                    // int8 G64 K (rotated in cache): the 8 code bytes for dims
                    // [d0, d0+8) are contiguous (d-major), already in the gfx12 i8 B
                    // layout: lane L<16 = k row L, n0..7; lane L>=16 = k row L-16,
                    // n8..15. Read raw, integer-contract on the WMMA.
                    const int d0 = ki * 16 + d_half;
                    const std::int64_t code_off = kv_cache_int8_quant_code_index<Geometry>(
                        key_page, kv_head, d0, page_off);
                    const std::int8_t* kp =
                        reinterpret_cast<const std::int8_t*>(cache_k) + code_off;
                    b_code[0] = *reinterpret_cast<const unsigned*>(kp);
                    b_code[1] = *reinterpret_cast<const unsigned*>(kp + 4);
#else
                    const int d0 = ki * 16 + d_half;
                    const std::int64_t code_off = kv_cache_int8_quant_code_index<Geometry>(
                        key_page, kv_head, d0, page_off);
                    const float ksf = __half2float(cache_k_scale[kv_cache_int8_quant_scale_index<
                        Geometry>(key_page, kv_head, d0 >> 6, page_off)]);
                    const int4 ki4 = kv_cache_int8_dequant_i8x8_from(
                        reinterpret_cast<const std::int8_t*>(cache_k) + code_off, ksf);
                    bw[0] = __uint_as_float(static_cast<unsigned>(ki4.x));
                    bw[1] = __uint_as_float(static_cast<unsigned>(ki4.y));
                    bw[2] = __uint_as_float(static_cast<unsigned>(ki4.z));
                    bw[3] = __uint_as_float(static_cast<unsigned>(ki4.w));
#endif
                } else if constexpr (KInt8Bf16) {
                    // KInt8Bf16: int8 G64 K codes dequantized to bf16 (8 codes at d0 are
                    // d-major contiguous; the G64 group d0>>6 is invariant across them).
                    // Same layout read as the native s8 path, but the contraction below
                    // uses the bf16 WMMA instead of mma_s8_hw_direct.
                    const int d0 = ki * 16 + d_half;
                    const std::int64_t code_off = kv_cache_int8_quant_code_index<Geometry>(
                        key_page, kv_head, d0, page_off);
                    const float ksf = __half2float(cache_k_scale[kv_cache_int8_quant_scale_index<
                        Geometry>(key_page, kv_head, d0 >> 6, page_off)]);
                    const int4 ki4 = kv_cache_int8_dequant_i8x8_from(
                        reinterpret_cast<const std::int8_t*>(cache_k) + code_off, ksf);
                    bw[0] = __uint_as_float(static_cast<unsigned>(ki4.x));
                    bw[1] = __uint_as_float(static_cast<unsigned>(ki4.y));
                    bw[2] = __uint_as_float(static_cast<unsigned>(ki4.z));
                    bw[3] = __uint_as_float(static_cast<unsigned>(ki4.w));
                } else if constexpr (KInt4) {
                    // RotatedInt4 K: 8 values = 4 code bytes + one G64 scale. d0 is a
                    // multiple of 16 and the K row is 128 packed bytes, so the 4 bytes
                    // never cross the row (d0>>1 in {0,4,...,124}).
                    const int d0 = ki * 16 + d_half;
                    const std::int64_t code_off =
                        kv_cache_int4_value_code_index<Geometry>(
                            key_page, kv_head, d0 >> 1, page_off);
                    const std::uint32_t codes4 = *reinterpret_cast<const std::uint32_t*>(
                        reinterpret_cast<const std::uint8_t*>(cache_k) + code_off);
                    const float ksf = __half2float(cache_k_scale[
                        kv_cache_int8_quant_scale_index<Geometry>(key_page, kv_head, d0 >> 6,
                                                                  page_off)]);
                    alignas(16) __nv_bfloat16 bv8[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int nib = (codes4 >> (8 * (i >> 1) + 4 * ((d0 + i) & 1))) & 0x0fu;
                        bv8[i] = __float2bfloat16_rn(static_cast<float>((nib ^ 8) - 8) * ksf);
                    }
                    const uint4 kv4 = *reinterpret_cast<const uint4*>(bv8);
                    bw[0] = __uint_as_float(kv4.x);
                    bw[1] = __uint_as_float(kv4.y);
                    bw[2] = __uint_as_float(kv4.z);
                    bw[3] = __uint_as_float(kv4.w);
                } else if constexpr (KQ4) {
                    // q4_0 K: 8 values = 8 code bytes + 1 scale, dequant per lane
                    // (d0%32 in {0,8,16,24}, window never crosses blocks).
                    const int d0 = ki * 16 + d_half;
                    const std::int64_t code_off =
                        q4_0_code_offset<Geometry::KVHeads>(
                            key_page, kv_head, page_off, (d0 / kQ4_0Block) * kQ4_0Block) +
                        (d0 & 15);
                    const std::int64_t scale_off =
                        q4_0_scale_offset<Geometry::KVHeads>(key_page, kv_head, page_off, d0);
                    const std::uint64_t codes8 = *reinterpret_cast<const std::uint64_t*>(
                        reinterpret_cast<const std::uint8_t*>(cache_k) + code_off);
                    const __half sc = cache_k_scale[scale_off];
                    float fv[8];
                    q4_0_dequant8(fv, codes8, (d0 & 16) == 0, sc);
                    alignas(16) __half hv8[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) { hv8[i] = __float2half_rn(fv[i]); }
                    const uint4 kv4 = *reinterpret_cast<const uint4*>(hv8);
                    bw[0] = __uint_as_float(kv4.x);
                    bw[1] = __uint_as_float(kv4.y);
                    bw[2] = __uint_as_float(kv4.z);
                    bw[3] = __uint_as_float(kv4.w);
                } else {
                    const __nv_bfloat16* k_ptr =
                        cache_k + paged_kv_element_offset<D, Geometry::KVHeads>(
                                      key_page, kv_head, page_off, ki * 16 + d_half);
                    const uint4 kv4 = *reinterpret_cast<const uint4*>(k_ptr);
                    bw[0] = __uint_as_float(kv4.x);
                    bw[1] = __uint_as_float(kv4.y);
                    bw[2] = __uint_as_float(kv4.z);
                    bw[3] = __uint_as_float(kv4.w);
                }
                if constexpr (KInt8 && !KInt8Bf16) {
#ifdef NINFER_HW_WMMMA
                    // Native int8 QK: integer s32 dot product on the WMMA, then scale
                    // by qs*ks. The G64 group is d0>>6 (64 dims), invariant across the
                    // 8-code window, so one scale pair per (ki, key) applies to all 8
                    // D slots.
                    const int d0 = ki * 16 + d_half;
                    const float qsf = __half2float(
                        q_scale[q_row * kKVCacheInt8Groups + (d0 >> 6)]);
                    const float ksf = __half2float(cache_k_scale[
                        kv_cache_int8_quant_scale_index<Geometry>(key_page, kv_head, d0 >> 6,
                                                                  page_off)]);
                    const float qk_scale = qsf * ksf;
                    const mma_hw_i32x2 av = {static_cast<int>(a_code[0]),
                                             static_cast<int>(a_code[1])};
                    const mma_hw_i32x2 bv = {static_cast<int>(b_code[0]),
                                             static_cast<int>(b_code[1])};
                    const mma_hw_i32x8 czero = {0, 0, 0, 0, 0, 0, 0, 0};
                    const mma_hw_i32x8 di = mma_s8_hw_direct(av, bv, czero);
                    // Hardware D layout: di[i] = score slot i (row i + 8*(L>=16)).
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        score[nt][i] = __fmaf_rn(qk_scale, static_cast<float>(di[i]), score[nt][i]);
                    }
#else
                    wmma16(score[nt], aw, bw, score[nt]);
#endif
                } else if constexpr (KFp16 || KQ4) {
                    wmma16_f16(score[nt], aw, bw, score[nt]);
                } else {
                    wmma16(score[nt], aw, bw, score[nt]);
                }
            }
        }

        // ---- online softmax over the whole Bc tile (scores all in registers) -----
        // Causal + validity mask, then per-slot max across this lane's 4 tiles,
        // then one 16-lane butterfly per slot (the row's 16 columns live in lanes
        // 0-15 for top-half slots and 16-31 for bottom-half slots).
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int key_abs = k0 + nt * 16 + (lane & 15);
                const bool keep   = (slot_qrow0 + i < tokens) && (key_abs <= slot_qabs0 + i);
                score[nt][i]      = keep ? score[nt][i] : -CUDART_INF_F;
            }
        }
        // bm/nm/alpha/bl used to be four 8-wide arrays live at once (32 VGPRs). The
        // running-state update is folded in here so only the new max (bm, reused as
        // nm) and the block row-sum (bl) stay live across the P loop.
        float bm[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v = score[0][i];
#pragma unroll
            for (int nt = 1; nt < QKNt; ++nt) { v = fmaxf(v, score[nt][i]); }
            bm[i] = warp_max<16>(v, my_mask);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float prev  = m_run[i];
            const float nm    = fmaxf(prev, bm[i]);
            const float alpha = (prev == -CUDART_INF_F)
                                    ? 0.0f
                                    : exp2_approx(__fmaf_rn(prev, scale_l2, -nm * scale_l2));
            bm[i]    = nm;
            m_run[i] = nm;
            l_run[i] *= alpha;
#pragma unroll
            for (int n = 0; n < PVNt; ++n) { acc[n][i] *= alpha; }
        }
        // P = exp2(S - m) (bf16 for the PV A operand) + local block row-sums.
        float bl[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { bl[i] = 0.0f; }
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = (score[nt][i] > -CUDART_INF_F)
                                    ? exp2_approx(__fmaf_rn(score[nt][i], scale_l2,
                                                            -bm[i] * scale_l2))
                                    : 0.0f;
                bl[i] += p;
                if constexpr (KFp16 || KQ4) {
                    reinterpret_cast<__half*>(ps)[causal_prompt_wmma_ps(
                        ps_row_base + i, nt * 16 + (lane & 15))] = __float2half_rn(p);
                } else {
                    ps[causal_prompt_wmma_ps(ps_row_base + i, nt * 16 + (lane & 15))] =
                        __float2bfloat16_rn(p);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            l_run[i] += warp_sum<16>(bl[i], my_mask);
        }
        __syncthreads();  // Ps complete before the PV reads

        // ---- O += P V over the Bc keys -------------------------------------------
#pragma unroll
        for (int pk = 0; pk < PVKs; ++pk) {
            // A = P: one LDS.128 (token-major Ps; lanes 16-31 take keys +8).
            const uint4 pv = *reinterpret_cast<const uint4*>(
                &ps[causal_prompt_wmma_ps(q_row, pk * 16 + d_half)]);
            const float aw[4] = {__uint_as_float(pv.x), __uint_as_float(pv.y),
                                 __uint_as_float(pv.z), __uint_as_float(pv.w)};
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                // B = V: one LDS.128 from the d-major Vt (lane = d col, keys +8 half).
                const uint4 vv = *reinterpret_cast<const uint4*>(
                    &vt[causal_prompt_wmma_vt(n * 16 + (lane & 15), pk * 16 + d_half)]);
                const float bw[4] = {__uint_as_float(vv.x), __uint_as_float(vv.y),
                                     __uint_as_float(vv.z), __uint_as_float(vv.w)};
                if constexpr (KFp16 || KQ4) {
                    wmma16_f16(acc[n], aw, bw, acc[n]);
                } else {
                    wmma16(acc[n], aw, bw, acc[n]);
                }
            }
        }
        __syncthreads();  // Vt/Ps readers done before the next tile's staging
    }

    // ---- epilogue: normalize by the running row sum, store token-major ----------
    // l_run is ALREADY the row total: the block loop updates it as
    // l_run[i]*alpha + warp_sum<16>(bl[i]) (the warp_sum reduces the 16 key columns
    // and every lane of the 16-lane half-wave holds the row value). A second
    // warp_sum<16> here would multiply l by 16 and divide the output by 16 -- the
    // same bug decode_t1_wmma.cuh calls out. (Verified: output was exactly v/16.)
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float inv = (l_run[i] > 0.0f) ? __frcp_rn(l_run[i]) : 0.0f;
            const int qrow  = slot_qrow0 + i;
            const int d     = n * 16 + (lane & 15);
            if (qrow < tokens) {
                out[causal_prompt_q_index<Geometry>(q_head, d, qrow)] =
                    __float2bfloat16_rn(acc[n][i] * inv);
            }
        }
    }
    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                             Threads);
}

} // namespace ninfer::ops






