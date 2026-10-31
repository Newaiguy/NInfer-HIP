#pragma once

// AMD port (gfx1201): zero-shuffle hardware-WMMA rewrite of the T=1 (decode) causal
// attention partial kernel for BF16-K/FP16-V paged KV. The production
// causal_attention_small_t_tc_partial_bf16_kernel runs the NVIDIA m16n8k16 fragment
// pipeline on the SOFTWARE mma emulation (~0.95 TFLOPS shuffle-throughput ceiling on
// gfx12, 2 __syncthreads per 32-key block) and measured 93% of the whole decode step
// at 100k ctx (437.7 of 470 ms). This kernel keeps the split-KV contract of the
// small-T path (same grid semantics, same partial m/l/acc layout, same reducer) and
// swaps the contraction engine for the gfx12 hardware WMMA exactly like the prompt
// kernel (prompt_bf16_wmma.cuh):
//   * QK^T: A = Q fragment from a once-staged smem q tile (16 rows: the kv-head's
//     GroupSize valid rows + zero padding), B = K^T fragment built DIRECTLY from the
//     paged K cache (d contiguous). Zero shuffles.
//   * V: staged once per 64-key block into the TRANSPOSED bf16 Vt tile (fp16 -> bf16),
//     so the PV B fragment is one conflict-free LDS.128.
//   * P: written into a per-warp 16-row token-major smem tile (bf16); every warp owns
//     its own tile slice (warp*16 rows of the 64-row Ps buffer), so the padded warps
//     (rows 16..63 never hold valid queries at T=1) run the identical instruction
//     stream on zeroed data and their outputs are simply never stored.
//   * Softmax: online max/rescale per 16-row C slot, fp32, scale folded into exp2.
// One 64-key block costs 2 __syncthreads vs 2 per 32 keys + 128 software mma calls
// in the production kernel.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/q4_0_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

// B20 P0-2: device-side max-|K-scale| tracker (defined in ops/common/nan_triage_probe.cu).
extern __device__ float ninfer_kv_scale_absmax;
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>
#include <cstdio>

namespace ninfer::ops {

namespace decode_wmma_detail {

typedef float v4f __attribute__((vector_size(16)));
typedef float v8f __attribute__((vector_size(32)));

// D(hw) = A(hw) * B(hw) + C(hw); operands are float arrays carrying the packed bit
// patterns, identical to mma.cuh's validated mma_hw_call (which passes 64 random ops
// bit-exact). Callers must declare the operand arrays alignas(16)/alignas(32) so the
// vector carriers are legally formed.
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

// Fp16-KV twin: identical fragment layout (A/B 16x16x16 fp16 -> f32), for the Fp16
// paged-KV storage path (K and V planes are FP16).
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

// d-major Vt address with the 8-key group XOR-swizzled by the d row (identical family
// to the prompt kernel's causal_prompt_wmma_vt; bc is the key-tile width).
__device__ __forceinline__ int dvt(int d, int t, int bc) {
    return d * bc + (((t >> 3) ^ (d & 7)) << 3) | (t & 7);
}

// Token-major Ps address, same swizzle family as the prompt kernel's
// causal_prompt_wmma_ps (q row-major, 8-key group XOR-swizzled by the q row).
__device__ __forceinline__ int dps(int q, int t, int bc) {
    return q * bc + (((t >> 3) ^ (q & 7)) << 3) | (t & 7);
}

} // namespace decode_wmma_detail

// Block geometry: 4 warps; the WMMA wave is 32 lanes wide, so all four warps run the
// identical 16-row instruction stream (rows 16..63 are padding at T=1) and only
// warp 0's rows carry the kv-head's GroupSize valid queries.
inline constexpr int kDecodeWmmaThreads = 128;
inline constexpr int kDecodeWmmaWarpRows = 16;  // rows per warp (one WMMA C tile)
inline constexpr int kDecodeWmmaBr      = 64;   // total padded query rows (4 warps)
inline constexpr int kDecodeWmmaBc      = 64;   // keys per block
inline constexpr int kDecodeWmmaD       = 256;  // kCausalHeadDim

// Dynamic smem: Vt [D][Bc] bf16 d-major + Ps [Br][Bc] bf16 token-major (per-warp rows).
inline constexpr int kDecodeWmmaVtBytes =
    kDecodeWmmaD * kDecodeWmmaBc * static_cast<int>(sizeof(__nv_bfloat16));    // 32 KiB
inline constexpr int kDecodeWmmaPsBytes =
    kDecodeWmmaBr * kDecodeWmmaBc * static_cast<int>(sizeof(__nv_bfloat16));   // 8 KiB
inline constexpr int kDecodeWmmaSmemBytes = kDecodeWmmaVtBytes + kDecodeWmmaPsBytes;
static_assert(kDecodeWmmaSmemBytes <= 48 * 1024,
              "the decode WMMA kernel must fit the default dynamic-smem ceiling");

// Split-KV decode partial: one CTA per (kv_head, split). Same split-range contract as
// causal_attention_small_t_tc_partial_bf16_kernel (first_tile/units_per_split walking,
// causal_small_t_active_splits) and the same partial m/l/acc indexing, so the
// production reduce kernel consumes the partials unchanged.

// B22: stage the D256-rotated + G64-quantized Q ONCE per (layer, kv_head) into a
// global scratch tile [kv_heads][16 rows][256] + scales [kv_heads][16][4]. Previously
// every split CTA redundantly re-rotated and re-quantized the same Q (85 splits x
// 4 heads x 16 layers = 5440 stagings per decode step @100k), which both wasted ALU
// and saturated the split parallelism. Bit-identical math to the old in-kernel
// staging (same rotation, same fp16 scale boundary, padding rows zero).
template <typename Geometry>
__global__ __launch_bounds__(kDecodeWmmaThreads) void causal_decode_t1_q_stage_kernel(
    const __nv_bfloat16* __restrict__ q, std::int8_t* __restrict__ q_codes_g,
    __half* __restrict__ q_scale_g) {
    constexpr int D      = kDecodeWmmaD;
    constexpr int Groups = kKVCacheInt8Groups;
    constexpr unsigned FullMask = 0xffffffffu;
    const int kv_head = blockIdx.x;
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane0   = tid & 31;
    for (int row = warp; row < kDecodeWmmaWarpRows; row += kDecodeWmmaThreads / 32) {
        const int q_head0 = kv_head * Geometry::GroupSize + row;
        float qv[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int d = lane0 + 32 * u;
            qv[u] = (row < Geometry::GroupSize && causal_valid_q_head<Geometry>(kv_head, q_head0))
                        ? __bfloat162float(q[causal_q_index<Geometry>(q_head0, d, 0)])
                        : 0.0f;
        }
        normalized_hadamard_d256_inplace(qv, lane0);
        std::int8_t* rowp = q_codes_g + (kv_head * kDecodeWmmaWarpRows + row) * D;
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0   = grp * kKVCacheInt8Group + lane0;
            const int d1   = d0 + 32;
            const float x0 = qv[2 * grp];
            const float x1 = qv[2 * grp + 1];
            float absmax   = fmaxf(fabsf(x0), fabsf(x1));
            absmax         = warp_max(absmax, FullMask);
            const auto qp  = kv_cache_int8_quant_params(absmax);
            rowp[d0] = kv_cache_int8_quant_code(x0, qp.inverse_scale);
            rowp[d1] = kv_cache_int8_quant_code(x1, qp.inverse_scale);
            if (lane0 == 0) {
                q_scale_g[(kv_head * kDecodeWmmaWarpRows + row) * Groups + grp] = qp.scale;
            }
        }
    }
}

template <typename Geometry, typename CacheInput, bool KFp16 = false, bool KQ4 = false,
          bool KInt4E8 = false, bool KInt8 = false>
__launch_bounds__(kDecodeWmmaThreads, 1) __global__ void causal_decode_t1_wmma_partial_kernel(
    const __nv_bfloat16* __restrict__ q,
    const std::int8_t* __restrict__ q_codes_g, const __half* __restrict__ q_scale_g,
    CacheInput input, const std::int32_t* __restrict__ pos,
    __nv_bfloat16* __restrict__ cache_k, __half* __restrict__ cache_v,
    __half* __restrict__ cache_k_scale, __half* __restrict__ cache_v_scale,
    const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
    std::int32_t table_stride, std::int32_t logical_capacity, std::int32_t splits_alloc, float scale,
    float* partial_acc, float* partial_m, float* partial_l) {
    using decode_wmma_detail::wmma16;
    using decode_wmma_detail::wmma16_f16;
    constexpr int D       = kDecodeWmmaD;
    constexpr int Bc      = kDecodeWmmaBc;
    constexpr int Br      = kDecodeWmmaBr;
    constexpr int Threads = kDecodeWmmaThreads;
    constexpr int QKKs    = D / 16;  // 16 k-steps over head_dim
    constexpr int QKNt    = Bc / 16; // 4 score tiles per 64-key block
    constexpr int PVKs    = Bc / 16; // 4 PV contraction steps over keys
    constexpr int PVNt    = D / 16;  // 16 PV output n-tiles
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned FullMask  = 0xffffffffu;
    constexpr unsigned LowerMask = 0x0000ffffu;
    constexpr unsigned UpperMask = 0xffff0000u;
    constexpr int PageIds        = 64;
    constexpr int tokens         = 1;  // T=1 decode specialization

    extern __shared__ __align__(16) __nv_bfloat16 decode_wmma_smem[];
    __nv_bfloat16* vt = decode_wmma_smem;  // [D, Bc] d-major
    __nv_bfloat16* ps = vt + D * Bc;       // [Br, Bc] token-major, per-warp 16-row slices
    __shared__ __align__(16) __nv_bfloat16 q_s[kDecodeWmmaWarpRows * D];  // 8 KiB static
    __shared__ std::int32_t physical_pages_s[PageIds];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int row_count   = tokens * Geometry::GroupSize;  // 6 for the 27B shape

    const auto write_neutral = [&] {
        for (int idx = tid; idx < row_count; idx += Threads) {
            const int q_head = kv_head * Geometry::GroupSize + idx;
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, 0, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, 0, split, tokens)] = 0.0f;
                float* acc_row =
                    partial_acc + causal_partial_acc_index<Geometry>(q_head, 0, 0, split, tokens);
                for (int d = 0; d < D; ++d) { acc_row[d] = 0.0f; }
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    const int q_abs = pos[0];
    // The engine's envelope.max_visible_keys is a per-call window hint (often the
    // first window), NOT a growing decode capacity; a decode step's pos exceeds it as
    // soon as the second token. Guard only against garbage (hard upper bound).
    if (q_abs < 0 || q_abs >= (1 << 24)) { write_neutral(); return; }
    const int window = q_abs + 1;
    if constexpr (KInt8) {
#ifdef NINFER_HW_WMMMA
        // =====================================================================
        // B22 per-warp split body (Int8Group64 only): each of the 4 warps owns a
        // DIFFERENT split (grid.y = ceil(splits/4)), so QK/softmax/PV run on all
        // 4 warps concurrently instead of warp 0 alone (~2% card utilization).
        // Vt smem staging is DELETED — PV dequantizes V codes straight from the
        // paged cache into the WMMA B fragment (T=1 reads every V value exactly
        // once, so the smem round-trip bought nothing). Partials land in the
        // same [.., splits] slots the production reducer already consumes.
        // =====================================================================
        constexpr int Groups = kKVCacheInt8Groups;
        const int table_row = table_rows == nullptr ? 0 : table_rows[0];
        const std::int32_t* block_table =
            block_tables + static_cast<std::int64_t>(table_row) * table_stride;
        const int split_count_eff = gridDim.y * 4;
        const int split           = blockIdx.y * 4 + static_cast<int>(warp);
        const int active_split_count =
            causal_small_t_active_splits<Geometry, false>(window, split_count_eff, 1);
        const bool slot_ok = split < splits_alloc;
        const bool active  = split < active_split_count;
        if (active) {
            const int logical_tiles = div_up(window, Bc);
            const bool tile_split   = logical_tiles >= active_split_count;
            const int units_per_split = tile_split ? div_up(logical_tiles, active_split_count)
                                                   : div_up(window, active_split_count);
            const int split_start = split * units_per_split * (tile_split ? Bc : 1);
            const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
            const int split_end   = (split_limit < window) ? split_limit : window;
            const bool nonempty   = split_start < split_end;
            // Fresh K/V row append (exactly one warp's range contains q_abs). The
            // barrier afterwards orders the global write before every warp's QK.
            if constexpr (CacheInput::writes_cache) {
                constexpr unsigned FullMaskW = 0xffffffffu;
                const int p_tok = q_abs;
                const int p_off = p_tok & kPagedKVPageMask;
                if (p_tok >= split_start && p_tok < split_end) {
                    const int physical_page = paged_kv_physical_page(block_table, p_tok);
                    float k_values[8];
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int d = lane + 32 * r;
                        k_values[r] = __bfloat162float(
                            input.k[kv_cache_int8_quant_src_index<Geometry>(kv_head, d, 0)]);
                    }
                    normalized_hadamard_d256_inplace(k_values, lane);
#pragma unroll
                    for (int group = 0; group < Groups; ++group) {
                        const int d0 = group * kKVCacheInt8Group + lane;
                        const int d1 = d0 + 32;
                        const float kv0 = k_values[2 * group];
                        const float kv1 = k_values[2 * group + 1];
                        const std::int64_t src0 =
                            kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, 0);
                        const std::int64_t src1 =
                            kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, 0);
                        const float v0 = __bfloat162float(input.v[src0]);
                        const float v1 = __bfloat162float(input.v[src1]);
                        const float k_abs = warp_max(fmaxf(fabsf(kv0), fabsf(kv1)), FullMaskW);
                        const float v_abs = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMaskW);
                        const auto k_quant = kv_cache_int8_quant_params(k_abs);
                        const auto v_quant = kv_cache_int8_quant_params(v_abs);
                        const std::int64_t code_base =
                            kv_cache_int8_quant_code_index<Geometry>(
                                physical_page, kv_head, group * kKVCacheInt8Group, p_off);
                        cache_k[code_base + lane] =
                            kv_cache_int8_quant_code(kv0, k_quant.inverse_scale);
                        cache_k[code_base + lane + 32] =
                            kv_cache_int8_quant_code(kv1, k_quant.inverse_scale);
                        cache_v[code_base + lane] =
                            kv_cache_int8_quant_code(v0, v_quant.inverse_scale);
                        cache_v[code_base + lane + 32] =
                            kv_cache_int8_quant_code(v1, v_quant.inverse_scale);
                        if (lane == 0) {
                            const std::int64_t scale_off =
                                kv_cache_int8_quant_scale_index<Geometry>(
                                    physical_page, kv_head, group, p_off);
                            cache_k_scale[scale_off] = k_quant.scale;
                            cache_v_scale[scale_off] = v_quant.scale;
                        }
                    }
                }
            }
            __syncthreads();
            if (!nonempty) {
                if (slot_ok) { write_neutral(); }
            } else {
                // ---- per-warp attention over [split_start, split_end) ------------
                const int first_tile = (split_start / Bc) * Bc;
                const int key_blocks = div_up(split_end - first_tile, Bc);
                float acc[PVNt][8];
                float m_run[8];
                float l_run[8];
#pragma unroll
                for (int n = 0; n < PVNt; ++n) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) { acc[n][i] = 0.0f; }
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) { m_run[i] = -CUDART_INF_F; l_run[i] = 0.0f; }
                __nv_bfloat16* ps_w = ps + warp * (kDecodeWmmaWarpRows * Bc);
                const int d_half  = (lane >> 4) * 8;
                const int my_mask = (lane < 16) ? LowerMask : UpperMask;
                int slot_qrow[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { slot_qrow[i] = i + ((lane >= 16) ? 8 : 0); }
                bool slot_valid[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { slot_valid[i] = slot_qrow[i] < row_count; }

                for (int kb = 0; kb < key_blocks; ++kb) {
                    const int k0 = first_tile + kb * Bc;
#ifdef NINFER_K_TILE
                    // ---- K+V+scale tile staging (B26): full-width 128B reads. ----
                    // QK gathers 16 lanes x 8B from 16 DIFFERENT keys (256B stride in
                    // the paged cache) = 8B per 32B sector = 4x DRAM waste. Stage this
                    // block's 64 keys x 256B codes into Vt smem: 128 threads each pull
                    // 128B contiguous (one aligned 128B cache line per thread), then
                    // QK reads the tile with zero global scatter. K plane [0..16K),
                    // V plane [16K..32K), stride 256 (no padding: 32K must fit Vt).
                    // G64 scales (K+V, 512B each) go into the KInt8-unused q_s static.
                    std::int8_t* k_tile = reinterpret_cast<std::int8_t*>(vt);  // [128][256]
                    constexpr int kTileStride = kDecodeWmmaD;
                    const int stage_keys = ((k0 + Bc) <= split_end)
                                               ? Bc
                                               : (split_end > k0 ? split_end - k0 : 0);
                    // zero the padding-key rows (never staged from cache)
                    for (int kl = stage_keys; kl < Bc; ++kl) {
                        if (tid < kTileStride / 4) {
                            reinterpret_cast<int*>(k_tile)[kl * (kTileStride / 4) + tid] = 0;
                            reinterpret_cast<int*>(k_tile)[(Bc + kl) * (kTileStride / 4) + tid] = 0;
                        }
                    }
                    for (int i = tid; i < stage_keys * (kDecodeWmmaD / 4); i += Threads) {
                        const int kl = i / (kDecodeWmmaD / 4);
                        const int d4 = (i % (kDecodeWmmaD / 4)) * 4;
                        const int key    = k0 + kl;
                        const int key_page = block_table[key >> kPagedKVPageShift];
                        const std::int64_t code_off =
                            kv_cache_int8_quant_code_index<Geometry>(
                                key_page, kv_head, d4, key & kPagedKVPageMask);
                        const int4 v = *reinterpret_cast<const int4*>(
                            reinterpret_cast<const std::int8_t*>(cache_k) + code_off);
                        *reinterpret_cast<int4*>(&k_tile[kl * kTileStride + d4]) = v;
                        const int4 vv = *reinterpret_cast<const int4*>(
                            reinterpret_cast<const std::int8_t*>(cache_v) + code_off);
                        *reinterpret_cast<int4*>(&k_tile[(Bc + kl) * kTileStride + d4]) = vv;
                    }
                    // G64 scales: scale_tile[plane K/V][key][group] (half).
                    __half* scale_tile = reinterpret_cast<__half*>(q_s);
                    for (int i = tid; i < Bc * kKVCacheInt8Groups * 2; i += Threads) {
                        const int plane = i / (Bc * kKVCacheInt8Groups);
                        const int rem   = i - plane * (Bc * kKVCacheInt8Groups);
                        const int kl    = rem / kKVCacheInt8Groups;
                        const int grp   = rem - kl * kKVCacheInt8Groups;
                        const int key   = k0 + kl;
                        if (kl < stage_keys) {
                            const int key_page = block_table[key >> kPagedKVPageShift];
                            const std::int64_t so = kv_cache_int8_quant_scale_index<Geometry>(
                                key_page, kv_head, grp, key & kPagedKVPageMask);
                            scale_tile[plane * Bc * kKVCacheInt8Groups +
                                       kl * kKVCacheInt8Groups + grp] =
                                plane == 0 ? cache_k_scale[so] : cache_v_scale[so];
                        } else {
                            scale_tile[plane * Bc * kKVCacheInt8Groups +
                                       kl * kKVCacheInt8Groups + grp] = __float2half(0.0f);
                        }
                    }
                    __syncthreads();
#endif
                    // ---- QK: native int8 WMMA, K codes straight from the cache ----
                    // (B22 A/B: a per-warp staged K tile with XOR swizzle measured
                    // SLOWER -- the two half-warps of a key pull 16B contiguous per
                    // 32B sector (2x waste, not 4x), and the smem round trip +
                    // syncwarp + swizzle ALU cost more than the saved traffic.)
                    float score[QKNt][8];
#pragma unroll
                    for (int nt = 0; nt < QKNt; ++nt) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) { score[nt][i] = 0.0f; }
                    }
                    const std::int8_t* q_codes =
                        q_codes_g + kv_head * kDecodeWmmaWarpRows * D;
#pragma unroll
                    for (int ki = 0; ki < QKKs; ++ki) {
                        unsigned a_code[2] = {0, 0};
                        const int d0a = ki * 16 + d_half;
                        a_code[0] = *reinterpret_cast<const unsigned*>(
                            &q_codes[(lane & 15) * D + d0a]);
                        a_code[1] = *reinterpret_cast<const unsigned*>(
                            &q_codes[(lane & 15) * D + d0a + 4]);
                        const mma_hw_i32x2 av = {static_cast<int>(a_code[0]),
                                                 static_cast<int>(a_code[1])};
#pragma unroll
                        for (int nt = 0; nt < QKNt; ++nt) {
#ifdef NINFER_K_TILE
                            const int key_l    = nt * 16 + (lane & 15);
                            const int d0 = ki * 16 + d_half;
                            // read the staged tile (block-local key index).
                            const std::int8_t* kp = &k_tile[key_l * kTileStride + d0];
                            const unsigned b0 = *reinterpret_cast<const unsigned*>(kp);
                            const unsigned b1 = *reinterpret_cast<const unsigned*>(kp + 4);
                            const int key_abs = k0 + key_l;
                            const int key_page = block_table[key_abs >> kPagedKVPageShift];
#else
                            const int key_l    = k0 + nt * 16 + (lane & 15);
                            const int key_page = block_table[key_l >> kPagedKVPageShift];
                            const int d0 = ki * 16 + d_half;
                            const std::int64_t code_off =
                                kv_cache_int8_quant_code_index<Geometry>(
                                    key_page, kv_head, d0, key_l & kPagedKVPageMask);
                            const std::int8_t* kp =
                                reinterpret_cast<const std::int8_t*>(cache_k) + code_off;
                            const unsigned b0 = *reinterpret_cast<const unsigned*>(kp);
                            const unsigned b1 = *reinterpret_cast<const unsigned*>(kp + 4);
                            const int key_abs = key_l;
#endif
                            const float qsf = __half2float(q_scale_g[
                                (kv_head * kDecodeWmmaWarpRows + (lane & 15)) * Groups +
                                (d0 >> 6)]);
                            #ifdef NINFER_K_TILE
                            const float ksf = __half2float(
                                scale_tile[key_l * kKVCacheInt8Groups + (d0 >> 6)]);
#else
                            const float ksf = __half2float(cache_k_scale[
                                kv_cache_int8_quant_scale_index<Geometry>(
                                    key_page, kv_head, d0 >> 6, key_abs & kPagedKVPageMask)]);
#endif
                            const mma_hw_i32x2 bv = {static_cast<int>(b0),
                                                     static_cast<int>(b1)};
                            const mma_hw_i32x8 czero = {0, 0, 0, 0, 0, 0, 0, 0};
                            const mma_hw_i32x8 di = mma_s8_hw_direct(av, bv, czero);
                            const float qk_scale = qsf * ksf;
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                score[nt][i] = __fmaf_rn(qk_scale,
                                                         static_cast<float>(di[i]), score[nt][i]);
                            }
                        }
                    }
                    // ---- online softmax (warp-local) --------------------------------
#pragma unroll
                    for (int nt = 0; nt < QKNt; ++nt) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int key_abs = k0 + nt * 16 + (lane & 15);
                            const bool keep = slot_valid[i] && key_abs >= split_start &&
                                              key_abs < split_end && key_abs <= q_abs;
                            score[nt][i] = keep ? score[nt][i] : -CUDART_INF_F;
                        }
                    }
                    float bm[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float v = score[0][i];
#pragma unroll
                        for (int nt = 1; nt < QKNt; ++nt) { v = fmaxf(v, score[nt][i]); }
                        bm[i] = warp_max<16>(v, my_mask);
                    }
                    float nm[8];
                    float alpha[8];
                    const float scale_l2 = scale * Log2E;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        nm[i]    = fmaxf(m_run[i], bm[i]);
                        alpha[i] = (m_run[i] == -CUDART_INF_F)
                                       ? 0.0f
                                       : exp2_approx(__fmaf_rn(m_run[i], scale_l2,
                                                               -nm[i] * scale_l2));
                    }
                    float bl[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) { bl[i] = 0.0f; }
#pragma unroll
                    for (int nt = 0; nt < QKNt; ++nt) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float p = (score[nt][i] > -CUDART_INF_F)
                                                ? exp2_approx(__fmaf_rn(score[nt][i], scale_l2,
                                                                        -nm[i] * scale_l2))
                                                : 0.0f;
                            bl[i] += p;
                            ps_w[decode_wmma_detail::dps(slot_qrow[i],
                                                         nt * 16 + (lane & 15), Bc)] =
                                __float2bfloat16_rn(p);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        l_run[i] = __fmaf_rn(l_run[i], alpha[i], warp_sum<16>(bl[i], my_mask));
                        m_run[i] = nm[i];
                    }
#pragma unroll
                    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) { acc[n][i] *= alpha[i]; }
                    }
                    // ---- PV: P (warp smem slice) x V (codes dequantized inline) ----
                    // (B22 A/B: staged bf16 V tile measured slower -- see the QK note.)
#pragma unroll
                    for (int pk = 0; pk < PVKs; ++pk) {
                        const uint4 pv = *reinterpret_cast<const uint4*>(
                            &ps_w[decode_wmma_detail::dps(lane & 15, pk * 16 + d_half, Bc)]);
                        const float aw[4] = {__uint_as_float(pv.x), __uint_as_float(pv.y),
                                             __uint_as_float(pv.z), __uint_as_float(pv.w)};
#pragma unroll
                        for (int n = 0; n < PVNt; ++n) {
#ifdef NINFER_K_TILE
                            const int key_l  = pk * 16 + (lane & 15);
                            const int d0     = n * 16 + d_half;
                            const int key_abs = k0 + key_l;
                            const int key_pg = block_table[key_abs >> kPagedKVPageShift];
                            float bw[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                            if (key_abs < split_end && key_abs <= q_abs) {
                                // B26: read the staged V plane (row 64+key).
                                const std::int8_t* vp = &k_tile[(Bc + key_l) * kTileStride + d0];
                                const int4 vv = *reinterpret_cast<const int4*>(vp);
                                const float vsf = __half2float(
                                    scale_tile[Bc * kKVCacheInt8Groups +
                                               key_l * kKVCacheInt8Groups + (d0 >> 6)]);
                                const float s0 = vsf, s1 = vsf, s2 = vsf, s3 = vsf;
                                bw[0] = __uint_as_float(static_cast<unsigned>(
                                    static_cast<int>(reinterpret_cast<const std::int8_t*>(&vv)[0]) * s0));
                                bw[1] = __uint_as_float(static_cast<unsigned>(
                                    static_cast<int>(reinterpret_cast<const std::int8_t*>(&vv)[1]) * s1));
                                bw[2] = __uint_as_float(static_cast<unsigned>(
                                    static_cast<int>(reinterpret_cast<const std::int8_t*>(&vv)[2]) * s2));
                                bw[3] = __uint_as_float(static_cast<unsigned>(
                                    static_cast<int>(reinterpret_cast<const std::int8_t*>(&vv)[3]) * s3));
                            }
#else
                            const int key_l  = k0 + pk * 16 + (lane & 15);
                            const int key_pg = block_table[key_l >> kPagedKVPageShift];
                            const int d0     = n * 16 + d_half;
                            float bw[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                            if (key_l < split_end && key_l <= q_abs) {
                                const std::int64_t code_off =
                                    kv_cache_int8_quant_code_index<Geometry>(
                                        key_pg, kv_head, d0, key_l & kPagedKVPageMask);
                                const float vsf = __half2float(cache_v_scale[
                                    kv_cache_int8_quant_scale_index<Geometry>(
                                        key_pg, kv_head, d0 >> 6, key_l & kPagedKVPageMask)]);
                                const int4 vv = kv_cache_int8_dequant_i8x8_from(
                                    reinterpret_cast<const std::int8_t*>(cache_v) + code_off,
                                    vsf);
                                bw[0] = __uint_as_float(static_cast<unsigned>(vv.x));
                                bw[1] = __uint_as_float(static_cast<unsigned>(vv.y));
                                bw[2] = __uint_as_float(static_cast<unsigned>(vv.z));
                                bw[3] = __uint_as_float(static_cast<unsigned>(vv.w));
                            }
#endif
                            wmma16(acc[n], aw, bw, acc[n]);
                        }
                    }
                }
                // ---- epilogue: write this warp's split partials ------------------
                if (slot_ok) {
                    if (lane == 0) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int row = slot_qrow[i];
                            if (row < row_count) {
                                const int q_head = kv_head * Geometry::GroupSize + row;
                                if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                                    partial_m[causal_partial_stat_index<Geometry>(
                                        q_head, 0, split, tokens)] = m_run[i] * scale;
                                    partial_l[causal_partial_stat_index<Geometry>(
                                        q_head, 0, split, tokens)] = l_run[i];
                                }
                            }
                        }
                    }
#pragma unroll
                    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int row = slot_qrow[i];
                            if (row < row_count) {
                                const int q_head = kv_head * Geometry::GroupSize + row;
                                if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                                    const int d = n * 16 + (lane & 15);
                                    float* dst  = partial_acc +
                                        causal_partial_acc_index<Geometry>(q_head, d, 0, split,
                                                                           tokens);
                                    // acc is already the softmax-weighted V sum; the
                                    // reducer divides by l without any scale. The
                                    // softmax's own 1/sqrt(d) scale is folded into the
                                    // exp2 path and partial_m, never into acc.
                                    dst[0] = acc[n][i];
                                }
                            }
                        }
                    }
                }
            }
        }
        return;  // KInt8 body ends here — the single-warp body below is other storages
#endif // NINFER_HW_WMMMA
    }
    const int active_split_count =
        causal_small_t_active_splits<Geometry, false>(window, split_count, 1);
    if (split >= active_split_count) { return; }

    const int table_row             = table_rows == nullptr ? 0 : table_rows[0];
    const std::int32_t* block_table = block_tables + static_cast<std::int64_t>(table_row) * table_stride;

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) { write_neutral(); return; }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }
    __syncthreads();

    // ---- Stage this CTA's 16 q rows once (rows = the kv-head's q group, 6 valid). --
    // Rows are NOT swizzled: row stride is 256 bf16 = 512 B, so each row's 16-d window
    // is one aligned 16 B load and A-fragment reads never conflict (lane = row).
    // KInt8 reads the pre-staged rotated+quantized Q from the global scratch tile
    // (causal_decode_t1_q_stage_kernel runs once per layer instead of per CTA —
    // B22: the per-CTA redundancy saturated split parallelism); only KInt4E8 still
    // stages rotated bf16 Q locally.
    if constexpr (KInt4E8) {
        // rk4v4-e8 stores the D256-Hadamard-rotated K; the WMMA Q fragment must be
        // rotated identically. One lane holds its stride-32 column (8 dims lane+32u),
        // rotates in place exactly like the SIMT path, and stores bf16 (its K
        // dequantizes to bf16 per lane; there is no symmetric i8 Q carrier).
        for (int row = static_cast<int>(tid) / 32; row < kDecodeWmmaWarpRows;
             row += Threads / 32) {
            const int lane0 = static_cast<int>(tid) & 31;
            const int q_head0 = kv_head * Geometry::GroupSize + row;
            float qv[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int d = lane0 + 32 * u;
                qv[u] = (row < row_count && causal_valid_q_head<Geometry>(kv_head, q_head0))
                            ? __bfloat162float(q[causal_q_index<Geometry>(q_head0, d, 0)])
                            : 0.0f;
            }
            normalized_hadamard_d256_inplace(qv, lane0);
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                q_s[row * D + lane0 + 32 * u] = __float2bfloat16_rn(qv[u]);
            }
        }
    } else
    for (int idx = tid; idx < kDecodeWmmaWarpRows * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        if constexpr (KFp16 || KQ4) {
            // Q is bf16 in global; the fp16 WMMA A operand needs fp16 (rounded once
            // at staging, negligible vs the per-block wmma work).
            __half value = __ushort_as_half(0);
            if (row < row_count) {
                const int q_head = kv_head * Geometry::GroupSize + row;
                if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                    value = __float2half_rn(__bfloat162float(
                        q[causal_q_index<Geometry>(q_head, d, 0)]));
                }
            }
            reinterpret_cast<__half*>(q_s)[row * D + d] = value;
        } else {
            __nv_bfloat16 value = __float2bfloat16(0.0f);
            if (row < row_count) {
                const int q_head = kv_head * Geometry::GroupSize + row;
                if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                    value = q[causal_q_index<Geometry>(q_head, d, 0)];
                }
            }
            q_s[row * D + d] = value;
        }
    }
    __syncthreads();

    // HW fragment indices (identical mapping to the prompt WMMA kernel):
    // QK A / PV A rows: lane L<16 = row L, lanes 16-31 = the +8 half of the 16-d window
    // (QK A) or the +8 key half (PV A). B: lane L<16 = key col L, lanes 16-31 = d+8.
    const int a_row    = warp * kDecodeWmmaWarpRows + (lane & 15);
    const int d_half   = (lane >> 4) * 8;
    const int my_mask  = (lane < 16) ? LowerMask : UpperMask;

    // C-slot rows: slot i of this lane covers padded row warp*16 + i + 8*(lane>=16).
    // Only warp 0 has valid rows (row_count = GroupSize <= 16); the other warps run on
    // zeroed P/zeroed V columns and never store anything.
    int slot_qrow[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { slot_qrow[i] = warp * kDecodeWmmaWarpRows + i + ((lane >= 16) ? 8 : 0); }
    bool slot_valid[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { slot_valid[i] = slot_qrow[i] < row_count; }

    float acc[PVNt][8];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[n][i] = 0.0f; }
    }
    float m_run[8];
    float l_run[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { m_run[i] = -CUDART_INF_F; l_run[i] = 0.0f; }

    // Zero the whole Ps buffer once: padded warps then run on zeros (softmax of all
    // -inf lanes -> p=0 via the keep-gate below) and nothing leaks between tiles.
    for (int idx = tid; idx < Br * Bc; idx += Threads) { ps[idx] = __float2bfloat16(0.0f); }

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

        // ---- write the fresh K/V row (q_abs) into the paged cache (writes_cache) ----
        // The engine's decode KV update happens HERE (production small-T semantics): the
        // fresh row is stored into cache_k/cache_v at q_abs; QK/Vt below read everything
        // from the cache, so later decode steps see the full history.
        if constexpr (CacheInput::writes_cache) {
            if constexpr (KInt8) {
                // Int8Group64 planes: K is D256-Hadamard-rotated then G64-quantized,
                // V is plain G64-quantized (same codec, no rotation). One warp owns the
                // full D=256 row: lane l holds dims l + 32u, rotates/quantizes K and
                // quantizes V in registers.
                constexpr unsigned FullMaskW = 0xffffffffu;
                const int p_tok             = q_abs;
                const int p_off             = p_tok & kPagedKVPageMask;
                if (p_tok >= split_start && p_tok < split_end) {
                    const int physical_page = paged_kv_physical_page(block_table, p_tok);
                    // Load K row, rotate in place.
                    float k_values[8];
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int d = lane + 32 * r;
                        k_values[r] = __bfloat162float(
                            input.k[kv_cache_int8_quant_src_index<Geometry>(kv_head, d, 0)]);
                    }
                    normalized_hadamard_d256_inplace(k_values, lane);
#pragma unroll
                    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
                        const int d0 = group * kKVCacheInt8Group + lane;
                        const int d1 = d0 + 32;
                        const float k0 = k_values[2 * group];
                        const float k1 = k_values[2 * group + 1];
                        const std::int64_t src0 =
                            kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, 0);
                        const std::int64_t src1 =
                            kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, 0);
                        const float v0 = __bfloat162float(input.v[src0]);
                        const float v1 = __bfloat162float(input.v[src1]);
                        const float k_abs = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMaskW);
                        const float v_abs = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMaskW);
                        const auto k_quant = kv_cache_int8_quant_params(k_abs);
                        const auto v_quant = kv_cache_int8_quant_params(v_abs);
                        const std::int64_t code_base =
                            kv_cache_int8_quant_code_index<Geometry>(
                                physical_page, kv_head, group * kKVCacheInt8Group, p_off);
                        cache_k[code_base + lane] =
                            kv_cache_int8_quant_code(k0, k_quant.inverse_scale);
                        cache_k[code_base + lane + 32] =
                            kv_cache_int8_quant_code(k1, k_quant.inverse_scale);
                        cache_v[code_base + lane] =
                            kv_cache_int8_quant_code(v0, v_quant.inverse_scale);
                        cache_v[code_base + lane + 32] =
                            kv_cache_int8_quant_code(v1, v_quant.inverse_scale);
                        if (lane == 0) {
                            const std::int64_t scale_off =
                                kv_cache_int8_quant_scale_index<Geometry>(
                                    physical_page, kv_head, group, p_off);
                            cache_k_scale[scale_off] = k_quant.scale;
                            cache_v_scale[scale_off] = v_quant.scale;
                        }
                    }
                }
            }
            else if constexpr (KQ4) {
                // q4_0 planes: 8 blocks per row, one warp per block, two chunk passes.
                // Two passes over K/V and two over block halves (128 threads).
                constexpr int BlocksPerRow = D / kQ4_0Block;  // 8
                const int p_tok            = q_abs;
                const int lane             = tid & 31;
                // AMD port: resolve the physical page with a direct block-table read
                // (no thread0 + __shfl_sync broadcast; that pattern is unreliable on
                // this driver once the grid / page index grows).
                const int physical_page = paged_kv_physical_page(block_table, p_tok);
                if (p_tok >= split_start && p_tok < split_end) {
                    const std::int64_t code_base = q4_0_code_offset<Geometry::KVHeads>(
                        physical_page, kv_head, p_tok & kPagedKVPageMask, 0);
                    const std::int64_t scale_base = q4_0_scale_offset<Geometry::KVHeads>(
                        physical_page, kv_head, p_tok & kPagedKVPageMask, 0);
                    for (int kv = 0; kv < 2; ++kv) {
                        for (int half = 0; half < 2; ++half) {
                            const int block = (tid >> 5) + half * (Threads >> 5);
                            if (block >= BlocksPerRow) { continue; }
                            const int d = block * kQ4_0Block + lane;
                            const std::int64_t new_off =
                                kv_cache_int8_new_index<Geometry>(kv_head, d, 0);
                            if (kv == 0) {
                                q4_0_block_encode(
                                    __bfloat162float(input.k[new_off]), lane,
                                    reinterpret_cast<std::uint8_t*>(cache_k) + code_base +
                                        block * kQ4_0CodesPerBlock,
                                    cache_k_scale[scale_base + block]);
                            } else {
                                q4_0_block_encode(
                                    __bfloat162float(input.v[new_off]), lane,
                                    reinterpret_cast<std::uint8_t*>(cache_v) + code_base +
                                        block * kQ4_0CodesPerBlock,
                                    cache_v_scale[scale_base + block]);
                            }
                        }
                    }
                }
            } else {
                for (int chunk = tid; chunk < (D / 8); chunk += Threads) {
                    const int d     = chunk * 8;
                    const int p_tok = q_abs;
                    if (p_tok >= split_start && p_tok < split_end) {
                        const std::int64_t new_off =
                            kv_cache_int8_new_index<Geometry>(kv_head, d, 0);
                        const int physical_page = paged_kv_physical_page(block_table, p_tok);
                        const std::int64_t cache_off = causal_cache_index<Geometry>(
                            physical_page, kv_head, d, p_tok & kPagedKVPageMask);
                        if constexpr (KFp16) {
                            // K plane is FP16: convert the bf16 K row at store time.
                            store_vec(reinterpret_cast<__half*>(&cache_k[cache_off]),
                                      bf16x8_bits_to_f16x8_bits(load_vec<int4>(&input.k[new_off])));
                        } else {
                            store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                        }
                        store_vec(&cache_v[cache_off],
                                  bf16x8_bits_to_f16x8_bits(load_vec<int4>(&input.v[new_off])));
                    }
                }
            }
            __syncthreads();
        }

        // ---- stage V(kb) transposed (fp16 cache -> bf16 tile) ----------------------
        __syncthreads();  // previous tile's Vt/Ps consumers are done
        if constexpr (KInt8) {
            // Int8Group64 V: each thread owns ONE key's 32 contiguous dims (key-major).
            // Global V codes are d-contiguous per key, so this is a full-width 32B read
            // per thread and the key's G64 scales load once per 32-dim block (vs the
            // d-major mapping: 32 different keys at 256B stride = 32 scattered 8B
            // reads + one scale each). Probe-verified bit-exact on single-block AND
            // multi-block+padding (probe_vt_keymajor_multi.cu).
            constexpr int ChunksPerKey = D / 32;  // 8
            for (int chunk = tid; chunk < Bc * ChunksPerKey; chunk += Threads) {
                const int key_l = chunk / ChunksPerKey;
                const int d32   = (chunk - key_l * ChunksPerKey) * 32;
                const int key   = k0 + key_l;
                const bool valid = key < split_end && key <= q_abs;
                const auto lo = [](unsigned bits) {
                    return __ushort_as_bfloat16(static_cast<unsigned short>(bits & 0xffffu));
                };
                const auto hi = [](unsigned bits) {
                    return __ushort_as_bfloat16(
                        static_cast<unsigned short>((bits >> 16) & 0xffffu));
                };
                if (valid) {
                    const int key_page =
                        physical_pages_s[(key >> kPagedKVPageShift) - first_page];
                    const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
                        key_page, kv_head, d32, key & kPagedKVPageMask);
                    const std::int8_t* vp =
                        reinterpret_cast<const std::int8_t*>(cache_v) + code_base;
                    // 32 contiguous dims span two G64 groups (d32..d32+15, +16..+31).
                    const int grp0 = d32 >> 6;
                    const int grp1 = (d32 + 16) >> 6;
                    const float vsf0 = __half2float(cache_v_scale[
                        kv_cache_int8_quant_scale_index<Geometry>(
                            key_page, kv_head, grp0, key & kPagedKVPageMask)]);
                    const float vsf1 = __half2float(cache_v_scale[
                        kv_cache_int8_quant_scale_index<Geometry>(
                            key_page, kv_head, grp1, key & kPagedKVPageMask)]);
                    if (isnan(vsf0) || isinf(vsf0) || isnan(vsf1) || isinf(vsf1)) {
                        printf("[dec-t1w Vt] BAD V scale pos=%d head=%d key=%d d32=%d page=%d vsf=%f/%f\n",
                               q_abs, kv_head, key, d32, key_page, vsf0, vsf1);
                    }
#pragma unroll
                    for (int g = 0; g < 2; ++g) {
                        const int d     = d32 + g * 16;
                        const float vsf = (g == 0) ? vsf0 : vsf1;
                        const auto put8 = [&](int off, const int4& vv) {
                            vt[decode_wmma_detail::dvt(d + off + 0, key_l, Bc)] = lo(static_cast<unsigned>(vv.x));
                            vt[decode_wmma_detail::dvt(d + off + 1, key_l, Bc)] = hi(static_cast<unsigned>(vv.x));
                            vt[decode_wmma_detail::dvt(d + off + 2, key_l, Bc)] = lo(static_cast<unsigned>(vv.y));
                            vt[decode_wmma_detail::dvt(d + off + 3, key_l, Bc)] = hi(static_cast<unsigned>(vv.y));
                            vt[decode_wmma_detail::dvt(d + off + 4, key_l, Bc)] = lo(static_cast<unsigned>(vv.z));
                            vt[decode_wmma_detail::dvt(d + off + 5, key_l, Bc)] = hi(static_cast<unsigned>(vv.z));
                            vt[decode_wmma_detail::dvt(d + off + 6, key_l, Bc)] = lo(static_cast<unsigned>(vv.w));
                            vt[decode_wmma_detail::dvt(d + off + 7, key_l, Bc)] = hi(static_cast<unsigned>(vv.w));
                        };
                        put8(0, kv_cache_int8_dequant_i8x8_from(vp + g * 16, vsf));
                        put8(8, kv_cache_int8_dequant_i8x8_from(vp + g * 16 + 8, vsf));
                    }
                } else {
                    // padding key: explicit zero Vt row (B22 fix; stale smem values
                    // would otherwise leak into the PV product).
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        vt[decode_wmma_detail::dvt(d32 + i, key_l, Bc)] =
                            __float2bfloat16_rn(0.0f);
                    }
                }
            }
        } else {
            constexpr int VecPerRow = D / 8;  // 16B = 8 fp16
            for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
                const int key_l = chunk >> 5;
                const int d     = (chunk & 31) << 3;
                const int key   = k0 + key_l;
                int4 raw        = make_int4(0, 0, 0, 0);
                if (key < split_end && key <= q_abs) {
                    const int key_page =
                        physical_pages_s[(key >> kPagedKVPageShift) - first_page];
                    const __half* v_ptr =
                        cache_v + paged_kv_element_offset<D, Geometry::KVHeads>(
                                      key_page, kv_head, key & kPagedKVPageMask, d);
                    raw = load_vec<int4>(v_ptr);
                }
                if constexpr (KInt4E8) {
                    // rk4v4-e8 V plane: packed signed int4 + G32 scales. 8 values =
                    // 4 code bytes (uint32); byte (d+i)>>1 holds dim (d+i) in the low
                    // nibble when even, high when odd. Dequant to the bf16 Vt tile.
                    if (key < split_end && key <= q_abs) {
                        const int key_page =
                            physical_pages_s[(key >> kPagedKVPageShift) - first_page];
                        const std::int64_t code_off = kv_cache_int4_value_code_index<Geometry>(
                            key_page, kv_head, d >> 1, key & kPagedKVPageMask);
                        const std::uint32_t codes4 = *reinterpret_cast<const std::uint32_t*>(
                            reinterpret_cast<const std::uint8_t*>(cache_v) + code_off);
                        const float vsf = __half2float(cache_v_scale[
                            kv_cache_int4_value_scale_index<Geometry>(
                                key_page, kv_head, d >> 5, key & kPagedKVPageMask)]);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int byte_off = i >> 1;
                            const int nib      = (codes4 >> (8 * byte_off + 4 * ((d + i) & 1))) & 0x0fu;
                            vt[decode_wmma_detail::dvt(d + i, key_l, Bc)] =
                                __float2bfloat16_rn(static_cast<float>((nib ^ 8) - 8) * vsf);
                        }
                    }
                } else if constexpr (KQ4) {
                    // q4_0 V plane: 8 values = 4 code bytes + 1 scale; dequant to
                    // fp16 Vt tile (d%32 in {0,8,16,24}, window never crosses blocks).
                    __half* vt16 = reinterpret_cast<__half*>(vt);
                    if (key < split_end && key <= q_abs) {
                        const int key_page =
                            physical_pages_s[(key >> kPagedKVPageShift) - first_page];
                        const std::int64_t code_off = q4_0_code_offset<Geometry::KVHeads>(
                            key_page, kv_head, key & kPagedKVPageMask,
                            (d / kQ4_0Block) * kQ4_0Block) +
                            (d & 15);
                        const std::int64_t scale_off = q4_0_scale_offset<Geometry::KVHeads>(
                            key_page, kv_head, key & kPagedKVPageMask, d);
                        const std::uint64_t codes8 = *reinterpret_cast<const std::uint64_t*>(
                            reinterpret_cast<const std::uint8_t*>(cache_v) + code_off);
                        const __half sc = cache_v_scale[scale_off];
                        float fv[8];
                        q4_0_dequant8(fv, codes8, (d & 16) == 0, sc);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            vt16[decode_wmma_detail::dvt(d + i, key_l, Bc)] =
                                __float2half_rn(fv[i]);
                        }
                    }
                } else if constexpr (KFp16) {
                    // V plane is already FP16: raw copy into the fp16 Vt tile (no
                    // fp16->f32->bf16 scalar round-trip).
                    __half* vt16 = reinterpret_cast<__half*>(vt);
                    const auto half_from = [](unsigned b, int s) {
                        return __ushort_as_half(
                            static_cast<unsigned short>((b >> s) & 0xffffu));
                    };
                    const unsigned x = static_cast<unsigned>(raw.x);
                    const unsigned y = static_cast<unsigned>(raw.y);
                    const unsigned z = static_cast<unsigned>(raw.z);
                    const unsigned w = static_cast<unsigned>(raw.w);
                    vt16[decode_wmma_detail::dvt(d + 0, key_l, Bc)] = half_from(x, 0);
                    vt16[decode_wmma_detail::dvt(d + 1, key_l, Bc)] = half_from(x, 16);
                    vt16[decode_wmma_detail::dvt(d + 2, key_l, Bc)] = half_from(y, 0);
                    vt16[decode_wmma_detail::dvt(d + 3, key_l, Bc)] = half_from(y, 16);
                    vt16[decode_wmma_detail::dvt(d + 4, key_l, Bc)] = half_from(z, 0);
                    vt16[decode_wmma_detail::dvt(d + 5, key_l, Bc)] = half_from(z, 16);
                    vt16[decode_wmma_detail::dvt(d + 6, key_l, Bc)] = half_from(w, 0);
                    vt16[decode_wmma_detail::dvt(d + 7, key_l, Bc)] = half_from(w, 16);
                } else {
                    const __half2 h0 = half2_from_bits(static_cast<std::uint32_t>(raw.x));
                    const __half2 h1 = half2_from_bits(static_cast<std::uint32_t>(raw.y));
                    const __half2 h2 = half2_from_bits(static_cast<std::uint32_t>(raw.z));
                    const __half2 h3 = half2_from_bits(static_cast<std::uint32_t>(raw.w));
                    const float2 f0 = __half22float2(h0);
                    const float2 f1 = __half22float2(h1);
                    const float2 f2 = __half22float2(h2);
                    const float2 f3 = __half22float2(h3);
                    vt[decode_wmma_detail::dvt(d + 0, key_l, Bc)] = __float2bfloat16_rn(f0.x);
                    vt[decode_wmma_detail::dvt(d + 1, key_l, Bc)] = __float2bfloat16_rn(f0.y);
                    vt[decode_wmma_detail::dvt(d + 2, key_l, Bc)] = __float2bfloat16_rn(f1.x);
                    vt[decode_wmma_detail::dvt(d + 3, key_l, Bc)] = __float2bfloat16_rn(f1.y);
                    vt[decode_wmma_detail::dvt(d + 4, key_l, Bc)] = __float2bfloat16_rn(f2.x);
                    vt[decode_wmma_detail::dvt(d + 5, key_l, Bc)] = __float2bfloat16_rn(f2.y);
                    vt[decode_wmma_detail::dvt(d + 6, key_l, Bc)] = __float2bfloat16_rn(f3.x);
                    vt[decode_wmma_detail::dvt(d + 7, key_l, Bc)] = __float2bfloat16_rn(f3.y);
                }
            }
        }
        __syncthreads();

        // ---- S = Q K^T over this 64-key block (all in registers) -------------------
#ifdef NINFER_DECODE_PV_MARK
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // marker 1 = reached after Vt staging; sentinel at acc tail (host allocates +512)
            partial_acc[(static_cast<std::size_t>(gridDim.y) * Geometry::QHeads * D) + 0] = 1.0f;
            // Vt dump: vt[d][key] for d in {0,8,16,24,32,40,48,56}, key=0..3 at tail + 8..
            for (int dd = 0; dd < 8; ++dd) {
                for (int kk = 0; kk < 4; ++kk) {
                    partial_acc[(static_cast<std::size_t>(gridDim.y) * Geometry::QHeads * D) + 8 +
                                static_cast<std::size_t>(dd) * 4 + kk] =
                        static_cast<float>(vt[decode_wmma_detail::dvt(dd * 8, kk, Bc)]);
                }
            }
        }
#endif
        // T=1: only warp 0 carries the kv-head's 6 valid query rows; warps 1-3 would
        // run the exact same 16-row WMMA instruction stream on padding rows, wasting
        // 4x compute. Run the whole QK/softmax/PV/epilogue on warp 0 only (staging
        // above already used all 128 threads).
        if (warp == 0) {
        float score[QKNt][8];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { score[nt][i] = 0.0f; }
        }
#pragma unroll
        for (int ki = 0; ki < QKKs; ++ki) {
            // Rows wrap into the 16-row q tile: warps 1-3 (padding rows 16..63 at T=1)
            // read warp 0's rows instead of out-of-bounds shared memory. Their outputs
            // are never stored.
            float aw[4];
            unsigned a_code[2] = {0, 0};  // gfx12-layout raw int8 carriers (KInt8 HW path)
            if constexpr (KInt8) {
#ifdef NINFER_HW_WMMMA
                // B22: read the pre-staged rotated+quantized Q from the global scratch
                // tile [kv_heads][16][256] (staged once per layer by
                // causal_decode_t1_q_stage_kernel; padding rows are zero).
                const std::int8_t* q_codes =
                    q_codes_g + kv_head * kDecodeWmmaWarpRows * D;
                const int row_idx = ((warp * kDecodeWmmaWarpRows + (lane & 15)) &
                                     (kDecodeWmmaWarpRows - 1));
                // Rotated-Q int8 codes, d-contiguous (no swizzle): the 8 codes of the
                // 16-d window are contiguous at q_codes[row*D + ki*16 + d_half]. gfx12
                // i8 A layout: lane L<16 = row L, k0..7; lane L>=16 = row L-16, k8..15.
                const int d0 = ki * 16 + d_half;
                a_code[0] = *reinterpret_cast<const unsigned*>(&q_codes[row_idx * D + d0]);
                a_code[1] = *reinterpret_cast<const unsigned*>(&q_codes[row_idx * D + d0 + 4]);
#else
                const uint4 qv = *reinterpret_cast<const uint4*>(
                    &q_s[((warp * kDecodeWmmaWarpRows + (lane & 15)) &
                          (kDecodeWmmaWarpRows - 1)) * D + ki * 16 + d_half]);
                aw[0] = __uint_as_float(qv.x);
                aw[1] = __uint_as_float(qv.y);
                aw[2] = __uint_as_float(qv.z);
                aw[3] = __uint_as_float(qv.w);
#endif
            } else {
                const uint4 qv = *reinterpret_cast<const uint4*>(
                    &q_s[((warp * kDecodeWmmaWarpRows + (lane & 15)) &
                          (kDecodeWmmaWarpRows - 1)) * D + ki * 16 + d_half]);
                aw[0] = __uint_as_float(qv.x);
                aw[1] = __uint_as_float(qv.y);
                aw[2] = __uint_as_float(qv.z);
                aw[3] = __uint_as_float(qv.w);
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key_l    = k0 + nt * 16 + (lane & 15);
                const int key_page = physical_pages_s[(key_l >> kPagedKVPageShift) - first_page];
                float bw[4];
                unsigned b_code[2] = {0, 0};  // gfx12-layout raw int8 carriers (KInt8 HW)
                if constexpr (KInt8) {
#ifdef NINFER_HW_WMMMA
                    // int8 G64 K (rotated in cache): the 8 code bytes for dims
                    // [d0, d0+8) are contiguous (d-major), already in the gfx12 i8 B
                    // layout: lane L<16 = k row L, n0..7; lane L>=16 = k row L-16,
                    // n8..15. Read raw, integer-contract on the WMMA.
                    const int d0    = ki * 16 + d_half;
                    const std::int64_t code_off = kv_cache_int8_quant_code_index<Geometry>(
                        key_page, kv_head, d0, key_l & kPagedKVPageMask);
                    const std::int8_t* kp =
                        reinterpret_cast<const std::int8_t*>(cache_k) + code_off;
                    b_code[0] = *reinterpret_cast<const unsigned*>(kp);
                    b_code[1] = *reinterpret_cast<const unsigned*>(kp + 4);
#else
                    // int8 G64 K (rotated): 8 code bytes + one G64 scale, dequant to
                    // bf16 per lane (d0%64 window never crosses groups).
                    const int d0    = ki * 16 + d_half;
                    const std::int64_t code_off = kv_cache_int8_quant_code_index<Geometry>(
                        key_page, kv_head, d0, key_l & kPagedKVPageMask);
                    const float ksf = __half2float(cache_k_scale[kv_cache_int8_quant_scale_index<
                        Geometry>(key_page, kv_head, d0 >> 6, key_l & kPagedKVPageMask)]);
                    // B20 P0-2: track max |K scale| (fp16 precision check, CAS loop).
                    {
                        const float abs_s = fabsf(ksf);
                        float cur = ninfer_kv_scale_absmax;
                        while (abs_s > cur) {
                            const float prev =
                                atomicCAS(reinterpret_cast<int*>(&ninfer_kv_scale_absmax),
                                          __float_as_int(cur), __float_as_int(abs_s));
                            if (prev == __float_as_int(cur)) { break; }
                            cur = __int_as_float(prev);
                        }
                    }
                    const int4 ki4 = kv_cache_int8_dequant_i8x8_from(
                        reinterpret_cast<const std::int8_t*>(cache_k) + code_off, ksf);
                    bw[0] = __uint_as_float(static_cast<unsigned>(ki4.x));
                    bw[1] = __uint_as_float(static_cast<unsigned>(ki4.y));
                    bw[2] = __uint_as_float(static_cast<unsigned>(ki4.z));
                    bw[3] = __uint_as_float(static_cast<unsigned>(ki4.w));
#endif
                } else if constexpr (KInt4E8) {
                    // int4-E8 K: 8 values = 4 code bytes + one G64 scale. Byte
                    // (d0+i)>>1 holds dim (d0+i) in the low nibble when even, high
                    // when odd. The K row is 128 packed bytes per token head; d0 is
                    // a multiple of 16, so the 4 bytes never cross the 128-byte row
                    // (d0>>1 in {0,4,...,124}; +4 stays inside).
                    const int d0    = ki * 16 + d_half;
                    const std::int64_t code_off =
                        kv_cache_int4_value_code_index<Geometry>(
                            key_page, kv_head, d0 >> 1, key_l & kPagedKVPageMask);
                    const std::uint32_t codes4 = *reinterpret_cast<const std::uint32_t*>(
                        reinterpret_cast<const std::uint8_t*>(cache_k) + code_off);
                    const float ksf = __half2float(cache_k_scale[
                        kv_cache_int8_quant_scale_index<Geometry>(
                            key_page, kv_head, d0 >> 6, key_l & kPagedKVPageMask)]);
                    alignas(16) __nv_bfloat16 bv8[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int byte_off = i >> 1;
                        const int nib      = (codes4 >> (8 * byte_off + 4 * ((d0 + i) & 1))) & 0x0fu;
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
                    const int d0    = ki * 16 + d_half;
                    const std::int64_t code_off =
                        q4_0_code_offset<Geometry::KVHeads>(
                            key_page, kv_head, key_l & kPagedKVPageMask,
                            (d0 / kQ4_0Block) * kQ4_0Block) +
                        (d0 & 15);
                    const std::int64_t scale_off = q4_0_scale_offset<Geometry::KVHeads>(
                        key_page, kv_head, key_l & kPagedKVPageMask, d0);
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
                                      key_page, kv_head, key_l & kPagedKVPageMask,
                                      ki * 16 + d_half);
                    const uint4 kv4 = *reinterpret_cast<const uint4*>(k_ptr);
                    bw[0] = __uint_as_float(kv4.x);
                    bw[1] = __uint_as_float(kv4.y);
                    bw[2] = __uint_as_float(kv4.z);
                    bw[3] = __uint_as_float(kv4.w);
                }
                if constexpr (KInt8) {
#ifdef NINFER_HW_WMMMA
                    // Native int8 QK: integer s32 dot product on the WMMA, then scale
                    // by qs*ks. G64 group = d0>>6 (64 dims, invariant over the 8-code
                    // window), so one scale pair per (ki, key) applies to all 8 slots.
                    const int d0 = ki * 16 + d_half;
                    const int row_idx = ((warp * kDecodeWmmaWarpRows + (lane & 15)) &
                                         (kDecodeWmmaWarpRows - 1));
                    const float qsf = __half2float(
                        q_scale_g[(kv_head * kDecodeWmmaWarpRows + row_idx) *
                                      kKVCacheInt8Groups +
                                  (d0 >> 6)]);
                    const float ksf = __half2float(cache_k_scale[
                        kv_cache_int8_quant_scale_index<Geometry>(key_page, kv_head, d0 >> 6,
                                                                  key_l & kPagedKVPageMask)]);
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
                        score[nt][i] =
                            __fmaf_rn(qk_scale, static_cast<float>(di[i]), score[nt][i]);
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

        // ---- online softmax over the 64-key block (per C slot) ---------------------
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int key_abs = k0 + nt * 16 + (lane & 15);
                const bool keep =
                    slot_valid[i] && key_abs >= split_start && key_abs < split_end &&
                    key_abs <= q_abs;
                score[nt][i] = keep ? score[nt][i] : -CUDART_INF_F;
            }
        }
        float bm[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v = score[0][i];
#pragma unroll
            for (int nt = 1; nt < QKNt; ++nt) { v = fmaxf(v, score[nt][i]); }
            bm[i] = warp_max<16>(v, my_mask);
        }
        float nm[8];
        float alpha[8];
        const float scale_l2 = scale * Log2E;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            nm[i]    = fmaxf(m_run[i], bm[i]);
            alpha[i] = (m_run[i] == -CUDART_INF_F)
                           ? 0.0f
                           : exp2_approx(__fmaf_rn(m_run[i], scale_l2, -nm[i] * scale_l2));
        }
        float bl[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { bl[i] = 0.0f; }
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = (score[nt][i] > -CUDART_INF_F)
                                    ? exp2_approx(__fmaf_rn(score[nt][i], scale_l2,
                                                            -nm[i] * scale_l2))
                                    : 0.0f;
                bl[i] += p;
                // Ps row = this lane's slot row within its own warp slice; padded
                // slots (slot_valid false) got p = 0 above and store 0.
                if constexpr (KFp16 || KQ4) {
                    reinterpret_cast<__half*>(ps)[decode_wmma_detail::dps(
                        slot_qrow[i], nt * 16 + (lane & 15), Bc)] = __float2half_rn(p);
                } else {
                    ps[decode_wmma_detail::dps(slot_qrow[i], nt * 16 + (lane & 15), Bc)] =
                        __float2bfloat16_rn(p);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            l_run[i] = __fmaf_rn(l_run[i], alpha[i], warp_sum<16>(bl[i], my_mask));
            m_run[i] = nm[i];
        }
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc[n][i] *= alpha[i]; }
        }
        __syncthreads();  // Ps complete before the PV reads

        // ---- O += P V ---------------------------------------------------------------
        // P row padding keys (>= split_end) hold p = 0 and V columns are zeroed, so
        // padded warps accumulate exact zeros.
#ifndef NINFER_DECODE_NO_PV
#pragma unroll
        for (int pk = 0; pk < PVKs; ++pk) {
            const uint4 pv = *reinterpret_cast<const uint4*>(
                &ps[decode_wmma_detail::dps(a_row, pk * 16 + d_half, Bc)]);
            const float aw[4] = {__uint_as_float(pv.x), __uint_as_float(pv.y),
                                 __uint_as_float(pv.z), __uint_as_float(pv.w)};
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                const uint4 vv = *reinterpret_cast<const uint4*>(
                    &vt[decode_wmma_detail::dvt(n * 16 + (lane & 15), pk * 16 + d_half, Bc)]);
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
#endif
#ifdef NINFER_DECODE_PV_MARK
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            partial_acc[(static_cast<std::size_t>(gridDim.y) * Geometry::QHeads * D) + 1] = 2.0f;
        }
#endif
    }

    // ---- epilogue: write split partials (production layout, reducer unchanged) ----
    // C layout: acc[n][i] @ lane L = (row warp*16 + i + 8*(L>=16), col n*16 + L%16).
    // Per lane: 8 rows (i) x 16 n-tiles, one column each. m_run/l_run are per-row-slot
    // values ALREADY reduced across the 16 columns by warp_max/warp_sum<16> inside the
    // block loop (every lane of a half-wave holds the row-level value), so no second
    // warp_sum here: re-reducing would scale l by 16. lane 0 of each warp stores its
    // slice's 8 slot rows (rows >= row_count are padding and skipped).
    if (lane == 0) {
        for (int i = 0; i < 8; ++i) {
            const int row = slot_qrow[i];
            if (row < row_count) {
                const int q_head = kv_head * Geometry::GroupSize + row;
                if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                    // Production reducer merges splits with weight = exp(m-M) where m is
                    // the SCALED max (small_t_bf16 stages score*scale and stores that max).
                    // The kernel here keeps raw scores internally, so the published max
                    // must carry the same scale factor.
                    partial_m[causal_partial_stat_index<Geometry>(q_head, 0, split, tokens)] =
                        m_run[i] * scale;
                    partial_l[causal_partial_stat_index<Geometry>(q_head, 0, split, tokens)] =
                        l_run[i];
                }
            }
        }
    }
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = slot_qrow[i];
            if (row < row_count) {
                const int q_head = kv_head * Geometry::GroupSize + row;
                if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                    const int d        = n * 16 + (lane & 15);
                    float* dst         = partial_acc + causal_partial_acc_index<Geometry>(
                                                     q_head, d, 0, split, tokens);
                    // acc is already the softmax-weighted V sum; the reducer divides by
                    // l without any scale. The softmax's own 1/sqrt(d) scale is folded
                    // into the exp2 path and partial_m, never into acc.
                    dst[0]             = acc[n][i];
                }
            }
        }
    }
}   // end if (warp == 0)
}   // end kernel

} // namespace ninfer::ops
