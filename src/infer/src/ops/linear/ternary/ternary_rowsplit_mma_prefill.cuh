// HIP port of the ORIGINAL NInfer prefill kernel (q4_rowsplit_gemm_mma.cuh skeleton),
// adapted to the PQ2_0 ternary format used by Bonsai-2-27B.
//
// The skeleton is upstream's and is kept structurally identical (validated against
// REFERENCE_original_q4_gemm_mma.cuh line by line):
//   * 2D block tile (BlockRows x BlockCols) advancing K one quant group per step;
//   * S-stage pipeline with separate quant/scale/activation shared buffers;
//   * decode-to-smem (As) followed by canonical ldmatrix fragments + m16n8k16 mma;
//   * warp grid decomposition warp_row = warp / kWarpGridCols, warp_col = warp % ...;
//   * the upstream ldmatrix address helpers (a_matrix / a_inner_row / a_row_offset /
//     a_col_offset / b_inner_row / b_k_offset) -- these are ROW-ADDRESS patterns
//     (lane&7 rows, matrix-select bits), NOT fragment element indices. The first
//     port attempt used lane/4 + (lane%4)*2 here, which corrupts every fragment.
//   * pipeline shape: pre-stage S tiles, per tile cp_wait<S-1> -> decode -> mma ->
//     prefetch tile k_tile+S into the same stage buffer after the trailing barrier.
//
// The only structural change is the decode atom. Q4G64: one byte per lane holds two
// nibbles -> one bf16x2 stored at col 2*lane (32 lanes x 2 = 64 = group). PQ2_0: one
// byte holds FOUR 2-bit codes and the group is 128 wide, so the atom decodes the
// lane's byte through the shared LUT into TWO bf16x2 pairs stored at cols 4*lane and
// 4*lane+2 (32 lanes x 4 = 128 = group). The group scale is folded into the decoded
// bf16 exactly like upstream's decode_pair.
//
// Layouts (identical to upstream): x is token-major [T, K] with K contiguous
// (ne[0] == K), out is token-major [T, rows] with a runtime out_row_stride.

#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"
#include "ops/common/mma.cuh"

#include <cstdint>
#include <cstdlib>
#include <string>

namespace ninfer::ops::detail {

// iamwavecut v3 T2 codec: bit0 selects value, bit1 flips sign. code 0/1/2/3 -> 0/+1/0/-1.
// PQ2_0 uses the GGUF (code - 1) mapping: 0/1/2/3 -> -1/0/+1/+2. The WMMA kernels build a
// 256-entry byte LUT; t2_codec == 1 switches the LUT to the bit0/bit1 table.
__device__ __forceinline__ float t2_lut_value(int code, int t2_codec) {
    if (t2_codec) {
        return static_cast<float>((code & 1) * (1 - 2 * ((code >> 1) & 1)));
    }
    return static_cast<float>(code - 1);
}


// ---------------------------------------------------------------------------
// Schedule: BlockRows(64) x BlockCols(32) x BlockK(=128, one PQ2 group), warp
// tile 32x16 -> warp grid 2x2 = 4 warps (128 threads), each warp owning two
// m16 row tiles x two n8 col tiles per k-step.
// Shared memory: As 64*128*2B = 16 KB, Bs 2*32*128*2B = 16 KB,
// Cr 2*64*32B = 4 KB, Sr 2*64*2B = 0.25 KB -> 36.3 KB, inside the 48 KB budget.
// ---------------------------------------------------------------------------
struct PQ2RowSplitMmaGemmSchedule {
    static constexpr int kBlockRows = 64;
    static constexpr int kBlockCols = 32;
    static constexpr int kBlockK    = 128;  // one PQ2 group per K tile
    static constexpr int kWarpRows  = 32;
    static constexpr int kWarpCols  = 16;

    static constexpr int kPipelineStages = 2;

    static constexpr int kWarpGridRows = kBlockRows / kWarpRows;  // 2
    static constexpr int kWarpGridCols = kBlockCols / kWarpCols;  // 2
    static constexpr int kWarps        = kWarpGridRows * kWarpGridCols;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kMmaRows      = kWarpRows / 16;  // 2
    static constexpr int kMmaCols      = kWarpCols / 8;   // 2
    static constexpr int kMmaKSteps    = kBlockK / 16;    // 8

    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kPipelineStages * kBlockCols * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kPipelineStages * kBlockRows * PQ2RowSplitStorage::kCodeBytesPerGroup +
        kPipelineStages * kBlockRows * PQ2RowSplitStorage::kScaleBytesPerGroup;
    static_assert(kBlockRows % kWarpRows == 0 && kBlockCols % kWarpCols == 0,
                  "PQ2 MMA block tile must divide into warp tiles");
    static_assert(kBlockK == PQ2RowSplitStorage::kGroupK,
                  "PQ2 MMA stages exactly one quant group per K tile");
    static_assert(kSharedBytes <= 48 * 1024,
                  "PQ2 MMA staged shared memory exceeds the static 48 KiB budget");
};

// XOR swizzle for a [rows][128] BF16 tile (16 column groups of 8 elements).
// Same family as upstream's q4_mma_swizzle_k64, widened for the 128-wide group.
__device__ __forceinline__ int pq2_mma_swizzle_k128(int row, int col) {
    return (((col >> 3) ^ (row & 15)) << 3) | (col & 7);
}

template <bool Full>
__global__ __launch_bounds__(PQ2RowSplitMmaGemmSchedule::kThreads)
void pq2_rowsplit_gemm_mma_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                  std::int32_t k, std::int32_t cols,
                                  std::int32_t out_row_stride,
                                  std::int32_t t2_codec) {
    using Schedule = PQ2RowSplitMmaGemmSchedule;
    constexpr int BM   = Schedule::kBlockRows;
    constexpr int BN   = Schedule::kBlockCols;
    constexpr int BK   = Schedule::kBlockK;
    constexpr int WM   = Schedule::kWarpRows;
    constexpr int WN   = Schedule::kWarpCols;
    constexpr int MT   = Schedule::kMmaRows;
    constexpr int NT   = Schedule::kMmaCols;
    constexpr int KSUB = Schedule::kMmaKSteps;
    constexpr int S    = Schedule::kPipelineStages;

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[S][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * PQ2RowSplitStorage::kCodeBytesPerGroup];
    __shared__ __align__(16) std::uint16_t Sr[S][BM];

    const int groups_per_row = k / PQ2RowSplitStorage::kGroupK;
    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane           = tid & 31;
    const int warp_row       = warp / Schedule::kWarpGridCols;
    const int warp_col       = warp % Schedule::kWarpGridCols;
    const int mma_row        = lane >> 2;
    const int mma_col        = lane & 3;

    const int row0 = static_cast<int>(blockIdx.x) * BM;
    const int col0 = static_cast<int>(blockIdx.y) * BN;

    // Build the 256-entry LUT once per CTA (identical to the small-T kernel):
    // lut[byte].x = bf16 pair of the codes in bits 0-3 (weights 4b, 4b+1),
    // lut[byte].y = bf16 pair of the codes in bits 4-7 (weights 4b+2, 4b+3);
    // code c means the value (c - 1).
    __shared__ uint2 lut[256];
    for (int i = tid; i < 256; i += Schedule::kThreads) {
        const __nv_bfloat162 lo = __floats2bfloat162_rn(t2_lut_value(i & 3, t2_codec),
                                                        t2_lut_value((i >> 2) & 3, t2_codec));
        const __nv_bfloat162 hi = __floats2bfloat162_rn(
            t2_lut_value((i >> 4) & 3, t2_codec), t2_lut_value((i >> 6) & 3, t2_codec));
        uint2 entry;
        entry.x = *reinterpret_cast<const unsigned*>(&lo);
        entry.y = *reinterpret_cast<const unsigned*>(&hi);
        lut[i]  = entry;
    }
    __syncthreads();

    float accum[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            accum[mi][ni][0] = 0.0f;
            accum[mi][ni][1] = 0.0f;
            accum[mi][ni][2] = 0.0f;
            accum[mi][ni][3] = 0.0f;
        }
    }

    const int k_tiles = k / BK;  // BK == kGroupK, one group per tile

    // Canonical ldmatrix address helpers -- upstream expressions, unchanged.
    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;

    auto stage_activation = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Schedule::kThreads) {
            const int local_col = item / (BK / 8);
            const int k8        = item - local_col * (BK / 8);
            const int kk        = k0 + k8 * 8;
            const int col       = col0 + local_col;
            auto* dst = &Bs[stage][local_col * BK + pq2_mma_swizzle_k128(local_col, k8 * 8)];
            if (Full || (col < cols && kk + 8 <= k)) {
                cp_async<16>(dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
    };

    auto stage_quant = [&](int stage, int k_tile) {
        const int group0 = (k_tile * BK) / PQ2RowSplitStorage::kGroupK;  // == k_tile
#pragma unroll 1
        for (int item = tid; item < BM * 2; item += Schedule::kThreads) {  // 2 x 16B halves
            const int row_group = item >> 1;
            const int half      = item & 1;
            const int local_row = row_group;  // GPB == 1: one group per row per tile
            const int row       = row0 + local_row;
            auto* dst = &Cr[stage][row_group * PQ2RowSplitStorage::kCodeBytesPerGroup + half * 16];
            if (Full || row < rows) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + group0;
                cp_async<16>(dst,
                             &codes[group_index * PQ2RowSplitStorage::kCodeBytesPerGroup +
                                    half * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }

#pragma unroll 1
        for (int row_group = tid; row_group < BM; row_group += Schedule::kThreads) {
            const int local_row = row_group;
            const int row       = row0 + local_row;
            if (Full || row < rows) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + group0;
                Sr[stage][row_group] = *reinterpret_cast<const std::uint16_t*>(
                    &scales[group_index * PQ2RowSplitStorage::kScaleBytesPerGroup]);
            } else {
                Sr[stage][row_group] = 0;
            }
        }
    };

    auto stage_inputs = [&](int stage, int k_tile) {
        stage_activation(stage, k_tile);
        stage_quant(stage, k_tile);
    };

    // Decode the staged PQ2 group into As (bf16, scale folded) -- the ported atom.
    // One lane covers weights {4L, 4L+1, 4L+2, 4L+3} of the group from byte L.
    auto decode_weight = [&](int stage) {
        for (int local_row = warp; local_row < BM; local_row += Schedule::kWarps) {
            auto* dst = &As[local_row * BK];
            const float scale =
                __half2float(__ushort_as_half(Sr[stage][local_row]));  // GPB == 1
            const std::uint8_t packed =
                Cr[stage][local_row * PQ2RowSplitStorage::kCodeBytesPerGroup + lane];
            const uint2 entry = lut[packed];
            const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&entry.x);
            const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&entry.y);
            const __nv_bfloat162 lo_s = __floats2bfloat162_rn(__bfloat162float(lo.x) * scale,
                                                              __bfloat162float(lo.y) * scale);
            const __nv_bfloat162 hi_s = __floats2bfloat162_rn(__bfloat162float(hi.x) * scale,
                                                              __bfloat162float(hi.y) * scale);
            const int col = 4 * lane;
            store_vec(&dst[pq2_mma_swizzle_k128(local_row, col)], lo_s);
            store_vec(&dst[pq2_mma_swizzle_k128(local_row, col + 2)], hi_s);
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < k_tiles) { stage_inputs(stage, stage); }
        cp_commit();
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % S;
        cp_wait<S - 1>();
        __syncthreads();

        decode_weight(stage);
        __syncthreads();

        auto load_fragments = [&](int k_step, unsigned (&a_frag)[MT][4],
                                  unsigned (&b_frag)[NT][2]) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = warp_row * WM + mi * 16 + a_row_offset;
                const int col = k_step + a_col_offset;
                ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            smem_addr(&As[row * BK + pq2_mma_swizzle_k128(row, col)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = warp_col * WN + ni * 8 + b_inner_row;
                const int col = k_step + b_k_offset;
                ldmatrix_x2(b_frag[ni][0], b_frag[ni][1],
                            smem_addr(&Bs[stage][row * BK + pq2_mma_swizzle_k128(row, col)]));
            }
        };

        // Fragment ping-pong (upstream's Q4FragmentPipeline::PingPong path).
        unsigned a_frag[2][MT][4];
        unsigned b_frag[2][NT][2];
        load_fragments(0, a_frag[0], b_frag[0]);
#pragma unroll
        for (int ki = 0; ki < KSUB; ++ki) {
            const int current = ki & 1;
            const int next    = (ki + 1) & 1;
            if (ki + 1 < KSUB) { load_fragments((ki + 1) * 16, a_frag[next], b_frag[next]); }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2],
                             accum[mi][ni][3], a_frag[current][mi][0], a_frag[current][mi][1],
                             a_frag[current][mi][2], a_frag[current][mi][3],
                             b_frag[current][ni][0], b_frag[current][ni][1]);
                }
            }
        }

        __syncthreads();
        const int prefetch_tile = k_tile + S;
        if (prefetch_tile < k_tiles) { stage_inputs(stage, prefetch_tile); }
        cp_commit();
    }

    // Epilogue: accum[mi][ni][{0,1}] = C(warp_row*WM + mi*16 + mma_row, warp_col*WN +
    // ni*8 + 2*mma_col + {0,1}); [{2,3}] are the +8 row. Token-major output.
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int output_row0 = row0 + warp_row * WM + mi * 16 + mma_row;
        const int output_row1 = output_row0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int output_col0 = col0 + warp_col * WN + ni * 8 + 2 * mma_col;
            const int output_col1 = output_col0 + 1;
            const float* values   = accum[mi][ni];
            if (output_row0 < rows) {
                if (output_col0 < cols) {
                    out[static_cast<std::int64_t>(output_col0) * out_row_stride + output_row0] =
                        __float2bfloat16_rn(values[0]);
                }
                if (output_col1 < cols) {
                    out[static_cast<std::int64_t>(output_col1) * out_row_stride + output_row0] =
                        __float2bfloat16_rn(values[1]);
                }
            }
            if (output_row1 < rows) {
                if (output_col0 < cols) {
                    out[static_cast<std::int64_t>(output_col0) * out_row_stride + output_row1] =
                        __float2bfloat16_rn(values[2]);
                }
                if (output_col1 < cols) {
                    out[static_cast<std::int64_t>(output_col1) * out_row_stride + output_row1] =
                        __float2bfloat16_rn(values[3]);
                }
            }
        }
    }
}

// ============================================================================
// Native gfx12 WMMA prefill kernel (zero-shuffle fragments).
//
// WHY A SECOND KERNEL: the software mma emulation costs 32 shuffles per
// m16n8k16, and this card's shuffle throughput caps ANY NVIDIA-fragment path
// (software emulation AND the shim's mma_bf16_hw repacker alike) at ~0.95
// TFLOPS -- measured standalone (probe/bench_prefill_kernel.cu): reference 1.12,
// 2D-tiled software 0.82. The hardware builtin
// __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12 must therefore be fed
// fragments built DIRECTLY in the hardware layout from shared memory, with zero
// cross-lane traffic:
//   * A (16 rows x 16 k): lane L<16 holds row L's k0..7 as 4 u32 (one 16B LDS
//     read at As[row][kstep*16]); lane L+16 holds row L-16's k8..15. The k128
//     XOR swizzle makes the 16 concurrent row reads conflict-free.
//   * B (16 k x 16 tokens): lane L<15+1 holds COLUMN L's k0..7 (u32[i] =
//     {B(2i,L), B(2i+1,L)}) per the kmap probe; lane L+16 holds column L's
//     k8..15. Column = token, so the fragment reads are TOKEN-major 16B loads
//     from a [token][k] tile -- the same layout the software path stages, no
//     transpose needed. (The mma.cuh pack_ab comment claims the transposed
//     interpretation; pack_ab is not independent evidence -- it and the builtin
//     would cancel. The kmap probe is, and it says column-major-per-lane.)
//   * C (16 x 16 f32): f32[i] at lane L = D(i + 8*(L>=16), L%16). The
//     accumulator lives in this layout across the whole K walk and the
//     epilogue stores straight out of it -- no seed/readback shuffles at all.
// Layouts verified by probe/lds_relay_probe.cu + raw_wmma_kmap_probe.cu and by
// the bit-exact engine layer dumps of the seventh session.
// ============================================================================

// Roll back to the software-fragment kernel with NINFER_PREFILL_SW=1.
[[nodiscard]] inline bool pq2_prefill_wmma_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_PREFILL_SW");
        return value == nullptr || std::string(value) != "1";
    }();
    return enabled;
}

namespace pq2_wmma_detail {

typedef float v4f __attribute__((vector_size(16)));
typedef float v8f __attribute__((vector_size(32)));

// D(hw) = A(hw) * B(hw) + C(hw); operands are float arrays carrying the packed
// bit patterns (__uint_as_float) exactly like mma.cuh's validated mma_hw_call.
// NOTE: out and cw alias at the call site (acc fed straight back in), so the
// result is staged through a local temporary -- never write out[] while cw[]
// might still be consumed by a reordered unrolled copy loop.
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

} // namespace pq2_wmma_detail

struct PQ2RowSplitWmmaSchedule {
    static constexpr int kBlockRows = 64;
    static constexpr int kBlockCols = 32;  // tokens
    static constexpr int kBlockK    = 128;
    static constexpr int kWarpRows  = 16;  // warp grid 4x1: each warp owns 16 rows
    static constexpr int kWarpCols  = 32;  // and all 32 tokens (two n16 tiles)

    static constexpr int kPipelineStages = 2;
    static constexpr int kWarps          = kBlockRows / kWarpRows;  // 4
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kNTiles        = kBlockCols / 16;          // 2 hw n-tiles
    static constexpr int kKSteps        = kBlockK / 16;             // 8 hw k-steps
    // Transposed activation tile row stride: 40 tokens = 80B rows (16B aligned,
    // padded past the 32 real tokens).
    static constexpr int kTokStride     = kBlockCols + 8;

    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kPipelineStages * kBlockK * kTokStride * static_cast<int>(sizeof(__nv_bfloat16)) +
        kPipelineStages * kBlockRows * PQ2RowSplitStorage::kCodeBytesPerGroup +
        kPipelineStages * kBlockRows * PQ2RowSplitStorage::kScaleBytesPerGroup;
    static_assert(kSharedBytes <= 48 * 1024,
                  "PQ2 WMMA staged shared memory exceeds the static 48 KiB budget");
};

__device__ __forceinline__ int pq2_mma_swizzle_k128_w(int row, int col) {
    return (((col >> 3) ^ (row & 15)) << 3) | (col & 7);
}

__global__ __launch_bounds__(PQ2RowSplitWmmaSchedule::kThreads)
void pq2_rowsplit_wmma_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ scales,
                              __nv_bfloat16* __restrict__ out, std::int32_t rows,
                              std::int32_t k, std::int32_t cols,
                              std::int32_t out_row_stride,
                              std::int32_t t2_codec) {
    using Schedule = PQ2RowSplitWmmaSchedule;
    constexpr int BM = Schedule::kBlockRows;
    constexpr int BN = Schedule::kBlockCols;
    constexpr int BK = Schedule::kBlockK;
    constexpr int TS = Schedule::kTokStride;
    constexpr int S  = Schedule::kPipelineStages;

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bst[S][BN * BK];  // [token][k], token-major
    __shared__ __align__(16) std::uint8_t Cr[S][BM * PQ2RowSplitStorage::kCodeBytesPerGroup];
    __shared__ __align__(16) std::uint16_t Sr[S][BM];

    const int groups_per_row = k / PQ2RowSplitStorage::kGroupK;
    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane           = tid & 31;

    const int row0 = static_cast<int>(blockIdx.x) * BM;
    const int col0 = static_cast<int>(blockIdx.y) * BN;

    // LUT (identical to the small-T and software-fragment kernels).
    __shared__ uint2 lut[256];
    for (int i = tid; i < 256; i += Schedule::kThreads) {
        const __nv_bfloat162 lo = __floats2bfloat162_rn(t2_lut_value(i & 3, t2_codec),
                                                        t2_lut_value((i >> 2) & 3, t2_codec));
        const __nv_bfloat162 hi = __floats2bfloat162_rn(
            t2_lut_value((i >> 4) & 3, t2_codec), t2_lut_value((i >> 6) & 3, t2_codec));
        uint2 entry;
        entry.x = *reinterpret_cast<const unsigned*>(&lo);
        entry.y = *reinterpret_cast<const unsigned*>(&hi);
        lut[i]  = entry;
    }
    __syncthreads();

    const int k_tiles = k / BK;

    auto stage_activation = [&](int stage, int k_tile) {
        // 32 tokens x 128 k staged token-major (16B chunks), exactly like the
        // software-fragment kernel's Bs. The B fragment reads are also token-major,
        // so no transpose happens anywhere.
        const int k0 = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Schedule::kThreads) {
            const int tok = item / (BK / 8);
            const int k8  = (item % (BK / 8)) * 8;
            const int col = col0 + tok;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (col < cols) {
                v = *reinterpret_cast<const uint4*>(&x[static_cast<std::int64_t>(col) * k + k0 + k8]);
            }
            *reinterpret_cast<uint4*>(&Bst[stage][tok * BK + pq2_mma_swizzle_k128_w(tok, k8)]) = v;
        }
    };

    auto stage_quant = [&](int stage, int k_tile) {
        const int group0 = k_tile;  // BK == kGroupK: one group per tile
#pragma unroll 1
        for (int item = tid; item < BM * 2; item += Schedule::kThreads) {
            const int row_group = item >> 1;
            const int half      = item & 1;
            const int row       = row0 + row_group;
            auto* dst = &Cr[stage][row_group * PQ2RowSplitStorage::kCodeBytesPerGroup + half * 16];
            if (row < rows) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + group0;
                cp_async<16>(dst,
                             &codes[group_index * PQ2RowSplitStorage::kCodeBytesPerGroup +
                                    half * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
#pragma unroll 1
        for (int row_group = tid; row_group < BM; row_group += Schedule::kThreads) {
            const int row = row0 + row_group;
            Sr[stage][row_group] =
                (row < rows) ? *reinterpret_cast<const std::uint16_t*>(
                                   &scales[(static_cast<std::int64_t>(row) * groups_per_row +
                                             group0) *
                                          PQ2RowSplitStorage::kScaleBytesPerGroup])
                             : 0;
        }
    };

    // Decode the staged PQ2 group into As -- identical to the software kernel
    // (probe-validated bit-exact against the fold oracle).
    auto decode_weight = [&](int stage) {
        for (int local_row = warp; local_row < BM; local_row += Schedule::kWarps) {
            auto* dst = &As[local_row * BK];
            const float scale = __half2float(__ushort_as_half(Sr[stage][local_row]));
            const std::uint8_t packed =
                Cr[stage][local_row * PQ2RowSplitStorage::kCodeBytesPerGroup + lane];
            const uint2 entry = lut[packed];
            const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&entry.x);
            const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&entry.y);
            const __nv_bfloat162 lo_s = __floats2bfloat162_rn(__bfloat162float(lo.x) * scale,
                                                              __bfloat162float(lo.y) * scale);
            const __nv_bfloat162 hi_s = __floats2bfloat162_rn(__bfloat162float(hi.x) * scale,
                                                              __bfloat162float(hi.y) * scale);
            const int col = 4 * lane;
            store_vec(&dst[pq2_mma_swizzle_k128_w(local_row, col)], lo_s);
            store_vec(&dst[pq2_mma_swizzle_k128_w(local_row, col + 2)], hi_s);
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < k_tiles) { stage_activation(stage, stage); stage_quant(stage, stage); }
        cp_commit();
    }

    // HW-layout accumulators, living across the WHOLE K walk (like the software
    // kernel's): acc[ni][i] = D(i + 8*(lane>=16), ni*16 + lane%16).
    float acc[Schedule::kNTiles][8];
#pragma unroll
    for (int ni = 0; ni < Schedule::kNTiles; ++ni) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[ni][i] = 0.0f; }
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % S;
        cp_wait<S - 1>();
        __syncthreads();

        decode_weight(stage);
        __syncthreads();

        const int a_row    = warp * 16 + (lane & 15);      // this lane's A row (mi == 0)
        const int a_khalf  = (lane >> 4) * 8;              // lanes 0-15: k0..7, 16-31: k8..15
        const int b_tok    = lane & 15;                    // this lane's B column (token)
        const int b_khalf  = (lane >> 4) * 8;              // lanes 0-15: k0..7, 16-31: k8..15

#pragma unroll
        for (int ki = 0; ki < Schedule::kKSteps; ++ki) {
            // A fragment: one 16B LDS read per lane, conflict-free through the swizzle.
            const uint4 abits = *reinterpret_cast<const uint4*>(
                &As[a_row * BK + pq2_mma_swizzle_k128_w(a_row, ki * 16 + a_khalf)]);
            const float aw[4] = {__uint_as_float(abits.x), __uint_as_float(abits.y),
                                 __uint_as_float(abits.z), __uint_as_float(abits.w)};
#pragma unroll
            for (int ni = 0; ni < Schedule::kNTiles; ++ni) {
                // B fragment: one 16B LDS read per lane, token-major (kmap layout).
                const int brow = ni * 16 + b_tok;  // local token row
                const uint4 bbits = *reinterpret_cast<const uint4*>(
                    &Bst[stage][brow * BK + pq2_mma_swizzle_k128_w(brow, ki * 16 + b_khalf)]);
                const float bw[4] = {__uint_as_float(bbits.x), __uint_as_float(bbits.y),
                                     __uint_as_float(bbits.z), __uint_as_float(bbits.w)};
                pq2_wmma_detail::wmma16(acc[ni], aw, bw, acc[ni]);
            }
        }

        __syncthreads();
        const int prefetch_tile = k_tile + S;
        if (prefetch_tile < k_tiles) {
            stage_activation(stage, prefetch_tile);
            stage_quant(stage, prefetch_tile);
        }
        cp_commit();
    }

    // Epilogue straight out of the HW C layout, once after the whole K walk:
    // lane L stores rows (i, i+8) of column ni*16 + lane%16. No readback shuffles.
#pragma unroll
    for (int ni = 0; ni < Schedule::kNTiles; ++ni) {
        const int out_col = col0 + ni * 16 + (lane & 15);
        const int row_half = (lane >> 4) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_row = row0 + warp * 16 + i + row_half;
            if (out_row < rows && out_col < cols) {
                out[static_cast<std::int64_t>(out_col) * out_row_stride + out_row] =
                    __float2bfloat16_rn(acc[ni][i]);
            }
        }
    }
}

// ============================================================================
// WMMA v2: register-decoded fragments -- no As, no staging, no barriers.
//
// MOTIVATION (measured): v1 runs 22.2 TFLOPS but a T=2048 GEMM's DRAM floor is
// ~0.7 ms against 4.8 ms measured -- latency/occupancy bound. v1's 38 KB of
// static LDS (As 16K + Bst 16K + Cr 4K + Sr + lut) keeps only ONE CTA (4 warps)
// resident per SM. v2 removes every staging buffer:
//   * A fragment: each lane decodes ITS OWN (row, 16k) chunk straight from the
//     global code stream -- one u32 code load per k-step, two LUT LDS.64, zero
//     arithmetic (the LUT holds UNSCALED {-1,0,+1} bf16 pairs). The 128 lanes
//     cover the 64 rows x 2 k-halves exactly once, so nothing is redundant.
//   * B fragment: the kmap layout (lane L = column L's k-pairs) is token-major
//     16B -- read DIRECTLY from the global activation x[token][k]. 8 KB per CTA
//     per group, L2-shared by all row-block CTAs of the token-block.
//   * Scale: applied POST-mma per group (the production small-T kernel's
//     strategy): the group partial lives in HW C layout, then
//     acc += group * scale(row). This also removes the bf16 fold precision loss.
// Static LDS: the 2 KB LUT alone -> occupancy is register/CTA-limit, not LDS.
// ============================================================================

struct PQ2RowSplitWmma2Schedule {
    static constexpr int kBlockRows = 64;
    static constexpr int kBlockCols = 32;
    static constexpr int kWarps     = 4;
    static constexpr int kThreads   = kWarps * 32;
    static constexpr int kNTiles    = kBlockCols / 16;
    static constexpr int kKSteps    = 8;
};

// AMD port: the VERIFY-SIZED schedule. MEASURED (probe/bench_small_t.cu, gate_up
// N=34816 K=5120, 47.3 MB of weights):
//
//   T=1  0.216 ms   T=4  0.213 ms   T=16 0.236 ms   T=32 0.264 ms
//
// i.e. v2 is essentially FLAT in T up to a full 32-token block, so with BN=32 a
// 4-16 token verify pass pays for 32 tokens of mma work. Halving the token block
// halves the A-operand decode and the B traffic, which is why this schedule exists.
//
// The ROW block matters just as much on the NARROW projections. The row tile is
// kBlockRows wide, so N=5120 (the MLP down projection, K=17408) gets only 5120/64 =
// 80 CTAs -- barely one per SM -- and measures 125 GB/s where the 34816-row gate_up
// shape reaches 240 GB/s with the same kernel. Letting the schedule pick the row
// block is what fixes that; it is a pure scheduling change, the body is shared.
template <int kRows, int kCols>
struct PQ2RowSplitWmma2Shape {
    static constexpr int kBlockRows = kRows;
    static constexpr int kBlockCols = kCols;
    static constexpr int kWarpRows  = 16;
    static constexpr int kWarps     = kBlockRows / kWarpRows;
    static constexpr int kThreads   = kWarps * 32;
    static constexpr int kNTiles    = kBlockCols / 16;
    static constexpr int kKSteps    = 8;
    static_assert(kBlockRows % kWarpRows == 0, "row block must be a whole number of warps");
    static_assert(kBlockCols % 16 == 0, "token block must be a whole number of n-tiles");
};

struct PQ2RowSplitWmma2Schedule16 {
    static constexpr int kBlockRows = 64;
    static constexpr int kBlockCols = 16;
    static constexpr int kWarps     = 4;
    static constexpr int kThreads   = kWarps * 32;
    static constexpr int kNTiles    = kBlockCols / 16;
    static constexpr int kKSteps    = 8;
};

template <class Schedule>
__device__ __forceinline__ void pq2_rowsplit_wmma2_body(const __nv_bfloat16* __restrict__ x,
                                                        const std::uint8_t* __restrict__ codes,
                                                        const std::uint8_t* __restrict__ scales,
                                                        __nv_bfloat16* __restrict__ out,
                                                        std::int32_t rows, std::int32_t k,
                                                        std::int32_t cols,
                                                        std::int32_t out_row_stride,
                                                        uint2* __restrict__ lut_storage,
                                                        std::int32_t t2_codec) {
    constexpr int kGroupK = PQ2RowSplitStorage::kGroupK;

    // The LUT is declared by the CALLING kernel, not here: a __shared__ array inside a
    // __device__ function that is instantiated by several kernels is a static shared
    // object with one definition per instantiation, and the placement/occupancy that
    // results is not the same as declaring it in the kernel itself (measured: the
    // prefill pass lost 30% when it moved in here).
    uint2* lut = lut_storage;
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < 256; i += Schedule::kThreads) {
        const __nv_bfloat162 lo = __floats2bfloat162_rn(t2_lut_value(i & 3, t2_codec),
                                                        t2_lut_value((i >> 2) & 3, t2_codec));
        const __nv_bfloat162 hi = __floats2bfloat162_rn(
            t2_lut_value((i >> 4) & 3, t2_codec), t2_lut_value((i >> 6) & 3, t2_codec));
        uint2 entry;
        entry.x = *reinterpret_cast<const unsigned*>(&lo);
        entry.y = *reinterpret_cast<const unsigned*>(&hi);
        lut[i]  = entry;
    }
    __syncthreads();

    const int groups = k / kGroupK;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int row0   = static_cast<int>(blockIdx.x) * Schedule::kBlockRows;
    const int col0   = static_cast<int>(blockIdx.y) * Schedule::kBlockCols;

    const std::int64_t code_row_stride =
        static_cast<std::int64_t>(groups) * PQ2RowSplitStorage::kCodeBytesPerGroup;
    const std::int64_t scale_row_stride =
        static_cast<std::int64_t>(groups) * PQ2RowSplitStorage::kScaleBytesPerGroup;

    // A side: this lane decodes weight row (warp*16 + lane&15), k-half (lane>>4).
    const int a_row       = row0 + warp * 16 + (lane & 15);
    const bool a_ok       = a_row < rows;
    const int a_row_safe  = a_ok ? a_row : row0;  // bounded load; the store is guarded
    const std::uint8_t* row_codes =
        codes + static_cast<std::int64_t>(a_row_safe) * code_row_stride;
    const unsigned a_byte_shift = (lane >> 4) * 16;  // lanes 0-15: bytes 0-1, 16-31: 2-3
    const int a_khalf           = (lane >> 4) * 8;

    // C side: slot i of this lane is output row row0 + warp*16 + (lane>>4)*8 + i.
    const int c_row0 = row0 + warp * 16 + (lane >> 4) * 8;

    float acc[Schedule::kNTiles][8];
#pragma unroll
    for (int ni = 0; ni < Schedule::kNTiles; ++ni) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[ni][i] = 0.0f; }
    }

    for (int g = 0; g < groups; ++g) {
        // Group scale per C-row slot (8 consecutive rows; L2-cached u16 loads).
        float sf[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = c_row0 + i;
            sf[i] = (r < rows) ? __half2float(__ushort_as_half(*reinterpret_cast<
                       const std::uint16_t*>(&scales[static_cast<std::int64_t>(r) *
                                                       scale_row_stride + g * 2])))
                               : 0.0f;
        }

        float grp[Schedule::kNTiles][8];
#pragma unroll
        for (int ni = 0; ni < Schedule::kNTiles; ++ni) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { grp[ni][i] = 0.0f; }
        }

#pragma unroll
        for (int ki = 0; ki < Schedule::kKSteps; ++ki) {
            // One u32 of code bytes covers this k-step's 16 weights; both k-halves
            // select their byte pair from it (same load, L2 broadcast).
            const std::uint32_t cw =
                *reinterpret_cast<const std::uint32_t*>(row_codes + g * PQ2RowSplitStorage::kCodeBytesPerGroup + ki * 4);
            const uint2 e0 = lut[(cw >> a_byte_shift) & 0xFFu];
            const uint2 e1 = lut[(cw >> (a_byte_shift + 8)) & 0xFFu];
            const float aw[4] = {__uint_as_float(e0.x), __uint_as_float(e0.y),
                                 __uint_as_float(e1.x), __uint_as_float(e1.y)};
#pragma unroll
            for (int ni = 0; ni < Schedule::kNTiles; ++ni) {
                const int tok      = col0 + ni * 16 + (lane & 15);
                const int tok_safe = (tok < cols) ? tok : col0;
                const uint4 bv = *reinterpret_cast<const uint4*>(
                    &x[static_cast<std::int64_t>(tok_safe) * k + g * kGroupK + ki * 16 +
                       a_khalf]);
                const float bw[4] = {__uint_as_float(bv.x), __uint_as_float(bv.y),
                                     __uint_as_float(bv.z), __uint_as_float(bv.w)};
                pq2_wmma_detail::wmma16(grp[ni], aw, bw, grp[ni]);
            }
        }

#pragma unroll
        for (int ni = 0; ni < Schedule::kNTiles; ++ni) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc[ni][i] = fmaf(grp[ni][i], sf[i], acc[ni][i]); }
        }
    }

#pragma unroll
    for (int ni = 0; ni < Schedule::kNTiles; ++ni) {
        const int out_col  = col0 + ni * 16 + (lane & 15);
        const int row_half = (lane >> 4) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int out_row = row0 + warp * 16 + i + row_half;
            if (out_row < rows && out_col < cols) {
                out[static_cast<std::int64_t>(out_col) * out_row_stride + out_row] =
                    __float2bfloat16_rn(acc[ni][i]);
            }
        }
    }
}

__global__ __launch_bounds__(PQ2RowSplitWmma2Schedule::kThreads)
void pq2_rowsplit_wmma2_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, std::int32_t rows,
                               std::int32_t k, std::int32_t cols,
                               std::int32_t out_row_stride,
                               std::int32_t t2_codec) {
    __shared__ uint2 lut_storage[256];
    pq2_rowsplit_wmma2_body<PQ2RowSplitWmma2Schedule>(x, codes, scales, out, rows, k, cols,
                                                      out_row_stride, lut_storage, t2_codec);
}

// Token block 16: the verify pass (T = 2..16) rides this one.
__global__ __launch_bounds__(PQ2RowSplitWmma2Schedule16::kThreads)
void pq2_rowsplit_wmma2_bn16_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, std::int32_t rows,
                               std::int32_t k, std::int32_t cols,
                               std::int32_t out_row_stride,
                               std::int32_t t2_codec) {
    __shared__ uint2 lut_storage[256];
    pq2_rowsplit_wmma2_body<PQ2RowSplitWmma2Schedule16>(x, codes, scales, out, rows, k, cols,
                                                      out_row_stride, lut_storage, t2_codec);
}

// Narrow-N tiles: 32 and 16 rows per CTA, so a 5120-row projection launches 160 / 320
// CTAs instead of 80. The thread counts go through named constants because
// __launch_bounds__ does not accept a dependent template-id expression here.
inline constexpr int kPQ2Wmma2R32C16Threads = PQ2RowSplitWmma2Shape<32, 16>::kThreads;
inline constexpr int kPQ2Wmma2R16C16Threads = PQ2RowSplitWmma2Shape<16, 16>::kThreads;
inline constexpr int kPQ2Wmma2R32C32Threads = PQ2RowSplitWmma2Shape<32, 32>::kThreads;

__global__ __launch_bounds__(kPQ2Wmma2R32C16Threads)
void pq2_rowsplit_wmma2_r32c16_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, std::int32_t rows,
                               std::int32_t k, std::int32_t cols,
                               std::int32_t out_row_stride,
                               std::int32_t t2_codec) {
    __shared__ uint2 lut_storage[256];
    pq2_rowsplit_wmma2_body<PQ2RowSplitWmma2Shape<32, 16>>(x, codes, scales, out, rows, k, cols,
                                                      out_row_stride, lut_storage, t2_codec);
}

__global__ __launch_bounds__(kPQ2Wmma2R16C16Threads)
void pq2_rowsplit_wmma2_r16c16_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, std::int32_t rows,
                               std::int32_t k, std::int32_t cols,
                               std::int32_t out_row_stride,
                               std::int32_t t2_codec) {
    __shared__ uint2 lut_storage[256];
    pq2_rowsplit_wmma2_body<PQ2RowSplitWmma2Shape<16, 16>>(x, codes, scales, out, rows, k, cols,
                                                      out_row_stride, lut_storage, t2_codec);
}

__global__ __launch_bounds__(kPQ2Wmma2R32C32Threads)
void pq2_rowsplit_wmma2_r32c32_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, std::int32_t rows,
                               std::int32_t k, std::int32_t cols,
                               std::int32_t out_row_stride,
                               std::int32_t t2_codec) {
    __shared__ uint2 lut_storage[256];
    pq2_rowsplit_wmma2_body<PQ2RowSplitWmma2Shape<32, 32>>(x, codes, scales, out, rows, k, cols,
                                                      out_row_stride, lut_storage, t2_codec);
}

} // namespace ninfer::ops::detail
