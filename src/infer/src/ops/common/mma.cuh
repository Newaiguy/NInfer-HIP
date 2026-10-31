#pragma once

// ============================================================================
// AMD/HIP port of ops/common/mma.cuh (gfx1201, wave32)
//
// Phase 1 implementation. Semantics are bit-exact PTX:
//   - ldmatrix.x2/x4 (+.trans): canonical NVIDIA fragment distribution,
//     built on cross-lane address exchange + addrspace(3) LDS reads.
//     `addr` is an LDS byte offset as produced by smem_addr() (the ROCm
//     __cvta_generic_to_shared returns the addrspace(3) pointer value,
//     i.e. a true 32-bit LDS offset), so a shuffled offset can be turned
//     back into a dereferenceable LDS pointer directly.
//   - mma_bf16 / mma_f16: m16n8k16 fragment layout emulation with warp
//     shuffles + fp32 FMA. ~4-6x slower than a hardware MMA would be, but
//     layout-identical to the CUDA kernels, so every call site (q4/q5/q6
//     ksplit + rowsplit mma, ternary mma, GDN, attention) works unchanged.
//     Phase 2 candidate: v_mfma_f32_16x16x16 with a redistribution shim.
//   - Every format without an RDNA4 path now traps loudly instead of
//     silently returning zeros (the original census behavior produced
//     all-zero MMA outputs, which looked like a model/garbage-output bug).
//   - Wave32 assumed (kernels are 32-lane NVIDIA code); lane = threadIdx.x & 31.
// ============================================================================

#include "ops/common/memory.cuh"

#include <cstdint>

namespace ninfer::ops {

namespace mma_port {

__device__ __forceinline__ std::uint32_t lane_id() { return threadIdx.x & 31u; }

// Read 4 bytes from an LDS byte offset (addrspace 3), independent of lane.
__device__ __forceinline__ std::uint32_t lds_load_u32(std::uint32_t off) {
    return *reinterpret_cast<const std::uint32_t __attribute__((address_space(3)))*>(off);
}

__device__ __forceinline__ std::uint32_t lds_load_u16(std::uint32_t off) {
    return *reinterpret_cast<const std::uint16_t __attribute__((address_space(3)))*>(off);
}

// b16 element at (lane-group row, element index) packed into the low 16 bits.
__device__ __forceinline__ float lo_bf16(std::uint32_t u) {
    return __uint_as_float(u << 16);
}
__device__ __forceinline__ float hi_bf16(std::uint32_t u) {
    return __uint_as_float(u & 0xffff0000u);
}

} // namespace mma_port

// --- ldmatrix (shared -> register fragment load) ----------------------------
// PTX ldmatrix.sync.aligned.m8n8.xN.shared.b16:
//   lanes 8*m .. 8*m+7 supply the 16-byte row addresses of matrix m;
//   lane l receives, for each matrix m, the b32
//     {M_m[l/4][2*(l%4)], M_m[l/4][2*(l%4)+1]}.
// With .trans the b32 is {M_m[2*(l%4)][l/4], M_m[2*(l%4)+1][l/4]} instead.

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    using namespace mma_port;
    const std::uint32_t l = lane_id();
    const std::uint32_t g = l >> 2; // row inside each 8x8 matrix
    const std::uint32_t t = l & 3u; // b32 quad within the 16-byte row
    const unsigned a0 = __shfl_sync(0xffffffffu, addr, g);
    const unsigned a1 = __shfl_sync(0xffffffffu, addr, 8u + g);
    r0 = lds_load_u32(a0 + 4u * t);
    r1 = lds_load_u32(a1 + 4u * t);
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    using namespace mma_port;
    const std::uint32_t l = lane_id();
    const std::uint32_t g = l >> 2;
    const std::uint32_t t = l & 3u;
    const unsigned a0 = __shfl_sync(0xffffffffu, addr, g);
    const unsigned a1 = __shfl_sync(0xffffffffu, addr, 8u + g);
    const unsigned a2 = __shfl_sync(0xffffffffu, addr, 16u + g);
    const unsigned a3 = __shfl_sync(0xffffffffu, addr, 24u + g);
    r0 = lds_load_u32(a0 + 4u * t);
    r1 = lds_load_u32(a1 + 4u * t);
    r2 = lds_load_u32(a2 + 4u * t);
    r3 = lds_load_u32(a3 + 4u * t);
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    using namespace mma_port;
    const std::uint32_t l = lane_id();
    const std::uint32_t g = l >> 2;
    const std::uint32_t t = l & 3u;
    const unsigned m = 0xffffffffu;
    const unsigned a_lo0 = __shfl_sync(m, addr, 2u * t);
    const unsigned a_hi0 = __shfl_sync(m, addr, 2u * t + 1u);
    const unsigned a_lo1 = __shfl_sync(m, addr, 8u + 2u * t);
    const unsigned a_hi1 = __shfl_sync(m, addr, 8u + 2u * t + 1u);
    r0 = (lds_load_u16(a_lo0 + 2u * g)) | (lds_load_u16(a_hi0 + 2u * g) << 16);
    r1 = (lds_load_u16(a_lo1 + 2u * g)) | (lds_load_u16(a_hi1 + 2u * g) << 16);
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                              unsigned addr) {
    using namespace mma_port;
    const std::uint32_t l = lane_id();
    const std::uint32_t g = l >> 2;
    const std::uint32_t t = l & 3u;
    const unsigned m = 0xffffffffu;
    unsigned base[4];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        base[mi] = __shfl_sync(m, addr, 8u * mi + 2u * t);
    }
    unsigned base1[4];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        base1[mi] = __shfl_sync(m, addr, 8u * mi + 2u * t + 1u);
    }
    r0 = (lds_load_u16(base[0] + 2u * g)) | (lds_load_u16(base1[0] + 2u * g) << 16);
    r1 = (lds_load_u16(base[1] + 2u * g)) | (lds_load_u16(base1[1] + 2u * g) << 16);
    r2 = (lds_load_u16(base[2] + 2u * g)) | (lds_load_u16(base1[2] + 2u * g) << 16);
    r3 = (lds_load_u16(base[3] + 2u * g)) | (lds_load_u16(base1[3] + 2u * g) << 16);
}

// --- mma.m16n8k16, fp32 accumulate ------------------------------------------
// NVIDIA fragment layout (lane l, g = l/4, t = l%4):
//   A 16x16: a0={A[g][2t],A[g][2t+1]}      a1={A[g+8][2t],A[g+8][2t+1]}
//            a2={A[g][2t+8],A[g][2t+9]}    a3={A[g+8][2t+8],A[g+8][2t+9]}
//   B 16x8:  b0={B[2t][g],B[2t+1][g]}      b1={B[2t+8][g],B[2t+9][g]}
//   D 16x8:  c0=D[g][2t] c1=D[g][2t+1] c2=D[g+8][2t] c3=D[g+8][2t+1]
//
// Emulation: gather the four quad members of the A rows and the eight lanes
// holding the two B output columns, then fp32-FMA the 16 products.
// NOTE: noinline on purpose - the -fgpu-rdc link re-optimizes all device
// bitcode and a per-callsite inlined copy of this body makes that link
// blow up (>20GB RSS, >40min). A plain call keeps code size sane.

#define NINFER_MMA_NOINLINE __attribute__((noinline))

// ============================================================================
// Hardware WMMA fast path (gfx1201, RDNA4 matrix cores)
//
// mma_bf16 mapped onto __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12 with
// register-level repacking of the NVIDIA m16n8k16 fragments. No LDS round
// trip, no rocwmma dependency (plain attribute-vector carriers).
//
// Hardware layouts (empirically verified, probe/lds_relay_probe.cu):
//   A: lane L<16  = row L     K0..7  (u32[i] = {A(L,2i),     A(L,2i+1)})
//      lane L>=16 = row L-16  K8..15 (u32[i] = {A(L-16,8+2i), A(L-16,9+2i)})
//   B: lane L<16  = K L     col 0..7  (u32[i] = {B(L,2i),     B(L,2i+1)})
//      lane L>=16 = K L-16  col 8..15 (u32[i] = {B(L-16,8+2i), B(L-16,9+2i)})
//   C: f32[i] @ lane L = D(i + 8*(L>=16), L%16)
//
// Validation: 64 random ops vs full CPU reference, max_err = 0.
// ============================================================================
#ifdef NINFER_HW_WMMMA

// ---- A/B operand packing (the only shuffles the hardware path needs) --------
static __device__ __forceinline__ void mma_hw_pack_ab(float aw[4], float bw[4], unsigned a0,
                                                      unsigned a1, unsigned a2, unsigned a3,
                                                      unsigned b0, unsigned b1) {
    const unsigned L = threadIdx.x & 31u;
    const unsigned m = 0xffffffffu;
    const unsigned rc = (L < 16u) ? L : (L - 16u);      // A row 0..15
    const unsigned src = 4u * (rc & 7u);                // NVIDIA lane base
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // shfl carriers must stay unsigned: a float assignment would VALUE-convert the
        // packed bf16-pair bits (0x3F803F80 -> 1065353216.0f) producing garbage/inf.
        const unsigned v0 = __shfl_sync(m, a0, src + i);
        const unsigned v1 = __shfl_sync(m, a1, src + i);
        const unsigned v2 = __shfl_sync(m, a2, src + i);
        const unsigned v3 = __shfl_sync(m, a3, src + i);
        aw[i] = __uint_as_float((L < 16u) ? ((rc < 8u) ? v0 : v1) : ((rc < 8u) ? v2 : v3));
    }
    {
        const unsigned k = (L < 16u) ? L : (L - 16u);   // B: K index 0..15 (destination)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned cc = (L < 16u) ? (2u * i) : (8u + 2u * i);
            const unsigned s0 = (4u * cc + (k >> 1u)) & 31u;
            const unsigned s1 = (4u * (cc + 1u) + (k >> 1u)) & 31u;
            // NOTE: both registers are shuffled and the k<8 choice is made at the
            // DESTINATION. Selecting inside the shfl argument would be evaluated with
            // the SOURCE lane's k (shfl evaluates var per source lane) — wrong register.
            const unsigned a0w = __shfl_sync(m, b0, s0);
            const unsigned a1w = __shfl_sync(m, b0, s1);
            const unsigned b0w = __shfl_sync(m, b1, s0);
            const unsigned b1w = __shfl_sync(m, b1, s1);
            const unsigned w0 = (k < 8u) ? a0w : b0w;
            const unsigned w1 = (k < 8u) ? a1w : b1w;
            const unsigned lo16 = (k & 1u) ? (w0 >> 16) : (w0 & 0xffffu);
            const unsigned hi16 = (k & 1u) ? (w1 >> 16) : (w1 & 0xffffu);
            bw[i] = __uint_as_float((hi16 << 16) | lo16);
        }
    }
}

typedef float mma_hw_f32x4 __attribute__((vector_size(16)));
typedef float mma_hw_f32x8 __attribute__((vector_size(32)));

// Raw gfx12 WMMA: D(hw layout) = A(hw) * B(hw) + C(hw).
static __device__ __forceinline__ void mma_hw_call(float out[8], const float aw[4],
                                                   const float bw[4], const float cw[8]) {
    mma_hw_f32x4 av;
    mma_hw_f32x4 bv;
    mma_hw_f32x8 cv;
#pragma unroll
    for (int i = 0; i < 4; ++i) { av[i] = aw[i]; }
#pragma unroll
    for (int i = 0; i < 4; ++i) { bv[i] = bw[i]; }
#pragma unroll
    for (int i = 0; i < 8; ++i) { cv[i] = cw[i]; }
    const mma_hw_f32x8 r = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bv, cv);
#pragma unroll
    for (int i = 0; i < 8; ++i) { out[i] = r[i]; }
}

// ---- Chained-accumulator API (hot k-loops) --------------------------------
// C seed / readback are inverse permutations; doing them EVERY k-step wastes
// 64 shuffles per call. Keep the accumulator in hardware layout across the loop:
//   float hw[8] = {};                       // zero seed, no shuffles
//   for (k) mma_bf16_hw_chain(hw, a0..a3, b0, b1);
//   mma_bf16_hw_read(hw, c0, c1, c2, c3);   // one readback at the end
static __device__ __forceinline__ void mma_bf16_hw_chain(float hw[8], unsigned a0, unsigned a1,
                                                        unsigned a2, unsigned a3, unsigned b0,
                                                        unsigned b1) {
    float aw[4], bw[4];
    mma_hw_pack_ab(aw, bw, a0, a1, a2, a3, b0, b1);
    float out[8];
    mma_hw_call(out, aw, bw, hw);
#pragma unroll
    for (int i = 0; i < 8; ++i) { hw[i] = out[i]; }
}

// NVIDIA-layout accumulator -> hardware layout (32 shuffles; use once per loop).
static __device__ __forceinline__ void mma_bf16_hw_seed(float hw[8], float c0, float c1, float c2,
                                                       float c3) {
    const unsigned L = threadIdx.x & 31u;
    const unsigned m = 0xffffffffu;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned r = (L < 16u) ? i : (8u + i);
        const unsigned c = L & 15u;
        const unsigned s = (4u * (r & 7u) + (c >> 1u)) & 31u;
        const float d0 = __shfl_sync(m, c0, s);
        const float d1 = __shfl_sync(m, c1, s);
        const float d2 = __shfl_sync(m, c2, s);
        const float d3 = __shfl_sync(m, c3, s);
        hw[i] = (c < 8u) ? ((c & 1u) ? ((r < 8u) ? d1 : d3) : ((r < 8u) ? d0 : d2)) : 0.0f;
    }
}

// hardware layout -> NVIDIA-layout accumulator (32 shuffles; use once per loop).
static __device__ __forceinline__ void mma_bf16_hw_read(const float hw[8], float& c0, float& c1,
                                                        float& c2, float& c3) {
    const unsigned L = threadIdx.x & 31u;
    const unsigned m = 0xffffffffu;
    const unsigned gg = L >> 2u, t = L & 3u;
    float lo[8], hi[8], lo8[8], hi8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        lo[i] = __shfl_sync(m, hw[i], 2u * t);
        hi[i] = __shfl_sync(m, hw[i], 2u * t + 1u);
        lo8[i] = __shfl_sync(m, hw[i], 2u * t + 16u);
        hi8[i] = __shfl_sync(m, hw[i], 2u * t + 17u);
    }
    c0 = lo[gg];
    c1 = hi[gg];
    c2 = lo8[gg];
    c3 = hi8[gg];
}

// NOTE: forceinline (the probe used forceinline; the noinline attribute is only needed
// for the software emulation's RDC-link size concern, and it breaks the attribute-vector
// builtin operands across the call boundary).
static __device__ __forceinline__ void mma_bf16_hw(float& c0, float& c1, float& c2, float& c3,
                                                       unsigned a0, unsigned a1, unsigned a2,
                                                       unsigned a3, unsigned b0, unsigned b1) {
    float aw[4], bw[4], cw[8];
    mma_hw_pack_ab(aw, bw, a0, a1, a2, a3, b0, b1);
    mma_bf16_hw_seed(cw, c0, c1, c2, c3);
    float out[8];
    mma_hw_call(out, aw, bw, cw);
    mma_bf16_hw_read(out, c0, c1, c2, c3);
}

// ============================================================================
// mma_s8: NVIDIA m16n8k32.s8 (s32 accumulate) on gfx12 native int8 WMMA.
//
// NVIDIA fragment layout (lane l, g = l/4, t = l%4), PTX mma.m16n8k32.s8:
//   A 16x32 row-major: a0 = {A[g][4t+0..3]}     a1 = {A[g+8][4t+0..3]}
//                      a2 = {A[g][4t+16..19]}   a3 = {A[g+8][4t+16..19]}
//   B 32x8 col-major:  b0 = {B[4t+0..3][g]}     b1 = {B[4t+16..19][g]}
//   D 16x8:  c0 = D[g][2t]  c1 = D[g][2t+1]  c2 = D[g+8][2t]  c3 = D[g+8][2t+1]
//
// gfx12 i8 WMMA is 16x16x16 (V2i x V2i -> V8i), so the NVIDIA k32 op becomes TWO
// chained hardware calls: k0..15 from a0/a1, then k16..31 from a2/a3. Lane layout
// (verified by probe/bench_wmma_i8_fp8_gfx12.cu pattern test, same shape as bf16):
//   A: lane L L%16 = row, k0..7 for L<16, k8..15 for L>=16 (V2i = 8 i8)
//   B: lane L L%16 = k row, n0..7 for L<16, n8..15 for L>=16
//   D: s32[j] @ lane L = D(j + 8*(L>=16), L%16)
// NVIDIA's B only has 8 columns, so the gfx12 B lanes for n8..15 are zeroed and
// the matching D columns are discarded at readback. Both neg flags (signedness)
// are compile-time true and the accumulator is signed.
// ============================================================================
typedef int mma_hw_i32x2 __attribute__((vector_size(8)));
typedef int mma_hw_i32x8 __attribute__((vector_size(32)));

// NVIDIA m16n8k32 A -> two gfx12 V2i A operands (aout[seg][0/1], seg = k16 half).
// For target row r = L&15 with k window base kw = 8*(L>=16): the 4-i8 groups
// {r, kw+0..3} / {r, kw+4..7} live in a0/a2 (row r<8) or a1/a3 (row r>=8) on
// NVIDIA lane 4*(r&7) + kw/4 + {0,1}. Both candidates are shuffled (srcLane is the
// same for every target), then the row-half select happens in-register — no shfl
// divergence.
static __device__ __forceinline__ void mma_s8_hw_pack_a(int aout[2][2], unsigned a0, unsigned a1,
                                                        unsigned a2, unsigned a3) {
    const unsigned L = threadIdx.x & 31u;
    const unsigned m = 0xffffffffu;
    const unsigned r = L & 15u;                 // gfx12 A row 0..15
    const unsigned kw = (L >= 16u) ? 8u : 0u;   // k window base
    const unsigned src_g = r & 7u;              // NVIDIA lane g for both halves
    const unsigned lo = 4u * src_g + (kw >> 2u);
#pragma unroll
    for (int seg = 0; seg < 2; ++seg) {
        const unsigned rlo = (seg == 0) ? a0 : a2;   // row r<8 half carrier
        const unsigned rhi = (seg == 0) ? a1 : a3;   // row r>=8 half carrier
        const unsigned v0 = __shfl_sync(m, rlo, lo);
        const unsigned w0 = __shfl_sync(m, rhi, lo);
        const unsigned v1 = __shfl_sync(m, rlo, lo + 1u);
        const unsigned w1 = __shfl_sync(m, rhi, lo + 1u);
        aout[seg][0] = (r >= 8u) ? w0 : v0;
        aout[seg][1] = (r >= 8u) ? w1 : v1;
    }
}

// NVIDIA m16n8k32 B -> one gfx12 V2i B operand for one k16 window (kwin=0 -> b0,
// kwin=1 -> b1). The B rows k = kwin*16 + (L%16) with columns n=0..7 are spread
// byte-wise: B[k][n] is byte (k&3) of the carrier on NVIDIA lane 4n + (k>>2).
// Lanes L>=16 (n=8..15) have no NVIDIA column -> zero (D cols 8..15 discarded).
static __device__ __forceinline__ void mma_s8_hw_pack_b(int bout[2], unsigned b0, unsigned b1,
                                                        int kwin) {
    const unsigned L = threadIdx.x & 31u;
    const unsigned m = 0xffffffffu;
    bout[0] = 0;
    bout[1] = 0;
    if (L < 16u) {
        const unsigned k = (kwin == 0) ? L : (16u + L);   // absolute B k row
        const unsigned src = (kwin == 0) ? b0 : b1;
        const unsigned t = (k & 15u) >> 2u;   // 4-k group within the window
        const unsigned sh = (k & 3u) << 3;    // byte offset in the carrier
#pragma unroll
        for (int b = 0; b < 4; ++b) {         // n = b
            const unsigned v = __shfl_sync(m, src, 4u * b + t);
            bout[0] |= (v >> sh) & 0xffu;
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {         // n = b + 4
            const unsigned v = __shfl_sync(m, src, 4u * (b + 4) + t);
            bout[1] |= (v >> sh) & 0xffu;
        }
    }
}

// One full m16n8k32.s8 op (2 hardware WMMAs chained + NVIDIA D readback).
static __device__ __forceinline__ void mma_s8_hw(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                                 unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                                 unsigned b1) {
    int aout[2][2];
    mma_s8_hw_pack_a(aout, a0, a1, a2, a3);
    int bw[2][2];
    mma_s8_hw_pack_b(bw[0], b0, b1, 0);
    mma_s8_hw_pack_b(bw[1], b0, b1, 1);
    const mma_hw_i32x2 av0 = {aout[0][0], aout[0][1]};
    const mma_hw_i32x2 av1 = {aout[1][0], aout[1][1]};
    const mma_hw_i32x2 bv0 = {bw[0][0], bw[0][1]};
    const mma_hw_i32x2 bv1 = {bw[1][0], bw[1][1]};
    const mma_hw_i32x8 czero = {0, 0, 0, 0, 0, 0, 0, 0};
    const mma_hw_i32x8 d0 =
        __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av0, true, bv0, czero, false);
    const mma_hw_i32x8 d1 =
        __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av1, true, bv1, d0, false);

    // NVIDIA D readback (identical math to mma_bf16_hw_read, s32 variant).
    const unsigned L = threadIdx.x & 31u;
    const unsigned m = 0xffffffffu;
    const unsigned gg = L >> 2u, t = L & 3u;
    int lo[8], hi[8], lo8[8], hi8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        lo[i]  = __shfl_sync(m, d1[i], 2u * t);
        hi[i]  = __shfl_sync(m, d1[i], 2u * t + 1u);
        lo8[i] = __shfl_sync(m, d1[i], 2u * t + 16u);
        hi8[i] = __shfl_sync(m, d1[i], 2u * t + 17u);
    }
    c0 = lo[gg];
    c1 = hi[gg];
    c2 = lo8[gg];
    c3 = hi8[gg];
}

// NVIDIA-layout mma_s8 dispatch: hardware fast path when enabled, trap otherwise.
static __device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                              unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                              unsigned b1) {
#ifdef NINFER_HW_WMMMA
    mma_s8_hw(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
    (void)c0; (void)c1; (void)c2; (void)c3; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0;
    (void)b1;
    __builtin_trap(); // int8 WMMA needs -DNINFER_HW_WMMMA on gfx1201
#endif
}

// gfx12-layout int8 WMMA, one 16x16x16 step. A/B carriers are raw 8-int8 V2i
// fragments ALREADY in hardware lane layout (the prompt KInt8 QK builds them
// layout-direct from the rotated int8 caches); the output is the hardware-layout
// s32 V8i D. neg flags = signed int8, accumulator signed.
static __device__ __forceinline__ mma_hw_i32x8 mma_s8_hw_direct(const mma_hw_i32x2& a,
                                                                const mma_hw_i32x2& b,
                                                                const mma_hw_i32x8& c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}

#endif // NINFER_HW_WMMMA

static NINFER_MMA_NOINLINE __device__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#ifdef NINFER_HW_WMMMA
    mma_bf16_hw(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    return;
#endif
    using namespace mma_port;
    const std::uint32_t l = lane_id();
    const std::uint32_t g = l >> 2;
    const std::uint32_t t = l & 3u;
    const unsigned m = 0xffffffffu;

    // A rows g (a0/a2) and g+8 (a1/a3) live across the quad 4*g..4*g+3.
    unsigned a0q[4], a1q[4], a2q[4], a3q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned src = 4u * g + j;
        a0q[j] = __shfl_sync(m, a0, src);
        a1q[j] = __shfl_sync(m, a1, src);
        a2q[j] = __shfl_sync(m, a2, src);
        a3q[j] = __shfl_sync(m, a3, src);
    }
    // B columns 2t (lanes 8*t..8*t+3) and 2t+1 (lanes 8*t+4..8*t+7).
    unsigned b0e[4], b1e[4], b0o[4], b1o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        b0e[j] = __shfl_sync(m, b0, 8u * t + j);
        b1e[j] = __shfl_sync(m, b1, 8u * t + j);
        b0o[j] = __shfl_sync(m, b0, 8u * t + 4u + j);
        b1o[j] = __shfl_sync(m, b1, 8u * t + 4u + j);
    }

    float d00 = c0, d01 = c1, d10 = c2, d11 = c3;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float a00 = lo_bf16(a0q[j]), a01 = hi_bf16(a0q[j]);
        const float a20 = lo_bf16(a2q[j]), a21 = hi_bf16(a2q[j]);
        const float a10 = lo_bf16(a1q[j]), a11 = hi_bf16(a1q[j]);
        const float a30 = lo_bf16(a3q[j]), a31 = hi_bf16(a3q[j]);
        // even output column 2t
        d00 = fmaf(a00, lo_bf16(b0e[j]), d00);
        d00 = fmaf(a01, hi_bf16(b0e[j]), d00);
        d00 = fmaf(a20, lo_bf16(b1e[j]), d00);
        d00 = fmaf(a21, hi_bf16(b1e[j]), d00);
        // odd output column 2t+1
        d01 = fmaf(a00, lo_bf16(b0o[j]), d01);
        d01 = fmaf(a01, hi_bf16(b0o[j]), d01);
        d01 = fmaf(a20, lo_bf16(b1o[j]), d01);
        d01 = fmaf(a21, hi_bf16(b1o[j]), d01);
        // row g+8
        d10 = fmaf(a10, lo_bf16(b0e[j]), d10);
        d10 = fmaf(a11, hi_bf16(b0e[j]), d10);
        d10 = fmaf(a30, lo_bf16(b1e[j]), d10);
        d10 = fmaf(a31, hi_bf16(b1e[j]), d10);
        d11 = fmaf(a10, lo_bf16(b0o[j]), d11);
        d11 = fmaf(a11, hi_bf16(b0o[j]), d11);
        d11 = fmaf(a30, lo_bf16(b1o[j]), d11);
        d11 = fmaf(a31, hi_bf16(b1o[j]), d11);
    }
    c0 = d00;
    c1 = d01;
    c2 = d10;
    c3 = d11;
}

// Exact f16 -> f32 (self-contained; cuda_fp16.h helpers are not uniformly
// visible across the kernels that include this header).
__device__ __forceinline__ float f16_bits_to_float(unsigned h) {
    const unsigned sign = (h & 0x8000u) << 16;
    const unsigned exp = (h >> 10) & 0x1fu;
    const unsigned man = h & 0x3ffu;
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
            } while (!(m & 0x400u));
            m &= 0x3ffu;
            bits = sign | (static_cast<unsigned>(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | (static_cast<unsigned>(exp - 15 + 127) << 23) | (man << 13);
    }
    return __uint_as_float(bits);
}

__device__ __forceinline__ float lo_f16(unsigned u) { return f16_bits_to_float(u & 0xffffu); }
__device__ __forceinline__ float hi_f16(unsigned u) { return f16_bits_to_float(u >> 16); }

static NINFER_MMA_NOINLINE __device__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    using namespace mma_port;
    const std::uint32_t l = lane_id();
    const std::uint32_t g = l >> 2;
    const std::uint32_t t = l & 3u;
    const unsigned m = 0xffffffffu;

    unsigned a0q[4], a1q[4], a2q[4], a3q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned src = 4u * g + j;
        a0q[j] = __shfl_sync(m, a0, src);
        a1q[j] = __shfl_sync(m, a1, src);
        a2q[j] = __shfl_sync(m, a2, src);
        a3q[j] = __shfl_sync(m, a3, src);
    }
    unsigned b0e[4], b1e[4], b0o[4], b1o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        b0e[j] = __shfl_sync(m, b0, 8u * t + j);
        b1e[j] = __shfl_sync(m, b1, 8u * t + j);
        b0o[j] = __shfl_sync(m, b0, 8u * t + 4u + j);
        b1o[j] = __shfl_sync(m, b1, 8u * t + 4u + j);
    }

    float d00 = c0, d01 = c1, d10 = c2, d11 = c3;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float a00 = lo_f16(a0q[j]), a01 = hi_f16(a0q[j]);
        const float a20 = lo_f16(a2q[j]), a21 = hi_f16(a2q[j]);
        const float a10 = lo_f16(a1q[j]), a11 = hi_f16(a1q[j]);
        const float a30 = lo_f16(a3q[j]), a31 = hi_f16(a3q[j]);
        d00 = fmaf(a00, lo_f16(b0e[j]), d00);
        d00 = fmaf(a01, hi_f16(b0e[j]), d00);
        d00 = fmaf(a20, lo_f16(b1e[j]), d00);
        d00 = fmaf(a21, hi_f16(b1e[j]), d00);
        d01 = fmaf(a00, lo_f16(b0o[j]), d01);
        d01 = fmaf(a01, hi_f16(b0o[j]), d01);
        d01 = fmaf(a20, lo_f16(b1o[j]), d01);
        d01 = fmaf(a21, hi_f16(b1o[j]), d01);
        d10 = fmaf(a10, lo_f16(b0e[j]), d10);
        d10 = fmaf(a11, hi_f16(b0e[j]), d10);
        d10 = fmaf(a30, lo_f16(b1e[j]), d10);
        d10 = fmaf(a31, hi_f16(b1e[j]), d10);
        d11 = fmaf(a10, lo_f16(b0o[j]), d11);
        d11 = fmaf(a11, hi_f16(b0o[j]), d11);
        d11 = fmaf(a30, lo_f16(b1o[j]), d11);
        d11 = fmaf(a31, hi_f16(b1o[j]), d11);
    }
    c0 = d00;
    c1 = d01;
    c2 = d10;
    c3 = d11;
}

// --- formats without an RDNA4 path: trap loudly ------------------------------
// NOTE: mma_s8 lives in the NINFER_HW_WMMMA block above (hardware i8 WMMA path
// since B21); mma_fp8_e4m3 / mma_nvfp4_e4m3 are still unported on gfx12.

__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    (void)c0; (void)c1; (void)c2; (void)c3; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0;
    (void)b1;
    __builtin_trap(); // fp8 WMMA not ported yet
}

// fp32 -> tf32 (round-to-nearest-even), matching cvt.rna.tf32.f32. Operands
// that are already tf32-exact (e.g. bf16 bit patterns unpacked into the high
// half) pass through unchanged.
__device__ __forceinline__ float tf32_rn(unsigned bits) {
    const unsigned exp = (bits >> 23) & 0xffu;
    if (exp == 0x00u) { return 0.0f; } // zero or fp32 denormal: tf32 has none
    if (exp == 0xffu) { return __uint_as_float(bits); } // inf / nan
    const unsigned low = bits & 0x1fffu;
    unsigned out       = bits & 0xffffe000u;
    if (low > 0x1000u || (low == 0x1000u && (out & 0x2000u))) { out += 0x2000u; }
    return __uint_as_float(out);
}

// --- mma.m16n8k8, tf32 inputs, fp32 accumulate -------------------------------
// NVIDIA fragment layout (lane l, g = l/4, t = l%4):
//   A 16x8: a0=A[g][t]   a1=A[g+8][t]   a2=A[g][t+4]   a3=A[g+8][t+4]
//   B 8x8:  b0=B[t][g]   b1=B[t+4][g]
//   D 16x8: c0=D[g][2t] c1=D[g][2t+1] c2=D[g+8][2t] c3=D[g+8][2t+1]
//
// Emulation mirrors mma_bf16/mma_f16: gather the quad members of the two A
// rows via shuffles, pick the two B columns from lanes 4j+2t / 4j+2t+1, then
// fp32-FMA the 8 products per output after tf32-rounding each operand.
static NINFER_MMA_NOINLINE __device__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                                         unsigned a0, unsigned a1, unsigned a2,
                                                         unsigned a3, unsigned b0, unsigned b1) {
    using namespace mma_port;
    const std::uint32_t l = lane_id();
    const std::uint32_t g = l >> 2;
    const std::uint32_t t = l & 3u;
    const unsigned m = 0xffffffffu;

    unsigned a0q[4], a2q[4], a1q[4], a3q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned src = 4u * g + j; // lanes holding row g / g+8 fragment regs
        a0q[j] = __shfl_sync(m, a0, src);
        a2q[j] = __shfl_sync(m, a2, src);
        a1q[j] = __shfl_sync(m, a1, src);
        a3q[j] = __shfl_sync(m, a3, src);
    }
    // B[k][col]  = b0 of lane 4*col + k  (k < 4),  b1 of the same lane for k >= 4.
    // (b0 of lane L is B[L&3][L>>2]: k index in the t-field, column in the g-field.)
    unsigned b0c0[4], b1c0[4], b0c1[4], b1c1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned src_even = 8u * t + j;      // B[j][2t] / B[j+4][2t]
        const unsigned src_odd  = 8u * t + 4u + j; // B[j][2t+1] / B[j+4][2t+1]
        b0c0[j] = __shfl_sync(m, b0, src_even);
        b1c0[j] = __shfl_sync(m, b1, src_even);
        b0c1[j] = __shfl_sync(m, b0, src_odd);
        b1c1[j] = __shfl_sync(m, b1, src_odd);
    }

    float d00 = c0, d01 = c1, d10 = c2, d11 = c3;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float a_0  = tf32_rn(a0q[j]); // A[g][j]
        const float a_0b = tf32_rn(a2q[j]); // A[g][j+4]
        const float a_1  = tf32_rn(a1q[j]); // A[g+8][j]
        const float a_1b = tf32_rn(a3q[j]); // A[g+8][j+4]
        const float b00 = tf32_rn(b0c0[j]), b10 = tf32_rn(b1c0[j]);
        const float b01 = tf32_rn(b0c1[j]), b11 = tf32_rn(b1c1[j]);
        d00 = fmaf(a_0b, b10, fmaf(a_0, b00, d00));
        d01 = fmaf(a_0b, b11, fmaf(a_0, b01, d01));
        d10 = fmaf(a_1b, b10, fmaf(a_1, b00, d10));
        d11 = fmaf(a_1b, b11, fmaf(a_1, b01, d11));
    }
    c0 = d00;
    c1 = d01;
    c2 = d10;
    c3 = d11;
}


__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    (void)c0; (void)c1; (void)c2; (void)c3; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0;
    (void)b1; (void)sfa; (void)sfb;
    __builtin_trap(); // fp4 block-scale MMA has no RDNA hardware
}

} // namespace ninfer::ops
