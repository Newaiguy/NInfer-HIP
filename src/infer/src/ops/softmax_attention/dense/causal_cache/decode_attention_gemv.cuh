// Dedicated T=1 decode attention: two-phase, one CTA per q-head.
// Phase 1: all threads cooperatively compute scores (key-parallel dot products)
// Phase 2: block softmax over smem scores
// Phase 3: d-parallel O accumulation (each thread owns one d, reads V column)
//
// Handles ctx up to 8192 keys (32KB score smem). Larger ctx falls back to chunked.
// smem: scores[max_keys] fp32 + Q[256] fp32 = 32KB + 1KB = 33KB for ctx 8192.
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <math_constants.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

constexpr int kDecAttnThreads = 256;
constexpr int kDecAttnMaxKeys = 8192;  // max visible keys for this kernel

template <typename Geometry, typename Metadata>
__global__ __launch_bounds__(kDecAttnThreads)
void causal_decode_attention_gemv_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ cache_k,
    const __half* __restrict__ cache_v, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale,
    __nv_bfloat16* __restrict__ out, std::int32_t width) {
    constexpr int D = kCausalPromptHeadDim;
    constexpr int KVH = Geometry::KVHeads;
    constexpr int QH = Geometry::QHeads;
    constexpr int GroupSize = QH / KVH;

    __shared__ __align__(16) float q_sh[D];            // 1 KB
    __shared__ __align__(16) float s_sh[kDecAttnMaxKeys];  // up to 32 KB
    __shared__ __align__(16) float reduce_sh[kDecAttnThreads];  // 1 KB for reductions

    const int q_head  = static_cast<int>(blockIdx.x);
    const int kv_head = q_head / GroupSize;
    const int tid     = static_cast<int>(threadIdx.x);
    const int base_pos = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int n_keys = base_pos + 1;  // keys [0, base_pos] inclusive

    if (n_keys <= 0 || base_pos >= kDecAttnMaxKeys) {
        // Should not reach here (route gates ctx); zero output as safety
        for (int d = tid; d < D; d += kDecAttnThreads) {
            out[causal_prompt_q_index<Geometry>(q_head, d, 0)] = __float2bfloat16_rn(0.0f);
        }
        return;
    }

    // Load Q into smem
    const __nv_bfloat16* q_ptr = q + causal_prompt_q_index<Geometry>(q_head, 0, 0);
    for (int d = tid; d < D; d += kDecAttnThreads) {
        q_sh[d] = __bfloat162float(q_ptr[d]);
    }
    __syncthreads();

    // ---- Phase 1: compute all scores (key-parallel) ----
    // Each thread handles keys [tid, tid+256, tid+512, ...] — strided for coalescing.
    // Per key: dot product Q[0..255] · K[key][0..255], reading K in 16B chunks.
    for (int key = tid; key < n_keys; key += kDecAttnThreads) {
        const int page = block_table[key >> kPagedKVPageShift];
        const int page_off = key & kPagedKVPageMask;
        const __nv_bfloat16* k_ptr = cache_k +
            paged_kv_element_offset<D, KVH>(page, kv_head, page_off, 0);
        float dot = 0.0f;
        // 16B vectorized loads: 8 bf16 per load, 32 loads for 256 d
#pragma unroll 8
        for (int vec = 0; vec < D / 8; ++vec) {
            const int d = vec * 8;
            const uint4 kv = *reinterpret_cast<const uint4*>(&k_ptr[d]);
            const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(&kv.x);
            const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(&kv.y);
            const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&kv.z);
            const __nv_bfloat162 b3 = *reinterpret_cast<const __nv_bfloat162*>(&kv.w);
            dot += q_sh[d+0] * __bfloat162float(b0.x) + q_sh[d+1] * __bfloat162float(b0.y)
                 + q_sh[d+2] * __bfloat162float(b1.x) + q_sh[d+3] * __bfloat162float(b1.y)
                 + q_sh[d+4] * __bfloat162float(b2.x) + q_sh[d+5] * __bfloat162float(b2.y)
                 + q_sh[d+6] * __bfloat162float(b3.x) + q_sh[d+7] * __bfloat162float(b3.y);
        }
        s_sh[key] = dot * scale;
    }
    __syncthreads();

    // ---- Phase 2: softmax ----
    // Block max
    float my_max = -CUDART_INF_F;
    for (int key = tid; key < n_keys; key += kDecAttnThreads) {
        my_max = fmaxf(my_max, s_sh[key]);
    }
    reduce_sh[tid] = my_max;
    __syncthreads();
    for (int stride = kDecAttnThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce_sh[tid] = fmaxf(reduce_sh[tid], reduce_sh[tid + stride]); }
        __syncthreads();
    }
    const float m = reduce_sh[0];

    // exp + block sum
    float my_sum = 0.0f;
    for (int key = tid; key < n_keys; key += kDecAttnThreads) {
        const float p = exp2_approx(__fmaf_rn(s_sh[key], 1.4426950408889634074f,
                                               -m * 1.4426950408889634074f));
        s_sh[key] = p;  // reuse score smem for P
        my_sum += p;
    }
    reduce_sh[tid] = my_sum;
    __syncthreads();
    for (int stride = kDecAttnThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce_sh[tid] += reduce_sh[tid + stride]; }
        __syncthreads();
    }
    const float l = reduce_sh[0];
    const float inv_l = (l > 0.0f) ? __frcp_rn(l) : 0.0f;

    // ---- Phase 3: O[my_d] = Σ_k P[k] · V[k][my_d] (d-parallel, re-read V from L2) ----
    {
        const int my_d = tid;
        if (my_d < D) {
            float o = 0.0f;
            for (int key = 0; key < n_keys; ++key) {
                const int page = block_table[key >> kPagedKVPageShift];
                const int page_off = key & kPagedKVPageMask;
                const __half* v_ptr = cache_v +
                    paged_kv_element_offset<D, KVH>(page, kv_head, page_off, my_d);
                o += s_sh[key] * __half2float(*v_ptr);
            }
            out[causal_prompt_q_index<Geometry>(q_head, my_d, 0)] =
                __float2bfloat16_rn(o * inv_l);
        }
    }
}

} // namespace ninfer::ops
