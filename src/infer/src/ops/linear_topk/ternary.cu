// AMD port: ternary (T2) full-head linear_topk fallback.
// The optimized Q4/Q8/Fp8 topk producers do not know the ternary codec. The DFlash2
// selector needs top-k candidates over the full ternary output head, so this path
// computes the complete BF16 logits with the regular ternary GEMV (which knows the
// folded basis and rotation) and then reduces each column's top-k candidates with a
// single-block scan. Columns are few (<= kDFlash maximum width), so one block per
// column is enough and the O(N * columns) scan is a small constant on top of the
// GEMV, which already walks every weight row once.
#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/score_id_order.cuh"
#include "ops/linear_topk/linear_topk_workspace.h"
#include "ops/linear/ternary/ternary_dispatch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Descending sorted insertion of `key` into `keys[0..n-1]` (0 = sentinel below every
// score/id pair). Keeps the top n of everything seen.
__device__ __forceinline__ void insert_candidate(std::uint64_t* keys, int n, std::uint64_t key) {
    int i = n - 1;
    while (i >= 0 && keys[i] < key) {
        keys[i] = (i == 0) ? key : keys[i - 1];
        --i;
    }
}

__global__ __launch_bounds__(256) void ternary_full_topk_kernel(
    const __nv_bfloat16* __restrict__ logits, std::int32_t rows, std::int32_t valid_rows,
    std::int32_t columns, const std::int32_t* __restrict__ id_map,
    std::int32_t* __restrict__ candidate_ids, float* __restrict__ candidate_scores) {
    const int column = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    constexpr int n  = kLinearTopK;

    __shared__ std::uint64_t best[256 * n];
    std::uint64_t local[n];
#pragma unroll
    for (int i = 0; i < n; ++i) { local[i] = 0ull; }

    const std::int64_t base = static_cast<std::int64_t>(column) * rows;
    for (int row = tid; row < valid_rows; row += 256) {
        const float value = __bfloat162float(logits[base + row]);
        insert_candidate(local, n, score_id_order_key(value, row));
    }
#pragma unroll
    for (int i = 0; i < n; ++i) { best[tid * n + i] = local[i]; }
    __syncthreads();

    if (tid == 0) {
        std::uint64_t top[n];
#pragma unroll
        for (int i = 0; i < n; ++i) { top[i] = 0ull; }
        for (int t = 0; t < 256; ++t) {
#pragma unroll
            for (int i = 0; i < n; ++i) { insert_candidate(top, n, best[t * n + i]); }
        }
#pragma unroll
        for (int i = 0; i < n; ++i) {
            const std::int64_t off = static_cast<std::int64_t>(column) * n + i;
            const int row          = id_from_order_key(top[i]);
            candidate_ids[off]     = id_map != nullptr ? id_map[row] : row;
            candidate_scores[off]  = score_from_order_key(top[i]);
        }
    }
}

} // namespace

void linear_topk_ternary_launch(const Tensor& hidden, const Weight& head,
                                std::int32_t valid_rows, std::int32_t columns,
                                WorkspaceArena& workspace, Tensor& logits,
                                Tensor& candidate_ids, Tensor& candidate_scores,
                                const Tensor* id_map, cudaStream_t stream) {
    ternary_dispatch(hidden, head, logits, LinearPolicy::AllowA8, &workspace, stream);
    linear_topk_ternary_reduce_launch(logits, valid_rows, columns, candidate_ids,
                                      candidate_scores, id_map, stream);
}

void linear_topk_ternary_reduce_launch(const Tensor& logits, std::int32_t valid_rows,
                                       std::int32_t columns, Tensor& candidate_ids,
                                       Tensor& candidate_scores, const Tensor* id_map,
                                       cudaStream_t stream) {
    ternary_full_topk_kernel<<<static_cast<unsigned>(columns), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), logits.ne[0], valid_rows, columns,
        id_map != nullptr ? static_cast<const std::int32_t*>(id_map->data) : nullptr,
        static_cast<std::int32_t*>(candidate_ids.data),
        static_cast<float*>(candidate_scores.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
