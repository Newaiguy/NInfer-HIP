#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void linear_topk_q8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                           const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_q8_m64_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_fp8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                            const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_fp8_m64_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                                const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_q4_launch(const Tensor& hidden, const Weight& head,
                           const Tensor& row_to_global_ids, const LinearTopKWorkspace& workspace,
                           cudaStream_t stream);
void linear_topk_q4_m64_launch(const Tensor& hidden, const Weight& head,
                               const Tensor& row_to_global_ids,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream);
// AMD port: ternary full-head fallback (compute full logits with the regular ternary
// GEMV, then reduce per-column top-k). Used when the DFlash2 selector targets a T2 head.
// `id_map` (optional) translates physical rows to global token ids for the optimized
// (indexed) head.
void linear_topk_ternary_launch(const Tensor& hidden, const Weight& head,
                                std::int32_t valid_rows, std::int32_t columns,
                                WorkspaceArena& workspace, Tensor& logits,
                                Tensor& candidate_ids, Tensor& candidate_scores,
                                const Tensor* id_map, cudaStream_t stream);
void linear_topk_ternary_reduce_launch(const Tensor& logits, std::int32_t valid_rows,
                                       std::int32_t columns, Tensor& candidate_ids,
                                       Tensor& candidate_scores, const Tensor* id_map,
                                       cudaStream_t stream);
void linear_topk_merge_launch(const LinearTopKWorkspace& workspace, Tensor& candidate_ids,
                              Tensor& candidate_scores, cudaStream_t stream);

} // namespace ninfer::ops::detail
