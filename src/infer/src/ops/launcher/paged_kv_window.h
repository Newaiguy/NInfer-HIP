#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Remaps one row's block table so the MTP draft attends to the first `sink_pages` pages plus the
// newest `window_keys` positions, and shifts the row's positions by the number of pages dropped
// from the front. Appended K/V rows still go through the cache's own table, so only the read view
// narrows.
void paged_kv_window_rows_launch(const Tensor& block_tables, const Tensor& table_rows,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 std::int32_t sink_pages, std::int32_t window_keys,
                                 Tensor& window_tables, Tensor& window_rows,
                                 Tensor& window_positions, cudaStream_t stream);

} // namespace ninfer::ops::detail
