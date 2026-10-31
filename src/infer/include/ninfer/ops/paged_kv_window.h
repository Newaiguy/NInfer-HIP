#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops {

// Remaps the draft's paged read view onto "first `sink_pages` pages + newest `window_keys`
// positions" so the MTP draft's attention cost stops growing with the context, and shifts the
// row's positions by the pages dropped from the front so the causal mask still lines up. The
// cache's own table is untouched: K/V appends keep landing in their absolute pages, only the read
// view narrows.
void paged_kv_window_rows(const Tensor& block_tables, const Tensor& table_rows,
                          const Tensor& positions, const Tensor& valid_columns,
                          std::int32_t sink_pages, std::int32_t window_keys, Tensor& window_tables,
                          Tensor& window_rows, Tensor& window_positions, cudaStream_t stream);

} // namespace ninfer::ops
