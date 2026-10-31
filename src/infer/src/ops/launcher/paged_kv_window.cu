#include "ops/launcher/paged_kv_window.h"

#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// One block per row: the row's window origin, then its table and positions.
__global__ void
paged_kv_window_rows_kernel(const std::int32_t* block_tables, const std::int32_t* table_rows,
                            const std::int32_t* positions, const std::int32_t* valid_columns,
                            std::int32_t pages, std::int32_t width, std::int32_t sink_pages,
                            std::int32_t window_keys, std::int32_t* window_tables,
                            std::int32_t* window_rows, std::int32_t* window_positions) {
    const std::int32_t b      = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t live   = valid_columns != nullptr ? valid_columns[b] : width;
    const std::int32_t p0     = live > 0 ? positions[static_cast<std::int64_t>(b) * width] : 0;
    const std::int32_t first  = (p0 > window_keys ? p0 - window_keys : 0) >> 6;
    const std::int32_t shift  = first > sink_pages ? first - sink_pages : 0;
    const std::int32_t* table = block_tables + static_cast<std::int64_t>(table_rows[b]) * pages;
    std::int32_t* out_table   = window_tables + static_cast<std::int64_t>(b) * pages;
    for (std::int32_t j = static_cast<std::int32_t>(threadIdx.x); j < pages;
         j += static_cast<std::int32_t>(blockDim.x)) {
        const std::int32_t source = j < sink_pages ? j : j + shift;
        out_table[j]              = table[source < pages ? source : j];
    }
    const std::int32_t delta = shift * 64;
    for (std::int32_t c = static_cast<std::int32_t>(threadIdx.x); c < width;
         c += static_cast<std::int32_t>(blockDim.x)) {
        const std::int64_t index = static_cast<std::int64_t>(b) * width + c;
        const std::int32_t value = positions[index];
        window_positions[index]  = value >= delta ? value - delta : value;
    }
    if (threadIdx.x == 0) { window_rows[b] = b; }
}

} // namespace

void paged_kv_window_rows_launch(const Tensor& block_tables, const Tensor& table_rows,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 std::int32_t sink_pages, std::int32_t window_keys,
                                 Tensor& window_tables, Tensor& window_rows,
                                 Tensor& window_positions, cudaStream_t stream) {
    const std::int32_t rows = table_rows.ne[0];
    paged_kv_window_rows_kernel<<<rows, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(block_tables.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<const std::int32_t*>(positions.data),
        valid_columns.data != nullptr ? static_cast<const std::int32_t*>(valid_columns.data)
                                      : nullptr,
        block_tables.ne[0], positions.ne[0], sink_pages, window_keys,
        static_cast<std::int32_t*>(window_tables.data),
        static_cast<std::int32_t*>(window_rows.data),
        static_cast<std::int32_t*>(window_positions.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
