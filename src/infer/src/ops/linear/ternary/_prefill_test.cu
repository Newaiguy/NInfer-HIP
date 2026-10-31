#include "ops/linear/ternary/ternary_rowsplit_mma_prefill.cuh"
namespace ninfer::ops::detail {
void prefill_test_compile_trigger() { (void)&pq2_rowsplit_gemm_mma_kernel<false>; }
} // namespace
