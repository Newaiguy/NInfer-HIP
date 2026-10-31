#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// AMD port: Q4 materialized dynamic grouped conv add. Same contract as the Q8 variant
// but reads a Q4_G64 branch projection weight.
void q4_dynamic_grouped_conv_add_materialized_launch(const Tensor& x, const Weight& weight,
                                                     const Tensor& base_kernel,
                                                     const Tensor& finish_delta, Tensor& residual,
                                                     Tensor& projected, cudaStream_t stream);

} // namespace ninfer::ops::detail
