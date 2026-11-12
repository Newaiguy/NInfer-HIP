#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Multiplies every element of a contiguous BF16 tensor by one scalar:
 *
 *   x[i] = x[i] * factor   for all i.
 *
 * Used to bring externally produced embeddings (the vision tower output, for example) into the same
 * numeric range as the token embeddings they get spliced next to. The tensor is updated in place on
 * `stream`; there is no workspace and no other state side effect.
 */
void scale_bf16(Tensor& x, float factor, cudaStream_t stream);

} // namespace ninfer::ops
