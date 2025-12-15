/**
 * Gradient Utilities
 * 
 * Tools for gradient manipulation during training
 */

#pragma once

#include "mm_rec/core/tensor.h"
#include <vector>

namespace mm_rec {

/**
 * Clip gradients by global norm
 * 
 * Algorithm:
 *   1. Compute global_norm = sqrt(sum(grad^2 for all grads))
 *   2. If global_norm > max_norm:
 *        scale = max_norm / global_norm
 *        for each grad: grad *= scale
 * 
 * @param gradients Vector of gradient tensors
 * @param max_norm Maximum allowed norm (typically 1.0)
 * @return Actual global norm before clipping
 */
float clip_gradients_by_norm(
    std::vector<Tensor>& gradients,
    float max_norm
);

} // namespace mm_rec
