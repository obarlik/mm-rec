/**
 * Gradient Utilities Implementation
 */

#include "mm_rec/training/gradient_utils.h"
#include <cmath>

namespace mm_rec {

float clip_gradients_by_norm(
    std::vector<Tensor>& gradients,
    float max_norm
) {
    // Compute global norm (L2 norm across all parameters)
    float global_norm_sq = 0.0f;
    
    for (const auto& grad : gradients) {
        const float* data = grad.data();
        int64_t numel = grad.numel();
        
        #pragma omp simd reduction(+:global_norm_sq)
        for (int64_t i = 0; i < numel; ++i) {
            global_norm_sq += data[i] * data[i];
        }
    }
    
    float global_norm = std::sqrt(global_norm_sq);
    
    // Clip if exceeds max
    if (global_norm > max_norm) {
        float scale = max_norm / global_norm;
        
        for (auto& grad : gradients) {
            float* data = grad.data();
            int64_t numel = grad.numel();
            
            #pragma omp simd
            for (int64_t i = 0; i < numel; ++i) {
                data[i] *= scale;
            }
        }
    }
    
    return global_norm;
}

} // namespace mm_rec
