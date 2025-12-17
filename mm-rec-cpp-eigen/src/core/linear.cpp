/**
 * Linear Layer Implementation
 */

#include "mm_rec/core/linear.h"
#include <cmath>

namespace mm_rec {

Linear::Linear(int64_t in_features, int64_t out_features)
    : in_features_(in_features), out_features_(out_features) {
    
    // Xavier/Glorot initialization
    float std = std::sqrt(2.0f / (in_features + out_features));
    
    weight_ = Tensor::randn({out_features, in_features}, 0.0f, std);
    bias_ = Tensor::zeros({out_features});
}

Tensor Linear::forward(const Tensor& input) {
    // input: [batch, in_features]
    // weight: [out_features, in_features]
    // output: [batch, out_features]
    
    // Multiply input by weight^T
    Tensor output = input.matmul(weight_.transpose());
    
    // Add bias (broadcast)
    // For now, simple loop (can optimize later)
    int64_t batch = input.size(0);
    float* out_data = output.data();
    const float* bias_data = bias_.data();
    
    #pragma omp parallel for if(batch > 1)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t f = 0; f < out_features_; ++f) {
            out_data[b * out_features_ + f] += bias_data[f];
        }
    }
    
    return output;
}

} // namespace mm_rec
