/**
 * Linear Layer (Pure C++)
 * 
 * Fully-connected layer using custom Tensor class and MKL.
 */

#pragma once

#include "mm_rec/core/tensor.h"

namespace mm_rec {

class Linear {
public:
    Linear(int64_t in_features, int64_t out_features);
    
    Tensor forward(const Tensor& input);
    
    // Getters
    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }
    
private:
    Tensor weight_;  // [out_features, in_features]
    Tensor bias_;    // [out_features]
    int64_t in_features_;
    int64_t out_features_;
};

} // namespace mm_rec
