#pragma once

#include "mm_rec/core/tensor.h"
#include "mm_rec/training/optimizer.h"
#include "mm_rec/training/optimizer.h"

#include "mm_rec/training/optimizer.h"

namespace mm_rec {

struct RMSNormGradients {
    Tensor d_weight;
};

class RMSNorm {
public:
    RMSNorm(int64_t dim, float eps = 1e-6);

    Tensor forward(const Tensor& x);

    Tensor backward(const Tensor& d_out, const Tensor& x, RMSNormGradients& grads);

    void update_parameters(Optimizer& optimizer, const RMSNormGradients& grads);
    
    // Accessors
    Tensor& weight() { return weight_; }
    const Tensor& weight() const { return weight_; }

private:
    int64_t dim_;
    float eps_;
    Tensor weight_; // Learnable parameter
};

} // namespace mm_rec
