/**
 * SGD Optimizer
 * 
 * Simple stochastic gradient descent
 */

#pragma once

#include "mm_rec/core/tensor.h"
#include <vector>

namespace mm_rec {

/**
 * Simple SGD optimizer
 */
class SGD {
public:
    explicit SGD(float learning_rate) : lr_(learning_rate) {}
    
    /**
     * Update parameters: W = W - lr * dW
     */
    void step(Tensor& param, const Tensor& grad) {
        for (int64_t i = 0; i < param.numel(); ++i) {
            param.data()[i] -= lr_ * grad.data()[i];
        }
    }
    
    /**
     * Update multiple parameters at once
     */
    void step(std::vector<Tensor*> params, const std::vector<Tensor>& grads) {
        for (size_t i = 0; i < params.size(); ++i) {
            step(*params[i], grads[i]);
        }
    }
    
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }
    
private:
    float lr_;
};

} // namespace mm_rec
