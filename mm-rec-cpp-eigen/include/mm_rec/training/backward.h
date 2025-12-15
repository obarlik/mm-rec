/**
 * Backward Pass Functions
 * 
 * Manual gradient computation (no autodiff!)
 */

#pragma once

#include "mm_rec/core/tensor.h"

namespace mm_rec {

/**
 * Linear layer backward
 * 
 * Forward:  y = Wx + b
 * Backward: dx = W^T @ dy
 *           dW = dy @ x^T  
 *           db = sum(dy, axis=0)
 * 
 * @param x Input tensor [batch, in_dim]
 * @param W Weight tensor [out_dim, in_dim]
 * @param dy Gradient from next layer [batch, out_dim]
 * @param dx Output: gradient for input [batch, in_dim]
 * @param dW Output: gradient for weights [out_dim, in_dim]
 * @param db Output: gradient for bias [out_dim]
 */
void linear_backward(
    const Tensor& x,
    const Tensor& W,
    const Tensor& dy,
    Tensor& dx,
    Tensor& dW,
    Tensor& db
);

/**
 * Sigmoid backward
 * 
 * Forward:  y = 1 / (1 + exp(-x))
 * Backward: dx = dy * y * (1 - y)
 * 
 * @param y Output from forward pass
 * @param dy Gradient from next layer
 * @return dx Gradient for input
 */
Tensor sigmoid_backward(const Tensor& y, const Tensor& dy);

/**
 * Tanh backward
 * 
 * Forward:  y = tanh(x)
 * Backward: dx = dy * (1 - y^2)
 * 
 * @param y Output from forward pass
 * @param dy Gradient from next layer
 * @return dx Gradient for input
 */
Tensor tanh_backward(const Tensor& y, const Tensor& dy);

/**
 * ReLU backward
 * 
 * Forward:  y = max(0, x)
 * Backward: dx = dy if x > 0 else 0
 *           (Using y > 0 as proxy since y=max(0,x))
 * 
 * @param y Output from forward pass
 * @param dy Gradient from next layer
 * @return dx Gradient for input
 */
Tensor relu_backward(const Tensor& y, const Tensor& dy);

/**
 * Softmax + Cross-Entropy backward (combined for numerical stability)
 * 
 * Forward:  y_pred = softmax(logits)
 *           loss = -sum(y_true * log(y_pred))
 * Backward: dlogits = y_pred - y_true
 * 
 * @param logits Input logits [batch, vocab]
 * @param targets Target indices [batch]
 * @return dlogits Gradient for logits [batch, vocab]
 */
Tensor softmax_cross_entropy_backward(
    const Tensor& logits,
    const Tensor& targets
);

} // namespace mm_rec
