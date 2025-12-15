/**
 * GRU Backward Pass
 * 
 * Most complex backward - through all gates
 */

#pragma once

#include "mm_rec/core/tensor.h"
#include "mm_rec/training/gradients.h"

namespace mm_rec {

/**
 * GRU backward pass
 * 
 * Forward reminder:
 *   r = sigmoid(W_r @ x + U_r @ h_prev + b_r)     // Reset gate
 *   u = sigmoid(W_u @ x + U_u @ h_prev + b_u)     // Update gate  
 *   h_tilde = tanh(W_h @ x + U_h @ (r * h_prev) + b_h)  // Candidate
 *   h_new = u * h_prev + (1-u) * h_tilde          // New state
 * 
 * Backward:
 *   Given dL/dh_new, compute:
 *   - dL/dW_r, dL/dU_r, dL/db_r
 *   - dL/dW_u, dL/dU_u, dL/db_u
 *   - dL/dW_h, dL/dU_h, dL/db_h
 *   - dL/dx, dL/dh_prev
 * 
 * @param x Input [batch, in_dim]
 * @param h_prev Previous hidden state [batch, hidden_dim]
 * @param r Reset gate output (from forward) [batch, hidden_dim]
 * @param u Update gate output (from forward) [batch, hidden_dim]
 * @param h_tilde Candidate state (from forward) [batch, hidden_dim]
 * @param W_r, U_r, b_r Reset gate parameters
 * @param W_u, U_u, b_u Update gate parameters
 * @param W_h, U_h, b_h Candidate parameters
 * @param dh_new Gradient from next layer [batch, hidden_dim]
 * @param grads Output: all parameter gradients
 * @param dx Output: gradient for input
 * @param dh_prev Output: gradient for previous hidden state
 */
void gru_backward(
    // Forward pass inputs and outputs
    const Tensor& x,
    const Tensor& h_prev,
    const Tensor& r,
    const Tensor& u,
    const Tensor& h_tilde,
    // Parameters
    const Tensor& W_r, const Tensor& U_r, const Tensor& b_r,
    const Tensor& W_u, const Tensor& U_u, const Tensor& b_u,
    const Tensor& W_h, const Tensor& U_h, const Tensor& b_h,
    // Incoming gradient
    const Tensor& dh_new,
    // Outputs: gradients
    GRUGradients& grads,
    Tensor& dx,
    Tensor& dh_prev
);

} // namespace mm_rec
