/**
 * GRU Backward Implementation
 * 
 * Chain rule through all gates - Eigen optimized
 */

#include "mm_rec/training/gru_backward.h"
#include "mm_rec/training/backward.h"
#include "mm_rec/core/tensor.h"
#include <Eigen/Dense>

namespace mm_rec {

void gru_backward(
    const Tensor& x,
    const Tensor& h_prev,
    const Tensor& r,
    const Tensor& u,
    const Tensor& h_tilde,
    const Tensor& W_r, const Tensor& U_r, const Tensor& /*b_r*/,
    const Tensor& W_u, const Tensor& U_u, const Tensor& /*b_u*/,
    const Tensor& W_h, const Tensor& U_h, const Tensor& /*b_h*/,
    const Tensor& dh_new,
    GRUGradients& grads,
    Tensor& dx,
    Tensor& dh_prev
) {
    // Dimensions
    int64_t batch = x.size(0);
    // int64_t in_dim = x.size(1);
    int64_t hidden_dim = h_prev.size(1);
    
    // Map tensors to Eigen for vectorized operations
    Eigen::Map<const Eigen::ArrayXf> e_dh_new(dh_new.data(), dh_new.numel());
    Eigen::Map<const Eigen::ArrayXf> e_h_prev(h_prev.data(), h_prev.numel());
    Eigen::Map<const Eigen::ArrayXf> e_h_tilde(h_tilde.data(), h_tilde.numel());
    Eigen::Map<const Eigen::ArrayXf> e_u(u.data(), u.numel());
    Eigen::Map<const Eigen::ArrayXf> e_r(r.data(), r.numel());
    
    // ===== STEP 1: Gradient through h_new = u * h_prev + (1-u) * h_tilde =====
    
    // dL/du = dh_new * (h_prev - h_tilde) - Vectorized!
    Tensor du = Tensor::zeros(u.sizes());
    Eigen::Map<Eigen::ArrayXf> e_du(du.data(), du.numel());
    e_du = e_dh_new * (e_h_prev - e_h_tilde);
    
    // dL/dh_tilde = dh_new * (1 - u) - Vectorized!
    Tensor dh_tilde = Tensor::zeros(h_tilde.sizes());
    Eigen::Map<Eigen::ArrayXf> e_dh_tilde(dh_tilde.data(), dh_tilde.numel());
    e_dh_tilde = e_dh_new * (1.0f - e_u);
    
    // dL/dh_prev (partial) - Vectorized!
    dh_prev = Tensor::zeros(h_prev.sizes());
    Eigen::Map<Eigen::ArrayXf> e_dh_prev(dh_prev.data(), dh_prev.numel());
    e_dh_prev = e_dh_new * e_u;
    
    // ===== STEP 2: Gradient through h_tilde = tanh(z_h) =====
    // where z_h = W_h @ x + U_h @ (r * h_prev) + b_h
    
    Tensor dz_h = tanh_backward(h_tilde, dh_tilde);
    
    // ===== STEP 3: Gradient through z_h = W_h @ x + U_h @ (r * h_prev) + b_h =====
    
    // Compute r * h_prev - Vectorized!
    Tensor r_h_prev = Tensor::zeros({batch, hidden_dim});
    Eigen::Map<Eigen::ArrayXf> e_r_h_prev(r_h_prev.data(), r_h_prev.numel());
    e_r_h_prev = e_r * e_h_prev;
    
    // Linear backward for h gate
    Tensor dx_h, dr_h_prev;
    linear_backward(x, W_h, dz_h, dx_h, grads.dW_h, grads.db_h);
    linear_backward(r_h_prev, U_h, dz_h, dr_h_prev, grads.dU_h, grads.db_h);  // db_h accumulated
    
    // ===== STEP 4: Gradient through r * h_prev =====
    
    // dL/dr = dr_h_prev * h_prev - Vectorized!
    Tensor dr = Tensor::zeros(r.sizes());
    Eigen::Map<Eigen::ArrayXf> e_dr(dr.data(), dr.numel());
    Eigen::Map<const Eigen::ArrayXf> e_dr_h_prev(dr_h_prev.data(), dr_h_prev.numel());
    e_dr = e_dr_h_prev * e_h_prev;
    
    // Accumulate to dh_prev - Vectorized!
    e_dh_prev += e_dr_h_prev * e_r;
    
    // ===== STEP 5: Gradient through r = sigmoid(z_r) and u = sigmoid(z_u) =====
    
    Tensor dz_r = sigmoid_backward(r, dr);
    Tensor dz_u = sigmoid_backward(u, du);
    
    // ===== STEP 6: Linear backward for reset and update gates =====
    
    Tensor dx_r, dh_prev_r;
    linear_backward(x, W_r, dz_r, dx_r, grads.dW_r, grads.db_r);
    linear_backward(h_prev, U_r, dz_r, dh_prev_r, grads.dU_r, grads.db_r);
    
    Tensor dx_u, dh_prev_u;
    linear_backward(x, W_u, dz_u, dx_u, grads.dW_u, grads.db_u);
    linear_backward(h_prev, U_u, dz_u, dh_prev_u, grads.dU_u, grads.db_u);
    
    // ===== STEP 7: Accumulate all dx and dh_prev - Vectorized! =====
    
    dx = Tensor::zeros(x.sizes());
    Eigen::Map<Eigen::ArrayXf> e_dx(dx.data(), dx.numel());
    Eigen::Map<const Eigen::ArrayXf> e_dx_h(dx_h.data(), dx_h.numel());
    Eigen::Map<const Eigen::ArrayXf> e_dx_r(dx_r.data(), dx_r.numel());
    Eigen::Map<const Eigen::ArrayXf> e_dx_u(dx_u.data(), dx_u.numel());
    e_dx = e_dx_h + e_dx_r + e_dx_u;
    
    Eigen::Map<const Eigen::ArrayXf> e_dh_prev_r(dh_prev_r.data(), dh_prev_r.numel());
    Eigen::Map<const Eigen::ArrayXf> e_dh_prev_u(dh_prev_u.data(), dh_prev_u.numel());
    e_dh_prev += e_dh_prev_r + e_dh_prev_u;
}

} // namespace mm_rec
