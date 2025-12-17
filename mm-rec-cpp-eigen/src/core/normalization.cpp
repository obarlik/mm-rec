#include "mm_rec/core/normalization.h"
#include <cmath>
#include <iostream>

namespace mm_rec {

RMSNorm::RMSNorm(int64_t dim, float eps) 
    : dim_(dim), eps_(eps) {
    // Initialize weights to 1.0
    weight_ = Tensor::ones({dim});
}

Tensor RMSNorm::forward(const Tensor& x) {
    // x: [batch, seq, dim] or [batch, dim]
    // We normalize over the last dimension (dim_)
    
    int64_t last_dim = x.sizes().back();
    if (last_dim != dim_) {
        throw std::runtime_error("RMSNorm dimension mismatch");
    }
    
    int64_t numels = x.numel();
    int64_t num_vectors = numels / dim_;
    
    Tensor output = Tensor::zeros(x.sizes());
    
    const float* in_ptr = x.data();
    float* out_ptr = output.data();
    const float* w_ptr = weight_.data();
    
    // Parallelize over vectors
    #pragma omp parallel for
    for (int64_t i = 0; i < num_vectors; ++i) {
        const float* vec_in = in_ptr + i * dim_;
        float* vec_out = out_ptr + i * dim_;
        
        // 1. Calculate sum of squares
        float sum_sq = 0.0f;
        for (int64_t j = 0; j < dim_; ++j) {
            sum_sq += vec_in[j] * vec_in[j];
        }
        
        // 2. Calculate RMS
        float rms = std::sqrt((sum_sq / dim_) + eps_);
        float inv_rms = 1.0f / rms;
        
        // 3. Normalize and Scale
        for (int64_t j = 0; j < dim_; ++j) {
            vec_out[j] = vec_in[j] * inv_rms * w_ptr[j];
        }
    }
    
    return output;
}

Tensor RMSNorm::backward(const Tensor& d_out, const Tensor& x, RMSNormGradients& grads) {
    // x: [..., dim]
    // d_out: [..., dim] produces d_x and d_weight
    
    int64_t last_dim = x.sizes().back();
    int64_t numels = x.numel();
    int64_t num_vectors = numels / dim_;
    
    Tensor dx = Tensor::zeros(x.sizes());
    
    // Gradients for weight need accumulation
    if (grads.d_weight.numel() != dim_) {
        grads.d_weight = Tensor::zeros({dim_});
    }
    
    // Use thread-local accumulation for weight gradients to avoid locking
    // Note: Simple OMP reduction for arrays is messy in C++, so we might do it serially for weight
    // or use a reduction buffer. For simplicity/MVP, we accumulate straight into grads.d_weight 
    // BUT that's a race condition if parallel. 
    // Strategy: Serialize weight accumulation or atomic?
    // Let's do parallel dX and accumulate dW carefully or serially.
    // Given dim is usually small (128-1024), serial accumulation over batch might remain fast enough relative to dX.
    // Let's optimize dX parallel, and dW accumulation serially for now for safety.
    
    const float* dout_ptr = d_out.data();
    const float* in_ptr = x.data();
    const float* w_ptr = weight_.data();
    float* dx_ptr = dx.data();
    float* dw_ptr = grads.d_weight.data();
    
    // For dW accumulation
    std::vector<float> dw_buffer(dim_, 0.0f);
    
    // Pre-calculate per-vector stats
    std::vector<float> inv_rms_cache(num_vectors);
    
    #pragma omp parallel for
    for (int64_t i = 0; i < num_vectors; ++i) {
        const float* vec_in = in_ptr + i * dim_;
        float sum_sq = 0.0f;
        for (int64_t j = 0; j < dim_; ++j) sum_sq += vec_in[j] * vec_in[j];
        inv_rms_cache[i] = 1.0f / std::sqrt((sum_sq / dim_) + eps_);
    }

    // Compute dX and accumulate dW (Serial for safety on dW for now)
    // Actually, locking/atomic is needed for dW. 
    // Optim: Parallelize dX calculation, but accumulate dW separately? 
    // dL/dw_j = sum_vectors( dL/dy_vec * x_vec * inv_rms )
    
    // Reset dw
    grads.d_weight.zero_();
    
    // Parallelize with reduction? Or explicit locking.
    // Let's do dX fully parallel first.
    
    #pragma omp parallel for
    for (int64_t i = 0; i < num_vectors; ++i) {
        const float* vec_in = in_ptr + i * dim_;
        const float* vec_dout = dout_ptr + i * dim_;
        float* vec_dx = dx_ptr + i * dim_;
        
        float inv_rms = inv_rms_cache[i];
        
        // dL/dx computation for RMSNorm is tricky:
        // y_j = (x_j * inv_rms) * w_j_(constant for x)
        // dx_k = ... formula involves sum over j
        
        // Intermediate term: sum(dout * w * x)
        float dot_dout_w_x = 0.0f;
        for(int64_t j=0; j<dim_; ++j) {
            dot_dout_w_x += vec_dout[j] * w_ptr[j] * vec_in[j];
        }
        
        float factor = dot_dout_w_x * (inv_rms * inv_rms * inv_rms) / dim_;
        
        for(int64_t j=0; j<dim_; ++j) {
            // dL/dx_j = (dL/dy_j * w_j * inv_rms) - (x_j * factor)
            vec_dx[j] = (vec_dout[j] * w_ptr[j] * inv_rms) - (vec_in[j] * factor);
        }
    }
    
    // Accumulate dW (Serial is safe and okay for small dims)
    for (int64_t i = 0; i < num_vectors; ++i) {
        const float* vec_in = in_ptr + i * dim_;
        const float* vec_dout = dout_ptr + i * dim_;
        float inv_rms = inv_rms_cache[i];
        
        for(int64_t j=0; j<dim_; ++j) {
            // dL/dw_j = dL/dy_j * (x_j * inv_rms)
            dw_ptr[j] += vec_dout[j] * (vec_in[j] * inv_rms);
        }
    }
    
    return dx;
}

void RMSNorm::update_parameters(SGD& optimizer, const RMSNormGradients& grads) {
    optimizer.step(weight_, grads.d_weight);
}

} // namespace mm_rec
