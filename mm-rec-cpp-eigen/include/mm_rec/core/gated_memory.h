/**
 * GRU-Style Gated Memory (Pure C++)
 * 
 * NO LibTorch dependency - uses custom Tensor class
 */

#pragma once

#include "mm_rec/core/tensor.h"
#include "mm_rec/core/linear.h"
#include <memory>

namespace mm_rec {

class GatedMemoryUpdate {
public:
    GatedMemoryUpdate(int64_t hidden_dim, int64_t mem_dim);
    
    // Standard forward
    std::pair<Tensor, Tensor> forward(const Tensor& h_t, const Tensor& m_prev);

    // Forward with cache saving (for training)
    struct Cache {
        Tensor r, u, h_tilde, r_h_prev;
    };
    std::pair<Tensor, Tensor> forward(const Tensor& h_t, const Tensor& m_prev, Cache& cache);

    // Accessors for backward pass
    Linear* get_W_z() { return W_z.get(); }
    Linear* get_W_r() { return W_r.get(); }
    Linear* get_W_m() { return W_m.get(); }

private:
    std::unique_ptr<Linear> W_z;  // Update gate
    std::unique_ptr<Linear> W_r;  // Reset gate
    std::unique_ptr<Linear> W_m;  // Candidate
    
    int64_t hidden_dim_;
    int64_t mem_dim_;
    
    Tensor concat_tensors(const Tensor& a, const Tensor& b);
};

} // namespace mm_rec
