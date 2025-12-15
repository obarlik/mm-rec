/**
 * GRU-Style Gated Memory Implementation (Pure C++)
 */

#include "mm_rec/core/gated_memory.h"
#include <cstring>

namespace mm_rec {

GatedMemoryUpdate::GatedMemoryUpdate(int64_t hidden_dim, int64_t mem_dim)
    : hidden_dim_(hidden_dim), mem_dim_(mem_dim) {
    
    int64_t combined_dim = hidden_dim + mem_dim;
    
    W_z = std::make_unique<Linear>(combined_dim, mem_dim);
    W_r = std::make_unique<Linear>(combined_dim, mem_dim);
    W_m = std::make_unique<Linear>(combined_dim, mem_dim);
}

Tensor GatedMemoryUpdate::concat_tensors(const Tensor& a, const Tensor& b) {
    // a: [batch, dim_a], b: [batch, dim_b]
    // result: [batch, dim_a + dim_b]
    
    int64_t batch = a.size(0);
    int64_t dim_a = a.size(1);
    int64_t dim_b = b.size(1);
    
    Tensor result = Tensor::zeros({batch, dim_a + dim_b});
    
    const float* a_data = a.data();
    const float* b_data = b.data();
    float* result_data = result.data();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < batch; ++i) {
        // Copy a
        std::memcpy(
            result_data + i * (dim_a + dim_b),
            a_data + i * dim_a,
            dim_a * sizeof(float)
        );
        
        // Copy b
        std::memcpy(
            result_data + i * (dim_a + dim_b) + dim_a,
            b_data + i * dim_b,
            dim_b * sizeof(float)
        );
    }
    
    return result;
}

std::pair<Tensor, Tensor> GatedMemoryUpdate::forward(
    const Tensor& h_t,
    const Tensor& m_prev
) {
    Cache dummy;
    return forward(h_t, m_prev, dummy);
}

std::pair<Tensor, Tensor> GatedMemoryUpdate::forward(
    const Tensor& h_t,
    const Tensor& m_prev,
    Cache& cache
) {
    // Concatenate inputs
    Tensor concat = concat_tensors(h_t, m_prev);
    
    // Update gate
    Tensor z_t = W_z->forward(concat).sigmoid();
    
    // Reset gate
    Tensor r_t = W_r->forward(concat).sigmoid();
    
    // CRITICAL: Reset gate multiplies old memory BEFORE candidate
    Tensor reset_memory = r_t * m_prev;
    
    // Concatenate with reset memory
    Tensor concat_reset = concat_tensors(h_t, reset_memory);
    
    // Candidate state
    Tensor m_tilde = W_m->forward(concat_reset).tanh_activation();
    
    // Save to cache
    // Note: In implementation `m_t = (1 - z) * m_prev + z * m_tilde`
    // My cache naming: u = z_t
    cache.u = z_t;
    cache.r = r_t;
    cache.h_tilde = m_tilde;
    cache.r_h_prev = reset_memory; 
    
    // Final memory update
    // m_t = (1 - z_t) * m_prev + z_t * m_tilde
    Tensor one_minus_z = (z_t * -1.0f) + Tensor::ones(z_t.sizes());
    Tensor m_t = (one_minus_z * m_prev) + (z_t * m_tilde);
    
    return {m_t, z_t};
}

} // namespace mm_rec
