/**
 * GRU-Style Gated Memory Implementation
 * 
 * Based on knowledge transfer from 8+ months production training
 */

#include "mm_rec/core/gated_memory.h"

namespace mm_rec {

GatedMemoryUpdate::GatedMemoryUpdate(int64_t hidden_dim, int64_t mem_dim)
    : hidden_dim_(hidden_dim), mem_dim_(mem_dim) {
    
    // Combined input size for gates
    const int64_t combined_dim = hidden_dim + mem_dim;
    
    // Initialize linear layers
    W_z = register_module("W_z", torch::nn::Linear(combined_dim, mem_dim));
    W_r = register_module("W_r", torch::nn::Linear(combined_dim, mem_dim));
    W_m = register_module("W_m", torch::nn::Linear(combined_dim, mem_dim));
    
    // Initialize weights (Xavier/Glorot initialization)
    // Similar to PyTorch default but explicit for clarity
    const double std_z = std::sqrt(2.0 / (combined_dim + mem_dim));
    const double std_r = std::sqrt(2.0 / (combined_dim + mem_dim));
    const double std_m = std::sqrt(2.0 / (combined_dim + mem_dim));
    
    torch::nn::init::normal_(W_z->weight, 0.0, std_z);
    torch::nn::init::normal_(W_r->weight, 0.0, std_r);
    torch::nn::init::normal_(W_m->weight, 0.0, std_m);
    
    // Biases initialized to zero (standard)
    torch::nn::init::zeros_(W_z->bias);
    torch::nn::init::zeros_(W_r->bias);
    torch::nn::init::zeros_(W_m->bias);
}

std::pair<torch::Tensor, torch::Tensor> GatedMemoryUpdate::forward(
    const torch::Tensor& h_t,
    const torch::Tensor& m_prev
) {
    // Input validation
    TORCH_CHECK(h_t.dim() == 2, "h_t must be 2D [batch, hidden_dim]");
    TORCH_CHECK(m_prev.dim() == 2, "m_prev must be 2D [batch, mem_dim]");
    TORCH_CHECK(h_t.size(0) == m_prev.size(0), "Batch size mismatch");
    TORCH_CHECK(h_t.size(1) == hidden_dim_, "Hidden dim mismatch");
    TORCH_CHECK(m_prev.size(1) == mem_dim_, "Memory dim mismatch");
    
    // Concatenate inputs: [batch, hidden_dim + mem_dim]
    auto concat = torch::cat({h_t, m_prev}, /*dim=*/1);
    
    // Update gate: controls how much new info flows in
    // z_t = sigmoid(W_z @ [h_t, m_{t-1}])
    auto z_t = torch::sigmoid(W_z->forward(concat));
    
    // Reset gate: controls selective forgetting
    // r_t = sigmoid(W_r @ [h_t, m_{t-1}])
    auto r_t = torch::sigmoid(W_r->forward(concat));
    
    // CRITICAL: Reset gate multiplies old memory BEFORE candidate computation
    // This is Bug #4 from knowledge transfer - DO NOT CHANGE THIS ORDER!
    // 
    // Why? Reset gate allows model to selectively forget irrelevant past.
    // If you apply reset AFTER candidate, model cannot forget properly.
    auto reset_memory = r_t * m_prev;  // Element-wise multiplication
    
    // Concatenate with reset memory for candidate
    auto concat_reset = torch::cat({h_t, reset_memory}, /*dim=*/1);
    
    // Candidate memory state
    // m̃_t = tanh(W_m @ [h_t, r_t * m_{t-1}])
    auto m_tilde = torch::tanh(W_m->forward(concat_reset));
    
    // Final memory update: blend old and candidate using update gate
    // m_t = (1 - z_t) * m_{t-1} + z_t * m̃_t
    // 
    // Interpretation:
    // - When z_t ≈ 0: Keep old memory (m_{t-1})
    // - When z_t ≈ 1: Use new candidate (m̃_t)
    // - When z_t ≈ 0.5: Blend equally
    auto m_t = (1.0 - z_t) * m_prev + z_t * m_tilde;
    
    // CRITICAL: DO NOT use .detach() here!
    // Bug #3 from knowledge transfer: Detaching breaks gradient flow
    // Memory state must maintain computation graph for backprop
    
    return {m_t, z_t};  // Return new memory and update gate (for analysis)
}

} // namespace mm_rec
