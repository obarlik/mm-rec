/**
 * UBOO Output Implementation
 */

#include "mm_rec/layers/uboo_output.h"

namespace mm_rec {

UBOOOutput::UBOOOutput(int64_t hidden_dim, int64_t vocab_size)
    : hidden_dim_(hidden_dim), vocab_size_(vocab_size) {
    
    // Linear projection: hidden_dim -> vocab_size
    projection = register_module(
        "projection",
        torch::nn::Linear(hidden_dim, vocab_size)
    );
    
    // Initialize weights (similar to LM head in transformers)
    // Small std for stability
    const double std = 0.02;
    torch::nn::init::normal_(projection->weight, 0.0, std);
    torch::nn::init::zeros_(projection->bias);
}

torch::Tensor UBOOOutput::forward(const torch::Tensor& hidden) {
    // Input: [batch, seq_len, hidden_dim] or [batch, hidden_dim]
    // Output: [batch, seq_len, vocab_size] or [batch, vocab_size]
    
    TORCH_CHECK(
        hidden.size(-1) == hidden_dim_,
        "Hidden dim mismatch. Expected ", hidden_dim_, ", got ", hidden.size(-1)
    );
    
    return projection->forward(hidden);
}

} // namespace mm_rec
