/**
 * Forward Cache - Save activations for backward pass
 * 
 * CRITICAL: Backward needs ALL intermediate values from forward!
 */

#pragma once

#include "mm_rec/core/tensor.h"
#include <vector>

namespace mm_rec {

/**
 * Cache for a single MMRecBlock forward pass
 */
struct BlockCache {
    // Input to this block
    Tensor x;  // [batch, seq, hidden]
    
    // Memory states
    Tensor h_prev;  // Previous memory [batch, mem_dim]
    Tensor h_new;   // New memory [batch, mem_dim]
    
    // GRU gates (needed for backward)
    Tensor reset_gate;    // r [batch, mem_dim]
    Tensor update_gate;   // u [batch, mem_dim]
    Tensor candidate;     // h_tilde [batch, mem_dim]
    
    // GRU intermediate (r * h_prev)
    Tensor r_h_prev;  // [batch, mem_dim]
    
    // FFN activations
    Tensor ffn_input;   // [batch, seq, hidden]
    Tensor ffn_hidden;  // [batch, seq, ffn_dim]
    Tensor ffn_output;  // [batch, seq, hidden]
    
    // Output of this block
    Tensor output;  // [batch, seq, hidden]
    Tensor logits;  // [batch, seq, vocab]
};

/**
 * Cache for entire model forward pass
 */
struct ForwardCache {
    // Embedding
    Tensor input_ids;  // Original input [batch, seq]
    Tensor embedded;   // After embedding [batch, seq, hidden]
    
    // Per-block caches
    std::vector<BlockCache> block_caches;
    
    // All layer outputs for UBOO
    Tensor all_logits;  // [num_layers, batch, seq, vocab]
    
    // Initialize for given dimensions
    void init(int64_t num_layers) {
        block_caches.resize(num_layers);
    }
};

} // namespace mm_rec
