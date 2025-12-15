/**
 * Full MM-Rec Model
 * 
 * Stack of N MM-Rec blocks with:
 * - Token embedding
 * - Per-layer memory states (Bug #2 prevention)
 * - UBOO deep supervision
 */

#pragma once

#include "mm_rec/model/mm_rec_block.h"
#include "mm_rec/core/tensor.h"
#include <vector>
#include <memory>

namespace mm_rec {

struct MMRecModelConfig {
    int64_t vocab_size;
    int64_t hidden_dim;
    int64_t mem_dim;
    int64_t ffn_dim;
    int64_t num_layers;
};

class MMRecModel {
public:
    MMRecModel(const MMRecModelConfig& config);
    
    // Forward pass (training mode)
    // Returns: [num_layers, batch, seq, vocab_size]
    Tensor forward(Tensor input_ids);  // Changed: pass by value
    
    // Inference helper (only returns final layer logits)
    Tensor generate(const Tensor& input_ids);
    
    /**
     * Reset memory states (start new sequence)
     */
    void reset_memory(int64_t batch_size);

private:
    MMRecModelConfig config_;
    
    // Embedding
    Tensor embedding_weights_;  // [vocab_size, hidden_dim]
    
    // Layers
    std::vector<std::unique_ptr<MMRecBlock>> blocks_;
    
    // Per-layer memory states (Bug #2: isolated!)
    std::vector<Tensor> memory_states_;
    
    // Helper: embed tokens
    Tensor embed(const Tensor& input_ids);
};

} // namespace mm_rec
