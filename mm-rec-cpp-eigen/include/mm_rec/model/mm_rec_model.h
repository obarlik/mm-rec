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
#include "mm_rec/training/gradients.h"
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

struct ForwardCache; // Forward declaration
class SGD; // Forward decl

class MMRecModel {
public:
    MMRecModel(const MMRecModelConfig& config);
    
    // Forward pass (training mode)
    // Returns: [num_layers, batch, seq, vocab_size]
    Tensor forward(Tensor input_ids, ForwardCache* cache = nullptr);  // Changed: pass by value
    
    // Inference helper (only returns final layer logits)
    Tensor generate(const Tensor& input_ids);
    
    /**
     * Backward pass
     * 
     * @param targets Target indices [batch, seq]
     * @param cache Forward cache
     * @return ModelGradients
     */
    ModelGradients backward(const Tensor& targets, const ForwardCache& cache);

    /**
     * Update all model parameters
     */
    void update_parameters(SGD& optimizer, const ModelGradients& grads);

    /**
     * Reset memory states (start new sequence)
     */
    void reset_memory(int64_t batch_size);
    
    // Checkpoint accessors (for save/load)
    const MMRecModelConfig& get_config() const { return config_; }
    const Tensor& get_embedding_weights() const { return embedding_weights_; }
    Tensor& get_embedding_weights() { return embedding_weights_; }
    const std::vector<std::unique_ptr<MMRecBlock>>& get_blocks() const { return blocks_; }
    std::vector<std::unique_ptr<MMRecBlock>>& get_blocks() { return blocks_; }

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
