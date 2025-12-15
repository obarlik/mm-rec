/**
 * Full MM-Rec Model Implementation
 */

#include "mm_rec/model/mm_rec_model.h"
#include <random>
#include <iostream>

namespace mm_rec {

MMRecModel::MMRecModel(const MMRecModelConfig& config)
    : config_(config) {
    
    // Initialize embedding
    // Xavier initialization for embeddings
    float std = std::sqrt(2.0f / (config.vocab_size + config.hidden_dim));
    embedding_weights_ = Tensor::randn(
        {config.vocab_size, config.hidden_dim},
        0.0f,
        std
    );
    
    // Initialize layers
    MMRecBlockConfig block_config{
        config.hidden_dim,
        config.mem_dim,
        config.ffn_dim,
        config.vocab_size
    };
    
    for (int64_t i = 0; i < config.num_layers; ++i) {
        blocks_.push_back(std::make_unique<MMRecBlock>(block_config));
    }
    
    // Initialize memory states (will be resized on first forward)
    memory_states_.resize(config.num_layers);
}

Tensor MMRecModel::embed(const Tensor& input_ids) {
    // input_ids: [batch, seq] (integers)
    // output: [batch, seq, hidden_dim]
    
    std::cout << "[DEBUG embed] input_ids.ndim() = " << input_ids.ndim() << std::endl;
    std::cout << "[DEBUG embed] input_ids.numel() = " << input_ids.numel() << std::endl;
    
    int64_t batch = input_ids.size(0);
    int64_t seq = input_ids.size(1);
    
    Tensor embeddings = Tensor::zeros({batch, seq, config_.hidden_dim});
    
    // Lookup embeddings
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            int64_t token_id = static_cast<int64_t>(
                input_ids.data()[b * seq + s]
            );
            
            // Copy embedding vector
            for (int64_t h = 0; h < config_.hidden_dim; ++h) {
                embeddings.data()[b * seq * config_.hidden_dim + s * config_.hidden_dim + h] =
                    embedding_weights_.data()[token_id * config_.hidden_dim + h];
            }
        }
    }
    
    return embeddings;
}

void MMRecModel::reset_memory(int64_t batch_size) {
    // Reset all layer memories to zero
    for (int64_t i = 0; i < config_.num_layers; ++i) {
        memory_states_[i] = Tensor::zeros({batch_size, config_.mem_dim});
    }
}

Tensor MMRecModel::forward(Tensor input_ids) {  // Changed: pass by value
    // input_ids: [batch, seq]
    std::cout << "[DEBUG forward] START" << std::endl;
    std::cout << "[DEBUG forward] input_ids.ndim() = " << input_ids.ndim() << std::endl;
    std::cout << "[DEBUG forward] input_ids.shape_.size() = " << input_ids.sizes().size() << std::endl;
    
    int64_t batch = input_ids.size(0);
    int64_t seq = input_ids.size(1);
    
    // Initialize memory if needed
    if (memory_states_[0].numel() == 0 || 
        memory_states_[0].size(0) != batch) {
        reset_memory(batch);
    }
    
    // Embed tokens
    Tensor x = embed(input_ids);
    
    // Collect logits from all layers (UBOO!)
    std::vector<Tensor> all_layer_logits;
    
    // Forward through each layer
    for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
        auto [hidden, new_memory, logits] = blocks_[layer]->forward(
            x,
            memory_states_[layer]
        );
        
        // Update layer state (Bug #2: per-layer isolation)
        memory_states_[layer] = new_memory;
        
        // Next layer input
        x = hidden;
        
        // Collect logits for UBOO loss
        all_layer_logits.push_back(logits);
    }
    
    // Stack all layer logits: [num_layers, batch, seq, vocab]
    Tensor stacked_logits = Tensor::zeros({
        config_.num_layers,
        batch,
        seq,
        config_.vocab_size
    });
    
    for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
        const Tensor& layer_logits = all_layer_logits[layer];
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq; ++s) {
                for (int64_t v = 0; v < config_.vocab_size; ++v) {
                    int64_t idx = layer * batch * seq * config_.vocab_size +
                                  b * seq * config_.vocab_size +
                                  s * config_.vocab_size +
                                  v;
                    int64_t src_idx = b * seq * config_.vocab_size +
                                     s * config_.vocab_size +
                                     v;
                    stacked_logits.data()[idx] = layer_logits.data()[src_idx];
                }
            }
        }
    }
    
    // Debug: Check shape before return
    std::cout << "[DEBUG] stacked_logits.ndim() = " << stacked_logits.ndim() << std::endl;
    std::cout << "[DEBUG] stacked_logits.numel() = " << stacked_logits.numel() << std::endl;
    
    return stacked_logits;
}

Tensor MMRecModel::generate(const Tensor& input_ids) {
    // Inference mode: only return final layer logits
    Tensor all_logits = forward(input_ids);
    
    // Extract final layer: [batch, seq, vocab]
    int64_t batch = input_ids.size(0);
    int64_t seq = input_ids.size(1);
    
    Tensor final_logits = Tensor::zeros({batch, seq, config_.vocab_size});
    
    int64_t final_layer = config_.num_layers - 1;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t v = 0; v < config_.vocab_size; ++v) {
                int64_t src_idx = final_layer * batch * seq * config_.vocab_size +
                                  b * seq * config_.vocab_size +
                                  s * config_.vocab_size +
                                  v;
                int64_t dst_idx = b * seq * config_.vocab_size +
                                  s * config_.vocab_size +
                                  v;
                final_logits.data()[dst_idx] = all_logits.data()[src_idx];
            }
        }
    }
    
    return final_logits;
}

} // namespace mm_rec
