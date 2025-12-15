/**
 * Full MM-Rec Model Implementation
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/training/forward_cache.h"
#include "mm_rec/training/backward.h"
#include <random>
#include <iostream>

namespace mm_rec {

MMRecModel::MMRecModel(const MMRecModelConfig& config)
    : config_(config) {
    // ... same constructor ...
    // Initialize embedding
    // Xavier initialization for embeddings
    float std = std::sqrt(2.0f / (config.vocab_size + config.hidden_dim));
    embedding_weights_ = Tensor::randn(
        {config.vocab_size, config.hidden_dim},
        0.0f,
        std
    );
    
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

// ... embed() and reset_memory() same ...
Tensor MMRecModel::embed(const Tensor& input_ids) {
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
    for (int64_t i = 0; i < config_.num_layers; ++i) {
        memory_states_[i] = Tensor::zeros({batch_size, config_.mem_dim});
    }
}

Tensor MMRecModel::forward(Tensor input_ids, ForwardCache* cache) {
    // input_ids: [batch, seq]
    
    int64_t batch = input_ids.size(0);
    int64_t seq = input_ids.size(1);
    
    if (cache) {
        cache->input_ids = input_ids;
        cache->init(config_.num_layers);
    }
    
    // Initialize memory if needed
    if (memory_states_[0].ndim() == 0 ||
        memory_states_[0].size(0) != batch) {
        reset_memory(batch);
    }
    
    // Embed tokens
    Tensor x = embed(input_ids);
    if (cache) cache->embedded = x;
    
    // Collect logits from all layers (UBOO!)
    std::vector<Tensor> all_layer_logits;
    
    // Forward through each layer
    for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
        BlockCache* block_cache = cache ? &cache->block_caches[layer] : nullptr;
        
        auto [hidden, new_memory, logits] = blocks_[layer]->forward(
            x,
            memory_states_[layer],
            block_cache
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
    
    if (cache) cache->all_logits = stacked_logits;
    
    return stacked_logits;
}

ModelGradients MMRecModel::backward(const Tensor& targets, const ForwardCache& cache) {
    // 1. Initialize gradients
    ModelGradients grads;
    grads.init(
        config_.num_layers,
        config_.vocab_size,
        config_.hidden_dim,
        config_.mem_dim,
        config_.ffn_dim
    );
    
    // 2. Compute UBOO Loss Gradients (d_logits for all layers)
    // cache.all_logits: [num_layers, batch, seq, vocab]
    // targets: [batch, seq]
    
    int64_t layers = config_.num_layers;
    int64_t batch = cache.all_logits.size(1);
    int64_t seq = cache.all_logits.size(2);
    int64_t vocab = config_.vocab_size;
    
    // Flatten logits to [layers * batch * seq, vocab]
    // Flatten targets to [layers * batch * seq] (repeated)
    // Actually, calculate per-layer for easier slicing
    
    Tensor d_all_logits = Tensor::zeros(cache.all_logits.sizes());
    
    for (int64_t l = 0; l < layers; ++l) {
        // Extract layer logits [batch, seq, vocab] -> reshape to [batch*seq, vocab]
        // Extract targets [batch, seq] -> reshape to [batch*seq]
        
        // Manual slicing/mapping for now (simplest without advanced Tensor views)
        Tensor layer_logits_flat = Tensor::zeros({batch * seq, vocab});
        Tensor layer_targets_flat = Tensor::zeros({batch * seq});
        
        // Copy data
        for (int64_t i = 0; i < batch * seq; ++i) {
            // Copy targets
            layer_targets_flat.data()[i] = targets.data()[i]; // targets is [batch, seq] continuous
            
            // Copy logits
            for (int64_t v = 0; v < vocab; ++v) {
                layer_logits_flat.data()[i * vocab + v] = 
                    cache.all_logits.data()[l * batch * seq * vocab + i * vocab + v];
            }
        }
        
        // Compute gradients [batch*seq, vocab]
        Tensor d_layer_logits = softmax_cross_entropy_backward(
            layer_logits_flat,
            layer_targets_flat
        );
        
        // Copy back to d_all_logits
        for (int64_t i = 0; i < batch * seq; ++i) {
            for (int64_t v = 0; v < vocab; ++v) {
                d_all_logits.data()[l * batch * seq * vocab + i * vocab + v] =
                    d_layer_logits.data()[i * vocab + v];
            }
        }
    }
    
    // 3. Backward Pass through Layers (Top to Bottom)
    Tensor d_output_from_above = Tensor::zeros({batch, seq, config_.hidden_dim});
    
    for (int64_t l = layers - 1; l >= 0; --l) {
        // Get d_logits for this layer [batch, seq, vocab]
        Tensor d_logits_layer = Tensor::zeros({batch, seq, vocab});
        for (int64_t i = 0; i < batch * seq * vocab; ++i) {
            d_logits_layer.data()[i] = d_all_logits.data()[l * batch * seq * vocab + i];
        }
        
        // No gradient from future memory (Assuming detached between batches)
        Tensor d_memory_from_future = Tensor::zeros({batch, config_.mem_dim});
        
        // Block Backward
        auto [dx, dmemory_init] = blocks_[l]->backward(
            d_output_from_above,
            d_memory_from_future,
            d_logits_layer,
            cache.block_caches[l],
            grads.block_grads[l]
        );
        
        // Update gradients flowing down
        d_output_from_above = dx; // Becomes input gradient for previous layer
    }
    
    // 4. Embedding Backward
    // d_output_from_above is now d_embeddings [batch, seq, hidden]
    // inputs: input_ids [batch, seq]
    
    // grads.embedding_grads initialized to zeros
    float* emb_grads = grads.embedding_grads.data();
    const float* d_emb = d_output_from_above.data();
    const float* ids = cache.input_ids.data();
    int64_t hidden = config_.hidden_dim;
    
    for (int64_t i = 0; i < batch * seq; ++i) {
        int64_t token_id = static_cast<int64_t>(ids[i]);
        // Accumulate gradients
        for (int64_t h = 0; h < hidden; ++h) {
            emb_grads[token_id * hidden + h] += d_emb[i * hidden + h];
        }
    }
    
    return grads;
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
