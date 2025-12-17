/**
 * Gradient Structures
 * 
 * Stores gradients for all model parameters
 */

#pragma once

#include "mm_rec/core/tensor.h"
#include "mm_rec/model/moe.h"
#include "mm_rec/core/normalization.h"
#include <vector>

namespace mm_rec {

/**
 * Gradients for a single Linear layer
 */
struct LinearGradients {
    Tensor dW;  // Weight gradient [out_dim, in_dim]
    Tensor db;  // Bias gradient [out_dim]
    
    LinearGradients() = default;
    LinearGradients(const std::vector<int64_t>& w_shape, const std::vector<int64_t>& b_shape)
        : dW(Tensor::zeros(w_shape)), db(Tensor::zeros(b_shape)) {}
        
    LinearGradients clone() const {
        LinearGradients cp;
        cp.dW = dW.clone();
        cp.db = db.clone();
        return cp;
    }

    void zero() {
        dW.zero_();
        db.zero_();
    }
};

/**
 * Gradients for GRU gates
 */
struct GRUGradients {
    // Reset gate
    Tensor dW_r, dU_r, db_r;
    // Update gate
    Tensor dW_u, dU_u, db_u;
    // Candidate
    Tensor dW_h, dU_h, db_h;
    
    GRUGradients() = default;
    
    // Initialize with zeros
    void init(int64_t hidden_dim, int64_t in_dim) {
        dW_r = Tensor::zeros({hidden_dim, in_dim});
        dU_r = Tensor::zeros({hidden_dim, hidden_dim});
        db_r = Tensor::zeros({hidden_dim});
        
        dW_u = Tensor::zeros({hidden_dim, in_dim});
        dU_u = Tensor::zeros({hidden_dim, hidden_dim});
        db_u = Tensor::zeros({hidden_dim});
        
        dW_h = Tensor::zeros({hidden_dim, in_dim});
        dU_h = Tensor::zeros({hidden_dim, hidden_dim});
        db_h = Tensor::zeros({hidden_dim});
    }
    
    GRUGradients clone() const {
        GRUGradients cp;
        cp.dW_r = dW_r.clone(); cp.dU_r = dU_r.clone(); cp.db_r = db_r.clone();
        cp.dW_u = dW_u.clone(); cp.dU_u = dU_u.clone(); cp.db_u = db_u.clone();
        cp.dW_h = dW_h.clone(); cp.dU_h = dU_h.clone(); cp.db_h = db_h.clone();
        return cp;
    }

    void zero() {
        dW_r.zero_(); dU_r.zero_(); db_r.zero_();
        dW_u.zero_(); dU_u.zero_(); db_u.zero_();
        dW_h.zero_(); dU_h.zero_(); db_h.zero_();
    }
};

/**
 * Gradients for a full MMRecBlock
 */
struct BlockGradients {
    GRUGradients gru_grads;
    
    // MoE (replaces standard FFN)
    MoEGradients moe_grads;
    
#include "mm_rec/model/moe.h"
#include "mm_rec/core/normalization.h"
#include <vector>

// ... (in BlockGradients)

    // Output Projector (UBOO)
    LinearGradients output_proj_grads;
    
    // Norm Grads
    RMSNormGradients norm_grads;
    
    // ...
    
    void init(int64_t hidden_dim, int64_t mem_dim, int64_t ffn_dim, int64_t vocab_size, int64_t num_experts) {
        gru_grads.init(mem_dim, hidden_dim);
        
        MoEConfig moe_config;
        moe_config.hidden_dim = hidden_dim;
        moe_config.ffn_dim = ffn_dim;
        moe_config.num_experts = num_experts;
        moe_config.top_k = 1; // Not needed
        
        moe_grads.init(moe_config);
        
        output_proj_grads = LinearGradients({vocab_size, hidden_dim}, {vocab_size});
        
        norm_grads.d_weight = Tensor::zeros({hidden_dim});
    }
    
    BlockGradients clone() const {
        BlockGradients cp;
        cp.gru_grads = gru_grads.clone();
        cp.moe_grads = moe_grads.clone();
        cp.output_proj_grads = output_proj_grads.clone();
        cp.norm_grads.d_weight = norm_grads.d_weight.clone();
        return cp;
    }

    void zero() {
        gru_grads.zero();
        moe_grads.zero();
        output_proj_grads.zero();
        norm_grads.d_weight.zero_();
    }
};

/**
 * Complete model gradients
 */
struct ModelGradients {
    Tensor embedding_grads;
    std::vector<BlockGradients> block_grads;
    
    // Updated init with MoE support
    void init(int64_t num_layers, int64_t vocab, int64_t hidden, int64_t mem, int64_t ffn, int64_t num_experts) {
        embedding_grads = Tensor::zeros({vocab, hidden});
        block_grads.resize(num_layers);
        for (auto& bg : block_grads) {
            bg.init(hidden, mem, ffn, vocab, num_experts);
        }
    }

    // Shallow copy default, but we need Deep Copy helper
    ModelGradients clone() const {
        ModelGradients cp;
        cp.embedding_grads = embedding_grads.clone();
        cp.block_grads.reserve(block_grads.size());
        for (const auto& bg : block_grads) {
            cp.block_grads.push_back(bg.clone());
        }
        return cp;
    }

    void zero() {
        embedding_grads.zero_();
        for (auto& bg : block_grads) bg.zero();
    }
};

} // namespace mm_rec
