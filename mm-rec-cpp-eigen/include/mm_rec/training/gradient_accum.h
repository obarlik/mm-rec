/**
 * Gradient Accumulation Helper
 */

#pragma once

#include "mm_rec/training/gradients.h"

namespace mm_rec {

/**
 * Add gradients from src to dst (in-place)
 */
inline void accumulate_gradients(ModelGradients& dst, const ModelGradients& src) {
    // Embedding gradients
    for (int64_t i = 0; i < dst.embedding_grads.numel(); ++i) {
        dst.embedding_grads.data()[i] += src.embedding_grads.data()[i];
    }
    
    // Block gradients
    for (size_t block_idx = 0; block_idx < dst.block_grads.size(); ++block_idx) {
        auto& dst_block = dst.block_grads[block_idx];
        const auto& src_block = src.block_grads[block_idx];
        
        // GRU gradients
        for (int64_t i = 0; i < dst_block.gru_grads.dW_r.numel(); ++i) {
            dst_block.gru_grads.dW_r.data()[i] += src_block.gru_grads.dW_r.data()[i];
        }
        for (int64_t i = 0; i < dst_block.gru_grads.dU_r.numel(); ++i) {
            dst_block.gru_grads.dU_r.data()[i] += src_block.gru_grads.dU_r.data()[i];
        }
        for (int64_t i = 0; i < dst_block.gru_grads.db_r.numel(); ++i) {
            dst_block.gru_grads.db_r.data()[i] += src_block.gru_grads.db_r.data()[i];
        }
        
        for (int64_t i = 0; i < dst_block.gru_grads.dW_u.numel(); ++i) {
            dst_block.gru_grads.dW_u.data()[i] += src_block.gru_grads.dW_u.data()[i];
        }
        for (int64_t i = 0; i < dst_block.gru_grads.dU_u.numel(); ++i) {
            dst_block.gru_grads.dU_u.data()[i] += src_block.gru_grads.dU_u.data()[i];
        }
        for (int64_t i = 0; i < dst_block.gru_grads.db_u.numel(); ++i) {
            dst_block.gru_grads.db_u.data()[i] += src_block.gru_grads.db_u.data()[i];
        }
        
        for (int64_t i = 0; i < dst_block.gru_grads.dW_h.numel(); ++i) {
            dst_block.gru_grads.dW_h.data()[i] += src_block.gru_grads.dW_h.data()[i];
        }
        for (int64_t i = 0; i < dst_block.gru_grads.dU_h.numel(); ++i) {
            dst_block.gru_grads.dU_h.data()[i] += src_block.gru_grads.dU_h.data()[i];
        }
        for (int64_t i = 0; i < dst_block.gru_grads.db_h.numel(); ++i) {
            dst_block.gru_grads.db_h.data()[i] += src_block.gru_grads.db_h.data()[i];
        }
        
        // MoE gradients
        for (int64_t i = 0; i < dst_block.moe_grads.d_gate.numel(); ++i) {
            dst_block.moe_grads.d_gate.data()[i] += src_block.moe_grads.d_gate.data()[i];
        }
        
        for (size_t e = 0; e < dst_block.moe_grads.d_expert_up_weights.size(); ++e) {
            for (int64_t i = 0; i < dst_block.moe_grads.d_expert_up_weights[e].numel(); ++i) {
                dst_block.moe_grads.d_expert_up_weights[e].data()[i] += src_block.moe_grads.d_expert_up_weights[e].data()[i];
            }
        }
        
        for (size_t e = 0; e < dst_block.moe_grads.d_expert_down_weights.size(); ++e) {
            for (int64_t i = 0; i < dst_block.moe_grads.d_expert_down_weights[e].numel(); ++i) {
                dst_block.moe_grads.d_expert_down_weights[e].data()[i] += src_block.moe_grads.d_expert_down_weights[e].data()[i];
            }
        }
        
        // Output projection
        for (int64_t i = 0; i < dst_block.output_proj_grads.dW.numel(); ++i) {
            dst_block.output_proj_grads.dW.data()[i] += src_block.output_proj_grads.dW.data()[i];
        }
        for (int64_t i = 0; i < dst_block.output_proj_grads.db.numel(); ++i) {
            dst_block.output_proj_grads.db.data()[i] += src_block.output_proj_grads.db.data()[i];
        }
    }
}

/**
 * Scale gradients by a constant factor
 */
inline void scale_gradients(ModelGradients& grads, float scale) {
    // Embedding
    for (int64_t i = 0; i < grads.embedding_grads.numel(); ++i) {
        grads.embedding_grads.data()[i] *= scale;
    }
    
    // Blocks
    for (auto& block_grad : grads.block_grads) {
        // GRU
        for (int64_t i = 0; i < block_grad.gru_grads.dW_r.numel(); ++i) block_grad.gru_grads.dW_r.data()[i] *= scale;
        for (int64_t i = 0; i < block_grad.gru_grads.dU_r.numel(); ++i) block_grad.gru_grads.dU_r.data()[i] *= scale;
        for (int64_t i = 0; i < block_grad.gru_grads.db_r.numel(); ++i) block_grad.gru_grads.db_r.data()[i] *= scale;
        for (int64_t i = 0; i < block_grad.gru_grads.dW_u.numel(); ++i) block_grad.gru_grads.dW_u.data()[i] *= scale;
        for (int64_t i = 0; i < block_grad.gru_grads.dU_u.numel(); ++i) block_grad.gru_grads.dU_u.data()[i] *= scale;
        for (int64_t i = 0; i < block_grad.gru_grads.db_u.numel(); ++i) block_grad.gru_grads.db_u.data()[i] *= scale;
        for (int64_t i = 0; i < block_grad.gru_grads.dW_h.numel(); ++i) block_grad.gru_grads.dW_h.data()[i] *= scale;
        for (int64_t i = 0; i < block_grad.gru_grads.dU_h.numel(); ++i) block_grad.gru_grads.dU_h.data()[i] *= scale;
        for (int64_t i = 0; i < block_grad.gru_grads.db_h.numel(); ++i) block_grad.gru_grads.db_h.data()[i] *= scale;
        
        // MoE
        for (int64_t i = 0; i < block_grad.moe_grads.d_gate.numel(); ++i) block_grad.moe_grads.d_gate.data()[i] *= scale;
        for (auto& w : block_grad.moe_grads.d_expert_up_weights) {
            for (int64_t i = 0; i < w.numel(); ++i) w.data()[i] *= scale;
        }
        for (auto& w : block_grad.moe_grads.d_expert_down_weights) {
            for (int64_t i = 0; i < w.numel(); ++i) w.data()[i] *= scale;
        }
        
        // Output proj
        for (int64_t i = 0; i < block_grad.output_proj_grads.dW.numel(); ++i) block_grad.output_proj_grads.dW.data()[i] *= scale;
        for (int64_t i = 0; i < block_grad.output_proj_grads.db.numel(); ++i) block_grad.output_proj_grads.db.data()[i] *= scale;
    }
}

} // namespace mm_rec
