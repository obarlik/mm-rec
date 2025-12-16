/**
 * Gradient Clipping Utilities
 * Prevents gradient explosion by clipping gradients to max norm
 */

#pragma once

#include "mm_rec/training/gradients.h"
#include <cmath>

namespace mm_rec {

/**
 * Clip gradients by global norm
 * 
 * @param grads Model gradients to clip
 * @param max_norm Maximum allowed gradient norm
 * @return Actual gradient norm before clipping
 */
inline float clip_gradients_by_norm(ModelGradients& grads, float max_norm) {
    // Compute global gradient norm
    float total_norm_sq = 0.0f;
    
    // Embedding gradients
    for (int64_t i = 0; i < grads.embedding_grads.numel(); ++i) {
        float g = grads.embedding_grads.data()[i];
        total_norm_sq += g * g;
    }
    
    // Block gradients
    for (auto& block_grad : grads.block_grads) {
        // GRU gradients
        for (int64_t i = 0; i < block_grad.gru_grads.dW_u.numel(); ++i) {
            total_norm_sq += block_grad.gru_grads.dW_u.data()[i] * block_grad.gru_grads.dW_u.data()[i];
        }
        for (int64_t i = 0; i < block_grad.gru_grads.dW_r.numel(); ++i) {
            total_norm_sq += block_grad.gru_grads.dW_r.data()[i] * block_grad.gru_grads.dW_r.data()[i];
        }
        for (int64_t i = 0; i < block_grad.gru_grads.dW_h.numel(); ++i) {
            total_norm_sq += block_grad.gru_grads.dW_h.data()[i] * block_grad.gru_grads.dW_h.data()[i];
        }
        
        // MoE gradients
        for (int64_t i = 0; i < block_grad.moe_grads.d_gate.numel(); ++i) {
            total_norm_sq += block_grad.moe_grads.d_gate.data()[i] * block_grad.moe_grads.d_gate.data()[i];
        }
        for (auto& expert_grad : block_grad.moe_grads.d_expert_up_weights) {
            for (int64_t i = 0; i < expert_grad.numel(); ++i) {
                total_norm_sq += expert_grad.data()[i] * expert_grad.data()[i];
            }
        }
        for (auto& expert_grad : block_grad.moe_grads.d_expert_down_weights) {
            for (int64_t i = 0; i < expert_grad.numel(); ++i) {
                total_norm_sq += expert_grad.data()[i] * expert_grad.data()[i];
            }
        }
        
        // Output projection gradients
        for (int64_t i = 0; i < block_grad.output_proj_grads.dW.numel(); ++i) {
            total_norm_sq += block_grad.output_proj_grads.dW.data()[i] * block_grad.output_proj_grads.dW.data()[i];
        }
    }
    
    float total_norm = std::sqrt(total_norm_sq);
    
    // Clip if needed
    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);
        
        // Scale all gradients
        for (int64_t i = 0; i < grads.embedding_grads.numel(); ++i) {
            grads.embedding_grads.data()[i] *= clip_coef;
        }
        
        for (auto& block_grad : grads.block_grads) {
            // GRU gradients
            for (int64_t i = 0; i < block_grad.gru_grads.dW_u.numel(); ++i) {
                block_grad.gru_grads.dW_u.data()[i] *= clip_coef;
            }
            for (int64_t i = 0; i < block_grad.gru_grads.dW_r.numel(); ++i) {
                block_grad.gru_grads.dW_r.data()[i] *= clip_coef;
            }
            for (int64_t i = 0; i < block_grad.gru_grads.dW_h.numel(); ++i) {
                block_grad.gru_grads.dW_h.data()[i] *= clip_coef;
            }
            
            // MoE gradients
            for (int64_t i = 0; i < block_grad.moe_grads.d_gate.numel(); ++i) {
                block_grad.moe_grads.d_gate.data()[i] *= clip_coef;
            }
            for (auto& expert_grad : block_grad.moe_grads.d_expert_up_weights) {
                for (int64_t i = 0; i < expert_grad.numel(); ++i) {
                    expert_grad.data()[i] *= clip_coef;
                }
            }
            for (auto& expert_grad : block_grad.moe_grads.d_expert_down_weights) {
                for (int64_t i = 0; i < expert_grad.numel(); ++i) {
                    expert_grad.data()[i] *= clip_coef;
                }
            }
            
            // Output projection
            for (int64_t i = 0; i < block_grad.output_proj_grads.dW.numel(); ++i) {
                block_grad.output_proj_grads.dW.data()[i] *= clip_coef;
            }
        }
    }
    
    return total_norm;
}

} // namespace mm_rec
