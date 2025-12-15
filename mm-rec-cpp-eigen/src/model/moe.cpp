/**
 * Mixture of Experts (MoE) Implementation
 */

#include "mm_rec/model/moe.h"
#include "mm_rec/training/backward.h" // For linear_backward, relu_backward
#include "mm_rec/training/optimizer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <utility>

namespace mm_rec {

void MoEGradients::init(const MoEConfig& config) {
    d_gate = Tensor::zeros({config.hidden_dim, config.num_experts});
    db_gate = Tensor::zeros({1, config.num_experts}); // Broadcastable bias
    
    d_expert_up_weights.resize(config.num_experts);
    d_expert_up_biases.resize(config.num_experts);
    d_expert_down_weights.resize(config.num_experts);
    d_expert_down_biases.resize(config.num_experts);
    
    for (int i = 0; i < config.num_experts; ++i) {
        d_expert_up_weights[i] = Tensor::zeros({config.hidden_dim, config.ffn_dim});
        d_expert_up_biases[i] = Tensor::zeros({1, config.ffn_dim});
        d_expert_down_weights[i] = Tensor::zeros({config.ffn_dim, config.hidden_dim});
        d_expert_down_biases[i] = Tensor::zeros({1, config.hidden_dim});
    }
}

void MoEGradients::zero() {
    d_gate.zero_();
    db_gate.zero_();
    for (auto& t : d_expert_up_weights) t.zero_();
    for (auto& t : d_expert_up_biases) t.zero_();
    for (auto& t : d_expert_down_weights) t.zero_();
    for (auto& t : d_expert_down_biases) t.zero_();
}

MoELayer::MoELayer(const MoEConfig& config) : config_(config) {
    // Initialize Gate (Router)
    gate_ = std::make_unique<Linear>(config.hidden_dim, config.num_experts);
    
    // Initialize Experts
    for (int i = 0; i < config.num_experts; ++i) {
        auto up = std::make_unique<Linear>(config.hidden_dim, config.ffn_dim);
        auto down = std::make_unique<Linear>(config.ffn_dim, config.hidden_dim);
        experts_up_.push_back(std::move(up));
        experts_down_.push_back(std::move(down));
    }
}

// Helper: Softmax on last dimension
static void simple_softmax(std::vector<float>& vals) {
    float max_val = -1e9;
    for (float v : vals) max_val = std::max(max_val, v);
    
    float sum_exp = 0.0f;
    for (float& v : vals) {
        v = std::exp(v - max_val);
        sum_exp += v;
    }
    
    for (float& v : vals) {
        v /= sum_exp;
    }
}

Tensor MoELayer::forward(const Tensor& x, MoECache* cache) {
    int64_t batch = x.size(0);
    int64_t seq = x.size(1);
    int64_t hidden = x.size(2);
    int64_t num_experts = config_.num_experts;
    int64_t top_k = config_.top_k;
    
    // 1. Compute Router Logits: [batch*seq, experts]
    // Flatten input for simpler processing
    Tensor x_flat = x.reshape({batch * seq, hidden});
    Tensor logits = gate_->forward(x_flat); // [N, experts]
    
    // Output tensor
    Tensor output = Tensor::zeros({batch * seq, hidden});
    
    // Caching structures
    Tensor saved_logits, saved_weights, saved_indices;
    if (cache) {
        saved_logits = logits; // Copy
        saved_weights = Tensor::zeros({batch * seq, top_k});
        saved_indices = Tensor::zeros({batch * seq, top_k});
    }
    
    // 2. Select Experts & Route (CPU-based Dynamic Routing)
    // Iterate over each token
    for (int64_t i = 0; i < batch * seq; ++i) {
        // Extract logits for token i
        std::vector<std::pair<float, int>> token_logits(num_experts);
        float* row_ptr = logits.data() + i * num_experts;
        
        // Prepare for Softmax
        std::vector<float> soft_vals(num_experts);
        for(int e=0; e<num_experts; ++e) {
            token_logits[e] = {row_ptr[e], e};
            soft_vals[e] = row_ptr[e];
        }
        
        // Calculate softmax weights (for gradient scaling)
        simple_softmax(soft_vals);
        
        // Top-K Selection
        // Sort descending by logit
        std::partial_sort(
            token_logits.begin(),
            token_logits.begin() + top_k,
            token_logits.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
        
        // Process selected experts
        for (int k = 0; k < top_k; ++k) {
            int expert_idx = token_logits[k].second;
            float weight = soft_vals[expert_idx]; // Use softmax probability as weight
            
            if (cache) {
                saved_indices.data()[i * top_k + k] = (float)expert_idx;
                saved_weights.data()[i * top_k + k] = weight;
            }
            
            // Extract single token input [1, hidden]
            // We use a simplified localized forward here for the single token.
            // This is "Dynamic Routing" - essentially iterating.
            // For production speed, we would batch tokens per expert.
            // But for correctness/MVP, this per-token loop is fine.
            
            // Note: Our Linear layer expects batched input usually, but works on [1, H] too.
            // But `Linear::forward` creates new tensors.
            // To be efficient, we manually compute dense layer for this token and this expert.
            
            // Manual Expert Forward:
            // h_mid = Relu( x[i] @ W_up + b_up )
            // h_out = h_mid @ W_down + b_down
            // output[i] += weight * h_out
            
            // Pointers
            const float* w_up = experts_up_[expert_idx]->weight().data();
            const float* b_up = experts_up_[expert_idx]->bias().data();
            const float* w_down = experts_down_[expert_idx]->weight().data();
            const float* b_down = experts_down_[expert_idx]->bias().data();
            const float* in_ptr = x_flat.data() + i * hidden;
            
            // Up Projection
            int64_t ffn_dim = config_.ffn_dim;
            std::vector<float> h_mid(ffn_dim);
            
            for(int d=0; d<ffn_dim; ++d) {
                float sum = b_up[d];
                for(int h=0; h<hidden; ++h) {
                    sum += in_ptr[h] * w_up[h * ffn_dim + d]; // Transposed? Linear stores [in, out]
                }
                // ReLU
                h_mid[d] = sum > 0.0f ? sum : 0.0f;
            }
            
            // Down Projection & Accumulate
            float* out_ptr = output.data() + i * hidden;
            for(int h=0; h<hidden; ++h) {
                float sum = b_down[h];
                for(int d=0; d<ffn_dim; ++d) {
                    sum += h_mid[d] * w_down[d * hidden + h];
                }
                out_ptr[h] += weight * sum;
            }
        }
    }
    
    if (cache) {
        cache->router_logits = saved_logits;
        cache->routing_weights = saved_weights;
        cache->selected_indices = saved_indices;
    }
    
    return output.reshape({batch, seq, hidden});
}

// Check helper
static bool is_expert_active(int expert_idx, const float* indices, int top_k) {
    for(int k=0; k<top_k; ++k) {
        if ((int)indices[k] == expert_idx) return true;
    }
    return false;
}

Tensor MoELayer::backward(
    const Tensor& d_output,
    const Tensor& x,
    const MoECache& cache,
    MoEGradients& grads
) {
    int64_t batch = x.size(0);
    int64_t seq = x.size(1);
    int64_t hidden = x.size(2);
    int64_t top_k = config_.top_k;
    
    Tensor x_flat = x.reshape({batch*seq, hidden});
    Tensor do_flat = d_output.reshape({batch*seq, hidden});
    Tensor dx = Tensor::zeros(x_flat.sizes());
    
    // For Router Gradients
    // dL/dLogits = ... complicated because of TopK + Softmax
    // Simplified approximation: Gradient flows through the Softmax weights of SELECTED experts.
    // Ideally we need full Jacobian of sparse softmax.
    // For MVP: We assume gradient flows through the weighting term `weight * Expert(x)`.
    
    // We accumulate dlogits here
    Tensor dlogits = Tensor::zeros(cache.router_logits.sizes());
    
    // Iterate tokens
    for(int64_t i=0; i<batch*seq; ++i) {
        const float* idx_ptr = cache.selected_indices.data() + i * top_k;
        const float* w_ptr = cache.routing_weights.data() + i * top_k;
        
        // For each selected expert
        for(int k=0; k<top_k; ++k) {
            int expert_idx = (int)idx_ptr[k];
            float weight = w_ptr[k];
            
            // Recompute Expert Forward (needed for gradient)
            // Ideally we should cache this too, but for memory we recompute (Checkpointing style).
            // Pointers
            const float* w_up = experts_up_[expert_idx]->weight().data();
            const float* b_up = experts_up_[expert_idx]->bias().data();
            const float* w_down = experts_down_[expert_idx]->weight().data();
            const float* b_down = experts_down_[expert_idx]->bias().data();
            const float* in_ptr = x_flat.data() + i * hidden;
            
            int64_t ffn_dim = config_.ffn_dim;
            std::vector<float> h_mid(ffn_dim);
            std::vector<float> h_out(hidden);
            
            // Forward Recompute
            for(int d=0; d<ffn_dim; ++d) {
                float sum = b_up[d];
                for(int h=0; h<hidden; ++h) sum += in_ptr[h] * w_up[h * ffn_dim + d];
                h_mid[d] = sum > 0.0f ? sum : 0.0f; // ReLU
            }
            for(int h=0; h<hidden; ++h) {
                float sum = b_down[h];
                for(int d=0; d<ffn_dim; ++d) sum += h_mid[d] * w_down[d * hidden + h];
                h_out[h] = sum;
            }
            
            // Backward Flow
            // 1. Through weighting: d_output * weight -> d_expert_out
            // 2. Through expert: d_expert_out -> d_expert_params + d_input_expert
            // 3. Through weight: d_output * expert_out -> d_weight -> d_logits
            
            const float* do_ptr = do_flat.data() + i * hidden;
            
            // A. Gradient w.r.t Weight (for Router)
            float d_weight = 0.0f;
            for(int h=0; h<hidden; ++h) {
                d_weight += do_ptr[h] * h_out[h];
            }
            
            // Map d_weight back to logits (simplified: d_logit_j = d_weight_j * w_j * (1 - w_j))
            // This serves as an approximation of Softmax backward on the selected indices.
            dlogits.data()[i * config_.num_experts + expert_idx] += d_weight; // Raw gradient to weight
            // Proper softmax backward is complex with sparse selection. 
            // We use the proxy that dL/dLogit ~= weight * (1-weight) * dL/dWeight for current class
            // This is "Straight Through"ish.
            
            // B. Gradient w.r.t Expert 
            // d_expert_out = d_output * weight
            std::vector<float> d_h_out(hidden);
            for(int h=0; h<hidden; ++h) d_h_out[h] = do_ptr[h] * weight;
            
            // Down Layer Backprop
            // d_h_mid = d_h_out @ W_down.T
            std::vector<float> d_h_mid(ffn_dim);
            float* dw_down_ptr = grads.d_expert_down_weights[expert_idx].data();
            float* db_down_ptr = grads.d_expert_down_biases[expert_idx].data();
            
            for(int d=0; d<ffn_dim; ++d) {
                float sum = 0.0f;
                for(int h=0; h<hidden; ++h) {
                    sum += d_h_out[h] * w_down[d * hidden + h];
                    // Accumulate Gradients for Weights/Bias
                    dw_down_ptr[d * hidden + h] += d_h_out[h] * h_mid[d];
                }
                d_h_mid[d] = sum;
                // Bias grad? No, bias is sum of d_h_out over batch.
                // We do it per token:
            }
            for(int h=0; h<hidden; ++h) db_down_ptr[h] += d_h_out[h];
            
            // ReLU Backward
            for(int d=0; d<ffn_dim; ++d) {
                if(h_mid[d] <= 0.0f) d_h_mid[d] = 0.0f;
            }
            
            // Up Layer Backprop
            // d_in = d_h_mid @ W_up.T
            float* dw_up_ptr = grads.d_expert_up_weights[expert_idx].data();
            float* db_up_ptr = grads.d_expert_up_biases[expert_idx].data();
            float* dx_ptr = dx.data() + i * hidden;
            
            for(int h=0; h<hidden; ++h) {
                float sum = 0.0f;
                for(int d=0; d<ffn_dim; ++d) {
                    sum += d_h_mid[d] * w_up[h * ffn_dim + d];
                    dw_up_ptr[h * ffn_dim + d] += d_h_mid[d] * in_ptr[h];
                }
                dx_ptr[h] += sum;
            }
            for(int d=0; d<ffn_dim; ++d) db_up_ptr[d] += d_h_mid[d];
        }
    }
    
    // Propagate Router Gradients
    // dlogits has partial gradients.
    // Backprop through Gate Linear: dlogits @ W_gate.T -> dx_gate
    // Also accumulate dW_gate, db_gate
    
    // We treat 'dlogits' as dL/dZ (pre-activation)? 
    // We accumulated derivatives w.r.t probability weights roughly.
    // Let's assume we treat the router output as a linear layer + softmax.
    // Simpler: Just backprop dlogits through linear gate.
    
    // d_gate_weights += x.T @ dlogits
    // dx += dlogits @ W_gate.T
    
    // Manual Matrix Mul for Grad Gate (Batch accumulation)
    // d_gate = x_flat.T @ dlogits
    for(int i=0; i<batch*seq; ++i) {
        const float* in_ptr = x_flat.data() + i * hidden;
        const float* dl_ptr = dlogits.data() + i * config_.num_experts;
        
        for(int h=0; h<hidden; ++h) {
            for(int e=0; e<config_.num_experts; ++e) {
                grads.d_gate.data()[h * config_.num_experts + e] += in_ptr[h] * dl_ptr[e];
            }
            
            // Input grad addition from gate
            const float* wg_ptr = gate_->weight().data();
            for(int e=0; e<config_.num_experts; ++e) {
                dx.data()[i * hidden + h] += dl_ptr[e] * wg_ptr[h * config_.num_experts + e];
            }
        }
        for(int e=0; e<config_.num_experts; ++e) grads.db_gate.data()[e] += dl_ptr[e];
    }
    
    return dx.reshape(x.sizes());
}

void MoELayer::update_parameters(SGD& optimizer, const MoEGradients& grads) {
    // 1. Update Gate
    optimizer.step(gate_->weight(), grads.d_gate);
    optimizer.step(gate_->bias(), grads.db_gate);
    
    // 2. Update Experts
    for(int i=0; i<config_.num_experts; ++i) {
        optimizer.step(experts_up_[i]->weight(), grads.d_expert_up_weights[i]);
        optimizer.step(experts_up_[i]->bias(), grads.d_expert_up_biases[i]);
        optimizer.step(experts_down_[i]->weight(), grads.d_expert_down_weights[i]);
        optimizer.step(experts_down_[i]->bias(), grads.d_expert_down_biases[i]);
    }
}

} // namespace mm_rec
