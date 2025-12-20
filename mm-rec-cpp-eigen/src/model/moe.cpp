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
    
    // std::cout << "[MoE] Init Gradients for " << config.num_experts << " experts." << std::endl;
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
    int64_t total_tokens = batch * seq; // N
    
    // 1. Router (GPU Accelerated)
    Tensor x_flat = x.reshape({total_tokens, hidden});
    Tensor logits = gate_->forward(x_flat); // [N, experts]
    
    Tensor output = Tensor::zeros({total_tokens, hidden});
    
    // Caching
    if (cache) {
        cache->router_logits = logits; // Deep copy or move? Tensor is shared_ptr data usually or copy. 
        // Based on Tensor.h, it's pointer based copy? No, it has operator= deep copy if not careful.
        // Tensor class in this codebase has a pointer but copy constructor does shallow or deep?
        // Checking header: 'data_ptr_' is raw pointer. Copy constructor does DEEP copy?
        // Wait, if it does deep copy, this is slow. 
        // Assuming light copy or we optimize later.
    }

    // --- 2. GATHER (CPU) ---
    // Prepare batches for each expert
    // expert_inputs[e] -> vector of floats (features)
    // expert_indices[e] -> vector of global token indices to scatter back
    // expert_weights[e] -> vector of routing weights for scaling
    
    std::vector<std::vector<float>> expert_inputs(num_experts);
    std::vector<std::vector<int64_t>> expert_indices(num_experts);
    std::vector<std::vector<float>> expert_weights(num_experts);
    
    // Pre-reserve to avoid reallocs (Heuristic: Uniform distribution)
    int64_t estimated_capacity = (total_tokens * top_k) / num_experts * 1.2;
    for(int e=0; e<num_experts; ++e) {
        expert_inputs[e].reserve(estimated_capacity * hidden);
        expert_indices[e].reserve(estimated_capacity);
        expert_weights[e].reserve(estimated_capacity);
    }

    const float* logits_ptr = logits.data();
    const float* x_ptr = x_flat.data();

    // Cache structure optimization
    std::vector<float> saved_indices_vec(total_tokens * top_k);
    std::vector<float> saved_weights_vec(total_tokens * top_k);

    // Parallelize processing if N is large? 
    // Gathering is tricky to parallelize due to push_back. 
    // Keep single threaded for safety, memory bandwidth is bottleneck anyway.
    
    for (int64_t i = 0; i < total_tokens; ++i) {
        // Softmax & TopK for token i
        const float* row = logits_ptr + i * num_experts;
        
        // Softmax
        float max_val = -1e9;
        for(int e=0; e<num_experts; ++e) if(row[e] > max_val) max_val = row[e];
        
        // Small vector for probs
        // Can utilize stack array since experts usually < 64
        float probs[64]; // Hard limit or dynamic? config says experts=4
        // Use std::vector fallback if needed but static is faster
        std::vector<float> probs_dyn;
        float* p_ptr = (num_experts <= 64) ? probs : (probs_dyn.resize(num_experts), probs_dyn.data());
        
        float sum_exp = 0.0f;
        for(int e=0; e<num_experts; ++e) {
            p_ptr[e] = std::exp(row[e] - max_val);
            sum_exp += p_ptr[e];
        }
        float inv_sum = 1.0f / sum_exp;
        for(int e=0; e<num_experts; ++e) p_ptr[e] *= inv_sum;
        
        // TopK Selection
        // Pair: (prob, index)
        std::pair<float, int> top_candidates[64]; 
        for(int e=0; e<num_experts; ++e) top_candidates[e] = {p_ptr[e], e};
        
        std::partial_sort(
            top_candidates, 
            top_candidates + top_k, 
            top_candidates + num_experts, 
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
        
        // Assign to Experts
        const float* token_data = x_ptr + i * hidden;
        
        for(int k=0; k<top_k; ++k) {
            int expert_idx = top_candidates[k].second;
            float weight = top_candidates[k].first;
            
            // Gather Input
            auto& input_buf = expert_inputs[expert_idx];
            // Unrolling memcpy or loop
            // input_buf.insert(input_buf.end(), token_data, token_data + hidden);
            // Manual push_back might be slow? memcpy to resized vector is better?
            // Let's rely on insert for now, optimize if needed.
            // Actually, bulk insert is better.
            size_t current_size = input_buf.size();
            input_buf.resize(current_size + hidden);
            std::memcpy(input_buf.data() + current_size, token_data, hidden * sizeof(float));
            
            // Store Metadata
            expert_indices[expert_idx].push_back(i);
            expert_weights[expert_idx].push_back(weight);
            
            // Save for Cache/Backward
            if(cache) {
                saved_indices_vec[i * top_k + k] = (float)expert_idx;
                saved_weights_vec[i * top_k + k] = weight;
            }
        }
    }
    
    if(cache) {
        cache->selected_indices = Tensor::from_data(saved_indices_vec, {total_tokens, top_k});
        cache->routing_weights = Tensor::from_data(saved_weights_vec, {total_tokens, top_k});
    }

    // --- 3. EXECUTE (GPU Parallellism) ---
    // Launch experts async if possible?
    // Current Linear is blocking/hybrid. Sequential expert launch is fine, 
    // GPU queue will handle pipelining naturally.
    
    for (int e = 0; e < num_experts; ++e) {
        if (expert_indices[e].empty()) continue;
        
        int64_t num_tokens_e = expert_indices[e].size();
        
        // Create Input Tensor [Be, Hidden]
        // This copies from vector to new Tensor buffer. Zero-copy possible?
        // Tensor class owns data. So copy is needed. Memory bandwidth cost.
        Tensor x_e = Tensor::from_data(expert_inputs[e], {num_tokens_e, hidden});
        
        // Forward Pass (Features -> FFN -> Features)
        // Up: [Be, H] -> [Be, FFN]
        Tensor h = experts_up_[e]->forward(x_e);
        
        // ReLU
        h = h.relu(); 
        
        // Down: [Be, FFN] -> [Be, H]
        Tensor out_e = experts_down_[e]->forward(h);
        
        // --- 4. SCATTER (CPU) ---
        // Add results back to Global Output
        const float* out_e_ptr = out_e.data();
        float* global_out_ptr = output.data();
        const auto& indices = expert_indices[e];
        const auto& weights = expert_weights[e];
        
        // This loop adds: output[global_idx] += out_e[local_idx] * weight
        // Parallelize? This writes to random locations.
        // If experts are processed sequentially, race conditions are only if TopK > 1 selects same token twice?
        // A token CANNOT go to the same expert twice.
        // Can multiple threads write to 'output[i]'? 
        // Yes, if token 'i' went to Expert A and Expert B.
        // We are processing Experts sequentially (e=0, e=1...).
        // So this loop is safe to parallelize? No, different 'e' loops collide on 'i'.
        // But inside THIS loop (for one 'e'), each 'i' is unique! 
        // Expert A only sees token 'i' once.
        // So we can parallelize THIS loop!
        
        #pragma omp parallel for
        for(int64_t j=0; j<num_tokens_e; ++j) {
            int64_t global_idx = indices[j];
            float w = weights[j];
            
            const float* src = out_e_ptr + j * hidden;
            float* dst = global_out_ptr + global_idx * hidden;
            
            // Vectorized Add
            for(int h=0; h<hidden; ++h) {
                // dst[h] += src[h] * w;
                // Atomic needed? No. Only one thread handles 'global_idx' FOR THIS EXPERT.
                // Since experts run sequentially, no other expert is writing to 'global_idx' NOW.
                // So this is safe.
                dst[h] += src[h] * w;
            }
        }
    }
    
    // Aux Loss Calculation (Simplified for performance)
    if (cache) {
        // Reuse pre-computed densities if we cared to store them.
        // For now, re-implementing fast approx aux loss or skipping to save compute?
        // Let's implement minimal Calc.
        // ... (See next block or keep existing Logic if possible, but existing logic was naive)
        // We'll calculate Aux Loss in a separate block or method if needed.
        // For now, setting 0 to avoid crash, or recompute.
        cache->aux_loss = 0.0f; // TODO: Implement vectorized aux loss
    }
    
    return output.reshape({batch, seq, hidden});
}

// Check helper
/*
static bool is_expert_active(int expert_idx, const float* indices, int top_k) {
    for(int k=0; k<top_k; ++k) {
        if ((int)indices[k] == expert_idx) return true;
    }
    return false;
}
*/

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
    
    // 3. Propagate Router Gradients
    // dlogits has partial gradients from the main task (through weighting)
    
    // --- 4. Add Aux Loss Gradient (SOTA) ---
    // L_aux = N * sum(f_i * P_i)
    // We treating f_i (density) as constant (Stop Gradient).
    // So we only diff w.r.t P_i.
    // dL_aux / dP_i = N * f_i.
    // P_i = sum_batch(softmax(logits)_i) / total_tokens
    // So dL_aux / dLogits = (dL_aux / dP) * (dP / dLogits)
    
    // We need 'f_i' (density) from forward pass. 
    // We can re-calculate it or store it.
    // Let's re-calculate it quickly (or reuse saved_indices).
    // And we need Softmax Probabilities.
    
    // Param: Aux Loss Weight
    float aux_loss_weight = 0.01f; 
    
    std::vector<float> density_1(config_.num_experts, 0.0f);
    // Re-calc density
    for(int64_t i=0; i<batch*seq; ++i) {
        const float* idx_ptr = cache.selected_indices.data() + i * top_k;
        for(int k=0; k<top_k; ++k) {
            int idx = (int)idx_ptr[k];
            if(idx >= 0 && idx < config_.num_experts) density_1[idx] += 1.0f;
        }
    }
    // Normalize density
    for(auto& d : density_1) d /= (batch * seq);
    
    // Now accumulate gradient into dlogits
    // dL_aux / dLogit_ij  (token i, expert j)
    // = sum_k ( dL/dP_k * dP_k/dLogit_ij )
    // dL/dP_k = N * f_k * weight
    // P_k = sum_i(p_ik) / T
    // dP_k / dp_ik = 1/T
    // dp_ik / dz_ij = p_ik * (delta_jk - p_ij)  (Softmax derivative)
    
    // Combine:
    // dL_aux / dz_ij = weight * N * (1/T) * sum_k ( f_k * p_ik * (delta_jk - p_ij) )
    //                = C * [ f_j * p_ij - p_ij * sum_k(f_k * p_ik) ]
    //                = C * p_ij * ( f_j - sum_k(f_k * p_ik) )
    
    float C = aux_loss_weight * config_.num_experts; // Removed 1/T because gradients are summed approx? No, exact.
    // Actually typically we average gradients over batch.
    // Let's implement the formula:
    // grad_z_ij = C * p_ij * (f_j - sum_f_p_i)
    // where sum_f_p_i = sum_k (f_k * p_ik) which is expected density per token roughly.
    
    #pragma omp parallel for
    for(int64_t i=0; i<batch*seq; ++i) {
        // Re-compute Softmax P for this token
        // Use simplified recompute as in forward
        // const float* x_ptr = x_flat.data() + i * hidden;
        // Need to recompute logits? Or use Router forward?
        // We verified logits are needed. We didn't cache full logits, only sparse ones?
        // Wait, cache->router_logits IS full logits [Batch, Seq, Experts].
        const float* l_ptr = cache.router_logits.data() + i * config_.num_experts;
        
        float max_val = -1e9;
        for(int e=0; e<config_.num_experts; ++e) if(l_ptr[e] > max_val) max_val = l_ptr[e];
        
        std::vector<float> p(config_.num_experts);
        float sum_exp = 0.0f;
        for(int e=0; e<config_.num_experts; ++e) {
            p[e] = std::exp(l_ptr[e] - max_val);
            sum_exp += p[e];
        }
        for(int e=0; e<config_.num_experts; ++e) p[e] /= sum_exp;
        
        // Calculate term: sum_prob_density = sum_k (f_k * p_ik)
        float sum_prob_density = 0.0f;
        for(int e=0; e<config_.num_experts; ++e) sum_prob_density += density_1[e] * p[e];
        
        // Add gradient to dlogits
        float* dl_ptr = dlogits.data() + i * config_.num_experts;
        for(int j=0; j<config_.num_experts; ++j) {
            float grad_aux = C * p[j] * (density_1[j] - sum_prob_density);
            dl_ptr[j] += grad_aux;
        }
    }

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

void MoELayer::update_parameters(Optimizer& optimizer, const MoEGradients& grads) {
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
