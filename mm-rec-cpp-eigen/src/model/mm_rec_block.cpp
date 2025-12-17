/**
 * MM-Rec Block Implementation
 */

#include "mm_rec/model/mm_rec_block.h"
#include "mm_rec/training/forward_cache.h"
#include "mm_rec/training/gradients.h"
#include "mm_rec/training/backward.h"
#include "mm_rec/training/gru_backward.h"
#include "mm_rec/training/optimizer.h"
#include <iostream>

namespace mm_rec {

MMRecBlock::MMRecBlock(const MMRecBlockConfig& config)
    : config_(config) {
    
    // Initialize gated memory
    gated_memory_ = std::make_unique<GatedMemoryUpdate>(
        config.hidden_dim,
        config.mem_dim
    );
    
    // MoE Layer (replaces standard FFN)
    MoEConfig moe_config;
    moe_config.hidden_dim = config.hidden_dim;
    moe_config.ffn_dim = config.ffn_dim;
    moe_config.num_experts = config.num_experts;
    moe_config.top_k = config.top_k;
    moe_layer_ = std::make_unique<MoELayer>(moe_config);
    
    // RMSNorm (Pre-Norm)
    block_norm_ = std::make_unique<RMSNorm>(config.hidden_dim);
    
    // UBOO output projection
    output_proj_ = std::make_unique<Linear>(config.hidden_dim, config.vocab_size);
}

std::tuple<Tensor, Tensor, Tensor> MMRecBlock::forward(
    const Tensor& x,
    const Tensor& memory,
    BlockCache* cache
) {
    // x: [batch, seq, hidden_dim]
    // memory: [batch, mem_dim]
    
    int64_t batch = x.size(0);
    int64_t seq = x.size(1);
    int64_t hidden_dim = x.size(2);
    int64_t mem_dim = memory.size(1);
    
    // Inputs/Outputs for cache
    if (cache) {
        cache->x = x;
        cache->h_prev = Tensor::zeros({batch, seq, mem_dim});
        cache->h_new = Tensor::zeros({batch, seq, mem_dim});
        
        cache->reset_gate = Tensor::zeros({batch, seq, mem_dim});
        cache->update_gate = Tensor::zeros({batch, seq, mem_dim});
        cache->candidate = Tensor::zeros({batch, seq, mem_dim});
        cache->r_h_prev = Tensor::zeros({batch, seq, mem_dim}); // r * h_prev
        
        cache->ffn_input = x; // Same as x
        cache->ffn_hidden = Tensor::zeros({batch, seq, config_.ffn_dim});
        cache->ffn_output = Tensor::zeros({batch, seq, hidden_dim});
    }

    // 1. RMSNorm (Pre-Norm)
    // Normalize input BEFORE processing
    Tensor x_norm = block_norm_->forward(x);
    
    if (cache) {
        cache->x = x; // Store original for residual backward
        cache->x_norm = x_norm; // Store normalized for GRU/MoE backward
    }

    // Process each token in sequence with memory
    std::vector<Tensor> hidden_states;
    Tensor current_memory = memory;
    
    for (int64_t t = 0; t < seq; ++t) {
        // Get normalized input at t [batch, hidden]
        Tensor x_t_norm = Tensor::zeros({batch, hidden_dim});
        // Also need original x_t for residual if we wanted post-norm, but we do Pre-Norm:
        // y = x + Layer(Norm(x))
        // So we need x_t (original) for addition later.
        Tensor x_t_orig = Tensor::zeros({batch, hidden_dim});
        
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < hidden_dim; ++h) {
                x_t_norm.data()[b * hidden_dim + h] = 
                    x_norm.data()[b * seq * hidden_dim + t * hidden_dim + h];
                x_t_orig.data()[b * hidden_dim + h] = 
                    x.data()[b * seq * hidden_dim + t * hidden_dim + h];
            }
        }
        
        // Save previous memory to cache
        if (cache) {
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t m = 0; m < mem_dim; ++m) {
                    cache->h_prev.data()[b * seq * mem_dim + t * mem_dim + m] = 
                        current_memory.data()[b * mem_dim + m];
                }
            }
        }
        
        // GRU Step: Input -> Memory
        // Uses NORMALIZED input
        Tensor new_memory, update_gate;
        
        if (cache) {
            GatedMemoryUpdate::Cache step_cache;
            auto res = gated_memory_->forward(x_t_norm, current_memory, step_cache);
            new_memory = res.first;
            update_gate = res.second;
            
            // Serialize step cache
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t m = 0; m < mem_dim; ++m) {
                    int64_t idx = b * seq * mem_dim + t * mem_dim + m;
                    int64_t step = b * mem_dim + m;
                    cache->h_new.data()[idx] = new_memory.data()[step];
                    cache->reset_gate.data()[idx] = step_cache.r.data()[step];
                    cache->update_gate.data()[idx] = step_cache.u.data()[step];
                    cache->candidate.data()[idx] = step_cache.h_tilde.data()[step];
                    cache->r_h_prev.data()[idx] = step_cache.r_h_prev.data()[step];
                }
            }
        } else {
            auto res = gated_memory_->forward(x_t_norm, current_memory);
            new_memory = res.first;
            update_gate = res.second;
        }
        current_memory = new_memory; // Updated memory state (h_t)
        
        // MoE Step: Memory -> Expert -> Output
        // The "Thought" comes from Memory. MoE processes the thought.
        // We use 'new_memory' as input to MoE. 
        // Note: mem_dim must equal hidden_dim for this direct connection, 
        // or MoE config.hidden_dim must match mem_dim.
        // Config check: In our config, hidden=128, mem=128. Matches.
        
        Tensor h_gru_seq = new_memory.reshape({batch, 1, hidden_dim});
        Tensor moe_out_seq;
        
        if (cache) {
             MoECache step_cache;
             moe_out_seq = moe_layer_->forward(h_gru_seq, &step_cache);
             
             // Copy MoE Cache
             int64_t num_experts = config_.num_experts;
             int64_t top_k = config_.top_k;
             
             if(t==0) {
                 cache->moe_cache.router_logits = Tensor::zeros({batch, seq, num_experts});
                 cache->moe_cache.routing_weights = Tensor::zeros({batch, seq, top_k});
                 cache->moe_cache.selected_indices = Tensor::zeros({batch, seq, top_k});
             }
             
             for(int b=0; b<batch; ++b) {
                 for(int e=0; e<num_experts; ++e)
                     cache->moe_cache.router_logits.data()[b*seq*num_experts + t*num_experts + e] = 
                         step_cache.router_logits.data()[b*num_experts + e];
                 for(int k=0; k<top_k; ++k) {
                     cache->moe_cache.routing_weights.data()[b*seq*top_k + t*top_k + k] = 
                         step_cache.routing_weights.data()[b*top_k + k];
                     cache->moe_cache.selected_indices.data()[b*seq*top_k + t*top_k + k] = 
                         step_cache.selected_indices.data()[b*top_k + k];
                 }
             }
        } else {
            moe_out_seq = moe_layer_->forward(h_gru_seq, nullptr);
        }
        
        Tensor moe_out = moe_out_seq.reshape({batch, hidden_dim});
        
        // Residual Connection: Output = Original + MoE(GRU(Norm(Original)))
        Tensor t_output = Tensor::zeros({batch, hidden_dim});
        for(int64_t i=0; i<t_output.numel(); ++i) {
            t_output.data()[i] = x_t_orig.data()[i] + moe_out.data()[i];
        }
        
        hidden_states.push_back(t_output);
    }
    
    // Stack outputs
    Tensor output_hidden = Tensor::zeros({batch, seq, hidden_dim});
    for (int64_t t = 0; t < seq; ++t) {
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < hidden_dim; ++h) {
                output_hidden.data()[b * seq * hidden_dim + t * hidden_dim + h] =
                    hidden_states[t].data()[b * hidden_dim + h];
            }
        }
    }
    
    // UBOO Projection
    Tensor output_hidden_flat = output_hidden.reshape({batch * seq, hidden_dim});
    Tensor logits_flat = output_proj_->forward(output_hidden_flat);
    Tensor output_logits = logits_flat.reshape({batch, seq, config_.vocab_size});
    
    if (cache) {
        cache->output = output_hidden;
        cache->logits = output_logits;
    }
    
    return {output_hidden, current_memory, output_logits};
}

// Helper to slice weights [out, in_total] -> [out, in_sub]
static Tensor slice_cols(const Tensor& W, int64_t start_col, int64_t num_cols) {
    int64_t rows = W.size(0);
    int64_t full_cols = W.size(1);
    Tensor slice = Tensor::zeros({rows, num_cols});
    
    for(int64_t r=0; r<rows; ++r) {
        for(int64_t c=0; c<num_cols; ++c) {
            slice.data()[r*num_cols + c] = W.data()[r*full_cols + (start_col + c)];
        }
    }
    return slice;
}

std::pair<Tensor, Tensor> MMRecBlock::backward(
    const Tensor& d_output,
    const Tensor& d_memory_next,
    const Tensor& d_logits,
    const BlockCache& cache,
    BlockGradients& grads
) {
    int64_t batch = config_.hidden_dim > 0 ? d_output.size(0) : 0;
    if (batch == 0) batch = cache.x.size(0);
    int64_t seq = cache.x.size(1);
    int64_t hidden_dim = config_.hidden_dim;
    int64_t mem_dim = config_.mem_dim;
    int64_t ffn_dim = config_.ffn_dim;
    int64_t vocab = config_.vocab_size;
    
    // Gradients w.r.t input x and memory
    Tensor dx = Tensor::zeros({batch, seq, hidden_dim});
    Tensor dmemory = d_memory_next; // From future
    
    // Pre-slice weights (Optimization)
    const Tensor& W_z_total = gated_memory_->get_W_z()->weight();
    const Tensor& b_z = gated_memory_->get_W_z()->bias();
    Tensor W_u_slice = slice_cols(W_z_total, 0, hidden_dim);
    Tensor U_u_slice = slice_cols(W_z_total, hidden_dim, mem_dim);
    
    const Tensor& W_r_total = gated_memory_->get_W_r()->weight();
    const Tensor& b_r = gated_memory_->get_W_r()->bias();
    Tensor W_r_slice = slice_cols(W_r_total, 0, hidden_dim);
    Tensor U_r_slice = slice_cols(W_r_total, hidden_dim, mem_dim);
    
    const Tensor& W_m_total = gated_memory_->get_W_m()->weight();
    const Tensor& b_m = gated_memory_->get_W_m()->bias();
    Tensor W_h_slice = slice_cols(W_m_total, 0, hidden_dim);
    Tensor U_h_slice = slice_cols(W_m_total, hidden_dim, mem_dim);
    
    // Accumulator for RMSNorm backprop
    // Since dx_norm is computed per step, we can accumulate d_x_total
    
    // We iterate backwards
    for (int64_t t = seq - 1; t >= 0; --t) {
        // Fetch Cache Steps
        auto get_slice = [&](const Tensor& src, int64_t t_idx, int64_t dim) {
            Tensor slice = Tensor::zeros({batch, dim});
            for(int64_t b=0; b<batch; ++b) {
                for(int64_t d=0; d<dim; ++d) slice.data()[b*dim + d] = src.data()[b*seq*dim + t_idx*dim + d];
            }
            return slice;
        };
        
        // Current Inputs/States
        Tensor x_t_norm = get_slice(cache.x_norm, t, hidden_dim); // Normalized input
        Tensor h_prev_t = get_slice(cache.h_prev, t, mem_dim);
        Tensor h_new_t = get_slice(cache.h_new, t, mem_dim); // GRU output, Input to MoE
        
        Tensor r_t = get_slice(cache.reset_gate, t, mem_dim);
        Tensor u_t = get_slice(cache.update_gate, t, mem_dim);
        Tensor h_tilde_t = get_slice(cache.candidate, t, mem_dim);
        
        // 1. Gradients from Output (Residual + UBOO)
        Tensor d_out_t = get_slice(d_output, t, hidden_dim);
        Tensor d_logits_t = get_slice(d_logits, t, vocab);
        
        // A. Backprop UBOO Projection
        Tensor d_h_out_uboo = Tensor::zeros({batch, hidden_dim});
        Tensor dW_out_t, db_out_t;
        linear_backward(
            get_slice(cache.output, t, hidden_dim), // Output of block
            output_proj_->weight(),
            d_logits_t,
            d_h_out_uboo,
            dW_out_t,
            db_out_t
        );
        // Accumulate Output Proj Grads
        float* dW_param = grads.output_proj_grads.dW.data();
        float* db_param = grads.output_proj_grads.db.data();
        for(int64_t i=0; i<dW_out_t.numel(); ++i) dW_param[i] += dW_out_t.data()[i];
        for(int64_t i=0; i<db_out_t.numel(); ++i) db_param[i] += db_out_t.data()[i];
        
        // Total Gradient at Output
        Tensor d_block_out = d_out_t + d_h_out_uboo;
        
        // B. Residual Connection split
        // y = x + MoE(GRU(Norm(x)))
        // d_MoE = d_block_out
        // d_x_skip = d_block_out (Direct path)
        
        // 2. Backprop MoE
        // MoE Input was h_new_t (GRU output)
        // MoE Output gradient is d_block_out
        
        // Reconstruct Step Cache for MoE
        MoECache step_cache;
        int64_t num_experts = config_.num_experts;
        int64_t top_k = config_.top_k;
        step_cache.router_logits = Tensor::zeros({batch, 1, num_experts});
        step_cache.routing_weights = Tensor::zeros({batch, 1, top_k});
        step_cache.selected_indices = Tensor::zeros({batch, 1, top_k});
        for(int b=0; b<batch; ++b) {
            for(int e=0; e<num_experts; ++e) step_cache.router_logits.data()[b*num_experts+e] = cache.moe_cache.router_logits.data()[b*seq*num_experts+t*num_experts+e];
            for(int k=0; k<top_k; ++k) {
                step_cache.routing_weights.data()[b*top_k+k] = cache.moe_cache.routing_weights.data()[b*seq*top_k+t*top_k+k];
                step_cache.selected_indices.data()[b*top_k+k] = cache.moe_cache.selected_indices.data()[b*seq*top_k+t*top_k+k];
            }
        }
        
        Tensor d_moe_out_seq = d_block_out.reshape({batch, 1, hidden_dim});
        Tensor h_new_t_seq = h_new_t.reshape({batch, 1, hidden_dim});
        
        Tensor d_h_gru_seq = moe_layer_->backward(
            d_moe_out_seq,
            h_new_t_seq,
            step_cache,
            grads.moe_grads
        );
        Tensor d_h_gru = d_h_gru_seq.reshape({batch, hidden_dim});
        
        // 3. Backprop GRU
        // GRU Output gradient = d_h_gru + dmemory (from future step)
        // GRU Input was x_t_norm
        
        Tensor d_h_total = d_h_gru + dmemory;
        Tensor d_x_norm_t, d_h_prev_t;
        
        gru_backward(
            x_t_norm, h_prev_t, r_t, u_t, h_tilde_t,
            W_r_slice, U_r_slice, b_r,
            W_u_slice, U_u_slice, b_z,
            W_h_slice, U_h_slice, b_m,
            d_h_total,
            grads.gru_grads,
            d_x_norm_t,  // Gradient w.r.t input to GRU (x_norm)
            d_h_prev_t   // Gradient w.r.t prev memory
        );
        
        dmemory = d_h_prev_t; // Propagate to past
        
        // 4. Backprop RMSNorm
        // Input to RMSNorm was x_t (slice of x)
        // Gradient coming back is d_x_norm_t
        // But RMSNorm is applied to whole tensor 'x'. We can do it per slice if we treat vectors independently.
        // Yes, RMSNorm is row-wise independent.
        // d_x_norm_t -> RMSNormBackward -> d_x_pre_norm
        
        // We need efficient RMSNorm backward for just this slice? 
        // Or we can accumulate d_x_norm into a full tensor and run full backward later.
        // Full backward later is better for batching, but for this loop structure, slice backward is needed for dx accumulation.
        // But we can just write d_x_norm_t into a d_x_norm_total buffer and run 1 big backward at end.
        // Let's assume we have d_x_norm_total tensor.
        // Wait, 'backward' returns dx, we need to populate dx.
        // Let's accumulate into d_x_norm storage, then post-loop run norm backward.
        
        // Wait, we don't have storage for d_x_norm in 'backward' scope unless we create it.
        // Optimization: Run Norm backward per step? Or Allocate big tensor.
        // Let's compute it now.
        // RMSNorm::backward expects full tensor relative to its batch size.
        // Here batch=BatchSize. Norm was on HiddenDim. It works.
        // x argument: x_t_orig (un-normalized input slice)
        
        Tensor x_t_orig = get_slice(cache.x, t, hidden_dim);
        
        // We use a temporary grads struct to catch dWeights for this step, then accumulate
        RMSNormGradients step_norm_grads; 
        step_norm_grads.d_weight = Tensor::zeros({hidden_dim});
        
        Tensor d_x_pre_norm = block_norm_->backward(d_x_norm_t, x_t_orig, step_norm_grads);
        
        // Accumulate Norm Weights
        float* d_nw = grads.norm_grads.d_weight.data();
        float* d_nw_step = step_norm_grads.d_weight.data();
        for(int i=0; i<hidden_dim; ++i) d_nw[i] += d_nw_step[i];
        
        // 5. Total Gradient w.r.t X
        // dx = d_x_skip + d_x_pre_norm
        for(int b=0; b<batch; ++b) {
            for(int h=0; h<hidden_dim; ++h) {
                dx.data()[b*seq*hidden_dim + t*hidden_dim + h] = 
                    d_block_out.data()[b*hidden_dim + h] + d_x_pre_norm.data()[b*hidden_dim + h];
            }
        }
    }
    
    return {dx, dmemory};
}

void MMRecBlock::update_parameters(SGD& optimizer, const BlockGradients& grads) {
    // 1. Update MoE
    moe_layer_->update_parameters(optimizer, grads.moe_grads);
    
    // 2. Update Output Projection (UBOO)
    optimizer.step(output_proj_->weight(), grads.output_proj_grads.dW);
    optimizer.step(output_proj_->bias(), grads.output_proj_grads.db);
    
    // 3. Update RMSNorm
    block_norm_->update_parameters(optimizer, grads.norm_grads);
    
    // 4. Update GRU
    // Helper to concat grads [out, in1] + [out, in2] -> [out, in1+in2]
    auto concat_grads = [](const Tensor& dW, const Tensor& dU) {
        int64_t rows = dW.size(0);
        int64_t cols1 = dW.size(1);
        int64_t cols2 = dU.size(1);
        
        Tensor d_total = Tensor::zeros({rows, cols1 + cols2});
        
        // Copy dW
        for(int64_t r=0; r<rows; ++r) {
            for(int64_t c=0; c<cols1; ++c) {
                d_total.data()[r*(cols1+cols2) + c] = dW.data()[r*cols1 + c];
            }
        }
        // Copy dU
        for(int64_t r=0; r<rows; ++r) {
            for(int64_t c=0; c<cols2; ++c) {
                d_total.data()[r*(cols1+cols2) + cols1 + c] = dU.data()[r*cols2 + c];
            }
        }
        return d_total;
    };
    
    // Update W_z (Update Gate)
    {
        Tensor d_total = concat_grads(grads.gru_grads.dW_u, grads.gru_grads.dU_u);
        optimizer.step(gated_memory_->get_W_z()->weight(), d_total);
        optimizer.step(gated_memory_->get_W_z()->bias(), grads.gru_grads.db_u);
    }
    
    // Update W_r (Reset Gate)
    {
        Tensor d_total = concat_grads(grads.gru_grads.dW_r, grads.gru_grads.dU_r);
        optimizer.step(gated_memory_->get_W_r()->weight(), d_total);
        optimizer.step(gated_memory_->get_W_r()->bias(), grads.gru_grads.db_r);
    }
    
    // Update W_m (Candidate)
    {
        Tensor d_total = concat_grads(grads.gru_grads.dW_h, grads.gru_grads.dU_h);
        optimizer.step(gated_memory_->get_W_m()->weight(), d_total);
        optimizer.step(gated_memory_->get_W_m()->bias(), grads.gru_grads.db_h);
    }
}

} // namespace mm_rec

