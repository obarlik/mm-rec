/**
 * MM-Rec Block Implementation
 */

#include "mm_rec/model/mm_rec_block.h"
#include "mm_rec/training/forward_cache.h"
#include "mm_rec/training/gradients.h"
#include "mm_rec/training/backward.h"
#include "mm_rec/training/gru_backward.h"
#include <iostream>

namespace mm_rec {

MMRecBlock::MMRecBlock(const MMRecBlockConfig& config)
    : config_(config) {
    
    // Initialize gated memory
    gated_memory_ = std::make_unique<GatedMemoryUpdate>(
        config.hidden_dim,
        config.mem_dim
    );
    
    // FFN (expand then contract)
    ffn_up_ = std::make_unique<Linear>(config.hidden_dim, config.ffn_dim);
    ffn_down_ = std::make_unique<Linear>(config.ffn_dim, config.hidden_dim);
    
    // UBOO output projection (every layer!)
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

    // Process each token in sequence with memory
    std::vector<Tensor> hidden_states;
    std::vector<Tensor> all_logits;
    
    Tensor current_memory = memory;
    
    for (int64_t t = 0; t < seq; ++t) {
        // Get token at position t: [batch, hidden_dim]
        Tensor h_t = Tensor::zeros({batch, hidden_dim});
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < hidden_dim; ++h) {
                h_t.data()[b * hidden_dim + h] = 
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
        
        // Update memory with GRU-style gates
        Tensor new_memory, update_gate;
        
        if (cache) {
            GatedMemoryUpdate::Cache step_cache;
            auto res = gated_memory_->forward(h_t, current_memory, step_cache);
            new_memory = res.first;
            update_gate = res.second;
            
            // Copy step cache to block cache (sequence)
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t m = 0; m < mem_dim; ++m) {
                    int64_t idx = b * seq * mem_dim + t * mem_dim + m;
                    int64_t step_idx = b * mem_dim + m;
                    
                    cache->h_new.data()[idx] = new_memory.data()[step_idx];
                    cache->reset_gate.data()[idx] = step_cache.r.data()[step_idx];
                    cache->update_gate.data()[idx] = step_cache.u.data()[step_idx];
                    cache->candidate.data()[idx] = step_cache.h_tilde.data()[step_idx];
                    cache->r_h_prev.data()[idx] = step_cache.r_h_prev.data()[step_idx];
                }
            }
        } else {
            auto res = gated_memory_->forward(h_t, current_memory);
            new_memory = res.first;
            update_gate = res.second;
        }
        current_memory = new_memory;
        
        // FFN: expand and contract
        Tensor ffn_hidden = ffn_up_->forward(h_t).relu();
        Tensor h_out = ffn_down_->forward(ffn_hidden);
        
        if (cache) {
           for (int64_t b = 0; b < batch; ++b) {
               // ffn_hidden [batch, ffn_dim]
               for (int64_t f = 0; f < config_.ffn_dim; ++f) {
                   cache->ffn_hidden.data()[b * seq * config_.ffn_dim + t * config_.ffn_dim + f] =
                       ffn_hidden.data()[b * config_.ffn_dim + f];
               }
               // ffn_output [batch, hidden_dim]
               for (int64_t h = 0; h < hidden_dim; ++h) {
                   cache->ffn_output.data()[b * seq * hidden_dim + t * hidden_dim + h] =
                       h_out.data()[b * hidden_dim + h];
               }
           }
        }
        
        // UBOO: output projection (every layer predicts!)
        Tensor logits = output_proj_->forward(h_out);
        
        hidden_states.push_back(h_out);
        all_logits.push_back(logits);
    }
    
    // Stack outputs: [batch, seq, hidden_dim] and [batch, seq, vocab]
    Tensor output_hidden = Tensor::zeros({batch, seq, hidden_dim});
    Tensor output_logits = Tensor::zeros({batch, seq, config_.vocab_size});
    
    for (int64_t t = 0; t < seq; ++t) {
        for (int64_t b = 0; b < batch; ++b) {
            // Copy hidden state
            for (int64_t h = 0; h < hidden_dim; ++h) {
                output_hidden.data()[b * seq * hidden_dim + t * hidden_dim + h] =
                    hidden_states[t].data()[b * hidden_dim + h];
            }
            
            // Copy logits
            for (int64_t v = 0; v < config_.vocab_size; ++v) {
                output_logits.data()[b * seq * config_.vocab_size + t * config_.vocab_size + v] =
                    all_logits[t].data()[b * config_.vocab_size + v];
            }
        }
    }
    
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
    // Dimensions
    int64_t batch = config_.hidden_dim > 0 ? d_output.size(0) : 0; // Check safety
    if (batch == 0) batch = cache.x.size(0);
    
    int64_t seq = cache.x.size(1);
    int64_t hidden_dim = config_.hidden_dim;
    int64_t mem_dim = config_.mem_dim;
    int64_t ffn_dim = config_.ffn_dim;
    int64_t vocab = config_.vocab_size;
    
    Tensor dx = Tensor::zeros({batch, seq, hidden_dim});
    Tensor dmemory = d_memory_next; // Start with gradient from future (or loss)
    
    // Pre-slice weights for GRU (Optimization: do once)
    // W_z (Update Gate)
    const Tensor& W_z_total = gated_memory_->get_W_z()->weight();
    const Tensor& b_z = gated_memory_->get_W_z()->bias();
    Tensor W_u_slice = slice_cols(W_z_total, 0, hidden_dim);
    Tensor U_u_slice = slice_cols(W_z_total, hidden_dim, mem_dim);
    
    // W_r (Reset Gate)
    const Tensor& W_r_total = gated_memory_->get_W_r()->weight();
    const Tensor& b_r = gated_memory_->get_W_r()->bias();
    Tensor W_r_slice = slice_cols(W_r_total, 0, hidden_dim);
    Tensor U_r_slice = slice_cols(W_r_total, hidden_dim, mem_dim);
    
    // W_m (Candidate) - Note: In code it's W_m, in math usually W_h
    const Tensor& W_m_total = gated_memory_->get_W_m()->weight();
    const Tensor& b_m = gated_memory_->get_W_m()->bias();
    Tensor W_h_slice = slice_cols(W_m_total, 0, hidden_dim);
    Tensor U_h_slice = slice_cols(W_m_total, hidden_dim, mem_dim);
    
    // Loop BACKWARDS through time
    for (int64_t t = seq - 1; t >= 0; --t) {
        // 1. Get step tensors from cache (slicing manually)
        // Helper to extract step slice [batch, dim]
        auto get_slice = [&](const Tensor& src, int64_t t_idx, int64_t dim) {
            Tensor slice = Tensor::zeros({batch, dim});
            for(int64_t b=0; b<batch; ++b) {
                for(int64_t d=0; d<dim; ++d) {
                    slice.data()[b*dim + d] = src.data()[b*seq*dim + t_idx*dim + d];
                }
            }
            return slice;
        };
        
        Tensor x_t = get_slice(cache.x, t, hidden_dim);
        Tensor h_prev_t = get_slice(cache.h_prev, t, mem_dim);
        Tensor ffn_hidden_t = get_slice(cache.ffn_hidden, t, ffn_dim);
        
        // GRU gate values
        Tensor r_t = get_slice(cache.reset_gate, t, mem_dim);
        Tensor u_t = get_slice(cache.update_gate, t, mem_dim);
        Tensor h_tilde_t = get_slice(cache.candidate, t, mem_dim);
        
        // 2. Gradients at this step
        Tensor d_out_t = get_slice(d_output, t, hidden_dim);
        Tensor d_logits_t = get_slice(d_logits, t, vocab);
        
        // 3. Output Projection Backward
        Tensor d_h_out_1 = Tensor::zeros({batch, hidden_dim});
        Tensor dW_out_t, db_out_t;
        
        linear_backward(
            get_slice(cache.ffn_output, t, hidden_dim), // Input to projection was h_out (ffn_output)
            output_proj_->weight(),
            d_logits_t,
            d_h_out_1,
            dW_out_t,
            db_out_t
        );
        
        // Accumulate Output Gradients
        // Note: This is slow manually, but safe for now
        // Using Eigen Map would be faster, but keeping it simple/safe
        float* dW_param = grads.output_proj_grads.dW.data();
        float* db_param = grads.output_proj_grads.db.data();
        const float* dW_step = dW_out_t.data();
        const float* db_step = db_out_t.data();
        int64_t size_W = dW_out_t.numel();
        int64_t size_b = db_out_t.numel();
        
        // #pragma omp parallel for // Optional
        for(int64_t i=0; i<size_W; ++i) dW_param[i] += dW_step[i];
        for(int64_t i=0; i<size_b; ++i) db_param[i] += db_step[i];
        
        // 4. Combine gradients for h_out
        Tensor d_h_out = Tensor::zeros(d_out_t.sizes());
        for(int64_t i=0; i<d_h_out.numel(); ++i) d_h_out.data()[i] = d_out_t.data()[i] + d_h_out_1.data()[i];
        
        // 5. FFN Backward
        Tensor d_ffn_hidden_relu = Tensor::zeros({batch, ffn_dim});
        Tensor dW_down_t, db_down_t;
        linear_backward(
            ffn_hidden_t, 
            ffn_down_->weight(),
            d_h_out,
            d_ffn_hidden_relu,
            dW_down_t,
            db_down_t
        );
        
        // Accumulate FFN Down
        dW_param = grads.ffn_down_grads.dW.data();
        db_param = grads.ffn_down_grads.db.data();
        dW_step = dW_down_t.data();
        db_step = db_down_t.data();
        size_W = dW_down_t.numel();
        size_b = db_down_t.numel();
        for(int64_t i=0; i<size_W; ++i) dW_param[i] += dW_step[i];
        for(int64_t i=0; i<size_b; ++i) db_param[i] += db_step[i];
        
        // Backprop through ReLU
        Tensor d_ffn_up_out = relu_backward(ffn_hidden_t, d_ffn_hidden_relu);
        
        // Backprop through ffn_up
        Tensor d_x_ffn = Tensor::zeros({batch, hidden_dim});
        Tensor dW_up_t, db_up_t;
        linear_backward(
            x_t,
            ffn_up_->weight(),
            d_ffn_up_out,
            d_x_ffn,
            dW_up_t,
            db_up_t
        );
        
        // Accumulate FFN Up
        dW_param = grads.ffn_up_grads.dW.data();
        db_param = grads.ffn_up_grads.db.data();
        dW_step = dW_up_t.data();
        db_step = db_up_t.data();
        size_W = dW_up_t.numel();
        size_b = db_up_t.numel();
        for(int64_t i=0; i<size_W; ++i) dW_param[i] += dW_step[i];
        for(int64_t i=0; i<size_b; ++i) db_param[i] += db_step[i];
        
        // 6. GRU Backward
        Tensor d_x_gru, d_h_prev_gru;
        
        // Pass dmemory as incoming gradient
        gru_backward(
            x_t, h_prev_t, r_t, u_t, h_tilde_t,
            W_r_slice, U_r_slice, b_r,
            W_u_slice, U_u_slice, b_z, // W_z is Update gate
            W_h_slice, U_h_slice, b_m, // W_m is Candidate
            dmemory,
            grads.gru_grads,
            d_x_gru,
            d_h_prev_gru
        );
        
        // 7. Combine gradients w.r.t input x_t
        for(int64_t b=0; b<batch; ++b) {
            for(int64_t h=0; h<hidden_dim; ++h) {
                dx.data()[b*seq*hidden_dim + t*hidden_dim + h] = 
                    d_x_ffn.data()[b*hidden_dim + h] + d_x_gru.data()[b*hidden_dim + h];
            }
        }
        
        // 8. Update dmemory for next iteration
        dmemory = d_h_prev_gru;
    }
    
    return {dx, dmemory};
}

} // namespace mm_rec
