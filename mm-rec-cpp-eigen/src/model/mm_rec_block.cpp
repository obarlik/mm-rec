/**
 * MM-Rec Block Implementation
 */

#include "mm_rec/model/mm_rec_block.h"

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
    const Tensor& memory
) {
    // x: [batch, seq, hidden_dim]
    // memory: [batch, mem_dim]
    
    int64_t batch = x.size(0);
    int64_t seq = x.size(1);
    int64_t hidden_dim = x.size(2);
    
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
        
        // Update memory with GRU-style gates
        auto [new_memory, update_gate] = gated_memory_->forward(h_t, current_memory);
        current_memory = new_memory;
        
        // FFN: expand and contract
        Tensor ffn_hidden = ffn_up_->forward(h_t).relu();
        Tensor h_out = ffn_down_->forward(ffn_hidden);
        
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
    
    return {output_hidden, current_memory, output_logits};
}

} // namespace mm_rec
