/**
 * MM-Rec Block
 * 
 * Combines:
 * - GRU-style gated memory update
 * - Feed-forward network (FFN)
 * - UBOO output projection (every layer predicts!)
 */

#pragma once

#include "mm_rec/core/tensor.h"
#include "mm_rec/core/linear.h"
#include "mm_rec/core/gated_memory.h"
#include <memory>

namespace mm_rec {

struct MMRecBlockConfig {
    int64_t hidden_dim;
    int64_t mem_dim;
    int64_t ffn_dim;
    int64_t vocab_size;
};

class MMRecBlock {
public:
    MMRecBlock(const MMRecBlockConfig& config);
    
    /**
     * Forward pass
     * 
     * @param x Input hidden state [batch, seq, hidden_dim]
     * @param memory Previous memory state [batch, mem_dim]
     * @return Tuple of (new_hidden, new_memory, logits)
     */
    std::tuple<Tensor, Tensor, Tensor> forward(
        const Tensor& x,
        const Tensor& memory
    );

private:
    MMRecBlockConfig config_;
    
    // Components
    std::unique_ptr<GatedMemoryUpdate> gated_memory_;
    std::unique_ptr<Linear> ffn_up_;
    std::unique_ptr<Linear> ffn_down_;
    std::unique_ptr<Linear> output_proj_;  // UBOO: vocab projection
};

} // namespace mm_rec
