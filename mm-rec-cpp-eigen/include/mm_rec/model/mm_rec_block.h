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
#include "mm_rec/model/moe.h"
#include "mm_rec/core/normalization.h"
#include <memory>

namespace mm_rec {

struct MMRecBlockConfig {
    int64_t hidden_dim;
    int64_t mem_dim;
    int64_t ffn_dim;
    int64_t vocab_size;
    int64_t num_experts = 4; // Default to 4 experts
    int64_t top_k = 2;       // Default to Top-2
};

struct BlockCache; // Forward decl
struct BlockGradients; // Forward decl
class SGD; // Forward decl

class MMRecBlock {
public:
    MMRecBlock(const MMRecBlockConfig& config);
    
    /**
     * Forward pass
     * 
     * @param x Input hidden state [batch, seq, hidden_dim]
     * @param memory Previous memory state [batch, mem_dim]
     * @param cache Optional pointer to cache for training
     * @return Tuple of (new_hidden, new_memory, logits)
     */
    std::tuple<Tensor, Tensor, Tensor> forward(
        const Tensor& x,
        const Tensor& memory,
        BlockCache* cache = nullptr
    );

    /**
     * Backward pass
     * 
     * @param d_output       Gradient w.r.t output hidden states [batch, seq, hidden]
     * @param d_memory_next  Gradient w.r.t final memory state [batch, mem_dim]
     * @param d_logits       Gradient w.r.t output logits [batch, seq, vocab]
     * @param cache          Forward cache
     * @param grads          Output: gradients for parameters
     * @return Pair of (dx, dmemory)
     */
    std::pair<Tensor, Tensor> backward(
        const Tensor& d_output,
        const Tensor& d_memory_next,
        const Tensor& d_logits,
        const BlockCache& cache,
        BlockGradients& grads
    );

    /**
     * Update parameters using optimizer and gradients
     */
    void update_parameters(SGD& optimizer, const BlockGradients& grads);

private:
    MMRecBlockConfig config_;
    
    // Components
    std::unique_ptr<RMSNorm> block_norm_; // Pre-norm
    std::unique_ptr<GatedMemoryUpdate> gated_memory_;
    // Replaced standard FFN with MoE Layer
    std::unique_ptr<MoELayer> moe_layer_;
    
    std::unique_ptr<Linear> output_proj_;  // UBOO: vocab projection
};

} // namespace mm_rec
