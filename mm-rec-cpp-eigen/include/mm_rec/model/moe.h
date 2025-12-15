/**
 * Mixture of Experts (MoE) Layer
 * 
 * Replaces standard FFN with a set of experts and a router.
 * Implements Dynamic Routing (sparse execution).
 */

#pragma once

#include "mm_rec/core/tensor.h"
#include "mm_rec/core/linear.h"
#include <vector>
#include <memory>
#include <tuple>

namespace mm_rec {

class SGD; // Forward declaration

struct MoEConfig {
    int64_t hidden_dim;
    int64_t ffn_dim;
    int64_t num_experts;
    int64_t top_k;      // Number of experts to select per token
    int64_t vocab_size; // Needed if we want to add aux loss? Not strictly for the layer.
};

// Cache for backward pass
struct MoECache {
    Tensor router_logits;     // [batch, seq, num_experts]
    Tensor routing_weights;   // [batch, seq, top_k] (Softmax output for selected)
    Tensor selected_indices;  // [batch, seq, top_k] (Indices of selected experts)
};

struct MoEGradients {
    Tensor d_gate; // [hidden, num_experts]
    Tensor db_gate; // [num_experts]
    
    // Gradients for experts are stored densely for simplicity of the struct,
    // but in practice we might want a sparse accumulation if N is huge.
    // For now (N=8/16), vector of Tensors is fine.
    std::vector<Tensor> d_expert_up_weights;
    std::vector<Tensor> d_expert_up_biases;
    std::vector<Tensor> d_expert_down_weights;
    std::vector<Tensor> d_expert_down_biases;
    
    void init(const MoEConfig& config);
    void zero();
};

class MoELayer {
public:
    MoELayer(const MoEConfig& config);
    
    /**
     * Sparse Forward Pass
     * 
     * 1. Compute router logits
     * 2. Select top-k experts
     * 3. Route inputs to experts (Dynamic Routing - no padding)
     * 4. Combine outputs
     * 
     * @param x Input [batch, seq, hidden]
     * @param cache Cache to store routing info for backward
     * @return Output [batch, seq, hidden]
     */
    Tensor forward(const Tensor& x, MoECache* cache = nullptr);
    
    /**
     * Backward Pass
     * 
     * Propagates gradients through:
     * - Active experts (weighted by routing weights)
     * - Router (gate)
     */
    Tensor backward(
        const Tensor& d_output,
        const Tensor& x,
        const MoECache& cache,
        MoEGradients& grads
    );
    
    /**
     * Initializer parameters
     */
    void update_parameters(SGD& optimizer, const MoEGradients& grads);

private:
    MoEConfig config_;
    
    // Router: Maps input -> logits per expert
    std::unique_ptr<Linear> gate_;
    
    // Experts: Array of FFNs (Up + Down)
    // Expert i: x -> Up -> ReLU -> Down -> y
    std::vector<std::unique_ptr<Linear>> experts_up_;
    std::vector<std::unique_ptr<Linear>> experts_down_;
};

} // namespace mm_rec
