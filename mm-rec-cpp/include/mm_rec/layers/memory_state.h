/**
 * Per-Layer Memory State Management
 * 
 * CRITICAL: Based on Bug #2 from production (8+ months experience)
 * Each layer MUST have its own isolated memory state.
 * Sharing memory across layers causes model to fail (loss stuck at 8.0).
 * 
 * Why isolated states?
 * - Lower layers: Syntax, immediate context
 * - Higher layers: Semantics, long-range dependencies
 * - Hierarchical abstraction emerges naturally
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>

namespace mm_rec {

/**
 * Per-Layer Memory State Container
 * 
 * Stores isolated memory state tensors for each layer.
 * Ensures no sharing between layers (Bug #2 prevention).
 */
class PerLayerMemoryState {
public:
    /**
     * Constructor
     * 
     * @param num_layers Number of layers in the model
     * @param batch_size Batch size for memory tensors
     * @param mem_dim Memory dimension per layer
     * @param device Device to allocate tensors on (CPU/CUDA)
     */
    PerLayerMemoryState(
        int64_t num_layers,
        int64_t batch_size,
        int64_t mem_dim,
        torch::Device device = torch::kCPU
    );

    /**
     * Get memory state for a specific layer
     * 
     * @param layer_idx Layer index [0, num_layers)
     * @return Memory tensor [batch_size, mem_dim]
     */
    torch::Tensor get_layer_state(int64_t layer_idx) const;

    /**
     * Update memory state for a specific layer
     * 
     * CRITICAL: Never use .detach() before calling this!
     * Bug #3: Detaching breaks gradient flow
     * 
     * @param layer_idx Layer index [0, num_layers)
     * @param new_state New memory state [batch_size, mem_dim]
     */
    void update_layer_state(int64_t layer_idx, const torch::Tensor& new_state);

    /**
     * Reset all layer states to zero
     * Useful for starting new sequence/session
     */
    void reset_all_states();

    /**
     * Get number of layers
     */
    int64_t num_layers() const { return num_layers_; }

    /**
     * Get batch size
     */
    int64_t batch_size() const { return batch_size_; }

    /**
     * Get memory dimension
     */
    int64_t mem_dim() const { return mem_dim_; }

    /**
     * Change batch size (e.g., for different batch in inference)
     * Reallocates all state tensors
     */
    void resize_batch(int64_t new_batch_size);

private:
    int64_t num_layers_;
    int64_t batch_size_;
    int64_t mem_dim_;
    torch::Device device_;
    
    // CRITICAL: std::vector ensures separate tensors per layer
    // DO NOT use single tensor with indexing - risk of sharing!
    std::vector<torch::Tensor> layer_states_;
};

} // namespace mm_rec
