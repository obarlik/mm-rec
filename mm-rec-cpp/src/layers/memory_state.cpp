/**
 * Per-Layer Memory State Implementation
 */

#include "mm_rec/layers/memory_state.h"

namespace mm_rec {

PerLayerMemoryState::PerLayerMemoryState(
    int64_t num_layers,
    int64_t batch_size,
    int64_t mem_dim,
    torch::Device device
) : num_layers_(num_layers),
    batch_size_(batch_size),
    mem_dim_(mem_dim),
    device_(device) {
    
    // CRITICAL: Reserve space to avoid reallocation
    layer_states_.reserve(num_layers);
    
    // Initialize separate tensor for each layer
    // Bug #2 Prevention: Each layer gets its OWN tensor
    for (int64_t i = 0; i < num_layers; ++i) {
        auto state = torch::zeros(
            {batch_size, mem_dim},
            torch::TensorOptions().dtype(torch::kFloat32).device(device)
        );
        layer_states_.push_back(state);
    }
}

torch::Tensor PerLayerMemoryState::get_layer_state(int64_t layer_idx) const {
    // Bounds checking
    TORCH_CHECK(
        layer_idx >= 0 && layer_idx < num_layers_,
        "Layer index ", layer_idx, " out of range [0, ", num_layers_, ")"
    );
    
    return layer_states_[layer_idx];
}

void PerLayerMemoryState::update_layer_state(
    int64_t layer_idx,
    const torch::Tensor& new_state
) {
    // Bounds checking
    TORCH_CHECK(
        layer_idx >= 0 && layer_idx < num_layers_,
        "Layer index ", layer_idx, " out of range [0, ", num_layers_, ")"
    );
    
    // Dimension validation
    TORCH_CHECK(
        new_state.dim() == 2 && 
        new_state.size(0) == batch_size_ && 
        new_state.size(1) == mem_dim_,
        "State shape mismatch. Expected [", batch_size_, ", ", mem_dim_, "], got [",
        new_state.size(0), ", ", new_state.size(1), "]"
    );
    
    // CRITICAL: DO NOT use .detach() here!
    // Bug #3: Detaching breaks gradient flow
    // The tensor should maintain its computation graph
    layer_states_[layer_idx] = new_state;
}

void PerLayerMemoryState::reset_all_states() {
    for (int64_t i = 0; i < num_layers_; ++i) {
        layer_states_[i].zero_();
    }
}

void PerLayerMemoryState::resize_batch(int64_t new_batch_size) {
    if (new_batch_size == batch_size_) {
        return;  // No change needed
    }
    
    batch_size_ = new_batch_size;
    
    // Reallocate all state tensors
    for (int64_t i = 0; i < num_layers_; ++i) {
        layer_states_[i] = torch::zeros(
            {batch_size_, mem_dim_},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_)
        );
    }
}

} // namespace mm_rec
