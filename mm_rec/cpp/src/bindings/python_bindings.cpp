/**
 * Python Bindings for C++ Optimized Functions
 * 
 * PyTorch C++ extension bindings for:
 * - Associative Scan (already in associative_scan_cpu.cpp)
 * - Core Recurrence Fused
 * - MDI Optimized
 */

#include <torch/extension.h>
#include <vector>

// Forward declarations
extern void core_recurrence_fused_cpu(
    const float* z_t,
    const float* h_prev,
    const float* W_g,
    const float* gamma,
    float* h_t,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_threads
);

extern void mdi_update_fused_simd(
    const float* h_new,
    const float* h_old,
    const float* gamma,
    const float* gate,
    float* h_updated,
    int batch_size,
    int seq_len,
    int model_dim,
    int num_threads
);

// ============================================================================
// Core Recurrence Fused - PyTorch Binding
// ============================================================================

torch::Tensor core_recurrence_fused_torch(
    torch::Tensor z_t,       // [batch, seq_len, hidden_dim]
    torch::Tensor h_prev,    // [batch, seq_len, hidden_dim]
    torch::Tensor W_g,       // [hidden_dim, hidden_dim]
    torch::Tensor gamma      // [batch, seq_len, hidden_dim]
) {
    // If gradients are required, fall back to PyTorch ops to preserve autograd.
    if (z_t.requires_grad() || h_prev.requires_grad() || W_g.requires_grad() || gamma.requires_grad()) {
        // Retain grads for non-leaf tensors so .grad is populated (tests expect this).
        if (z_t.requires_grad() && !z_t.is_leaf()) z_t.retain_grad();
        if (h_prev.requires_grad() && !h_prev.is_leaf()) h_prev.retain_grad();
        if (W_g.requires_grad() && !W_g.is_leaf()) W_g.retain_grad();
        if (gamma.requires_grad() && !gamma.is_leaf()) gamma.retain_grad();

        auto gate = torch::sigmoid(torch::matmul(h_prev, W_g.transpose(0, 1)));
        return z_t * gate + gamma * h_prev;
    }

    // Check inputs
    TORCH_CHECK(z_t.dim() == 3, "z_t must be 3D: [batch, seq_len, hidden_dim]");
    TORCH_CHECK(h_prev.dim() == 3, "h_prev must be 3D: [batch, seq_len, hidden_dim]");
    TORCH_CHECK(W_g.dim() == 2, "W_g must be 2D: [hidden_dim, hidden_dim]");
    TORCH_CHECK(gamma.dim() == 3, "gamma must be 3D: [batch, seq_len, hidden_dim]");
    
    int batch_size = z_t.size(0);
    int seq_len = z_t.size(1);
    int hidden_dim = z_t.size(2);
    
    // Get contiguous tensors
    auto z_t_cont = z_t.contiguous();
    auto h_prev_cont = h_prev.contiguous();
    auto W_g_cont = W_g.contiguous();
    auto gamma_cont = gamma.contiguous();
    
    // Allocate output
    auto h_t = torch::zeros_like(z_t);
    
    // Get data pointers
    const float* z_t_ptr = z_t_cont.data_ptr<float>();
    const float* h_prev_ptr = h_prev_cont.data_ptr<float>();
    const float* W_g_ptr = W_g_cont.data_ptr<float>();
    const float* gamma_ptr = gamma_cont.data_ptr<float>();
    float* h_t_ptr = h_t.data_ptr<float>();
    
    // Call C++ function
    core_recurrence_fused_cpu(
        z_t_ptr, h_prev_ptr, W_g_ptr, gamma_ptr, h_t_ptr,
        batch_size, seq_len, hidden_dim, 0
    );
    
    return h_t;
}

// ============================================================================
// MDI Update Fused - PyTorch Binding
// ============================================================================

torch::Tensor mdi_update_fused_torch(
    torch::Tensor h_new,     // [batch, seq_len, model_dim]
    torch::Tensor h_old,     // [batch, seq_len, model_dim]
    torch::Tensor gamma,     // [batch, seq_len, model_dim]
    torch::Tensor gate       // [batch, seq_len, model_dim]
) {
    // Check inputs
    TORCH_CHECK(h_new.dim() == 3, "h_new must be 3D: [batch, seq_len, model_dim]");
    TORCH_CHECK(h_old.dim() == 3, "h_old must be 3D: [batch, seq_len, model_dim]");
    TORCH_CHECK(gamma.dim() == 3, "gamma must be 3D: [batch, seq_len, model_dim]");
    TORCH_CHECK(gate.dim() == 3, "gate must be 3D: [batch, seq_len, model_dim]");
    
    int batch_size = h_new.size(0);
    int seq_len = h_new.size(1);
    int model_dim = h_new.size(2);
    
    // Get contiguous tensors
    auto h_new_cont = h_new.contiguous();
    auto h_old_cont = h_old.contiguous();
    auto gamma_cont = gamma.contiguous();
    auto gate_cont = gate.contiguous();
    
    // Allocate output
    auto h_updated = torch::zeros_like(h_new);
    
    // Get data pointers
    const float* h_new_ptr = h_new_cont.data_ptr<float>();
    const float* h_old_ptr = h_old_cont.data_ptr<float>();
    const float* gamma_ptr = gamma_cont.data_ptr<float>();
    const float* gate_ptr = gate_cont.data_ptr<float>();
    float* h_updated_ptr = h_updated.data_ptr<float>();
    
    // Call C++ function
    mdi_update_fused_simd(
        h_new_ptr, h_old_ptr, gamma_ptr, gate_ptr, h_updated_ptr,
        batch_size, seq_len, model_dim, 0
    );
    
    return h_updated;
}

// ============================================================================
// Python Module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("core_recurrence_fused", &core_recurrence_fused_torch,
          "Core recurrence formula fused kernel (C++ optimized)");
    m.def("mdi_update_fused", &mdi_update_fused_torch,
          "MDI update fused kernel (C++ optimized with SIMD)");
}
