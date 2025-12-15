/**
 * Test: GRU-Style Gated Memory
 * 
 * Validates:
 * 1. Correct dimensions
 * 2. Update/reset gate behavior
 * 3. Gradient flow (Bug #3 check)
 * 4. Numerical stability
 */

#include "mm_rec/core/gated_memory.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

using namespace mm_rec;

void test_dimensions() {
    std::cout << "[TEST] Dimension validation..." << std::endl;
    
    int64_t hidden_dim = 256;
    int64_t mem_dim = 512;
    int64_t batch_size = 4;
    
    auto gated_mem = GatedMemoryUpdate(hidden_dim, mem_dim);
    
    // Create inputs
    auto h_t = torch::randn({batch_size, hidden_dim});
    auto m_prev = torch::randn({batch_size, mem_dim});
    
    // Forward pass
    auto [m_t, z_t] = gated_mem.forward(h_t, m_prev);
    
    // Check dimensions
    assert(m_t.sizes() == torch::IntArrayRef({batch_size, mem_dim}));
    assert(z_t.sizes() == torch::IntArrayRef({batch_size, mem_dim}));
    
    std::cout << "✅ Dimensions correct" << std::endl;
    std::cout << "   m_t shape: [" << m_t.size(0) << ", " << m_t.size(1) << "]" << std::endl;
    std::cout << "   z_t shape: [" << z_t.size(0) << ", " << z_t.size(1) << "]" << std::endl;
}

void test_gradient_flow() {
    std::cout << "\n[TEST] Gradient flow (Bug #3 check)..." << std::endl;
    
    int64_t hidden_dim = 128;
    int64_t mem_dim = 128;
    int64_t batch_size = 2;
    
    auto gated_mem = GatedMemoryUpdate(hidden_dim, mem_dim);
    
    // Create inputs with gradient tracking
    auto h_t = torch::randn({batch_size, hidden_dim}, torch::requires_grad(true));
    auto m_prev = torch::randn({batch_size, mem_dim}, torch::requires_grad(true));
    
    // Forward pass
    auto [m_t, z_t] = gated_mem.forward(h_t, m_prev);
    
    // CRITICAL: m_t should require grad (Bug #3 check)
    assert(m_t.requires_grad());
    std::cout << "✅ Gradient enabled on m_t" << std::endl;
    
    // Compute dummy loss
    auto loss = m_t.sum();
    loss.backward();
    
    // Check gradients exist
    assert(h_t.grad().defined());
    assert(m_prev.grad().defined());
    
    std::cout << "✅ Gradients flow correctly" << std::endl;
    std::cout << "   h_t grad norm: " << h_t.grad().norm().template item<float>() << std::endl;
    std::cout << "   m_prev grad norm: " << m_prev.grad().norm().template item<float>() << std::endl;
}

void test_gate_behavior() {
    std::cout << "\n[TEST] Update gate behavior..." << std::endl;
    
    int64_t hidden_dim = 64;
    int64_t mem_dim = 64;
    int64_t batch_size = 1;
    
    auto gated_mem = GatedMemoryUpdate(hidden_dim, mem_dim);
    
    // Test 1: Update gate should be in [0, 1]
    auto h_t = torch::randn({batch_size, hidden_dim});
    auto m_prev = torch::randn({batch_size, mem_dim});
    
    auto [m_t, z_t] = gated_mem.forward(h_t, m_prev);
    
    // Check update gate range
    auto z_min = z_t.min().template item<float>();
    auto z_max = z_t.max().template item<float>();
    
    assert(z_min >= 0.0f && z_min <= 1.0f);
    assert(z_max >= 0.0f && z_max <= 1.0f);
    
    std::cout << "✅ Update gate in valid range [0,1]" << std::endl;
    std::cout << "   z_t range: [" << z_min << ", " << z_max << "]" << std::endl;
}

void test_numerical_stability() {
    std::cout << "\n[TEST] Numerical stability..." << std::endl;
    
    int64_t hidden_dim = 128;
    int64_t mem_dim = 128;
    int64_t batch_size = 4;
    
    auto gated_mem = GatedMemoryUpdate(hidden_dim, mem_dim);
    
    // Test with extreme values
    auto h_t = torch::randn({batch_size, hidden_dim}) * 100.0;  // Large values
    auto m_prev = torch::randn({batch_size, mem_dim}) * 100.0;
    
    auto [m_t, z_t] = gated_mem.forward(h_t, m_prev);
    
    // Check for NaN/Inf
    assert(!torch::isnan(m_t).any().template item<bool>());
    assert(!torch::isinf(m_t).any().template item<bool>());
    
    std::cout << "✅ No NaN/Inf with large inputs" << std::endl;
    std::cout << "   m_t stats: mean=" << m_t.mean().template item<float>() 
              << ", std=" << m_t.std().template item<float>() << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing GRU-Style Gated Memory" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_dimensions();
        test_gradient_flow();
        test_gate_behavior();
        test_numerical_stability();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "✅ ALL TESTS PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
