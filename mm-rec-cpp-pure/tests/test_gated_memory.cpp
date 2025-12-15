/**
 * Test: GRU-Style Gated Memory (Pure C++)
 */

#include "mm_rec/core/gated_memory.h"
#include <iostream>
#include <cassert>

using namespace mm_rec;

void test_dimensions() {
    std::cout << "[TEST] Dimension validation..." << std::endl;
    
    int64_t hidden_dim = 256;
    int64_t mem_dim = 512;
    int64_t batch_size = 4;
    
    GatedMemoryUpdate gated_mem(hidden_dim, mem_dim);
    
    auto h_t = Tensor::randn({batch_size, hidden_dim});
    auto m_prev = Tensor::randn({batch_size, mem_dim});
    
    auto [m_t, z_t] = gated_mem.forward(h_t, m_prev);
    
    assert(m_t.sizes()[0] == batch_size);
    assert(m_t.sizes()[1] == mem_dim);
    assert(z_t.sizes()[0] == batch_size);
    assert(z_t.sizes()[1] == mem_dim);
    
    std::cout << "✅ Dimensions correct" << std::endl;
    std::cout << "   m_t shape: [" << m_t.size(0) << ", " << m_t.size(1) << "]" << std::endl;
}

void test_gate_behavior() {
    std::cout << "\n[TEST] Update gate behavior..." << std::endl;
    
    int64_t hidden_dim = 64;
    int64_t mem_dim = 64;
    int64_t batch_size = 1;
    
    GatedMemoryUpdate gated_mem(hidden_dim, mem_dim);
    
    auto h_t = Tensor::randn({batch_size, hidden_dim});
    auto m_prev = Tensor::randn({batch_size, mem_dim});
    
    auto [m_t, z_t] = gated_mem.forward(h_t, m_prev);
    
    // Check update gate range [0, 1]
    const float* z_data = z_t.data();
    for (int64_t i = 0; i < z_t.numel(); ++i) {
        assert(z_data[i] >= 0.0f && z_data[i] <= 1.0f);
    }
    
    std::cout << "✅ Update gates in valid range [0,1]" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing GRU-Style Gated Memory (Pure C++)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_dimensions();
        test_gate_behavior();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "✅ ALL TESTS PASSED (NO LibTorch!)" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
