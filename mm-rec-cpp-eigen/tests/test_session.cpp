/**
 * Test: Session Persistence
 */

#include "mm_rec/application/session.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace mm_rec;

int main() {
    std::cout << "=== Session Persistence Test ===" << std::endl;
    
    // Create dummy memory states (3 layers)
    std::vector<Tensor> original_states;
    
    // Layer 0: [2, 64]
    Tensor mem0 = Tensor::zeros({2, 64});
    for (int i = 0; i < 128; ++i) {
        mem0.data()[i] = static_cast<float>(i) * 0.01f;
    }
    original_states.push_back(mem0);
    
    // Layer 1: [2, 64]
    Tensor mem1 = Tensor::zeros({2, 64});
    for (int i = 0; i < 128; ++i) {
        mem1.data()[i] = static_cast<float>(i) * 0.02f;
    }
    original_states.push_back(mem1);
    
    // Layer 2: [2, 64]
    Tensor mem2 = Tensor::zeros({2, 64});
    for (int i = 0; i < 128; ++i) {
        mem2.data()[i] = static_cast<float>(i) * 0.03f;
    }
    original_states.push_back(mem2);
    
    std::cout << "Original states created: " << original_states.size() << " layers" << std::endl;
    
    // Save session
    std::string session_file = "/tmp/test_session.mmrs";
    std::cout << "\nSaving session to: " << session_file << std::endl;
    
    try {
        SessionManager::save_session(session_file, original_states);
        std::cout << "✅ Session saved" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Save failed: " << e.what() << std::endl;
        return 1;
    }
    
    // Load session
    std::cout << "\nLoading session..." << std::endl;
    std::vector<Tensor> loaded_states;
    
    try {
        loaded_states = SessionManager::load_session(session_file);
        std::cout << "✅ Session loaded: " << loaded_states.size() << " layers" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Load failed: " << e.what() << std::endl;
        return 1;
    }
    
    // Verify
    std::cout << "\nVerifying data..." << std::endl;
    
    assert(loaded_states.size() == original_states.size());
    
    for (size_t layer = 0; layer < original_states.size(); ++layer) {
        const auto& orig = original_states[layer];
        const auto& loaded = loaded_states[layer];
        
        // Check shape
        assert(orig.ndim() == loaded.ndim());
        assert(orig.numel() == loaded.numel());
        
        // Check data
        for (int64_t i = 0; i < orig.numel(); ++i) {
            float diff = std::abs(orig.data()[i] - loaded.data()[i]);
            assert(diff < 1e-6f);
        }
        
        std::cout << "  Layer " << layer << ": ✅ Shape and data match" << std::endl;
    }
    
    // Cleanup
    std::remove(session_file.c_str());
    
    std::cout << "\n=== ALL SESSION TESTS PASSED ===" << std::endl;
    
    return 0;
}
