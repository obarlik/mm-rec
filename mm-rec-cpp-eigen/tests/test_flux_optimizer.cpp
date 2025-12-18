/**
 * Test: Flux Optimizer Verification
 * 
 * Verifies:
 * 1. Vectorized compilation and execution.
 * 2. Flux Scaling behavior.
 * 3. Safety Brake activation and telemetry.
 */

#include "mm_rec/training/optimizer.h"
#include "mm_rec/core/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace mm_rec;

const float EPSILON = 1e-6f;

void test_flux_scaling() {
    std::cout << "[Test 1] Flux Scaling..." << std::endl;
    
    // Create params and grads
    Tensor param = Tensor::ones({10}); // All 1.0
    Tensor grad = Tensor::ones({10});  // All 1.0
    
    Flux result_opt(0.1f); // LR = 0.1
    result_opt.set_flux_scale(2.0f); // Boost to 2.0x (LR effectively 0.2)
    
    // Step 1
    result_opt.step(param, grad);
    
    // We expect aggressive update.
    // m = 0.1, v = 0.001 (unbias -> larger)
    // Roughly Adam step is ~1.0? 
    // With Flux 2.0, it should be doubled.
    
    std::cout << "  Param[0] after step: " << param.data()[0] << std::endl;
    // Just ensure it ran and changed
    assert(param.data()[0] < 1.0f);
    
    std::cout << "  Passed." << std::endl;
}

void test_safety_brake() {
    std::cout << "[Test 2] Safety Brake (Adaptive Clipping)..." << std::endl;
    
    // Scenario: Small Weight, Huge Gradient -> Explosion
    Tensor param = Tensor::zeros({10});
    // Set one weight to be small but non-zero
    param.data()[0] = 0.1f;
    
    // Huge gradient -> Huge Adam update
    Tensor grad = Tensor::zeros({10});
    grad.data()[0] = 1000.0f; 
    
    // Flux opt(lr=1.0, beta1, beta2, eps, weight_decay=0.0)
    Flux opt(1.0f, 0.9f, 0.999f, 1e-8f, 0.0f); 
    opt.reset_brake_stats();
    
    opt.step(param, grad);
    
    // Expected behavior:
    // Threshold = max(0.01, 0.1 * 0.5) = 0.05
    // Update would be huge (~1.0)
    // Should be clamped to 0.05 change.
    
    float new_val = param.data()[0];
    float diff = std::abs(new_val - 0.1f);
    
    std::cout << "  Original: 0.1, New: " << new_val << ", Diff: " << diff << std::endl;
    
    // Check if stats recorded
    uint64_t brakes = opt.get_brake_stats();
    std::cout << "  Brakes triggered: " << brakes << std::endl;
    
    assert(brakes >= 1);
    // Tolerance for float math
    assert(diff <= 0.05f + 1e-5f);
    assert(diff >= 0.05f - 1e-5f);
    
    std::cout << "  Passed." << std::endl;
}

int main() {
    std::cout << "=== Flux Optimizer Test Suite ===" << std::endl;
    
    test_flux_scaling();
    test_safety_brake();
    
    std::cout << "\n✅ All Tests Passed." << std::endl;
    return 0;
}
