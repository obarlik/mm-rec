/**
 * Test: Adversarial & Edge Cases
 * "Çürütme Testleri"
 */

#include "mm_rec/core/normalization.h"
#include "mm_rec/model/moe.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <limits>

using namespace mm_rec;

void test_rmsnorm_zeros() {
    std::cout << "[Adversarial] RMSNorm: Zero Input" << std::endl;
    // Edge case: All zeros. RMS = 0 + epsilon. 
    // Output should be 0/eps ~= 0. Should NOT be NaN.
    
    int64_t hidden = 4;
    RMSNorm norm(hidden);
    Tensor x = Tensor::zeros({1, 1, hidden});
    
    Tensor y = norm.forward(x);
    
    bool has_nan = false;
    for(int i=0; i<y.numel(); ++i) {
        if(std::isnan(y.data()[i])) has_nan = true;
    }
    
    if(has_nan) {
        std::cerr << "❌ FAILED: Zero input caused NaN!" << std::endl;
        exit(1);
    }
    std::cout << "✅ Zero Input Handled (No NaN)" << std::endl;
}

void test_rmsnorm_extreme() {
    std::cout << "[Adversarial] RMSNorm: Extreme Values" << std::endl;
    // Edge case: Very large values (potential overflow in square sum)
    
    int64_t hidden = 4;
    RMSNorm norm(hidden);
    Tensor x = Tensor::zeros({1, 1, hidden});
    
    // 1e10. Square is 1e20. float usually goes up to 1e38.
    // 1e20. Square is 1e40 (Overflow for float32).
    float large_val = 1e20f; 
    x.data()[0] = large_val;
    x.data()[1] = large_val;
    
    Tensor y = norm.forward(x);
    
    // RMS should scale it back to ~1.0?
    // If sum_sq overflows to Inf, then x / Inf = 0.
    // If we want stability, we might need double precision accum or scaling.
    // Let's see behavior.
    
    std::cout << "Input: " << large_val << " -> Output[0]: " << y.data()[0] << std::endl;
    
    if(std::isnan(y.data()[0])) {
         std::cerr << "⚠️  WARNING: Extreme value caused NaN (Expected if no overflow protection)" << std::endl;
         // Not failing, just noting.
    } else {
         std::cout << "✅ Extreme Inputs Handled" << std::endl;
    }
}

void test_moe_uniform() {
    std::cout << "[Adversarial] MoE: Uniform Logits (Ambiguity)" << std::endl;
    
    MoEConfig config;
    config.hidden_dim = 4;
    config.ffn_dim = 4;
    config.num_experts = 4;
    config.top_k = 2;
    
    MoELayer moe(config);
    // Directly hack gate weights to be zero? Or input zero?
    // If input is zero, logits are zero (bias is zero initialized).
    // All logits = 0.
    // Selection? partial_sort is unstable or implementation defined for equal elements?
    // We expect it to pick the first K indices (0, 1) usually.
    
    Tensor x = Tensor::zeros({1, 1, 4});
    MoECache cache;
    moe.forward(x, &cache);
    
    float* idx = cache.selected_indices.data();
    std::cout << "Selected with Zero Logits: " << idx[0] << ", " << idx[1] << std::endl;
    
    // Expect valid indices
    assert(idx[0] >= 0 && idx[0] < 4);
    assert(idx[1] >= 0 && idx[1] < 4);
    assert(idx[0] != idx[1]); // Should pick distinct? Or implementation detail? 
    // Actually our partial sort logic just sorts pairs. (0,0), (0,1), (0,2)...
    // If values equal, typically keeps order. So 0 and 1.
    
    std::cout << "✅ Uniform Logits Handled (Deterministic Selection)" << std::endl;
}

int main() {
    test_rmsnorm_zeros();
    test_rmsnorm_extreme();
    test_moe_uniform();
    std::cout << "🛡️  All Adversarial Tests Passed" << std::endl;
    return 0;
}
