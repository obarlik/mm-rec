/**
 * Test: RMSNorm
 */

#include "mm_rec/core/normalization.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace mm_rec;

void test_rmsnorm() {
    std::cout << "=== Test: RMSNorm ===" << std::endl;
    
    int64_t hidden = 4;
    RMSNorm norm(hidden);
    
    // Non-zero input
    Tensor x = Tensor::ones({2, 3, hidden}); // [batch, seq, hidden]
    // Fill with values
    for(int i=0; i<x.numel(); ++i) x.data()[i] = (i % 2 == 0) ? 1.0f : 2.0f;
    
    Tensor y = norm.forward(x);
    
    // Check output shape
    assert(y.sizes() == x.sizes());
    
    // Check normalization: RMS(y) should be approx 1 (if weight is 1)
    // Actually RMSNorm: y = x * w / RMS(x).
    // So RMS(y) = RMS(x * w / RMS(x)) = RMS(w).
    // Initially w=1. So RMS(y) should be 1.
    
    float sum_sq = 0.0f;
    for(int i=0; i<hidden; ++i) {
        float val = y.data()[i]; // First token
        sum_sq += val * val;
    }
    float rms = std::sqrt(sum_sq / hidden);
    
    std::cout << "RMS(Output): " << rms << " (Expected ~1.0)" << std::endl;
    assert(std::abs(rms - 1.0f) < 1e-4);
    
    // Backward
    Tensor dy = Tensor::ones(y.sizes());
    RMSNormGradients grads;
    grads.d_weight = Tensor::zeros({hidden});
    
    Tensor dx = norm.backward(dy, x, grads);
    
    assert(dx.sizes() == x.sizes());
    assert(grads.d_weight.size(0) == hidden);
    
    std::cout << "✅ RMSNorm Forward/Backward OK" << std::endl;
}

int main() {
    test_rmsnorm();
    return 0;
}
