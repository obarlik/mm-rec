/**
 * Test: Gradient Clipping and LR Scheduler
 */

#include "mm_rec/training/gradient_utils.h"
#include "mm_rec/training/scheduler.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace mm_rec;

void test_gradient_clipping() {
    std::cout << "=== Test: Gradient Clipping ===" << std::endl;
    
    // Create some dummy gradients
    std::vector<Tensor> gradients;
    
    // Grad 1: [10] with values 0.1
    Tensor grad1 = Tensor::zeros({10});
    for (int i = 0; i < 10; ++i) {
        grad1.data()[i] = 0.1f;
    }
    gradients.push_back(grad1);
    
    // Grad 2: [10] with values 0.2
    Tensor grad2 = Tensor::zeros({10});
    for (int i = 0; i < 10; ++i) {
        grad2.data()[i] = 0.2f;
    }
    gradients.push_back(grad2);
    
    // Global norm = sqrt(10*0.01 + 10*0.04) = sqrt(0.5) = 0.707
    float expected_norm = std::sqrt(10 * 0.01f + 10 * 0.04f);
    
    std::cout << "Expected global norm: " << expected_norm << std::endl;
    
    // Clip with max_norm = 0.5 (should trigger clipping)
    float actual_norm = clip_gradients_by_norm(gradients, 0.5f);
    
    std::cout << "Actual global norm: " << actual_norm << std::endl;
    assert(std::abs(actual_norm - expected_norm) < 1e-4f);
    
    // Check that gradients were scaled
    float scale = 0.5f / expected_norm;
    for (int i = 0; i < 10; ++i) {
        assert(std::abs(gradients[0].data()[i] - 0.1f * scale) < 1e-4f);
        assert(std::abs(gradients[1].data()[i] - 0.2f * scale) < 1e-4f);
    }
    
    std::cout << "✅ Gradient clipping works" << std::endl;
}

void test_lr_scheduler() {
    std::cout << "\n=== Test: LR Scheduler ===" << std::endl;
    
    // Create scheduler: warmup=10, total=100, base_lr=1e-3, min_lr=1e-5
    CosineScheduler scheduler(1e-3f, 10, 100, 1e-5f);
    
    // Test warmup (step 5 should be 50% of base_lr)
    float lr_step5 = scheduler.get_lr(5);
    std::cout << "LR at step 5 (warmup): " << lr_step5 << std::endl;
    assert(std::abs(lr_step5 - 0.0005f) < 1e-6f);
    
    // Test end of warmup (step 10 should be base_lr)
    float lr_step10 = scheduler.get_lr(10);
    std::cout << "LR at step 10 (warmup end): " << lr_step10 << std::endl;
    assert(std::abs(lr_step10 - 0.001f) < 1e-6f);
    
    // Test mid-training (step 55 should be cosine midpoint)
    float lr_step55 = scheduler.get_lr(55);
    std::cout << "LR at step 55 (mid): " << lr_step55 << std::endl;
    // At progress=0.5, cosine=0, so lr = min_lr + 0.5*(base-min) = 5.05e-4
    float expected_mid = 1e-5f + 0.5f * (1e-3f - 1e-5f);
    assert(std::abs(lr_step55 - expected_mid) < 1e-5f);
    
    // Test end (step 100 should approach min_lr)
    float lr_step100 = scheduler.get_lr(100);
    std::cout << "LR at step 100 (end): " << lr_step100 << std::endl;
    assert(lr_step100 < 2e-5f);  // Close to min_lr
    
    std::cout << "✅ LR scheduler works" << std::endl;
}

int main() {
    std::cout << "=== Training Utilities Test ===" << std::endl;
    
    test_gradient_clipping();
    test_lr_scheduler();
    
    std::cout << "\n=== ALL TRAINING UTILITIES TESTS PASSED ===" << std::endl;
    
    return 0;
}
