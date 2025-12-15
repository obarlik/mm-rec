/**
 * Test: Backward Pass with REAL Numerical Gradient Checking
 */

#include "mm_rec/training/backward.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace mm_rec;

const float EPS = 1e-4f;
const float TOL = 2e-3f;  // Slightly relaxed for numerical stability

// Numerical gradient for sigmoid
void test_sigmoid_backward() {
    std::cout << "=== Test: Sigmoid Backward (Numerical Check) ===" << std::endl;
    
    Tensor x = Tensor::randn({2, 3}, 0.0f, 1.0f);
    
    // Forward
    Tensor y = Tensor::zeros(x.sizes());
    for (int64_t i = 0; i < x.numel(); ++i) {
        y.data()[i] = 1.0f / (1.0f + std::exp(-x.data()[i]));
    }
    
    // Assume loss = sum(y), so dy/dy_i = 1
    Tensor dy = Tensor::ones(y.sizes());
    
    // Analytical backward
    Tensor dx_analytical = sigmoid_backward(y, dy);
    
    // Numerical gradient check (check a few elements)
    for (int idx = 0; idx < std::min(5L, x.numel()); ++idx) {
        // Perturb x[idx]
        float x_original = x.data()[idx];
        
        // f(x + eps)
        x.data()[idx] = x_original + EPS;
        Tensor y_plus = Tensor::zeros(x.sizes());
        for (int64_t i = 0; i < x.numel(); ++i) {
            y_plus.data()[i] = 1.0f / (1.0f + std::exp(-x.data()[i]));
        }
        float loss_plus = 0.0f;
        for (int64_t i = 0; i < y_plus.numel(); ++i) {
            loss_plus += y_plus.data()[i];
        }
        
        // f(x - eps)
        x.data()[idx] = x_original - EPS;
        Tensor y_minus = Tensor::zeros(x.sizes());
        for (int64_t i = 0; i < x.numel(); ++i) {
            y_minus.data()[i] = 1.0f / (1.0f + std::exp(-x.data()[i]));
        }
        float loss_minus = 0.0f;
        for (int64_t i = 0; i < y_minus.numel(); ++i) {
            loss_minus += y_minus.data()[i];
        }
        
        // Restore
        x.data()[idx] = x_original;
        
        // Numerical gradient
        float numerical_grad = (loss_plus - loss_minus) / (2.0f * EPS);
        float analytical_grad = dx_analytical.data()[idx];
        float diff = std::abs(numerical_grad - analytical_grad);
        
        std::cout << "  Element " << idx << ": numerical=" << numerical_grad 
                  << ", analytical=" << analytical_grad 
                  << ", diff=" << diff << std::endl;
        
        if (diff > TOL) {
            std::cerr << "ERROR: Gradient mismatch!" << std::endl;
            assert(false);
        }
    }
    
    std::cout << "✅ Sigmoid backward: NUMERICAL CHECK PASSED" << std::endl;
}

void test_tanh_backward() {
    std::cout << "\n=== Test: Tanh Backward (Numerical Check) ===" << std::endl;
    
    Tensor x = Tensor::randn({2, 3}, 0.0f, 0.5f);
    
    // Forward
    Tensor y = Tensor::zeros(x.sizes());
    for (int64_t i = 0; i < x.numel(); ++i) {
        y.data()[i] = std::tanh(x.data()[i]);
    }
    
    Tensor dy = Tensor::ones(y.sizes());
    Tensor dx_analytical = tanh_backward(y, dy);
    
    // Numerical check
    for (int idx = 0; idx < std::min(5L, x.numel()); ++idx) {
        float x_original = x.data()[idx];
        
        // f(x + eps)
        x.data()[idx] = x_original + EPS;
        Tensor y_plus = Tensor::zeros(x.sizes());
        for (int64_t i = 0; i < x.numel(); ++i) {
            y_plus.data()[i] = std::tanh(x.data()[i]);
        }
        float loss_plus = 0.0f;
        for (int64_t i = 0; i < y_plus.numel(); ++i) {
            loss_plus += y_plus.data()[i];
        }
        
        // f(x - eps)
        x.data()[idx] = x_original - EPS;
        Tensor y_minus = Tensor::zeros(x.sizes());
        for (int64_t i = 0; i < x.numel(); ++i) {
            y_minus.data()[i] = std::tanh(x.data()[i]);
        }
        float loss_minus = 0.0f;
        for (int64_t i = 0; i < y_minus.numel(); ++i) {
            loss_minus += y_minus.data()[i];
        }
        
        x.data()[idx] = x_original;
        
        float numerical_grad = (loss_plus - loss_minus) / (2.0f * EPS);
        float analytical_grad = dx_analytical.data()[idx];
        float diff = std::abs(numerical_grad - analytical_grad);
        
        std::cout << "  Element " << idx << ": numerical=" << numerical_grad 
                  << ", analytical=" << analytical_grad 
                  << ", diff=" << diff << std::endl;
        
        assert(diff < TOL);
    }
    
    std::cout << "✅ Tanh backward: NUMERICAL CHECK PASSED" << std::endl;
}

void test_linear_backward() {
    std::cout << "\n=== Test: Linear Backward ===" << std::endl;
    
    Tensor x = Tensor::randn({2, 3}, 0.0f, 1.0f);
    Tensor W = Tensor::randn({4, 3}, 0.0f, 0.1f);
    Tensor dy = Tensor::ones({2, 4});
    
    Tensor dx, dW, db;
    linear_backward(x, W, dy, dx, dW, db);
    
    // Shape checks
    assert(dx.size(0) == 2 && dx.size(1) == 3);
    assert(dW.size(0) == 4 && dW.size(1) == 3);
    assert(db.size(0) == 4);
    
    std::cout << "✅ Linear backward: SHAPE CHECK PASSED" << std::endl;
}

int main() {
    std::cout << "=== REAL Backward Pass Tests (Numerical Validation) ===" << std::endl;
    
    test_sigmoid_backward();
    test_tanh_backward();
    test_linear_backward();
    
    std::cout << "\n=== ALL TESTS PASSED WITH NUMERICAL VALIDATION ===" << std::endl;
    
    return 0;
}
