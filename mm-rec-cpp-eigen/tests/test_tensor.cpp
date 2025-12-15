/**
 * Test: Pure C++ Tensor Class
 */

#include "mm_rec/core/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace mm_rec;

void test_basic_ops() {
    std::cout << "[TEST] Basic tensor operations..." << std::endl;
    
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::ones({2, 3});
    
    auto c = a + b;
    assert(std::abs(c.data()[0] - 2.0f) < 1e-5);
    
    std::cout << "✅ Addition works" << std::endl;
}

void test_matmul() {
    std::cout << "\n[TEST] Matrix multiplication (MKL)..." << std::endl;
    
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::ones({3, 4});
    
    auto c = a.matmul(b);
    
    assert(c.sizes()[0] == 2);
    assert(c.sizes()[1] == 4);
    assert(std::abs(c.data()[0] - 3.0f) < 1e-5);
    
    std::cout << "✅ Matmul works" << std::endl;
    std::cout << "   Result shape: [" << c.size(0) << ", " << c.size(1) << "]" << std::endl;
    std::cout << "   Result[0,0]: " << c.data()[0] << " (expected: 3.0)" << std::endl;
}

void test_sigmoid() {
    std::cout << "\n[TEST] Sigmoid activation..." << std::endl;
    
    auto x = Tensor::zeros({2, 3});
    auto y = x.sigmoid();
    
    assert(std::abs(y.data()[0] - 0.5f) < 1e-5);
    
    std::cout << "✅ Sigmoid works" << std::endl;
    std::cout << "   sigmoid(0) = " << y.data()[0] << " (expected: 0.5)" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing Pure C++ Tensor Class" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_basic_ops();
        test_matmul();
        test_sigmoid();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "✅ ALL TENSOR TESTS PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
