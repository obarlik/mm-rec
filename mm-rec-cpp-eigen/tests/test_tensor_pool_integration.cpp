#include "mm_rec/core/tensor.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>

using namespace mm_rec;

void test_tensor_lifecycle(int iterations) {
    std::cout << "[Tensor] Creating/Destroying " << iterations << " tensors..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simulate training loop behavior: Create temps, destroy them.
    for (int i = 0; i < iterations; ++i) {
        // Create small tensor (should go to pool)
        Tensor t = Tensor::zeros({4, 4}); // 16 floats = 64 bytes
        t.data()[0] = 1.0f;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "[Tensor] Time: " << diff.count() << " seconds (" 
              << (iterations / diff.count()) << " ops/sec)" << std::endl;
}

void test_tensor_large(int iterations) {
    std::cout << "[Tensor] Large Tensor (Malloc/Free) " << iterations << " iterations..." << std::endl;
    for (int i = 0; i < iterations; ++i) {
        // 1MB Tensor
        Tensor t = Tensor::zeros({1024, 256}); 
        t.data()[0] = 1.0f;
    } 
    std::cout << "[Tensor] Large Tensor OK." << std::endl;
}

int main() {
    std::cout << "=== Tensor Pool Integration Test ===" << std::endl;
    
    test_tensor_lifecycle(1000000);
    test_tensor_large(1000);
    
    std::cout << "PASS" << std::endl;
    return 0;
}
