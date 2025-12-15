/**
 * Test: Error Handling
 */

#include "mm_rec/core/tensor.h"
#include "mm_rec/utils/error_handling.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace mm_rec;

int main() {
    std::cout << "=== Error Handling Test ===" << std::endl;
    
    // Test 1: Bounds checking
    std::cout << "\n[TEST 1] Bounds checking..." << std::endl;
    
    Tensor t = Tensor::zeros({3, 4});
    
    // Valid access
    t[0] = 1.0f;
    std::cout << "  Valid access: OK" << std::endl;
    
    // Invalid access (should throw)
    bool caught_bounds_error = false;
    try {
        float x = t[100];  // Out of bounds
    } catch (const std::out_of_range& e) {
        caught_bounds_error = true;
        std::cout << "  ✅ Caught bounds error: " << e.what() << std::endl;
    }
    assert(caught_bounds_error);
    
    // Test 2: NaN detection
    std::cout << "\n[TEST 2] NaN detection..." << std::endl;
    
    Tensor nan_tensor = Tensor::zeros({2, 2});
    nan_tensor.data()[0] = NAN;
    
    bool caught_nan_error = false;
    try {
        nan_tensor.check_valid("nan_test");
    } catch (const std::runtime_error& e) {
        caught_nan_error = true;
        std::cout << "  ✅ Caught NaN error: " << e.what() << std::endl;
    }
    assert(caught_nan_error);
    
    // Test 3: Inf detection
    std::cout << "\n[TEST 3] Inf detection..." << std::endl;
    
    Tensor inf_tensor = Tensor::zeros({2, 2});
    inf_tensor.data()[0] = INFINITY;
    
    bool caught_inf_error = false;
    try {
        inf_tensor.check_valid("inf_test");
    } catch (const std::runtime_error& e) {
        caught_inf_error = true;
        std::cout << "  ✅ Caught Inf error: " << e.what() << std::endl;
    }
    assert(caught_inf_error);
    
    // Test 4: Shape mismatch
    std::cout << "\n[TEST 4] Shape mismatch detection..." << std::endl;
    
    bool caught_shape_error = false;
    try {
        check_shape_compatible(3, 5, "test_op");
    } catch (const std::runtime_error& e) {
        caught_shape_error = true;
        std::cout << "  ✅ Caught shape error: " << e.what() << std::endl;
    }
    assert(caught_shape_error);
    
    // Test 5: Valid tensor passes checks
    std::cout << "\n[TEST 5] Valid tensor passes..." << std::endl;
    
    Tensor valid_tensor = Tensor::zeros({3, 3});
    for (int i = 0; i < 9; ++i) {
        valid_tensor.data()[i] = static_cast<float>(i) * 0.1f;
    }
    
    try {
        valid_tensor.check_valid("valid_test");
        std::cout << "  ✅ Valid tensor passed checks" << std::endl;
    } catch (...) {
        std::cout << "  ❌ Valid tensor should not throw!" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== ALL ERROR HANDLING TESTS PASSED ===" << std::endl;
    
    return 0;
}
