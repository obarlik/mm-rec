// Minimal debug test
#include "mm_rec/core/tensor.h"
#include <iostream>

using namespace mm_rec;

int main() {
    std::cout << "Test 1: Create tensor..." << std::endl;
    Tensor t = Tensor::zeros({2, 3});
    std::cout << "  ndim: " << t.ndim() << std::endl;
    std::cout << "  size(0): " << t.size(0) << std::endl;
    std::cout << "  size(1): " << t.size(1) << std::endl;
    
    std::cout << "\nTest 2: Return from function..." << std::endl;
    auto func = []() -> Tensor {
        return Tensor::zeros({4, 5});
    };
    
    Tensor t2 = func();
    std::cout << "  ndim: " << t2.ndim() << std::endl;
    std::cout << "  size(0): " << t2.size(0) << std::endl;
    
    std::cout << "\n✅ All tests passed!" << std::endl;
    return 0;
}
