#include "mm_rec/core/linear.h"
#include <iostream>

using namespace mm_rec;

int main() {
    try {
        std::cout << "Testing Zero Batch..." << std::endl;
        Linear layer(128, 64);
        Tensor empty_input = Tensor::zeros({0, 128});
        Tensor out = layer.forward(empty_input);
        
        std::cout << "Output shape: [" << out.size(0) << ", " << out.size(1) << "]" << std::endl;
        if (out.size(0) != 0 || out.size(1) != 64) {
            throw std::runtime_error("Wrong output shape");
        }
        std::cout << "✅ Pass" << std::endl;
        
        std::cout << "Testing Small Batch (1)..." << std::endl;
        Tensor small = Tensor::randn({1, 128});
        out = layer.forward(small);
        std::cout << "Output shape: [" << out.size(0) << ", " << out.size(1) << "]" << std::endl;
        if (out.size(0) != 1) throw std::runtime_error("Wrong output shape");
        std::cout << "✅ Pass" << std::endl;

    } catch (std::exception& e) {
        std::cerr << "❌ Crash: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
