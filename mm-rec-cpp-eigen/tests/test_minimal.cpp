// Minimal test - exactly what test_model does
#include "mm_rec/core/tensor.h"
#include <iostream>

using namespace mm_rec;

int main() {
    std::cout << "Creating tensor [2, 8]..." << std::endl;
    Tensor input_ids = Tensor::zeros({2, 8});
    
    std::cout << "ndim: " << input_ids.ndim() << std::endl;
    std::cout << "numel: " << input_ids.numel() << std::endl;
    
    std::cout << "Calling size(0)..." << std::endl;
    int64_t batch = input_ids.size(0);
    std::cout << "batch: " << batch << std::endl;
    
    std::cout << "✅ Test passed!" << std::endl;
    return 0;
}
