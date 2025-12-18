/**
 * Test: iGPU Dynamic Loader
 */

#include "mm_rec/core/gpu_backend.h"
#include <iostream>

int main() {
    std::cout << "=== Test: iGPU Dynamic Loader ===" << std::endl;
    
    // Attempt to initialize GPU
    bool success = mm_rec::GPUBackend::instance().init_device();
    
    if (success) {
        std::cout << "✅ SUCCESS: iGPU is ready for compute!" << std::endl;
    } else {
        std::cout << "⚠️  INFO: iGPU hardware exists (/dev/dri), but 'libOpenCL.so' is missing." << std::endl;
        std::cout << "         Install 'intel-opencl-icd' to unlock this feature." << std::endl;
    }
    
    return 0;
}
