#include "mm_rec/core/vulkan_backend.h"
#include <iostream>

int main() {
    std::cout << "=== Test: Vulkan Backend Integration ===" << std::endl;
    
    bool ready = mm_rec::VulkanBackend::get().init();
    
    if (ready) {
        std::cout << "✅ SUCCESS: Vulkan engine initialized inside the library." << std::endl;
        std::cout << "   The Foundation for iGPU Matrix Ops is laid." << std::endl;
    } else {
        std::cout << "❌ FAILURE: Context creation failed." << std::endl;
    }
    
    return 0;
}
