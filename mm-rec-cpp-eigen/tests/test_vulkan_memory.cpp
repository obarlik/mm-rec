/**
 * Test: Vulkan Memory Allocation
 * 
 * Verifies we can:
 * 1. Create a logical device.
 * 2. Create a buffer (100MB).
 * 3. Allocate specialized graphics memory.
 * 4. Bind them together.
 * 
 * This proves we "Own" the memory on the GPU.
 */

#include "mm_rec/core/vulkan_backend.h"
#include <iostream>

int main() {
    std::cout << "=== Test: Vulkan GPU Memory ===" << std::endl;
    
    auto& backend = mm_rec::VulkanBackend::get();
    if (!backend.init()) {
         std::cout << "❌ Init failed." << std::endl;
         return 1;
    }

    std::cout << "Attempting to allocate 128MB on iGPU..." << std::endl;
    
    // Minimal Types (Forward declared in header, we need to treat them as opaque in main or expose them?
    // Actually, helper 'create_buffer' uses internal types. We just pass size.
    // But create_buffer returns references to opaque pointers.
    
    // We need to define the handle types here to hold the result, 
    // OR just trust the internal state test?
    // Implementation uses references to internal typdefs.
    // Let's modify the test to strictly use the public API if possible.
    // Ah, create_buffer signature exposes VkBuffer. We need those types visible.
    // They are visible in the header.
    
    mm_rec::VkBuffer buffer;
    mm_rec::VkDeviceMemory memory;
    
    size_t size = 128 * 1024 * 1024; // 128 MB
    
    bool success = backend.create_buffer(size, buffer, memory);
    
    if (success) {
        std::cout << "✅ SUCCESS: Allocated 128MB resident on GPU." << std::endl;
        std::cout << "   Buffer Handle: " << buffer << std::endl;
        std::cout << "   Memory Handle: " << memory << std::endl;
        std::cout << "   GPU is ready for data." << std::endl;
    } else {
        std::cout << "❌ FAILURE: Could not allocate memory." << std::endl;
        return 1;
    }
    
    return 0;
}
