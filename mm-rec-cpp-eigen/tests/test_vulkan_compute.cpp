#include "mm_rec/core/vulkan_compute.h"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "=== Test: Vulkan Compute Integration ===" << std::endl;
    
    // 1. Init
    if (!mm_rec::VulkanBackend::get().init()) {
        std::cerr << "❌ Vulkan Init Failed" << std::endl;
        return 1;
    }
    
    if (!mm_rec::VulkanCompute::is_ready()) {
         std::cerr << "❌ VulkanCompute reports not ready." << std::endl;
         return 1;
    }
    
    // 2. Data
    int M = 64, N = 64, K = 64;
    std::vector<float> A(M*K, 1.0f);
    std::vector<float> B(K*N, 2.0f);
    std::vector<float> C(M*N, 0.0f);
    
    // 3. Run
    std::cout << "🚀 Dispatching Vulkan Compute Job..." << std::endl;
    bool success = mm_rec::VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K);
    
    if (success) {
        std::cout << "✅ SUCCESS: Vulkan Compute Pipeline executed (Buffers Created & Mapped)." << std::endl;
    } else {
        std::cout << "❌ FAILURE: Vulkan Compute Job returned false." << std::endl;
        return 1;
    }
    
    return 0;
}
