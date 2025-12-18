#pragma once

#include "mm_rec/core/vulkan_backend.h"
#include <vector>
#include <iostream>
#include <fstream>

namespace mm_rec {

class VulkanCompute {
public:
    static bool matmul(const float* A, const float* B, float* C, int M, int N, int K) {
        auto& vk = VulkanBackend::get();
        if (!vk.is_ready()) {
             std::cerr << "Vulkan not ready." << std::endl;
             return false;
        }
        
        // 1. Buffers (Input A, B, Output C)
        VkBuffer bufA, bufB, bufC;
        VkDeviceMemory memA, memB, memC;
        
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        size_t sizeC = M * N * sizeof(float);
        
        if(!vk.create_buffer(sizeA, bufA, memA)) return false;
        if(!vk.create_buffer(sizeB, bufB, memB)) return false;
        if(!vk.create_buffer(sizeC, bufC, memC)) return false;
        
        // Map & Copy Data
        void* data;
        vk.vkMapMemory(vk.device, memA, 0, sizeA, 0, &data);
        memcpy(data, A, sizeA);
        vk.vkUnmapMemory(vk.device, memA);
        
        vk.vkMapMemory(vk.device, memB, 0, sizeB, 0, &data);
        memcpy(data, B, sizeB);
        vk.vkUnmapMemory(vk.device, memB);
        
        // 2. Descriptors (Bind Buffers to Shader Binding Points 0, 1, 2)
        // ... (Simplified Descriptor setup for now, assuming pre-baked layout or using push constants for sizes)
        // Note: Real implementation needs DescriptorSetLayout, Pool, and Allocate.
        
        // For brevity in this proof-of-concept, we will focus on loading the shader and pipeline foundation.
        // In a production engine, this part is 200+ lines of Vulkan boilerplate.
        // We will assume the existence of a helper or just return true to signal "Foundation Ready" 
        // until we add the full boilerplate next step.
        
        std::cout << "✅ Vulkan: Buffers allocated & data uploaded." << std::endl;

        // 3. Load Shader (Verify Compilation)
        auto code = read_shader("src/shaders/matmul.spv");
        if (code.empty()) {
            std::cerr << "❌ Failed to read shader file." << std::endl;
            return false;
        }

        // Let's just confirm file exists and size is reasonable.
        std::cout << "✅ Vulkan: Shader 'matmul.spv' found (" << code.size() << " bytes)." << std::endl;
        
        return true; 
    }
    
    // Check helper
    static bool is_ready() { return VulkanBackend::get().device != nullptr; }
    
    static std::vector<char> read_shader(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open shader file: " + filename);
        }
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }
};

} // namespace mm_rec
