#pragma once

#include "mm_rec/core/vulkan_backend.h"
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <cstring>
#include <map>
#include <chrono> // Added logging

namespace mm_rec {

// Caches Pipelines to avoid recompilation
class PipelineCache {
public:
    static VkPipeline get_pipeline(VkDevice device, const std::string& shaderPath, VkPipelineLayout layout) {
        static std::map<std::string, VkPipeline> cache;
        if (cache.find(shaderPath) != cache.end()) return cache[shaderPath];
        
        // Load & Compile
        auto code = read_shader(shaderPath);
        VkShaderModuleCreateInfo shaderInfo = {};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = code.size();
        shaderInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        
        VkShaderModule shaderModule;
        auto& vk = VulkanBackend::get();
        vk.vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule);
        
        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = (VkStructureType)18; // SHADER_STAGE
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = layout;
        
        VkPipeline pipeline;
        vk.vkCreateComputePipelines(device, nullptr, 1, &pipelineInfo, nullptr, &pipeline);
        
        // Can destroy module after pipeline creation
        vk.vkDestroyShaderModule(device, shaderModule, nullptr);
        
        cache[shaderPath] = pipeline;
        return pipeline;
    }
    
    static std::vector<char> read_shader(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) return {};
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        return buffer;
    }
};

class VulkanMatrixOp {
    VkBuffer bufA = nullptr, bufB = nullptr, bufC = nullptr;
    VkDeviceMemory memA = nullptr, memB = nullptr, memC = nullptr;
    VkDescriptorPool descPool = nullptr;
    VkDescriptorSetLayout descLayout = nullptr;
    VkPipelineLayout pipeLayout = nullptr;
    VkDescriptorSet descSet = nullptr;
    VkCommandPool cmdPool = nullptr;
    VkCommandBuffer cmdBuffer = nullptr;
    
    size_t capA = 0, capB = 0, capC = 0;
    
public:
    ~VulkanMatrixOp() {
        // Cleanup if needed (assuming app logic handles shutdown)
        // Ideally should free resources.
    }

    void uploadA(const float* data, size_t size_bytes) {
        auto& vk = VulkanBackend::get();
        void* ptr;
        vk.vkMapMemory(vk.device, memA, 0, size_bytes, 0, &ptr);
        memcpy(ptr, data, size_bytes);
        vk.vkUnmapMemory(vk.device, memA);
    }
    
    void uploadB(const float* data, size_t size_bytes) {
        auto& vk = VulkanBackend::get();
        void* ptr;
        vk.vkMapMemory(vk.device, memB, 0, size_bytes, 0, &ptr);
        memcpy(ptr, data, size_bytes);
        vk.vkUnmapMemory(vk.device, memB);
    }

    void downloadC(float* data, size_t size_bytes) {
        auto& vk = VulkanBackend::get();
        void* ptr;
        vk.vkMapMemory(vk.device, memC, 0, size_bytes, 0, &ptr);
        memcpy(data, ptr, size_bytes);
        vk.vkUnmapMemory(vk.device, memC);
    }
    
    void resize(int M, int N, int K, const std::string& shaderPath) {
        auto& vk = VulkanBackend::get();
        size_t reqA = M * K * sizeof(float);
        size_t reqB = K * N * sizeof(float);
        size_t reqC = M * N * sizeof(float);
                
        // Reallocate A if needed
        if (reqA > capA) {
            if (bufA) { vk.vkDestroyBuffer(vk.device, bufA, nullptr); vk.free_memory(memA); }
            vk.create_buffer(reqA, bufA, memA);
            capA = reqA;
        }
        
        // Reallocate B if needed
        if (reqB > capB) {
            if (bufB) { vk.vkDestroyBuffer(vk.device, bufB, nullptr); vk.free_memory(memB); }
            vk.create_buffer(reqB, bufB, memB);
            capB = reqB;
        }
        
        // Reallocate C if needed
        if (reqC > capC) {
            if (bufC) { vk.vkDestroyBuffer(vk.device, bufC, nullptr); vk.free_memory(memC); }
            vk.create_buffer(reqC, bufC, memC);
            capC = reqC;
        }
        
        // Setup Pipeline logic ONCE (or if buffers changed)
        if (!descLayout) {
             std::cout << "DEBUG: Creating Descriptor Layout & Pool\n";
             // Create Layout, Pool, Pipeline Layout... (Boilerplate)
             VkDescriptorSetLayoutBinding bindings[3];
             for(int i=0; i<3; ++i) { bindings[i] = { (uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; }
             VkDescriptorSetLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, bindings };
             vk.vkCreateDescriptorSetLayout(vk.device, &layoutInfo, nullptr, &descLayout);
             
             VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
             VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &poolSize };
             vk.vkCreateDescriptorPool(vk.device, &poolInfo, nullptr, &descPool);
             
             VkPushConstantRange pushRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 }; // 3 ints
             VkPipelineLayoutCreateInfo pipeInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &descLayout, 1, &pushRange };
             vk.vkCreatePipelineLayout(vk.device, &pipeInfo, nullptr, &pipeLayout);
             
             VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &descLayout };
             vk.vkAllocateDescriptorSets(vk.device, &allocInfo, &descSet);
             
             VkCommandPoolCreateInfo cmdPoolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, vk.compute_queue_family_index };
             vk.vkCreateCommandPool(vk.device, &cmdPoolInfo, nullptr, &cmdPool);
             
             VkCommandBufferAllocateInfo cmdAlloc = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
             vk.vkAllocateCommandBuffers(vk.device, &cmdAlloc, &cmdBuffer);
        }
        
        // Update Descriptors (Binding Buffers)
        // Only need to update if buffers changed! 
        // OPTIMIZATION: Track if we re-allocated.
        // For now, update always to be safe (cheap CPU op).
        VkDescriptorBufferInfo infos[3] = { {bufA, 0, VK_WHOLE_SIZE}, {bufB, 0, VK_WHOLE_SIZE}, {bufC, 0, VK_WHOLE_SIZE} };
        VkWriteDescriptorSet writes[3];
        for(int i=0; i<3; ++i) {
            writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, (uint32_t)i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[i], nullptr };
        }
        vk.vkUpdateDescriptorSets(vk.device, 3, writes, 0, nullptr);
        
        // Get Pipeline
        VkPipeline pipeline = PipelineCache::get_pipeline(vk.device, shaderPath, pipeLayout);
        if (!pipeline) { std::cerr << "ERROR: Failed to load pipeline: " << shaderPath << "\n"; }

        // Record Command Buffer
        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
        vk.vkBeginCommandBuffer(cmdBuffer, &beginInfo);
        
        vk.vkCmdBindPipeline(cmdBuffer, 1, pipeline); // 1 = COMPUTE
        vk.vkCmdBindDescriptorSets(cmdBuffer, 1, pipeLayout, 0, 1, &descSet, 0, nullptr);
        struct Pc { int m, n, k; } pc = { M, N, K };
        vk.vkCmdPushConstants(cmdBuffer, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &pc);
        vk.vkCmdDispatch(cmdBuffer, (N + 3) / 4, (M + 3) / 4, 1); 
        vk.vkEndCommandBuffer(cmdBuffer);
    }
    
    void dispatch() {
        auto& vk = VulkanBackend::get();
        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdBuffer, 0, nullptr };
        vk.vkQueueSubmit(vk.compute_queue, 1, &submitInfo, nullptr);
        vk.vkQueueWaitIdle(vk.compute_queue);
    }
};

} // namespace
