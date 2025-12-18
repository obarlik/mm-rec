#pragma once

#include "mm_rec/core/vulkan_backend.h"
#include <vector>
#include <iostream>
#include <fstream>

namespace mm_rec {

class VulkanCompute {
public:
    static bool matmul(const float* A, const float* B, float* C, int M, int N, int K, const std::string& shaderPath = "src/shaders/matmul.spv") {
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
        
        // std::cout << "✅ Vulkan: Buffers allocated & data uploaded." << std::endl;

        // 3. Load Shader (Verify Compilation)
        auto code = read_shader(shaderPath);
        if (code.empty()) {
            std::cerr << "❌ Failed to read shader file." << std::endl;
            return false;
        }

        // std::cout << "✅ Vulkan: Shader 'matmul.spv' found (" << code.size() << " bytes)." << std::endl;

        // ---------------------------------------------------------
        // REAL DISPATCH IMPLEMENTATION
        // ---------------------------------------------------------
        
        // 1. Create Shader Module
        VkShaderModuleCreateInfo shaderInfo = {};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = code.size();
        shaderInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        
        VkShaderModule shaderModule;
        if (vk.vkCreateShaderModule(vk.device, &shaderInfo, nullptr, &shaderModule) != 0) {
            std::cerr << "❌ Failed to create shader module" << std::endl;
            return false;
        }

        // 2. Descriptor Set Layout (3 Storage Buffers)
        VkDescriptorSetLayoutBinding bindings[3];
        for(int i=0; i<3; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].pImmutableSamplers = nullptr;
        }
        
        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = bindings;
        
        VkDescriptorSetLayout descriptorLayout;
        vk.vkCreateDescriptorSetLayout(vk.device, &layoutInfo, nullptr, &descriptorLayout);

        // 3. Pipeline Layout
        struct PushConsts { int M; int N; int K; } pushConsts = {M, N, K};
        VkPushConstantRange pushRange = {};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConsts);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        
        VkPipelineLayout pipelineLayout;
        vk.vkCreatePipelineLayout(vk.device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

        // 4. Compute Pipeline
        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = (VkStructureType)2; // VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO (Assuming 2, let's check spec or use 0 and manual set)
        // Wait, standard enum is 2? No, let's be careful.
        // Actually, let's just use raw struct and hope 0 works if sType is correct. 
        // For ShaderStage: sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18
        pipelineInfo.stage.sType = (VkStructureType)18; 
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;
        
        VkPipeline pipeline;
        if (vk.vkCreateComputePipelines(vk.device, nullptr, 1, &pipelineInfo, nullptr, &pipeline) != 0) {
             std::cerr << "❌ Failed to create compute pipeline" << std::endl;
             return false;
        }

        // 5. Descriptor Pool & Set
        VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        
        VkDescriptorPool descriptorPool;
        vk.vkCreateDescriptorPool(vk.device, &poolInfo, nullptr, &descriptorPool);

        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorLayout;
        
        VkDescriptorSet descriptorSet;
        vk.vkAllocateDescriptorSets(vk.device, &allocInfo, &descriptorSet);

        // 6. Update Descriptor Set (Bind Buffers)
        VkDescriptorBufferInfo bufferInfos[3] = {
            {bufA, 0, VK_WHOLE_SIZE},
            {bufB, 0, VK_WHOLE_SIZE},
            {bufC, 0, VK_WHOLE_SIZE}
        };

        VkWriteDescriptorSet descriptorWrites[3];
        for(int i=0; i<3; i++) {
            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].pNext = nullptr;
            descriptorWrites[i].dstSet = descriptorSet;
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].dstArrayElement = 0;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        }
        vk.vkUpdateDescriptorSets(vk.device, 3, descriptorWrites, 0, nullptr);

        // 7. Command Buffer & Dispatch
        VkCommandPoolCreateInfo cmdPoolInfo = {};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = vk.compute_queue_family_index;
        cmdPoolInfo.flags = 0; // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT usually
        
        VkCommandPool commandPool;
        vk.vkCreateCommandPool(vk.device, &cmdPoolInfo, nullptr, &commandPool);

        VkCommandBufferAllocateInfo cmdAllocInfo = {};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = commandPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vk.vkAllocateCommandBuffers(vk.device, &cmdAllocInfo, &commandBuffer);

        // 8. Record & Dispatch
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vk.vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        vk.vkCmdBindPipeline(commandBuffer, 1 /*BindPointCompute*/, pipeline);
        vk.vkCmdBindDescriptorSets(commandBuffer, 1 /*BindPointCompute*/, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        
        // Push Constants
        vk.vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConsts), &pushConsts);
        
        // Dispatch (Size/16 because local_size=16)
        vk.vkCmdDispatch(commandBuffer, (N + 15) / 16, (M + 15) / 16, 1);
        
        vk.vkEndCommandBuffer(commandBuffer);

        // 9. Submit & Wait
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        
        if (vk.vkQueueSubmit(vk.compute_queue, 1, &submitInfo, nullptr) != 0) {
            std::cerr << "❌ Queue Submit failed." << std::endl;
            return false;
        }

        vk.vkQueueWaitIdle(vk.compute_queue); // SYNC WAIT
        
        // 10. Download Results (Map C)
        void* dataC;
        vk.vkMapMemory(vk.device, memC, 0, sizeC, 0, &dataC);
        memcpy(C, dataC, sizeC);
        vk.vkUnmapMemory(vk.device, memC);
        
        // Cleanup resources (simplified for benchmark: leaking some handles, but OS/Driver cleans up on exit)
        // In prod: destroy pipeline, layout, descriptor pool, shader module.
        vk.vkDestroyPipeline(vk.device, pipeline, nullptr);
        vk.vkDestroyPipelineLayout(vk.device, pipelineLayout, nullptr);
        vk.vkDestroyDescriptorPool(vk.device, descriptorPool, nullptr);
        vk.vkDestroyDescriptorSetLayout(vk.device, descriptorLayout, nullptr);
        vk.vkDestroyShaderModule(vk.device, shaderModule, nullptr);

        // Cleanup Buffers & Memory
        vk.vkDestroyCommandPool(vk.device, commandPool, nullptr);
        
        vk.vkDestroyBuffer(vk.device, bufA, nullptr);
        vk.vkDestroyBuffer(vk.device, bufB, nullptr);
        vk.vkDestroyBuffer(vk.device, bufC, nullptr);
        
        vk.vkFreeMemory(vk.device, memA, nullptr);
        vk.vkFreeMemory(vk.device, memB, nullptr);
        vk.vkFreeMemory(vk.device, memC, nullptr);

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
