#pragma once

#include "mm_rec/core/vulkan_backend.h"
#include "mm_rec/core/embedded_shaders.h"  // Static shaders
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>  // For pipeline cache

namespace mm_rec {

// GPU Execution Context for Async Operations
struct GpuContext {
    VkBuffer bufA, bufB, bufC;
    VkDeviceMemory memA, memB, memC;
    VkFence fence;
    int M, N, K;
    size_t sizeC;
    void* mapped_result;  // For UMA zero-copy
    
    GpuContext() : bufA(nullptr), bufB(nullptr), bufC(nullptr),
                   memA(nullptr), memB(nullptr), memC(nullptr),
                   fence(nullptr), M(0), N(0), K(0), sizeC(0), mapped_result(nullptr) {}
};


// Cached Pipeline State (reused across calls for same shader)
struct PipelineCacheEntry {
    VkShaderModule shaderModule;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkCommandPool commandPool;
    
    PipelineCacheEntry() : shaderModule(nullptr), descriptorSetLayout(nullptr),
                           pipelineLayout(nullptr), pipeline(nullptr),
                           commandPool(nullptr) {}
};

// GPU Dispatch Context (per-call resources + cached pipeline reference)
struct GpuDispatchContext {
    PipelineCacheEntry* cachedPipeline;  // Reference to cached pipeline
    VkDescriptorPool descriptorPool;      // Per-call
    VkDescriptorSet descriptorSet;        // Per-call
    VkCommandBuffer commandBuffer;        // Per-call
    
    GpuDispatchContext() : cachedPipeline(nullptr), descriptorPool(nullptr),
                           descriptorSet(nullptr), commandBuffer(nullptr) {}
};


class VulkanCompute {
private:
    // Pipeline cache - static map for reuse
    static std::unordered_map<std::string, PipelineCacheEntry*>& get_pipeline_cache() {
        static std::unordered_map<std::string, PipelineCacheEntry*> cache;
        return cache;
    }
    
public:
    // Synchronous matmul (blocks until GPU completes)
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

        // Wait for GPU to finish (sync mode only for now)
        vk.vkQueueWaitIdle(vk.compute_queue);
        
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
    
    // === ASYNC GPU API (True Parallelism + UMA Zero-Copy) ===
    
    // Submit GPU work (non-blocking) - returns context for later download  
    static GpuContext* matmul_submit_async(const float* A, const float* B, int M, int N, int K, const std::string& shaderPath = "src/shaders/matmul.spv") {
        auto& vk = VulkanBackend::get();
        if (!vk.is_ready()) return nullptr;
        
        auto* ctx = new GpuContext();
        ctx->M = M;
        ctx->N = N; 
        ctx->K = K;
        ctx->sizeC = M * N * sizeof(float);
        
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        
        // 1. Allocate GPU buffers (HOST_VISIBLE for UMA)
        if(!vk.create_buffer(sizeA, ctx->bufA, ctx->memA)) { delete ctx; return nullptr; }
        if(!vk.create_buffer(sizeB, ctx->bufB, ctx->memB)) { delete ctx; return nullptr; }
        if(!vk.create_buffer(ctx->sizeC, ctx->bufC, ctx->memC)) { delete ctx; return nullptr; }
        
        // 2. Upload input data
        void* data;
        vk.vkMapMemory(vk.device, ctx->memA, 0, sizeA, 0, &data);
        memcpy(data, A, sizeA);
        vk.vkUnmapMemory(vk.device, ctx->memA);
        
        vk.vkMapMemory(vk.device, ctx->memB, 0, sizeB, 0, &data);
        memcpy(data, B, sizeB);
        vk.vkUnmapMemory(vk.device, ctx->memB);
        
        // 3. Get or create cached pipeline
        auto& cache = get_pipeline_cache();
        PipelineCacheEntry* pipeline = nullptr;
        
        auto it = cache.find(shaderPath);
        if (it != cache.end()) {
            pipeline = it->second;  // Cache HIT!
        } else {
            pipeline = create_pipeline_cache(shaderPath);  // Cache MISS - create & store
            if (pipeline) {
                cache[shaderPath] = pipeline;
            }
        }
        
        if (!pipeline) {
            std::cerr << "❌ Failed to get/create pipeline" << std::endl;
            // Cleanup and fail
            vk.vkDestroyBuffer(vk.device, ctx->bufA, nullptr);
            vk.vkDestroyBuffer(vk.device, ctx->bufB, nullptr);
            vk.vkDestroyBuffer(vk.device, ctx->bufC, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memA, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memB, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memC, nullptr);
            delete ctx;
            return nullptr;
        }
        
        // 4. Create per-call dispatch context (just descriptors + command buffer)
        auto* dispatch = new GpuDispatchContext();
        dispatch->cachedPipeline = pipeline;
        
        // 5. Allocate per-call descriptor pool and set
        VkDescriptorPoolSize poolSize = {};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 3;
        
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;
        
        if (vk.vkCreateDescriptorPool(vk.device, &poolInfo, nullptr, &dispatch->descriptorPool) != 0) {
            std::cerr << "❌ Failed to create descriptor pool" << std::endl;
            destroy_dispatch_context(dispatch);
            // cleanup ctx buffers
            vk.vkDestroyBuffer(vk.device, ctx->bufA, nullptr);
            vk.vkDestroyBuffer(vk.device, ctx->bufB, nullptr);
            vk.vkDestroyBuffer(vk.device, ctx->bufC, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memA, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memB, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memC, nullptr);
            delete ctx;
            return nullptr;
        }
        
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = dispatch->descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &pipeline->descriptorSetLayout;  // Use CACHED layout!
        
        VkResult result = vk.vkAllocateDescriptorSets(vk.device, &allocInfo, &dispatch->descriptorSet);
        if (result != 0) {
            std::cerr << "❌ Failed to allocate descriptor set, error: " << result << std::endl;
            destroy_dispatch_context(dispatch);
            vk.vkDestroyBuffer(vk.device, ctx->bufA, nullptr);
            vk.vkDestroyBuffer(vk.device, ctx->bufB, nullptr);
            vk.vkDestroyBuffer(vk.device, ctx->bufC, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memA, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memB, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memC, nullptr);
            delete ctx;
            return nullptr;
        }
        
        // 5. Bind buffers to descriptor set
        
        if (!ctx->bufA || !ctx->bufB || !ctx->bufC) {
            std::cerr << "❌ Invalid buffer handles!" << std::endl;
            return nullptr;
        }
        
        VkDescriptorBufferInfo bufferInfos[3] = {
            {ctx->bufA, 0, VK_WHOLE_SIZE},
            {ctx->bufB, 0, VK_WHOLE_SIZE},
            {ctx->bufC, 0, VK_WHOLE_SIZE}
        };
        
        VkWriteDescriptorSet descriptorWrites[3];
        for(int i=0; i<3; i++) {
            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].pNext = nullptr;
            descriptorWrites[i].dstSet = dispatch->descriptorSet;
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].dstArrayElement = 0;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        }
        vk.vkUpdateDescriptorSets(vk.device, 3, descriptorWrites, 0, nullptr);
        
        // 6. Allocate command buffer from CACHED command pool
        VkCommandBufferAllocateInfo cmdBufAllocInfo = {};
        cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdBufAllocInfo.commandPool = pipeline->commandPool;  // Use CACHED pool!
        cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdBufAllocInfo.commandBufferCount = 1;
        
        if (vk.vkAllocateCommandBuffers(vk.device, &cmdBufAllocInfo, &dispatch->commandBuffer) != 0) {
            std::cerr << "❌ Failed to allocate command buffer" << std::endl;
            delete dispatch;
            delete ctx;
            return nullptr;
        }
        
        // 7. Record command buffer
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        if (vk.vkBeginCommandBuffer(dispatch->commandBuffer, &beginInfo) != 0) {
            std::cerr << "❌ vkBeginCommandBuffer failed" << std::endl;
            return nullptr;
        }
        vk.vkCmdBindPipeline(dispatch->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
        
        vk.vkCmdBindDescriptorSets(dispatch->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline->pipelineLayout, 0, 1, &dispatch->descriptorSet, 0, nullptr);
        
        // Push constants (M, N, K)
        struct PushConsts { int M; int N; int K; } pushConsts = {M, N, K};
        vk.vkCmdPushConstants(dispatch->commandBuffer, pipeline->pipelineLayout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConsts), &pushConsts);
        
        // Dispatch workgroups (match sync matmul convention: X=N, Y=M)
        uint32_t groupsX = (N + 15) / 16;
        uint32_t groupsY = (M + 15) / 16;
        
        // Sanity check: Intel GPU might have workgroup limits
        const uint32_t MAX_GROUPS = 65535;
        if (groupsX > MAX_GROUPS || groupsY > MAX_GROUPS) {
            std::cerr << "❌ Workgroup count exceeds limits: " << groupsX << "x" << groupsY << std::endl;
            return nullptr;
        }
        
        vk.vkCmdDispatch(dispatch->commandBuffer, groupsX, groupsY, 1);
        
        VkResult endResult = vk.vkEndCommandBuffer(dispatch->commandBuffer);
        if (endResult != 0) {
            std::cerr << "❌ vkEndCommandBuffer failed with error: " << endResult << std::endl;
            return nullptr;
        }
        
        // 7. Create fence for async signaling
        ctx->fence = vk.create_fence();
        
        // 8. Submit to GPU (NON-BLOCKING!)
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &dispatch->commandBuffer;
        
        vk.vkQueueSubmit(vk.compute_queue, 1, &submitInfo, ctx->fence);
        
        // 9. Store dispatch context in GpuContext for cleanup later  
        // Note: We need to keep dispatch alive until download
        // For now, we'll clean it up in download. In production, embed it in GpuContext.
        ctx->mapped_result = dispatch; // HACK: reuse this field to store dispatch*
        
        return ctx;
    }
    
    // Download GPU results - waits for GPU, retrieves data, cleans up
    static bool matmul_download(GpuContext* ctx, float* C_out) {
        if (!ctx) return false;
        
        auto& vk = VulkanBackend::get();
        
        // 1. Wait for GPU
        if (ctx->fence) {
            vk.wait_fence(ctx->fence);
        }
        
        // 2. UMA Zero-Copy Optimization
        // On Intel iGPU (UMA), bufC memory is HOST_VISIBLE = shared RAM
        void* dataC;
        vk.vkMapMemory(vk.device, ctx->memC, 0, ctx->sizeC, 0, &dataC);
        
        if (C_out) {
            // TODO: Detect UMA and skip memcpy
            // For UMA: dataC IS the result (shared RAM), no copy needed!
            // For now: always copy for safety
            memcpy(C_out, dataC, ctx->sizeC);
        }
        
        vk.vkUnmapMemory(vk.device, ctx->memC);
        
        // 3. Cleanup dispatch context (stored in mapped_result hack)
        auto* dispatch = reinterpret_cast<GpuDispatchContext*>(ctx->mapped_result);
        if (dispatch) {
            destroy_dispatch_context(dispatch);
        }
        
        // 4. Cleanup buffers and fence
        if (ctx->fence) vk.destroy_fence(ctx->fence);
        vk.vkDestroyBuffer(vk.device, ctx->bufA, nullptr);
        vk.vkDestroyBuffer(vk.device, ctx->bufB, nullptr);
        vk.vkDestroyBuffer(vk.device, ctx->bufC, nullptr);
        vk.vkFreeMemory(vk.device, ctx->memA, nullptr);
        vk.vkFreeMemory(vk.device, ctx->memB, nullptr);
        vk.vkFreeMemory(vk.device, ctx->memC, nullptr);
        
        delete ctx;
        return true;
    }
    
    // Check helper
    static bool is_ready() { return VulkanBackend::get().device != nullptr; }
    
    static std::vector<char> read_shader(const std::string& filename) {
        // 1. Try Embedded Storage first (Static Linking)
        auto embedded = EmbeddedShaders::get(filename);
        if (embedded.data != nullptr) {
            // Found in memory!
            return std::vector<char>(embedded.data, embedded.data + embedded.length);
        }

        // 2. Fallback to Disk (Development / Custom Shader)
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            // Try relative to build dir fallback if not found
            std::cerr << "⚠️ Shader not found in memory or disk: " << filename << std::endl;
            throw std::runtime_error("Failed to open shader file: " + filename);
        }
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }
    
private:
    // === GPU Dispatch Helpers (Phase 1: Extraction) ===
    
    // Create dispatch context (pipeline, descriptors, etc.)
    // Create reusable pipeline cache (called once per shader)
    static PipelineCacheEntry* create_pipeline_cache(const std::string& shaderPath) {
        auto& vk = VulkanBackend::get();
        if (!vk.is_ready()) return nullptr;
        
        auto* cache = new PipelineCacheEntry();
        
        // 1. Load and create shader module
        auto code = read_shader(shaderPath);
        if (code.empty()) {
            delete cache;
            return nullptr;
        }
        
        VkShaderModuleCreateInfo shaderInfo = {};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = code.size();
        shaderInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        
        if (vk.vkCreateShaderModule(vk.device, &shaderInfo, nullptr, &cache->shaderModule) != 0) {
            delete cache;
            return nullptr;
        }
        
        // 2. Create descriptor set layout (3 storage buffers)
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
        
        // Create layout directly in cache (not local variable!)
        vk.vkCreateDescriptorSetLayout(vk.device, &layoutInfo, nullptr, &cache->descriptorSetLayout);
        
        // 3. Create pipeline layout with push constants
        VkPushConstantRange pushRange = {};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(int) * 3; // M, N, K
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &cache->descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        
        vk.vkCreatePipelineLayout(vk.device, &pipelineLayoutInfo, nullptr, &cache->pipelineLayout);
        
        // 4. Create compute pipeline
        VkPipelineShaderStageCreateInfo shaderStage = {};
        shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStage.module = cache->shaderModule;
        shaderStage.pName = "main";
        
        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = shaderStage;
        pipelineInfo.layout = cache->pipelineLayout;
        
        VkResult pipelineResult = vk.vkCreateComputePipelines(vk.device, nullptr, 1, &pipelineInfo, nullptr, &cache->pipeline);
        if (pipelineResult != 0) {
            std::cerr << "❌ Failed to create compute pipeline, error: " << pipelineResult << std::endl;
            delete cache;  // Simple delete for cache (no complex cleanup needed yet)
            return nullptr;
        }
        
        // 5. Create command pool (reused for command buffer allocation)
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        
        if (vk.vkCreateCommandPool(vk.device, &poolInfo, nullptr, &cache->commandPool) != 0) {
            std::cerr << "❌ Failed to create command pool" << std::endl;
            delete cache;
            return nullptr;
        }
        
        return cache;  // Return cached pipeline!
    }
    
    // Cleanup per-call dispatch context (descriptor pool/set, command buffer)
    // NOTE: Does NOT destroy cached pipeline - that lives in cache!
    static void destroy_dispatch_context(GpuDispatchContext* ctx) {
        if (!ctx) return;
        
        auto& vk = VulkanBackend::get();
        
        // Only clean per-call resources (descriptor pool also frees descriptor sets and we let command buffer cleanup happen with pool)
        if (ctx->descriptorPool) vk.vkDestroyDescriptorPool(vk.device, ctx->descriptorPool, nullptr);
        
        delete ctx;
    }
};

} // namespace mm_rec
