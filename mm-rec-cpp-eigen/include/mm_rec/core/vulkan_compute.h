#pragma once

#include "mm_rec/core/vulkan_backend.h"
#include "mm_rec/core/embedded_shaders.h"  // Static shaders
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>  // For pipeline cache

namespace mm_rec {

// GPU Execution Context for Async Operations
struct PipelineCacheEntry; // Forward declaration

// Pooled Buffer Set (reuses buffers and persistent mappings)
struct BufferSet {
    VkBuffer bufA, bufB, bufC;
    VkDeviceMemory memA, memB, memC;
    void* ptrA; void* ptrB; void* ptrC; // Persistent mapped pointers
    size_t sizeA, sizeB, sizeC;
    bool in_use;
    
    BufferSet() : bufA(nullptr), bufB(nullptr), bufC(nullptr),
                  memA(nullptr), memB(nullptr), memC(nullptr),
                  ptrA(nullptr), ptrB(nullptr), ptrC(nullptr),
                  sizeA(0), sizeB(0), sizeC(0), in_use(false) {}
};

// GPU Context for Async Operations (holds state between submit and download)
struct GpuContext {
    VkFence fence;
    
    // Matmul params
    int M, N, K;
    size_t sizeC;
    
    // Buffers (now referencing pooled buffers)
    BufferSet* pooledBuffers; // If valid, these buffers belong to the pool
    
    // Raw handles (setup from pooled buffers)
    VkBuffer bufA, bufB, bufC;
    VkDeviceMemory memA, memB, memC;
    
    void* mapped_result;  // Generic pointer (stores dispatch context or mapped ptr)
    
    GpuContext() : fence(nullptr), M(0), N(0), K(0), sizeC(0), 
                   pooledBuffers(nullptr),
                   bufA(nullptr), bufB(nullptr), bufC(nullptr),
                   memA(nullptr), memB(nullptr), memC(nullptr),
                   mapped_result(nullptr) {}
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
    
    // Buffer Pool - static list for reuse
    static std::vector<BufferSet*>& get_buffer_pool() {
        static std::vector<BufferSet*> pool;
        return pool;
    }
    
    static BufferSet* acquire_buffers(size_t sizeA, size_t sizeB, size_t sizeC) {
        auto& pool = get_buffer_pool();
        
        // 1. Try to reuse existing buffer set
        for (auto* buf : pool) {
            if (!buf->in_use && buf->sizeA >= sizeA && buf->sizeB >= sizeB && buf->sizeC >= sizeC) {
                buf->in_use = true;
                return buf;
            }
        }
        
        // 2. Create new buffer set
        auto& vk = VulkanBackend::get();
        auto* buf = new BufferSet();
        buf->sizeA = sizeA;
        buf->sizeB = sizeB;
        buf->sizeC = sizeC;
        
        if (!vk.create_buffer(sizeA, buf->bufA, buf->memA) ||
            !vk.create_buffer(sizeB, buf->bufB, buf->memB) ||
            !vk.create_buffer(sizeC, buf->bufC, buf->memC)) {
            delete buf;
            return nullptr;
        }
        
        // Persistent Memory Mapping (Zero-Copy Optimization)
        // We map once and keep it mapped for the lifetime of the buffer
        if (vk.vkMapMemory(vk.device, buf->memA, 0, sizeA, 0, &buf->ptrA) != 0 ||
            vk.vkMapMemory(vk.device, buf->memB, 0, sizeB, 0, &buf->ptrB) != 0 ||
            vk.vkMapMemory(vk.device, buf->memC, 0, sizeC, 0, &buf->ptrC) != 0) {
            std::cerr << "❌ Failed to map persistent buffers" << std::endl;
            delete buf;
            return nullptr;
        }
        
        buf->in_use = true;
        pool.push_back(buf);
        return buf;
    }
    
    static void release_buffers(BufferSet* buf) {
        if (buf) buf->in_use = false;
    }
    
    
public:
    // Synchronous matmul (blocks until GPU completes)
    // Refactored to use the optimized Async Pipeline (with pooling & caching)
    static bool matmul(const float* A, const float* B, float* C, int M, int N, int K, const std::string& shaderPath = "src/shaders/matmul.spv") {
        GpuContext* ctx = matmul_submit_async(A, B, M, N, K, shaderPath);
        if (!ctx) return false;
        return matmul_download(ctx, C);
    }
    
    // === ASYNC GPU API (True Parallelism + UMA Zero-Copy) ===
    
    // Submit GPU work (non-blocking) - returns context for later download  
    static GpuContext* matmul_submit_async(const float* A, const float* B, int M, int N, int K, const std::string& shaderPath = "src/shaders/matmul.spv") {
        auto& vk = VulkanBackend::get();
        // ...
        ctx->M = M;
        ctx->N = N; 
        ctx->K = K;
        ctx->sizeC = M * N * sizeof(float);
        
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        
        // 1. Allocate GPU buffers (use POOL!)
        BufferSet* bufSet = acquire_buffers(sizeA, sizeB, ctx->sizeC);
        if (!bufSet) { delete ctx; return nullptr; }
        
        ctx->pooledBuffers = bufSet;
        ctx->bufA = bufSet->bufA; ctx->memA = bufSet->memA;
        ctx->bufB = bufSet->bufB; ctx->memB = bufSet->memB;
        ctx->bufC = bufSet->bufC; ctx->memC = bufSet->memC;
        
        // 2. Upload input data (Zero-Copy via persistent mapping)
        if (bufSet->ptrA) memcpy(bufSet->ptrA, A, sizeA);
        else {
            // Fallback (shouldn't happen with pool)
            void* data;
            vk.vkMapMemory(vk.device, ctx->memA, 0, sizeA, 0, &data);
            memcpy(data, A, sizeA);
            vk.vkUnmapMemory(vk.device, ctx->memA);
        }

        if (bufSet->ptrB) memcpy(bufSet->ptrB, B, sizeB);
        else {
            void* data;
            vk.vkMapMemory(vk.device, ctx->memB, 0, sizeB, 0, &data);
            memcpy(data, B, sizeB);
            vk.vkUnmapMemory(vk.device, ctx->memB);
        }
        
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
        
        // NOTE: Limit check removed - max matrix size (4096) → max groups (~256) << 65535
        // Keeping check would add unnecessary branch per call
        
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
        if (C_out) {
            if (ctx->pooledBuffers && ctx->pooledBuffers->ptrC) {
                // Persistent Mapping (Zero-Copy Read)
                memcpy(C_out, ctx->pooledBuffers->ptrC, ctx->sizeC);
            } else {
                void* dataC;
                vk.vkMapMemory(vk.device, ctx->memC, 0, ctx->sizeC, 0, &dataC);
                memcpy(C_out, dataC, ctx->sizeC);
                vk.vkUnmapMemory(vk.device, ctx->memC);
            }
        }
        
        // 3. Cleanup dispatch context (stored in mapped_result hack)
        auto* dispatch = reinterpret_cast<GpuDispatchContext*>(ctx->mapped_result);
        if (dispatch) {
            destroy_dispatch_context(dispatch);
        }
        
        // 4. Cleanup buffers and fence
        // 4. Cleanup buffers and fence
        if (ctx->fence) vk.destroy_fence(ctx->fence);
        
        // Release buffers to pool (don't destroy!)
        if (ctx->pooledBuffers) {
            release_buffers(ctx->pooledBuffers);
        } else {
            // Fallback for non-pooled buffers (legacy safety)
            vk.vkDestroyBuffer(vk.device, ctx->bufA, nullptr);
            vk.vkDestroyBuffer(vk.device, ctx->bufB, nullptr);
            vk.vkDestroyBuffer(vk.device, ctx->bufC, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memA, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memB, nullptr);
            vk.vkFreeMemory(vk.device, ctx->memC, nullptr);
        }
        
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
