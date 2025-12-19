/**
 * Vulkan Backend (Dynamic Loader & Context)
 * 
 * Provides runtime access to the iGPU without build-time dependencies.
 * Loads 'libvulkan.so.1', initializes Instance, and selects Physical Device.
 */

#pragma once

#include <iostream>
#include <vector>
#include <dlfcn.h>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <vector>
#include <dlfcn.h>
#include <stdexcept>
#include <cstring>
#include <cstring>
#include <cstdint>
#include <vector>
#include <iostream>
#include <dlfcn.h>
#include <stdexcept>
#include <iomanip> // For VRAM formatting

// --- Vulkan Types (Minimal) ---
#define VK_MAKE_VERSION(major, minor, patch) \
    ((((uint32_t)(major)) << 22) | (((uint32_t)(minor)) << 12) | ((uint32_t)(patch)))

typedef struct VkInstance_T* VkInstance;
typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
typedef struct VkDevice_T* VkDevice;
typedef struct VkBuffer_T* VkBuffer;
typedef struct VkDeviceMemory_T* VkDeviceMemory;
typedef struct VkShaderModule_T* VkShaderModule;
typedef struct VkDescriptorSetLayout_T* VkDescriptorSetLayout;
typedef struct VkPipelineLayout_T* VkPipelineLayout;
typedef struct VkPipeline_T* VkPipeline;
typedef struct VkCommandPool_T* VkCommandPool;
typedef struct VkCommandBuffer_T* VkCommandBuffer;
typedef struct VkDescriptorPool_T* VkDescriptorPool;
typedef struct VkDescriptorSet_T* VkDescriptorSet;
typedef struct VkQueue_T* VkQueue;
typedef struct VkFence_T* VkFence;
typedef struct VkPipelineCache_T* VkPipelineCache;

typedef uint32_t VkFlags;
typedef uint32_t VkStructureType;
typedef int VkResult;
typedef uint64_t VkDeviceSize;

// Enum Constants (from spec)
const VkStructureType VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = (VkStructureType)1;
const VkStructureType VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = (VkStructureType)3;
const VkStructureType VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = (VkStructureType)12;
const VkStructureType VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = (VkStructureType)5;

const uint32_t VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002;
const uint32_t VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004;
const uint32_t VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020;
const uint32_t VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001;
const uint32_t VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002;
const uint32_t VK_SHARING_MODE_EXCLUSIVE = 0;

// Structs
typedef struct VkApplicationInfo {
    VkStructureType    sType;
    const void*        pNext;
    const char*        pApplicationName;
    uint32_t           applicationVersion;
    const char*        pEngineName;
    uint32_t           engineVersion;
    uint32_t           apiVersion;
} VkApplicationInfo;

typedef struct VkInstanceCreateInfo {
    VkStructureType             sType;
    const void*                 pNext;
    VkFlags                     flags;
    const VkApplicationInfo*    pApplicationInfo;
    uint32_t                    enabledLayerCount;
    const char* const*          ppEnabledLayerNames;
    uint32_t                    enabledExtensionCount;
    const char* const*          ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct VkDeviceQueueCreateInfo {
    VkStructureType             sType;
    const void*                 pNext;
    VkFlags                     flags;
    uint32_t                    queueFamilyIndex;
    uint32_t                    queueCount;
    const float*                pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct VkDeviceCreateInfo {
    VkStructureType                    sType;
    const void*                        pNext;
    VkFlags                            flags;
    uint32_t                           queueCreateInfoCount;
    const VkDeviceQueueCreateInfo*     pQueueCreateInfos;
    uint32_t                           enabledLayerCount;
    const char* const*                 ppEnabledLayerNames;
    uint32_t                           enabledExtensionCount;
    const char* const*                 ppEnabledExtensionNames;
    const void*                        pEnabledFeatures;
} VkDeviceCreateInfo;

typedef struct VkQueueFamilyProperties {
    VkFlags    queueFlags;
    uint32_t   queueCount;
    uint32_t   timestampValidBits;
    uint32_t   minImageTransferGranularity[3];
} VkQueueFamilyProperties;

const uint32_t VK_QUEUE_COMPUTE_BIT = 0x00000002;

typedef struct VkBufferCreateInfo {
    VkStructureType    sType;
    const void*        pNext;
    VkFlags            flags;
    VkDeviceSize       size;
    VkFlags            usage;
    uint32_t           sharingMode;
    uint32_t           queueFamilyIndexCount;
    const uint32_t*    pQueueFamilyIndices;
} VkBufferCreateInfo;

typedef struct VkMemoryAllocateInfo {
    VkStructureType    sType;
    const void*        pNext;
    VkDeviceSize       allocationSize;
    uint32_t           memoryTypeIndex;
} VkMemoryAllocateInfo;

typedef struct VkMemoryRequirements {
    VkDeviceSize    size;
    VkDeviceSize    alignment;
    uint32_t        memoryTypeBits;
} VkMemoryRequirements;

typedef struct VkPhysicalDeviceMemoryProperties {
    uint32_t        memoryTypeCount;
    struct {
        uint32_t    propertyFlags;
        uint32_t    heapIndex;
    } memoryTypes[32];
    uint32_t        memoryHeapCount;
    struct {
        VkDeviceSize    size;
        VkFlags         flags;
    } memoryHeaps[16];
} VkPhysicalDeviceMemoryProperties;

typedef struct VkPushConstantRange {
    VkFlags    stageFlags;
    uint32_t   offset;
    uint32_t   size;
} VkPushConstantRange;

// Properties
typedef struct VkPhysicalDeviceLimits {
    uint32_t maxImageDimension1D;
    uint32_t maxImageDimension2D;
    uint32_t maxImageDimension3D;
    uint32_t maxImageDimensionCube;
    uint32_t maxImageArrayLayers;
    uint32_t maxTexelBufferElements;
    uint32_t maxUniformBufferRange;
    uint32_t maxStorageBufferRange;
    uint32_t maxPushConstantsSize;
    uint32_t maxMemoryAllocationCount;
    uint32_t maxSamplerAllocationCount;
    VkDeviceSize bufferImageGranularity;
    VkDeviceSize sparseAddressSpaceSize;
    uint32_t maxBoundDescriptorSets;
    uint32_t maxPerStageDescriptorSamplers;
    uint32_t maxPerStageDescriptorUniformBuffers;
    uint32_t maxPerStageDescriptorStorageBuffers;
    uint32_t maxPerStageDescriptorSampledImages;
    uint32_t maxPerStageDescriptorStorageImages;
    uint32_t maxPerStageDescriptorInputAttachments;
    uint32_t maxPerStageResources;
    uint32_t maxDescriptorSetSamplers;
    uint32_t maxDescriptorSetUniformBuffers;
    uint32_t maxDescriptorSetUniformBuffersDynamic;
    uint32_t maxDescriptorSetStorageBuffers;
    uint32_t maxDescriptorSetStorageBuffersDynamic;
    uint32_t maxDescriptorSetSampledImages;
    uint32_t maxDescriptorSetStorageImages;
    uint32_t maxDescriptorSetInputAttachments;
    uint32_t maxVertexInputAttributes;
    uint32_t maxVertexInputBindings;
    uint32_t maxVertexInputAttributeOffset;
    uint32_t maxVertexInputBindingStride;
    uint32_t maxVertexOutputComponents;
    uint32_t maxTessellationGenerationLevel;
    uint32_t maxTessellationPatchSize;
    uint32_t maxTessellationControlPerVertexInputComponents;
    uint32_t maxTessellationControlPerVertexOutputComponents;
    uint32_t maxTessellationControlPerPatchOutputComponents;
    uint32_t maxTessellationControlTotalOutputComponents;
    uint32_t maxTessellationEvaluationInputComponents;
    uint32_t maxTessellationEvaluationOutputComponents;
    uint32_t maxGeometryShaderInvocations;
    uint32_t maxGeometryInputComponents;
    uint32_t maxGeometryOutputComponents;
    uint32_t maxGeometryOutputVertices;
    uint32_t maxGeometryTotalOutputComponents;
    uint32_t maxFragmentInputComponents;
    uint32_t maxFragmentOutputAttachments;
    uint32_t maxFragmentDualSrcAttachments;
    uint32_t maxFragmentCombinedOutputResources;
    uint32_t maxComputeSharedMemorySize;
    uint32_t maxComputeWorkGroupCount[3];
    uint32_t maxComputeWorkGroupInvocations;
    uint32_t maxComputeWorkGroupSize[3];
    uint32_t subPixelPrecisionBits;
    uint32_t subTexelPrecisionBits;
    uint32_t mipmapPrecisionBits;
    uint32_t maxDrawIndexedIndexValue;
    uint32_t maxDrawIndirectCount;
    float    maxSamplerLodBias;
    float    maxSamplerAnisotropy;
    uint32_t maxViewports;
    uint32_t maxViewportDimensions[2];
    float    viewportBoundsRange[2];
    uint32_t viewportSubPixelBits;
    size_t   minMemoryMapAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment;
    VkDeviceSize minUniformBufferOffsetAlignment;
    VkDeviceSize minStorageBufferOffsetAlignment;
    int32_t  minTexelOffset;
    uint32_t maxTexelOffset;
    int32_t  minTexelGatherOffset;
    uint32_t maxTexelGatherOffset;
    float    minInterpolationOffset;
    float    maxInterpolationOffset;
    uint32_t subPixelInterpolationOffsetBits;
    uint32_t maxFramebufferWidth;
    uint32_t maxFramebufferHeight;
    uint32_t maxFramebufferLayers;
    VkFlags  framebufferColorSampleCounts;
    VkFlags  framebufferDepthSampleCounts;
    VkFlags  framebufferStencilSampleCounts;
    VkFlags  framebufferNoAttachmentsSampleCounts;
    uint32_t maxColorAttachments;
    VkFlags  sampledImageColorSampleCounts;
    VkFlags  sampledImageIntegerSampleCounts;
    VkFlags  sampledImageDepthSampleCounts;
    VkFlags  sampledImageStencilSampleCounts;
    VkFlags  storageImageSampleCounts;
    uint32_t maxSampleMaskWords;
    uint32_t timestampComputeAndGraphics;
    float    timestampPeriod;
    uint32_t maxClipDistances;
    uint32_t maxCullDistances;
    uint32_t maxCombinedClipAndCullDistances;
    uint32_t discreteQueuePriorities;
    float    pointSizeRange[2];
    float    lineWidthRange[2];
    float    pointSizeGranularity;
    float    lineWidthGranularity;
    uint32_t strictLines;
    uint32_t standardSampleLocations;
    VkDeviceSize optimalBufferCopyOffsetAlignment;
    VkDeviceSize optimalBufferCopyRowPitchAlignment;
    VkDeviceSize nonCoherentAtomSize;
} VkPhysicalDeviceLimits;

typedef struct VkPhysicalDeviceSparseProperties {
    uint32_t residencyStandard2DBlockShape;
    uint32_t residencyStandard2DMultisampleBlockShape;
    uint32_t residencyStandard3DBlockShape;
    uint32_t residencyAlignedMipSize;
    uint32_t residencyNonResidentStrict;
} VkPhysicalDeviceSparseProperties;

typedef struct VkPhysicalDeviceProperties {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    VkPhysicalDeviceLimits limits;
    VkPhysicalDeviceSparseProperties sparseProperties;
} VkPhysicalDeviceProperties;

// --- Extended Structs for Compute ---

// Shader
typedef struct VkShaderModuleCreateInfo {
    VkStructureType    sType;
    const void*        pNext;
    VkFlags            flags;
    size_t             codeSize;
    const uint32_t*    pCode;
} VkShaderModuleCreateInfo;

// Pipeline Layout
typedef struct VkPipelineLayoutCreateInfo {
    VkStructureType                 sType;
    const void*                     pNext;
    VkFlags                         flags;
    uint32_t                        setLayoutCount;
    const VkDescriptorSetLayout*    pSetLayouts;
    uint32_t                        pushConstantRangeCount;
    const void*                     pPushConstantRanges;
} VkPipelineLayoutCreateInfo;

// Descriptors
typedef struct VkDescriptorSetLayoutBinding {
    uint32_t              binding;
    uint32_t              descriptorType; // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
    uint32_t              descriptorCount;
    uint32_t              stageFlags;     // VK_SHADER_STAGE_COMPUTE_BIT = 32
    const void*           pImmutableSamplers;
} VkDescriptorSetLayoutBinding;

typedef struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType                        sType;
    const void*                            pNext;
    VkFlags                                flags;
    uint32_t                               bindingCount;
    const VkDescriptorSetLayoutBinding*    pBindings;
} VkDescriptorSetLayoutCreateInfo;

typedef struct VkDescriptorPoolSize {
    uint32_t    type;
    uint32_t    descriptorCount;
} VkDescriptorPoolSize;

typedef struct VkDescriptorPoolCreateInfo {
    VkStructureType                sType;
    const void*                    pNext;
    VkFlags                        flags;
    uint32_t                       maxSets;
    uint32_t                       poolSizeCount;
    const VkDescriptorPoolSize*    pPoolSizes;
} VkDescriptorPoolCreateInfo;

typedef struct VkDescriptorSetAllocateInfo {
    VkStructureType                 sType;
    const void*                     pNext;
    VkDescriptorPool                descriptorPool;
    uint32_t                        descriptorSetCount;
    const VkDescriptorSetLayout*    pSetLayouts;
} VkDescriptorSetAllocateInfo;

typedef struct VkDescriptorBufferInfo {
    VkBuffer        buffer;
    VkDeviceSize    offset;
    VkDeviceSize    range;
} VkDescriptorBufferInfo;

typedef struct VkWriteDescriptorSet {
    VkStructureType                  sType;
    const void*                      pNext;
    VkDescriptorSet                  dstSet;
    uint32_t                         dstBinding;
    uint32_t                         dstArrayElement;
    uint32_t                         descriptorCount;
    uint32_t                         descriptorType;
    const void*                      pImageInfo;
    const VkDescriptorBufferInfo*    pBufferInfo;
    const void*                      pTexelBufferView;
} VkWriteDescriptorSet;

// Pipeline
typedef struct VkPipelineShaderStageCreateInfo {
    VkStructureType     sType;
    const void*         pNext;
    VkFlags             flags;
    uint32_t            stage; // VK_SHADER_STAGE_COMPUTE_BIT
    VkShaderModule      module;
    const char*         pName;
    const void*         pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct VkComputePipelineCreateInfo {
    VkStructureType                    sType;
    const void*                        pNext;
    VkFlags                            flags;
    VkPipelineShaderStageCreateInfo    stage;
    VkPipelineLayout                   layout;
    VkPipeline                         basePipelineHandle;
    int32_t                            basePipelineIndex;
} VkComputePipelineCreateInfo;

// Commands
typedef struct VkCommandBufferAllocateInfo {
    VkStructureType    sType;
    const void*        pNext;
    VkCommandPool      commandPool;
    uint32_t           level; // VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
    uint32_t           commandBufferCount;
} VkCommandBufferAllocateInfo;

typedef struct VkCommandBufferBeginInfo {
    VkStructureType    sType;
    const void*        pNext;
    VkFlags            flags; // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 1
    const void*        pInheritanceInfo;
} VkCommandBufferBeginInfo;

typedef struct VkSubmitInfo {
    VkStructureType                sType;
    const void*                    pNext;
    uint32_t                       waitSemaphoreCount;
    const void*                    pWaitSemaphores;
    const uint32_t*                pWaitDstStageMask;
    uint32_t                       commandBufferCount;
    const VkCommandBuffer*         pCommandBuffers;
    uint32_t                       signalSemaphoreCount;
    const void*                    pSignalSemaphores;
} VkSubmitInfo;

typedef struct VkCommandPoolCreateInfo {
    VkStructureType    sType;
    const void*        pNext;
    VkFlags            flags;
    uint32_t           queueFamilyIndex;
} VkCommandPoolCreateInfo;

// CONSTANTS
const uint32_t VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
const uint32_t VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020;
const uint32_t VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
const uint32_t VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001;
const VkStructureType VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = (VkStructureType)16;
const VkStructureType VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = (VkStructureType)30;
const VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = (VkStructureType)32;
const VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = (VkStructureType)33;
const VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = (VkStructureType)34;
const VkStructureType VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = (VkStructureType)35;
const VkStructureType VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = (VkStructureType)29;
const VkStructureType VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = (VkStructureType)39;
const VkStructureType VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = (VkStructureType)40;
const VkStructureType VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = (VkStructureType)42;
const VkStructureType VK_STRUCTURE_TYPE_SUBMIT_INFO = (VkStructureType)4;
const VkDeviceSize VK_WHOLE_SIZE = (~0ULL);

namespace mm_rec {

class VulkanBackend {
    friend class VulkanCompute;
    friend class VulkanMatrixOp;
    friend class PipelineCache;
private:
    void* lib_handle = nullptr;
    VkInstance instance = nullptr;
    VkPhysicalDevice physical_device = nullptr;
    VkDevice device = nullptr; // Logical device
    VkQueue compute_queue = nullptr;
    VkPhysicalDeviceMemoryProperties memory_properties;
    uint32_t compute_queue_family_index = 0;

    // --- Function Pointers ---
    VkResult (*vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*) = nullptr;
    void (*vkDestroyInstance)(VkInstance, const void*) = nullptr; 
    VkResult (*vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*) = nullptr;
    void (*vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*) = nullptr;
    void (*vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*) = nullptr;
    void (*vkGetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties*) = nullptr;
    VkResult (*vkCreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*) = nullptr;
    void (*vkDestroyDevice)(VkDevice, const void*) = nullptr;
    void (*vkGetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue*) = nullptr;
    
    // Memory & Buffer
    VkResult (*vkCreateBuffer)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*) = nullptr;
    void (*vkDestroyBuffer)(VkDevice, VkBuffer, const void*) = nullptr;
    void (*vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements*) = nullptr;
    VkResult (*vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*) = nullptr;
    void (*vkFreeMemory)(VkDevice, VkDeviceMemory, const void*) = nullptr;
    VkResult (*vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) = nullptr;
    VkResult (*vkMapMemory)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**) = nullptr;
    void (*vkUnmapMemory)(VkDevice, VkDeviceMemory) = nullptr;

    // --- Compute Pipeline Symbols ---
    VkResult (*vkCreateShaderModule)(VkDevice, const void*, const void*, VkShaderModule*) = nullptr; // void* for CreateInfo to lazy struct
    void (*vkDestroyShaderModule)(VkDevice, VkShaderModule, const void*) = nullptr;
    VkResult (*vkCreateComputePipelines)(VkDevice, VkPipelineCache, uint32_t, const void*, const void*, VkPipeline*) = nullptr;
    void (*vkDestroyPipeline)(VkDevice, VkPipeline, const void*) = nullptr;
    VkResult (*vkCreatePipelineLayout)(VkDevice, const void*, const void*, VkPipelineLayout*) = nullptr;
    void (*vkDestroyPipelineLayout)(VkDevice, VkPipelineLayout, const void*) = nullptr;
    
    // --- Descriptors ---
    VkResult (*vkCreateDescriptorSetLayout)(VkDevice, const void*, const void*, VkDescriptorSetLayout*) = nullptr;
    void (*vkDestroyDescriptorSetLayout)(VkDevice, VkDescriptorSetLayout, const void*) = nullptr;
    VkResult (*vkCreateDescriptorPool)(VkDevice, const void*, const void*, VkDescriptorPool*) = nullptr;
    void (*vkDestroyDescriptorPool)(VkDevice, VkDescriptorPool, const void*) = nullptr;
    VkResult (*vkAllocateDescriptorSets)(VkDevice, const void*, VkDescriptorSet*) = nullptr;
    void (*vkUpdateDescriptorSets)(VkDevice, uint32_t, const void*, uint32_t, const void*) = nullptr;

// --- Commands ---
    VkResult (*vkCreateCommandPool)(VkDevice, const void*, const void*, VkCommandPool*) = nullptr;
    void (*vkDestroyCommandPool)(VkDevice, VkCommandPool, const void*) = nullptr;
    VkResult (*vkAllocateCommandBuffers)(VkDevice, const void*, VkCommandBuffer*) = nullptr;
    VkResult (*vkBeginCommandBuffer)(VkCommandBuffer, const void*) = nullptr;
    VkResult (*vkEndCommandBuffer)(VkCommandBuffer) = nullptr;
    VkResult (*vkQueueSubmit)(VkQueue, uint32_t, const void*, VkFence) = nullptr;
    VkResult (*vkQueueWaitIdle)(VkQueue) = nullptr;
    void (*vkCmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t) = nullptr;
    void (*vkCmdBindPipeline)(VkCommandBuffer, int, VkPipeline) = nullptr;
    void (*vkCmdBindDescriptorSets)(VkCommandBuffer, int, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*) = nullptr;
    void (*vkCmdPushConstants)(VkCommandBuffer, VkPipelineLayout, uint32_t, uint32_t, uint32_t, const void*) = nullptr;

public:
    inline bool is_ready() const { return device != nullptr; }

    static VulkanBackend& get() {
        static VulkanBackend backend;
        return backend;
    }

    ~VulkanBackend() {
        if (device && vkDestroyDevice) vkDestroyDevice(device, nullptr);
        if (instance && vkDestroyInstance) vkDestroyInstance(instance, nullptr);
        if (lib_handle) dlclose(lib_handle);
    }

    // Helper: Find Memory Type
    uint32_t catchMemoryType(uint32_t typeBits, VkFlags properties) {
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
            if ((typeBits & 1) == 1) {
                if ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }
            typeBits >>= 1;
        }
        throw std::runtime_error("Vulkan: Could not find suitable memory type!");
    }

    bool create_buffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory) {
         if(!device) return false;

         VkBufferCreateInfo bufferInfo = {};
         bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
         bufferInfo.size = size;
         bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
         bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

         if(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != 0) return false;

         VkMemoryRequirements memRequirements;
         vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

         VkMemoryAllocateInfo allocInfo = {};
         allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
         allocInfo.allocationSize = memRequirements.size;
         allocInfo.memoryTypeIndex = catchMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

         if(vkAllocateMemory(device, &allocInfo, nullptr, &memory) != 0) return false;

         vkBindBufferMemory(device, buffer, memory, 0);
         return true;
    }


    bool init() {
        if (lib_handle) return true; // Already init

        // 1. Dynamic Load
        lib_handle = dlopen("libvulkan.so.1", RTLD_LAZY);
        if (!lib_handle) {
            std::cerr << "❌ Vulkan: libvulkan.so.1 not found." << std::endl;
            return false;
        }

        // 2. Load Symbols (Instance)
        *(void**)(&vkCreateInstance) = dlsym(lib_handle, "vkCreateInstance");
        *(void**)(&vkDestroyInstance) = dlsym(lib_handle, "vkDestroyInstance");
        *(void**)(&vkEnumeratePhysicalDevices) = dlsym(lib_handle, "vkEnumeratePhysicalDevices");
        *(void**)(&vkGetPhysicalDeviceQueueFamilyProperties) = dlsym(lib_handle, "vkGetPhysicalDeviceQueueFamilyProperties");
        *(void**)(&vkGetPhysicalDeviceMemoryProperties) = dlsym(lib_handle, "vkGetPhysicalDeviceMemoryProperties");
        *(void**)(&vkGetPhysicalDeviceProperties) = dlsym(lib_handle, "vkGetPhysicalDeviceProperties");

        // ... Load Device pointers using separate specific resolver usually, but here globally for simplicity
        *(void**)(&vkCreateDevice) = dlsym(lib_handle, "vkCreateDevice");
        *(void**)(&vkDestroyDevice) = dlsym(lib_handle, "vkDestroyDevice");
        *(void**)(&vkGetDeviceQueue) = dlsym(lib_handle, "vkGetDeviceQueue");
        *(void**)(&vkCreateBuffer) = dlsym(lib_handle, "vkCreateBuffer");
        *(void**)(&vkDestroyBuffer) = dlsym(lib_handle, "vkDestroyBuffer");
        *(void**)(&vkGetBufferMemoryRequirements) = dlsym(lib_handle, "vkGetBufferMemoryRequirements");
        *(void**)(&vkAllocateMemory) = dlsym(lib_handle, "vkAllocateMemory");
        *(void**)(&vkFreeMemory) = dlsym(lib_handle, "vkFreeMemory");
        *(void**)(&vkBindBufferMemory) = dlsym(lib_handle, "vkBindBufferMemory");
        *(void**)(&vkMapMemory) = dlsym(lib_handle, "vkMapMemory");
        *(void**)(&vkUnmapMemory) = dlsym(lib_handle, "vkUnmapMemory");
        
        // Compute Loading
        *(void**)(&vkCreateShaderModule) = dlsym(lib_handle, "vkCreateShaderModule");
        *(void**)(&vkDestroyShaderModule) = dlsym(lib_handle, "vkDestroyShaderModule");
        *(void**)(&vkCreateComputePipelines) = dlsym(lib_handle, "vkCreateComputePipelines");
        *(void**)(&vkDestroyPipeline) = dlsym(lib_handle, "vkDestroyPipeline");
        *(void**)(&vkCreatePipelineLayout) = dlsym(lib_handle, "vkCreatePipelineLayout");
        *(void**)(&vkDestroyPipelineLayout) = dlsym(lib_handle, "vkDestroyPipelineLayout");
        *(void**)(&vkCreateDescriptorSetLayout) = dlsym(lib_handle, "vkCreateDescriptorSetLayout");
        *(void**)(&vkDestroyDescriptorSetLayout) = dlsym(lib_handle, "vkDestroyDescriptorSetLayout");
        *(void**)(&vkCreateDescriptorPool) = dlsym(lib_handle, "vkCreateDescriptorPool");
        *(void**)(&vkDestroyDescriptorPool) = dlsym(lib_handle, "vkDestroyDescriptorPool");
        *(void**)(&vkAllocateDescriptorSets) = dlsym(lib_handle, "vkAllocateDescriptorSets");
        *(void**)(&vkUpdateDescriptorSets) = dlsym(lib_handle, "vkUpdateDescriptorSets");
        *(void**)(&vkCreateCommandPool) = dlsym(lib_handle, "vkCreateCommandPool");
        *(void**)(&vkDestroyCommandPool) = dlsym(lib_handle, "vkDestroyCommandPool");
        *(void**)(&vkAllocateCommandBuffers) = dlsym(lib_handle, "vkAllocateCommandBuffers");
        *(void**)(&vkBeginCommandBuffer) = dlsym(lib_handle, "vkBeginCommandBuffer");
        *(void**)(&vkEndCommandBuffer) = dlsym(lib_handle, "vkEndCommandBuffer");
        *(void**)(&vkQueueSubmit) = dlsym(lib_handle, "vkQueueSubmit");
        *(void**)(&vkQueueWaitIdle) = dlsym(lib_handle, "vkQueueWaitIdle");
        *(void**)(&vkCmdDispatch) = dlsym(lib_handle, "vkCmdDispatch");
        *(void**)(&vkCmdBindPipeline) = dlsym(lib_handle, "vkCmdBindPipeline");
        *(void**)(&vkCmdDispatch) = dlsym(lib_handle, "vkCmdDispatch");
        *(void**)(&vkCmdBindPipeline) = dlsym(lib_handle, "vkCmdBindPipeline");
        *(void**)(&vkCmdBindDescriptorSets) = dlsym(lib_handle, "vkCmdBindDescriptorSets");
        *(void**)(&vkCmdPushConstants) = dlsym(lib_handle, "vkCmdPushConstants");

        if (!vkCreateInstance || !vkEnumeratePhysicalDevices) {
             std::cerr << "❌ Vulkan: Symbols missing." << std::endl;
             return false;
        }

        // 3. Create Instance
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; // Wait, appInfo is APPLICATION_INFO (0) but passed to createInfo
        // Correction: Structure Type for ApplicationInfo is 0. 
        // Using raw 0 for safety as my enum might be wrong.
        appInfo.sType = (VkStructureType)0; 
        appInfo.pApplicationName = "MM-Rec Engine";
        appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

       if (vkCreateInstance(&createInfo, nullptr, &instance) != 0) {
            std::cerr << "❌ Vulkan: vkCreateInstance failed." << std::endl;
            return false;
        }

        // 4. Find Physical Device
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) return false;

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        physical_device = devices[0]; 

        // 5. Get Memory Properties
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

        // 6. Create Logical Device (with 1 Compute Queue)
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = (VkStructureType)2; // VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
        queueCreateInfo.queueFamilyIndex = 0; // Assume 0 is compute for now (usually Gfx+Compute)
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
        
        if (vkCreateDevice(physical_device, &deviceCreateInfo, nullptr, &device) != 0) {
             std::cerr << "❌ Vulkan: vkCreateDevice failed." << std::endl;
             return false;
        }

        std::cout << "🚀 Vulkan: Initialized! Logical Device Ready." << std::endl;
        
        // Print GPU Info
        if (vkGetPhysicalDeviceProperties) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physical_device, &props);
            
            // Calculate Total VRAM (Heap with DEVICE_LOCAL bit)
            VkDeviceSize total_vram = 0;
             for (uint32_t i = 0; i < memory_properties.memoryHeapCount; i++) {
                if (memory_properties.memoryHeaps[i].flags & 1) { // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 1
                    total_vram += memory_properties.memoryHeaps[i].size;
                }
            }
            
            std::cout << "   GPU: " << props.deviceName << " | VRAM: " 
                      << std::fixed << std::setprecision(1) << (total_vram / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
            std::cout << "   GPU Threads: " << props.limits.maxComputeWorkGroupInvocations << " (Max Concurrency per Block)" << std::endl;
        }
        
        // Get Queue
        vkGetDeviceQueue(device, 0, 0, &compute_queue);
        
        return true;
    }
};

} // namespace mm_rec
