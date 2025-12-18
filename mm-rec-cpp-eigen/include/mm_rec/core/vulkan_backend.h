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
#include <cstdint>

// Minimal Vulkan Types to avoid <vulkan/vulkan.h> dependency
#define VK_MAKE_VERSION(major, minor, patch) \
    ((((uint32_t)(major)) << 22) | (((uint32_t)(minor)) << 12) | ((uint32_t)(patch)))

typedef struct VkInstance_T* VkInstance;
typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
typedef struct VkDevice_T* VkDevice;
typedef uint32_t VkFlags;
typedef uint32_t VkStructureType;

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

namespace mm_rec {

class VulkanBackend {
private:
    void* lib_handle = nullptr;
    VkInstance instance = nullptr;
    VkPhysicalDevice physical_device = nullptr;

    // Function Pointers
    typedef int VkResult;
    VkResult (*vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*) = nullptr;
    void (*vkDestroyInstance)(VkInstance, const void*) = nullptr; 
    VkResult (*vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*) = nullptr;

public:
    static VulkanBackend& get() {
        static VulkanBackend backend;
        return backend;
    }

    ~VulkanBackend() {
        if (instance && vkDestroyInstance) {
            vkDestroyInstance(instance, nullptr);
        }
        if (lib_handle) dlclose(lib_handle);
    }

    bool init() {
        if (lib_handle) return true; // Already init

        // 1. Dynamic Load
        lib_handle = dlopen("libvulkan.so.1", RTLD_LAZY);
        if (!lib_handle) {
            std::cerr << "❌ Vulkan: libvulkan.so.1 not found." << std::endl;
            return false;
        }

        // 2. Load Symbols
        *(void**)(&vkCreateInstance) = dlsym(lib_handle, "vkCreateInstance");
        *(void**)(&vkDestroyInstance) = dlsym(lib_handle, "vkDestroyInstance");
        *(void**)(&vkEnumeratePhysicalDevices) = dlsym(lib_handle, "vkEnumeratePhysicalDevices");

        if (!vkCreateInstance || !vkEnumeratePhysicalDevices) {
             std::cerr << "❌ Vulkan: Symbols missing." << std::endl;
             return false;
        }

        // 3. Create Instance
        VkApplicationInfo appInfo = {};
        appInfo.sType = 1; // VK_STRUCTURE_TYPE_APPLICATION_INFO = 0, but usually 1? Wait, checking spec. 
                           // VK_STRUCTURE_TYPE_APPLICATION_INFO is 0 in some enums, 1 in others? 
                           // Actually it's 0. Let's use 0.
        appInfo.sType = 0; 
        appInfo.pApplicationName = "MM-Rec Engine";
        appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = 1; // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
        createInfo.pApplicationInfo = &appInfo;

        if (vkCreateInstance(&createInfo, nullptr, &instance) != 0) {
            std::cerr << "❌ Vulkan: vkCreateInstance failed." << std::endl;
            return false;
        }

        // 4. Find Device
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            std::cerr << "⚠️  Vulkan: No devices found." << std::endl;
            return false;
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        physical_device = devices[0]; // Pick first

        std::cout << "🚀 Vulkan: Initialized! Found " << deviceCount << " GPU(s)." << std::endl;
        return true;
    }
};

} // namespace mm_rec
