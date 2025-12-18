/**
 * GPU Backend (Dynamic OpenCL Loader)
 * 
 * Enables iGPU acceleration even if OpenCL SDK is not present at build time.
 * Uses dlopen/dlsym to load libOpenCL.so at runtime.
 */

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <memory> 

// Minimal OpenCL Definitions (to avoid needing <CL/cl.h>)
typedef struct _cl_platform_id *    cl_platform_id;
typedef struct _cl_device_id *      cl_device_id;
typedef struct _cl_context *        cl_context;
typedef struct _cl_command_queue *  cl_command_queue;
typedef struct _cl_mem *            cl_mem;
typedef struct _cl_program *        cl_program;
typedef struct _cl_kernel *         cl_kernel;
typedef struct _cl_event *          cl_event;
typedef int                         cl_int;
typedef unsigned int                cl_uint;
typedef unsigned long               cl_ulong;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)

namespace mm_rec {

class GPUBackend {
private:
    void* lib_handle = nullptr;
    
    // Function Pointers
    cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
    cl_int (*clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*) = nullptr;
    cl_context (*clCreateContext)(const void*, cl_uint, const cl_device_id*, void(*)(const char*, const void*, size_t, void*), void*, cl_int*) = nullptr;
    cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*) = nullptr;
    cl_int (*clReleaseContext)(cl_context) = nullptr;

public:
    static GPUBackend& instance() {
        static GPUBackend inst;
        return inst;
    }

    bool load_library() {
        if (lib_handle) return true;
        
        // Try common names
        const char* libs[] = {"libOpenCL.so.1", "libOpenCL.so", "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1"};
        
        for(const char* name : libs) {
            lib_handle = dlopen(name, RTLD_LAZY);
            if(lib_handle) {
                std::cout << "✅ iGPU: Loaded OpenCL library from " << name << std::endl;
                break;
            }
        }
        
        if (!lib_handle) {
            std::cout << "⚠️  iGPU: Could not load OpenCL library. GPU acceleration unavailable." << std::endl;
            return false;
        }
        
        // Load Symbols
        *(void**)(&clGetPlatformIDs) = dlsym(lib_handle, "clGetPlatformIDs");
        *(void**)(&clGetDeviceIDs) = dlsym(lib_handle, "clGetDeviceIDs");
        *(void**)(&clCreateContext) = dlsym(lib_handle, "clCreateContext");
        // ... Load more as needed
        
        if (!clGetPlatformIDs) {
            std::cout << "❌ iGPU: Failed to load symbols." << std::endl;
            return false;
        }
        
        return true;
    }

    bool init_device() {
        if (!load_library()) return false;
        
        cl_platform_id platform;
        cl_uint num_platforms;
        if (clGetPlatformIDs(1, &platform, &num_platforms) != CL_SUCCESS) return false;
        
        cl_device_id device;
        cl_uint num_devices;
        // Check specifically for GPU
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices) != CL_SUCCESS) {
             std::cout << "⚠️  iGPU: OpenCL Platform found, but NO GPU device detected." << std::endl;
             return false;
        }
        
        std::cout << "🚀 iGPU: Intel/Compatible GPU Device Initialized!" << std::endl;
        return true;
    }
};

} // namespace mm_rec
