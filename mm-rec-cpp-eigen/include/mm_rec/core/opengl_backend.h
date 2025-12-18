/**
 * OpenGL Backend (Legacy GPGPU Fallback)
 * 
 * When Vulkan and OpenCL are missing, we turn to the oldest trick in the book:
 * Render-to-Texture.
 * 
 * This class dynamically loads libGL / libEGL to ensure we can access the GPU.
 */

#pragma once

#include <iostream>
#include <dlfcn.h>
#include <string>

namespace mm_rec {

class OpenGLBackend {
private:
    void* lib_gl = nullptr;
    void* lib_egl = nullptr;
    
public:
    static OpenGLBackend& get() {
        static OpenGLBackend backend;
        return backend;
    }
    
    ~OpenGLBackend() {
        if (lib_gl) dlclose(lib_gl);
        if (lib_egl) dlclose(lib_egl);
    }

    bool init() {
        if (lib_gl || lib_egl) return true;

        // 1. Try EGL (Modern Headless)
        lib_egl = dlopen("libEGL.so.1", RTLD_LAZY);
        if (lib_egl) {
            std::cout << "🛡️ OpenGL Backend: Loaded libEGL (Headless capable)." << std::endl;
            // In a real impl, we would dlsym eglGetDisplay etc.
            return true;
        }
        
        // 2. Try Legacy GLX (Desktop)
        lib_gl = dlopen("libGL.so.1", RTLD_LAZY);
        if (lib_gl) {
            std::cout << "🛡️ OpenGL Backend: Loaded libGL (Desktop)." << std::endl;
            return true;
        }

        std::cerr << "❌ OpenGL Backend: Failed to load EGL or GL." << std::endl;
        return false;
    }
};

} // namespace mm_rec
