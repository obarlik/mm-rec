#include "mm_rec/core/opengl_backend.h"
#include <iostream>

int main() {
    std::cout << "=== Test: OpenGL Backend Integration ===" << std::endl;
    
    // Attempt Fallback Init
    bool ready = mm_rec::OpenGLBackend::get().init();
    
    if (ready) {
         std::cout << "✅ SUCCESS: OpenGL Backend is online." << std::endl;
         std::cout << "   We are NEVER gathering pears. We calculate." << std::endl;
    } else {
         std::cout << "❌ FAILURE: Even OpenGL is missing. Are we on a toaster?" << std::endl;
         return 1;
    }
    
    return 0;
}
