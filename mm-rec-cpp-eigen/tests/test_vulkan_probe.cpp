#include "mm_rec/application/service_configurator.h" // DI Configurator
#include "mm_rec/core/i_compute_backend.h"       // Interface
#include <iostream>

using namespace mm_rec;

int main() {
    std::cout << "=== Test: Vulkan Backend Integration (DI) ===" << std::endl;
    
    // 1. Initialize DI Container
    ServiceConfigurator::initialize();
    
    // 2. Resolve IComputeBackend
    auto backend = ServiceConfigurator::container().resolve<IComputeBackend>();
    
    // 3. Use Backend
    bool ready = backend->init();
    
    if (ready) {
        std::cout << "✅ SUCCESS: Vulkan engine initialized via DI." << std::endl;
        std::cout << "   Device Name: " << backend->get_device_name() << std::endl;
    } else {
        std::cout << "❌ FAILURE: Context creation failed." << std::endl;
    }
    
    return 0;
}
