#include "mm_rec/infrastructure/http_server.h"
#include "mm_rec/application/service_configurator.h"
#include "mm_rec/infrastructure/request_context.h"
#include "mm_rec/infrastructure/logger.h"
#include <iostream>

using namespace mm_rec;
using namespace mm_rec::net;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  RequestContext Demo" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Setup DI Container
    DIContainer container;
    ServiceConfigurator::configure_services(container);
    
    // Simulate 3 HTTP requests
    for (int i = 1; i <= 3; i++) {
        std::cout << "\n--- Request #" << i << " ---" << std::endl;
        
        // Create request scope
        Scope request_scope(container);
        
        // Simulate handler - inject RequestContext
        auto ctx = request_scope.resolve<RequestContext>();
        auto logger = request_scope.resolve<Logger>();
        
        // Set request metadata
        ctx->method = "GET";
        ctx->path = "/api/users";
        ctx->ip_address = "192.168.1." + std::to_string(100 + i);
        ctx->user_agent = "Mozilla/5.0";
        
        // Use context for logging
        std::cout << ctx->log_prefix() << "Started: " << ctx->method << " " << ctx->path << std::endl;
        std::cout << ctx->log_prefix() << "Client: " << ctx->ip_address << std::endl;
        
        // Simulate work
        std::this_thread::sleep_for(std::chrono::milliseconds(50 + i * 10));
        
        // Log with timing
        std::cout << ctx->log_prefix() << "Completed in " << ctx->elapsed_ms() << "ms" << std::endl;
        
        // Multiple resolves in same scope return SAME instance
        auto ctx2 = request_scope.resolve<RequestContext>();
        std::cout << ctx->log_prefix() << "Same instance? " << (ctx.get() == ctx2.get() ? "YES ✓" : "NO ✗") << std::endl;
        
    } // Scope destroyed → RequestContext cleaned up (RAII)
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ✅ Demo Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n💡 Benefits:" << std::endl;
    std::cout << "  - Unique request ID per request" << std::endl;
    std::cout << "  - Automatic timing tracking" << std::endl;
    std::cout << "  - Thread-safe (per-request scope)" << std::endl;
    std::cout << "  - Easy to inject into services" << std::endl;
    std::cout << "  - RAII cleanup (no manual delete)" << std::endl;
    
    return 0;
}
