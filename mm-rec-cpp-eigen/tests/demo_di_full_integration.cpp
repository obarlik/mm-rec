// Example: Using DI in Dashboard Handlers
// Demonstrates Logger + RequestContext injection

#include "mm_rec/utils/service_configurator.h"
#include "mm_rec/utils/request_context.h"
#include "mm_rec/utils/logger.h"
#include <iostream>

using namespace mm_rec;
using namespace mm_rec::net;

// Mock Request for demo
struct MockRequest {
    Scope* scope = nullptr;
    std::string method;
    std::string path;
    
    template<typename T>
    std::shared_ptr<T> get() const {
        return scope->template resolve<T>();
    }
};

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Full DI Integration Example" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // 1. Setup DI Container (ServiceConfigurator pattern)
    DIContainer container;
    ServiceConfigurator::configure_services(container);
    
    std::cout << "✅ DI Container configured with:" << std::endl;
    std::cout << "   Singleton: Logger, Config, EventBus, MetricsManager" << std::endl;
    std::cout << "   Scoped: RequestContext" << std::endl;
    std::cout << std::endl;
    
    // 2. Simulate HTTP request handlers
    auto status_handler = [](const MockRequest& req) {
        // ✨ DEPENDENCY INJECTION ✨
        auto ctx = req.get<RequestContext>();
        auto logger = req.get<Logger>();
        
        // Use injected services
        logger->info(ctx->log_prefix() + "Status request from " + ctx->ip_address);
        
        std::string response = "{\n";
        response += "  \"request_id\": \"" + ctx->request_id + "\",\n";
        response += "  \"elapsed_ms\": " + std::to_string(ctx->elapsed_ms()) + ",\n";
        response += "  \"path\": \"" + ctx->path + "\"\n";
        response += "}";
        
        std::cout << ctx->log_prefix() << "Response: " << response << std::endl;
        logger->info(ctx->log_prefix() + "Completed in " + std::to_string(ctx->elapsed_ms()) + "ms");
    };
    
    // 3. Simulate requests
    std::cout << "📡 Simulating HTTP Requests...\n" << std::endl;
    
    for (int i = 1; i <= 3; i++) {
        std::cout << "--- Request #" << i << " ---" << std::endl;
        
        // Create request scope (RAII)
        Scope request_scope(container);
        
        MockRequest req;
        req.scope = &request_scope;
        req.method = "GET";
        req.path = "/api/status";
        
        // Set request metadata
        auto ctx = request_scope.resolve<RequestContext>();
        ctx->ip_address = "192.168.1." + std::to_string(100 + i);
        ctx->method = req.method;
        ctx->path = req.path;
        
        // Execute handler
        status_handler(req);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::cout << std::endl;
        
    } // Scope destroyed → RequestContext cleaned up
    
    std::cout << "========================================" << std::endl;
    std::cout << "  ✅ All Services Working!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n🎯 Registered Services in DI:" << std::endl;
    std::cout << "\n  📦 Singleton (App-wide):" << std::endl;
    std::cout << "     - Logger (logging)" << std::endl;
    std::cout << "     - Config (configuration)" << std::endl;
    std::cout << "     - EventBus (pub/sub)" << std::endl;
    std::cout << "     - MetricsManager (metrics)" << std::endl;
    
    std::cout << "\n  🔄 Request-Scoped (Per-request):" << std::endl;
    std::cout << "     - RequestContext (tracing, timing)" << std::endl;
    
    std::cout << "\n💡 Benefits:" << std::endl;
    std::cout << "  ✓ No global singletons in handlers" << std::endl;
    std::cout << "  ✓ Easy to test (inject mocks)" << std::endl;
    std::cout << "  ✓ Explicit dependencies via constructor" << std::endl;
    std::cout << "  ✓ Per-request isolation" << std::endl;
    std::cout << "  ✓ Automatic cleanup (RAII)" << std::endl;
    
    std::cout << "\n📝 Usage in real handler:" << std::endl;
    std::cout << R"(
  server->register_handler("/api/data", [](const Request& req, auto res) {
      auto ctx = req.get<RequestContext>();    // Injected ✨
      auto logger = req.get<Logger>();          // Injected ✨
      
      logger->info(ctx->log_prefix() + "Processing...");
      // ... business logic ...
      res->send("OK");
  });
)" << std::endl;
    
    return 0;
}
