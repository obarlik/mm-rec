// Example: Manual Scope Creation (ASP.NET Core CreateScope pattern)

#include "mm_rec/application/service_configurator.h"
#include "mm_rec/infrastructure/request_context.h"
#include "mm_rec/infrastructure/logger.h"
#include <iostream>
#include <thread>

using namespace mm_rec;
using namespace mm_rec::net;

// Background job example
void process_background_job(int job_id) {
    std::cout << "\n🔧 Background Job #" << job_id << " started" << std::endl;
    
    // ✨ Create scope manually (like ASP.NET Core CreateScope)
    auto scope = ServiceConfigurator::create_scope();
    
    // Resolve scoped services
    auto ctx = scope->resolve<RequestContext>();
    auto logger = scope->resolve<Logger>();
    
    // Set job context
    ctx->path = "/jobs/process";
    ctx->method = "JOB";
    
    logger->info(ctx->log_prefix() + "Processing job #" + std::to_string(job_id));
    
    // Simulate work
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    logger->info(ctx->log_prefix() + "Job completed in " + std::to_string(ctx->elapsed_ms()) + "ms");
    
    std::cout << "  [" << ctx->request_id << "] Job #" << job_id << " completed" << std::endl;
    
} // Scope destroyed here (RAII) → RequestContext cleaned up

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Manual Scope Creation Demo" << std::endl;
    std::cout << "  (ASP.NET Core CreateScope Pattern)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Initialize DI container
    ServiceConfigurator::initialize();
    
    std::cout << "✅ DI Container initialized\n" << std::endl;
    
    // === Use Case 1: HTTP Request (Auto Scope) ===
    std::cout << "📡 Use Case 1: HTTP Request Handler" << std::endl;
    {
        // Middleware creates scope automatically
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        ctx->method = "GET";
        ctx->path = "/api/users";
        
        std::cout << "  [" << ctx->request_id << "] HTTP request handled" << std::endl;
    } // Auto cleanup
    
    std::cout << std::endl;
    
    // === Use Case 2: Background Jobs (Manual Scope) ===
    std::cout << "⚙️  Use Case 2: Background Jobs" << std::endl;
    for (int i = 1; i <= 3; i++) {
        process_background_job(i);
    }
    
    std::cout << std::endl;
    
    // === Use Case 3: Console Command (Manual Scope) ===
    std::cout << "💻 Use Case 3: Console Command" << std::endl;
    {
        auto scope = ServiceConfigurator::create_scope();
        auto logger = scope->resolve<Logger>();
        auto ctx = scope->resolve<RequestContext>();
        
        ctx->method = "CMD";
        ctx->path = "/commands/migrate";
        
        logger->info(ctx->log_prefix() + "Running database migration...");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        logger->info(ctx->log_prefix() + "Migration completed in " + std::to_string(ctx->elapsed_ms()) + "ms");
        
        std::cout << "  [" << ctx->request_id << "] Command executed" << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ✅ All Scopes Completed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n💡 Use Cases for CreateScope:" << std::endl;
    std::cout << "  1. Background jobs/tasks" << std::endl;
    std::cout << "  2. Console commands" << std::endl;
    std::cout << "  3. Scheduled jobs (cron)" << std::endl;
    std::cout << "  4. Message queue consumers" << std::endl;
    std::cout << "  5. WebSocket connections" << std::endl;
    
    std::cout << "\n📝 Pattern (like ASP.NET Core):" << std::endl;
    std::cout << R"(
  // C# ASP.NET Core:
  using (var scope = serviceProvider.CreateScope())
  {
      var service = scope.ServiceProvider.GetService<IMyService>();
      // ...
  }
  
  // C++ mm-rec:
  {
      auto scope = ServiceConfigurator::create_scope();
      auto service = scope->resolve<MyService>();
      // ...
  } // Auto cleanup (RAII)
)" << std::endl;
    
    return 0;
}
