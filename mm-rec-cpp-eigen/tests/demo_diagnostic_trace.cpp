// Demo: Error-Triggered Diagnostic Logging

#include "mm_rec/utils/request_context.h"
#include "mm_rec/utils/service_configurator.h"
#include <iostream>

using namespace mm_rec;
using namespace mm_rec::net;

// Simulate service layers with diagnostic tracing

void simulate_database_call(RequestContext* ctx, bool will_fail) {
    ctx->add_trace("INFO", "Database", "Connecting to database...");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    ctx->add_trace("INFO", "Database", "Connection established");
    
    ctx->add_trace("INFO", "Database", "Executing query: SELECT * FROM users");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    
    if (will_fail) {
        ctx->add_trace("ERROR", "Database", "Query timeout after 5000ms!");
        ctx->add_trace("ERROR", "Database", "Connection lost");
    } else {
        ctx->add_trace("INFO", "Database", "Query completed successfully");
        ctx->add_trace("INFO", "Database", "Returning 42 rows");
    }
}

void simulate_auth_service(RequestContext* ctx, bool will_fail) {
    ctx->add_trace("INFO", "Auth", "Validating JWT token...");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    ctx->add_trace("INFO", "Auth", "Token signature verified");
    ctx->add_trace("INFO", "Auth", "User: alice (ID: 123)");
    
    // Call database
    simulate_database_call(ctx, will_fail);
    
    if (!will_fail) {
        ctx->add_trace("INFO", "Auth", "Authorization successful");
    } else {
        ctx->add_trace("ERROR", "Auth", "Authorization failed due to database error");
    }
}

void simulate_api_handler(RequestContext* ctx, bool will_fail) {
    ctx->add_trace("INFO", "Gateway", "Request received: POST /api/users");
    ctx->add_trace("INFO", "Gateway", "Client IP: 192.168.1.100");
    
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    
    ctx->add_trace("INFO", "Gateway", "Routing to auth service");
    
    // Call auth service
    simulate_auth_service(ctx, will_fail);
    
    if (!will_fail) {
        ctx->add_trace("INFO", "Gateway", "200 OK - Response sent");
    } else {
        ctx->add_trace("ERROR", "Gateway", "500 Internal Server Error - Response sent");
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Error-Triggered Diagnostic Logging" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    ServiceConfigurator::initialize();
    
    // ========================================
    // Scenario 1: SUCCESSFUL Request
    // ========================================
    std::cout << "✅ Request #1: SUCCESSFUL (No disk I/O)\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        
        ctx->method = "POST";
        ctx->path = "/api/users";
        
        // Simulate request processing
        simulate_api_handler(ctx.get(), false);  // will_fail = false
        
        std::cout << "📊 Diagnostic trace collected: " << ctx->trace_count() << " entries\n";
        std::cout << "💾 Disk writes: 0 (success - trace discarded)\n";
        std::cout << "⏱️  Duration: " << ctx->elapsed_ms() << "ms\n" << std::endl;
        
        // On success: trace is discarded (no flush to disk)
        ctx->clear_trace();
        
        std::cout << "✓ Request completed successfully, trace discarded\n" << std::endl;
    }
    
    // ========================================
    // Scenario 2: FAILED Request (Error-Triggered Flush)
    // ========================================
    std::cout << "\n❌ Request #2: FAILED (Full trace written to disk)\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        
        ctx->method = "POST";
        ctx->path = "/api/users";
        
        // Simulate request processing with error
        simulate_api_handler(ctx.get(), true);  // will_fail = true
        
        std::cout << "📊 Diagnostic trace collected: " << ctx->trace_count() << " entries\n";
        std::cout << "⏱️  Duration: " << ctx->elapsed_ms() << "ms\n" << std::endl;
        
        // On error: flush full trace to disk/log
        std::cout << "🔥 ERROR DETECTED! Flushing diagnostic trace...\n";
        
        std::string trace_dump = ctx->flush_trace();
        
        // This would normally go to log file
        std::cout << trace_dump << std::endl;
        
        std::cout << "💾 Disk write: 1 error report with full trace\n" << std::endl;
        std::cout << "✓ Full diagnostic context saved for debugging\n" << std::endl;
    }
    
    // ========================================
    // Scenario 3: Performance Comparison
    // ========================================
    std::cout << "\n📈 Performance Comparison\n" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    int successful_requests = 100;
    int failed_requests = 5;
    
    std::cout << "Scenario: " << successful_requests << " successful + " << failed_requests << " failed requests\n\n";
    
    std::cout << "Traditional Logging (all requests write to disk):\n";
    std::cout << "  Disk writes: " << (successful_requests + failed_requests) << " (every request)\n";
    std::cout << "  I/O overhead: HIGH\n\n";
    
    std::cout << "Error-Triggered Diagnostic Logging:\n";
    std::cout << "  Disk writes: " << failed_requests << " (only failures)\n";
    std::cout << "  I/O overhead: LOW (" << (failed_requests * 100 / (successful_requests + failed_requests)) << "% of traditional)\n";
    std::cout << "  Context: FULL trace for failures, ZERO overhead for success\n";
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ✅ Demo Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n💡 Benefits:" << std::endl;
    std::cout << "  ✓ Zero disk I/O for successful requests (performance!)" << std::endl;
    std::cout << "  ✓ Full diagnostic trace for failures (debugging!)" << std::endl;
    std::cout << "  ✓ Timing information for each step" << std::endl;
    std::cout << "  ✓ Component-level tracing" << std::endl;
    std::cout << "  ✓ Memory-bounded (max 1000 entries)" << std::endl;
    std::cout << "  ✓ Correlation ID for distributed tracing" << std::endl;
    
    std::cout << "\n📝 Usage Pattern:" << std::endl;
    std::cout << R"(
  // During request processing:
  ctx->add_trace("INFO", "Auth", "Validating token");
  ctx->add_trace("INFO", "Database", "Query executed");
  
  // On success:
  ctx->clear_trace();  // Discard, no disk I/O
  
  // On error:
  LOG_ERROR(ctx->flush_trace());  // Write FULL trace to disk
)" << std::endl;
    
    return 0;
}
