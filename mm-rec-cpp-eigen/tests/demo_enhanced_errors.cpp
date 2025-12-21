// Demo: Enhanced Error Handling with Correlation ID

#include "mm_rec/infrastructure/result.h"
#include "mm_rec/infrastructure/request_context.h"
#include "mm_rec/application/service_configurator.h"
#include "mm_rec/infrastructure/logger.h"
#include <iostream>

using namespace mm_rec;
using namespace mm_rec::net;

// Simulate database layer
Result<std::string> database_query(const std::string& query, RequestContext* ctx) {
    // Simulate database timeout
    if (query.find("slow_query") != std::string::npos) {
        return Result<std::string>::err(
            ErrorContext("Database connection timeout")
                .with_correlation(ctx->correlation_id)
                .with_trace_id(ctx->trace_id)
                .with_code("DB_TIMEOUT")
                .at("DatabaseLayer::execute_query")
                .with_metadata("query", query)
                .with_metadata("timeout_ms", "5000")
                .with_trace("DatabaseConnection::connect() - timeout after 5s")
                .with_trace("DatabaseLayer::execute_query() - failed to get connection")
        );
    }
    
    return Result<std::string>::ok("Query result: SUCCESS");
}

// Simulate service layer
Result<std::string> get_user(int user_id, RequestContext* ctx) {
    auto result = database_query("SELECT * FROM users WHERE id = " + std::to_string(user_id) + " slow_query", ctx);
    
    if (result.is_err()) {
        // Add context to error chain
        return Result<std::string>::err(
            result.error()
                .with_trace("UserService::get_user(" + std::to_string(user_id) + ") - database query failed")
        );
    }
    
    return result;
}

// Simulate API handler
void handle_request(RequestContext* ctx, ILogger* logger) {
    logger->info(ctx->log_prefix() + "Processing user request");
    
    auto result = get_user(123, ctx);
    
    if (result.is_err()) {
        // Full error report (for admins/developers)
        std::cout << "\n" << result.error().to_string() << std::endl;
        
        // Compact log line (for log files)
        logger->error(result.error().to_log_line());
        
        // User-friendly message
        std::cout << "❌ Failed to load user\n" << std::endl;
    } else {
        logger->info(ctx->log_prefix() + "User loaded successfully");
        std::cout << "✅ " << result.value() << "\n" << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Enhanced Error Handling Demo" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    ServiceConfigurator::initialize();
    
    // ========================================
    // Scenario 1: Successful Request
    // ========================================
    std::cout << "📡 Request #1: Successful Query\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        auto logger = scope->resolve<ILogger>();
        
        ctx->method = "GET";
        ctx->path = "/api/users/123";
        
        auto result = database_query("SELECT * FROM users WHERE id = 123", ctx.get());
        
        if (result.is_ok()) {
            std::cout << "✅ " << result.value() << "\n" << std::endl;
        }
    }
    
    // ========================================
    // Scenario 2: Error with Full Context
    // ========================================
    std::cout << "\n📡 Request #2: Database Timeout (Full Error Tracking)\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        auto logger = scope->resolve<ILogger>();
        
        ctx->method = "GET";
        ctx->path = "/api/users/456";
        ctx->user_id = "admin_user";
        
        handle_request(ctx.get(), logger.get());
    }
    
    // ========================================
    // Scenario 3: Error Grep Simulation
    // ========================================
    std::cout << "\n📋 Log Grep Simulation (Finding Error Flow)\n" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        auto logger = scope->resolve<ILogger>();
        
        std::string corr_id = ctx->correlation_id;
        
        // Simulate multiple log entries
        logger->info(ctx->log_prefix() + "Gateway: Request received");
        logger->info(ctx->log_prefix() + "Auth: Token validated");
        logger->info(ctx->log_prefix() + "UserService: Loading user 789");
        
        auto result = database_query("SELECT * FROM users WHERE id = 789 slow_query", ctx.get());
        
        if (result.is_err()) {
            logger->error(result.error().to_log_line());
        }
        
        logger->info(ctx->log_prefix() + "Gateway: 500 Internal Server Error sent");
        
        std::cout << "\n💡 Grep Command to Find Full Request Flow:" << std::endl;
        std::cout << "   $ grep '" << corr_id.substr(0, 15) << "' app.log\n" << std::endl;
        std::cout << "   This would show ALL logs for this request!" << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ✅ Enhanced Error Handling Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n💡 Benefits:" << std::endl;
    std::cout << "  ✓ Correlation ID - Track errors across services" << std::endl;
    std::cout << "  ✓ Error Stack Trace - See error propagation chain" << std::endl;
    std::cout << "  ✓ Error Codes - Machine-readable (DB_TIMEOUT, etc.)" << std::endl;
    std::cout << "  ✓ Metadata - Additional context (query, timeout, etc.)" << std::endl;
    std::cout << "  ✓ Timestamps - When error occurred" << std::endl;
    std::cout << "  ✓ Location - Where error originated" << std::endl;
    std::cout << "  ✓ Full Error Report - For debugging" << std::endl;
    std::cout << "  ✓ Compact Log Line - For log aggregation" << std::endl;
    
    std::cout << "\n📝 Usage Pattern:" << std::endl;
    std::cout << R"(
  // Create error with full context:
  return Result<T>::err(
      ErrorContext("Database timeout")
          .with_correlation(ctx->correlation_id)
          .with_code("DB_TIMEOUT")
          .at("DatabaseLayer::query")
          .with_metadata("timeout_ms", "5000")
          .with_trace("Step 1: Connect failed")
          .with_trace("Step 2: Query aborted")
  );
  
  // Handle error:
  if (result.is_err()) {
      // Full report for debugging
      std::cout << result.error().to_string();
      
      // Compact for logs
      LOG_ERROR(result.error().to_log_line());
  }
  
  // Grep logs by correlation ID:
  $ grep 'COR:abc123' app.log
)" << std::endl;
    
    return 0;
}
