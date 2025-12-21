// Demo: Distributed Tracing with Correlation ID

#include "mm_rec/application/service_configurator.h"
#include "mm_rec/infrastructure/request_context.h"
#include "mm_rec/infrastructure/logger.h"
#include <iostream>

using namespace mm_rec;
using namespace mm_rec::net;

// Simulate microservice call chain
void database_service(std::shared_ptr<RequestContext> ctx, std::shared_ptr<ILogger> logger) {
    logger->info(ctx->log_prefix() + "📊 Database: Executing query");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    logger->info(ctx->log_prefix() + "✅ Database: Query complete");
}

void auth_service(std::shared_ptr<RequestContext> ctx, std::shared_ptr<ILogger> logger) {
    logger->info(ctx->log_prefix() + "🔐 Auth: Validating token");
    
    // Simulate user context
    ctx->user_id = "user_42";
    ctx->username = "alice";
    ctx->session_id = "sess_abc123";
    
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    logger->info(ctx->log_prefix() + "✅ Auth: User authenticated");
    
    // Call downstream service
    database_service(ctx, logger);
}

void api_gateway(std::shared_ptr<RequestContext> ctx, std::shared_ptr<ILogger> logger) {
    logger->info(ctx->log_prefix() + "🌐 Gateway: Request received");
    logger->info(ctx->log_prefix() + "   W3C Trace: " + ctx->to_traceparent());
    
    // Set baggage
    ctx->set_baggage("experiment_id", "exp_001");
    ctx->set_baggage("feature_flag", "new_ui_enabled");
    
    // Call auth service
    auth_service(ctx, logger);
    
    logger->info(ctx->log_prefix() + "✅ Gateway: Response sent (" + std::to_string(ctx->elapsed_ms()) + "ms)");
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Distributed Tracing Demo" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Initialize DI
    ServiceConfigurator::initialize();
    
    // ========================================
    // Scenario 1: Single Request Flow
    // ========================================
    std::cout << "📡 Request #1: Complete Flow\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        auto logger = scope->resolve<ILogger>();
        
        // Set request metadata
        ctx->method = "POST";
        ctx->path = "/api/users/create";
        ctx->ip_address = "192.168.1.100";
        
        // Execute flow
        api_gateway(ctx, logger);
        
        std::cout << "\n📋 Request Summary:" << std::endl;
        std::cout << "   Correlation ID: " << ctx->correlation_id << std::endl;
        std::cout << "   Trace ID:       " << ctx->trace_id << std::endl;
        std::cout << "   User:           " << ctx->username << " (ID: " << ctx->user_id << ")" << std::endl;
        std::cout << "   Baggage:        " << ctx->get_baggage("experiment_id") << std::endl;
        std::cout << "   Total Time:     " << ctx->elapsed_ms() << "ms" << std::endl;
    }
    
    // ========================================
    // Scenario 2: Trace Propagation (Incoming Request)
    // ========================================
    std::cout << "\n\n📡 Request #2: Incoming with W3C Trace Header\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        auto logger = scope->resolve<ILogger>();
        
        // Simulate incoming W3C traceparent header from upstream service
        std::string incoming_traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        
        std::cout << "🔗 Received traceparent: " << incoming_traceparent << std::endl;
        
        if (ctx->from_traceparent(incoming_traceparent)) {
            logger->info(ctx->log_prefix() + "✅ Trace context propagated!");
            logger->info(ctx->log_prefix() + "   Trace ID preserved: " + ctx->trace_id);
            logger->info(ctx->log_prefix() + "   Parent Span: " + ctx->parent_span_id);
            logger->info(ctx->log_prefix() + "   New Span: " + ctx->span_id);
        }
    }
    
    // ========================================
    // Scenario 3: Multiple Concurrent Requests
    // ========================================
    std::cout << "\n\n📡 Concurrent Requests (Correlation ID Isolation)\n" << std::endl;
    
    for (int i = 1; i <= 3; i++) {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        auto logger = scope->resolve<ILogger>();
        
        ctx->method = "GET";
        ctx->path = "/api/users/" + std::to_string(i);
        
        logger->info(ctx->log_prefix() + "Request #" + std::to_string(i) + " started");
        logger->info(ctx->log_prefix() + "Processing...");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        logger->info(ctx->log_prefix() + "Completed in " + std::to_string(ctx->elapsed_ms()) + "ms");
        std::cout << std::endl;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "  ✅ Distributed Tracing Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n💡 Features Demonstrated:" << std::endl;
    std::cout << "  ✓ Correlation ID - End-to-end tracking" << std::endl;
    std::cout << "  ✓ Trace ID - Distributed tracing (W3C compliant)" << std::endl;
    std::cout << "  ✓ Span ID - Operation hierarchy" << std::endl;
    std::cout << "  ✓ User Context - Authentication tracking" << std::endl;
    std::cout << "  ✓ Baggage - Context propagation" << std::endl;
    std::cout << "  ✓ W3C Trace Context - Standard headers" << std::endl;
    std::cout << "  ✓ Per-request isolation - Concurrent safety" << std::endl;
    
    std::cout << "\n📝 Log Grep Example:" << std::endl;
    std::cout << "  # Find all logs for specific request:" << std::endl;
    std::cout << "  $ grep 'COR:abc123' application.log" << std::endl;
    std::cout << "\n  # Find all logs for distributed trace:" << std::endl;
    std::cout << "  $ grep 'TR:0af76519' application.log" << std::endl;
    
    return 0;
}
