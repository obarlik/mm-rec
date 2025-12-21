// Example: Using Diagnostic Services with DI

#include "mm_rec/utils/service_configurator.h"
#include "mm_rec/utils/diagnostic_dashboard.h"
#include "mm_rec/utils/http_server.h"

using namespace mm_rec;
using namespace mm_rec::diagnostics;

int main() {
    // Initialize DI container
    ServiceConfigurator::initialize();
    
    // Create application scope
    auto scope = ServiceConfigurator::create_scope();
    
    // Resolve services from DI (NO MORE SINGLETONS!)
    auto diag_mgr = scope->resolve<DiagnosticManager>();
    auto alert_mgr = scope->resolve<AlertManager>();
    auto logger = scope->resolve<ILogger>();
    
    // Configure diagnostic settings
    diag_mgr->set_tracing_enabled(true);
    diag_mgr->set_max_error_traces(100);
    
    alert_mgr->set_error_rate_threshold(5.0);
    alert_mgr->set_slow_request_threshold(1000.0);
    
    // Create HTTP server
    HttpServer server(8080);
    
    // Register diagnostic endpoints (WITH DEPENDENCY INJECTION!)
    register_diagnostic_endpoints(&server, diag_mgr.get(), alert_mgr.get());
    
    // Add middleware for automatic tracking
    server.use([&](const Request& req, auto res, auto next) {
        auto start = std::chrono::steady_clock::now();
        
        // Create request-scoped context
        Scope req_scope(ServiceConfigurator::container());
        auto ctx = req_scope.resolve<RequestContext>();
        
        // Process request
        next(req, res);
        
        // Calculate response time
        auto end = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        ).count();
        
        // Record performance
        alert_mgr->record_response_time(duration_ms);
        
        // Log
        std::string level = (res->status_code >= 400) ? "ERROR" : "INFO";
        alert_mgr->add_log(
            level,
            "Gateway",
            req.method + " " + req.path + " - " + std::to_string(res->status_code),
            ctx->correlation_id
        );
        
        // On error: Store full trace
        if (res->status_code >= 400) {
            ErrorTrace trace;
            trace.correlation_id = ctx->correlation_id;
            trace.trace_id = ctx->trace_id;
            trace.error_message = "Request failed";
            trace.error_code = "HTTP_" + std::to_string(res->status_code);
            trace.timestamp = std::chrono::system_clock::now();
            trace.full_trace = ctx->flush_trace();
            trace.request_path = ctx->path;
            trace.request_method = ctx->method;
            trace.status_code = res->status_code;
            trace.duration_ms = duration_ms;
            
            diag_mgr->record_error_trace(trace);
        } else {
            ctx->clear_trace();
        }
        
        // Periodic health check
        static std::atomic<int> counter{0};
        if (++counter % 100 == 0) {
            alert_mgr->check_health(diag_mgr.get());
        }
    });
    
    logger->info("🚀 Server starting on http://localhost:8080");
    logger->info("📊 Operations Dashboard: http://localhost:8080/ops");
    
    server.start();
    
    return 0;
}
