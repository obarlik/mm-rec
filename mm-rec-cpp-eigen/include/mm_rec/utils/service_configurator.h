#pragma once

#include "mm_rec/utils/service_layers.h"  // Service layer classification guide
#include "mm_rec/utils/di_container.h"
#include "mm_rec/utils/event_bus.h"
#include "mm_rec/utils/metrics.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/config.h"
#include "mm_rec/utils/request_context.h"
#include "mm_rec/utils/http_server.h"      // For HttpServer registration
#include "mm_rec/utils/diagnostic_manager.h"
#include "mm_rec/utils/alert_manager.h"

// Forward declaration to avoid circular dependency
namespace mm_rec { class DashboardManager; }

namespace mm_rec {

/**
 * Central DI Configuration (like ASP.NET Core Startup.cs)
 * 
 * All service registrations in one place for:
 * - Visibility: See all dependencies at a glance
 * - Maintainability: Single source of truth
 * - Testability: Easy to mock for tests
 * 
 * Usage:
 *   // At app startup:
 *   ServiceConfigurator::initialize();
 *   
 *   // In handlers:
 *   auto& container = ServiceConfigurator::container();
 */
class ServiceConfigurator {
public:
    /**
     * Initialize global DI container.
     * Call once at application startup.
     */
    static void initialize() {
        configure_services(instance());
    }
    
    /**
     * Get global DI container instance.
     */
    static DIContainer& container() {
        return instance();
    }
    
    /**
     * Create a new DI scope (like ASP.NET Core CreateScope).
     * Use this for background jobs, commands, or manual scope control.
     * 
     * Example:
     *   auto scope = ServiceConfigurator::create_scope();
     *   auto service = scope->resolve<MyService>();
     *   // scope destroyed automatically (RAII)
     */
    static std::unique_ptr<Scope> create_scope() {
        return std::make_unique<Scope>(container());
    }
    
    /**
     * Register all application services.
     * Called once at application startup.
     */
    static void configure_services(DIContainer& container) {
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // INFRASTRUCTURE LAYER (Stable Platform Services)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // These are framework/platform services that are:
        // - Stable and rarely change
        // - No business logic
        // - Single obvious implementation
        // - Can be used as concrete classes (no interface needed)
        
        // Configuration Management (key-value store)
        container.bind_singleton<Config>();
        
        // Event Bus (internal pub/sub messaging)
        container.bind_singleton<EventBus>();
        
        // Metrics Collection (stats tracking)
        container.bind_singleton<MetricsManager>();
        
        // Logging (interface for pluggable destinations: file, console, remote)
        container.bind_singleton<ILogger, Logger>();
        
        // HTTP Server (for dashboard and APIs)
        // Config-driven: reads from config instead of hard-coded values
        container.bind_singleton<mm_rec::net::HttpServer>([](DIContainer& c) {
            auto cfg = c.resolve<Config>();
            
            // Infrastructure Config (Domain reads, Infrastructure executes)
            int port = cfg->get_int("server.port", 8085);
            int threads = cfg->get_int("server.threads", 4);
            int timeout = cfg->get_int("server.timeout", 3);
            int max_conn = cfg->get_int("server.max_connections", 100);
            int rate_limit = cfg->get_int("server.rate_limit", 1000);
            
            // Create config struct
            mm_rec::net::HttpServerConfig config;
            config.port = port;
            config.threads = threads;
            config.timeout_sec = timeout;
            config.max_connections = max_conn;
            config.max_req_per_min = rate_limit;
            
            return std::make_shared<mm_rec::net::HttpServer>(config);
        });
        
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // BUSINESS LOGIC LAYER (Domain Services)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // These are domain-specific services that:
        // - Contain business rules and logic
        // - May have multiple implementations
        // - Need mocking in unit tests
        // - MUST use interface-based registration
        
        // Diagnostic Tracing & Error Management
        container.bind_singleton<diagnostics::IDiagnosticManager, diagnostics::DiagnosticManager>();
        
        // Alert Generation & System Health Monitoring
        container.bind_singleton<diagnostics::IAlertManager, diagnostics::AlertManager>();
        
        // Future domain services (examples):
        // container.bind_singleton<IUserService, UserService>();
        // container.bind_singleton<IOrderService, OrderService>();
        // container.bind_singleton<IPaymentService, StripePaymentService>();
        
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // APPLICATION LAYER (Facades / Coordinators)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // These services compose and coordinate other services
        // Facade pattern - dependencies are INJECTED (not owned)
        
        // Dashboard (depends on HttpServer via DI)
        container.bind_singleton<DashboardManager>([](DIContainer& c) {
            auto server = c.resolve<mm_rec::net::HttpServer>();
            return std::make_shared<DashboardManager>(server);
        });
        
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // REQUEST-SCOPED SERVICES (Per-Request State)
        // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        // These services are created fresh for each HTTP request:
        // - Hold per-request data (correlation ID, timing, etc.)
        // - Lifetime managed by request scope
        // - Usually concrete (just data holders)
        
        // HTTP Request Context (timing, tracing, correlation)
        container.bind_scoped<mm_rec::net::RequestContext>();
        
        // Additional request-scoped services (examples):
        // container.bind_scoped<UserContext>();
        // container.bind_scoped<AuthContext>();
    }
    
    /**
     * Configure services for testing (with mocks).
     * Override real services with test doubles.
     */
    static void configure_test_services(DIContainer& container) {
        // Example test configuration:
        // container.bind_singleton<ILogger, MockLogger>();
        // container.bind_singleton<IDatabase, InMemoryDatabase>();
        // container.bind_singleton<Config, TestConfig>();
    }

private:
    /**
     * Global DI container instance (singleton).
     */
    static DIContainer& instance() {
        static DIContainer global_container;
        return global_container;
    }
};

} // namespace mm_rec

// Include DashboardManager after ServiceConfigurator to avoid circular dependency
#include "mm_rec/utils/dashboard_manager.h"
