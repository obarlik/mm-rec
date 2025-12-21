#pragma once

#include "mm_rec/utils/di_container.h"
#include "mm_rec/utils/event_bus.h"
#include "mm_rec/utils/metrics.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/config.h"
#include "mm_rec/utils/request_context.h"

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
        // ========================================
        // Core Infrastructure Services (Singleton)
        // ========================================
        
        // Logger - app-wide logging
        container.bind_singleton<Logger>();
        
        // Config - configuration management
        container.bind_singleton<Config>();
        
        // Event Bus - pub/sub communication
        container.bind_singleton<EventBus>();
        
        // Metrics tracking
        container.bind_singleton<MetricsManager>();
        
        // Dashboard manager - web UI (registered in production, not in demos)
        // container.bind_singleton<DashboardManager>();
        
        // ========================================
        // HTTP Request-Scoped Services
        // ========================================
        
        // Request context - timing, tracing, metadata (per HTTP request)
        container.bind_scoped<mm_rec::net::RequestContext>();
        
        // Add more request-scoped services:
        // container.bind_scoped<UserService>();
        // container.bind_scoped<AuthService>();
        
        // ========================================
        // Transient Services (New instance each time)
        // ========================================
        
        // Example transient services
        // container.bind_transient<EmailService>();
        // container.bind_transient<ValidationService>();
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
