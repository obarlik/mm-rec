/**
 * Service Layer Classification
 * 
 * This document defines the architectural layers for dependency injection.
 * Services are categorized to maintain clean separation of concerns.
 */

#pragma once

namespace mm_rec {

/**
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * INFRASTRUCTURE LAYER
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * 
 * Platform services that provide stable, reusable utilities.
 * 
 * CHARACTERISTICS:
 * ✓ Framework/platform level (not domain-specific)
 * ✓ Stable implementation (rarely changes)
 * ✓ No business logic
 * ✓ Single obvious implementation
 * ✓ Tests use real instances (mocking not needed)
 * 
 * REGISTRATION PATTERN:
 *   container.bind_singleton<ConcreteClass>();
 * 
 * EXAMPLES:
 * - Config:           Key-value configuration store
 * - EventBus:         Internal pub/sub messaging
 * - MetricsManager:   Statistics collection
 * - Logger:           I/O abstraction (may use interface for destinations)
 * - DashboardManager: UI/presentation layer
 */
namespace infrastructure {
    // Example:
    // class Config { ... };
    // class EventBus { ... };
    // class MetricsManager { ... };
}

/**
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * BUSINESS LOGIC LAYER
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * 
 * Domain services that implement business rules and logic.
 * 
 * CHARACTERISTICS:
 * ✓ Domain-specific behavior
 * ✓ Complex business rules
 * ✓ May have multiple implementations
 * ✓ Requires mocking in unit tests
 * ✓ Changes with business requirements
 * 
 * REGISTRATION PATTERN:
 *   container.bind_singleton<IInterface, Implementation>();
 * 
 * MANDATORY:
 * ⚠️  MUST use interface-based registration!
 * ⚠️  MUST define interface (IServiceName)
 * ⚠️  Consumers MUST depend on interface, not concrete class
 * 
 * EXAMPLES:
 * - IDiagnosticManager: Error tracking & diagnostic tracing
 * - IAlertManager:      Alert generation & health monitoring  
 * - IUserService:       User management (domain logic)
 * - IOrderService:      Order processing (business rules)
 * - IPaymentService:    Payment processing (external integration)
 */
namespace diagnostics {
    // Example:
    // class IDiagnosticManager { ... };  // Interface
    // class DiagnosticManager : public IDiagnosticManager { ... };  // Implementation
}

/**
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * DATA LAYER (Request-Scoped)
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * 
 * Per-request state containers with minimal behavior.
 * 
 * CHARACTERISTICS:
 * ✓ Holds per-request data only
 * ✓ No complex logic (just getters/setters)
 * ✓ Lifetime = HTTP request scope
 * ✓ New instance per request
 * 
 * REGISTRATION PATTERN:
 *   container.bind_scoped<ConcreteClass>();
 * 
 * EXAMPLES:
 * - RequestContext: Correlation ID, timing, trace buffer
 * - UserContext:    Current user info for request
 * - AuthContext:    Authentication state for request
 */
namespace net {
    // Example:
    // class RequestContext { ... };
}

/**
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * DECISION TREE
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * 
 * Use this to determine the correct layer for a new service:
 * 
 * ┌─────────────────────────────────────┐
 * │  Does it contain business logic?    │
 * └──────────┬──────────────────────────┘
 *            │
 *      ┌─────┴─────┐
 *      │           │
 *     YES         NO
 *      │           │
 *      ▼           ▼
 * ┌─────────┐  ┌──────────────┐
 * │ Business│  │ Is it per-   │
 * │ Logic   │  │ request data?│
 * │ Layer   │  └──────┬───────┘
 * │         │         │
 * │ MUST    │    ┌────┴────┐
 * │ use     │    │         │
 * │ Interface│  YES       NO
 * └─────────┘    │         │
 *                ▼         ▼
 *          ┌──────────┐ ┌──────────────┐
 *          │ Data     │ │Infrastructure│
 *          │ Layer    │ │ Layer        │
 *          │ (Scoped) │ │ (Singleton)  │
 *          │ Concrete │ │ Concrete     │
 *          └──────────┘ └──────────────┘
 * 
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * EXAMPLES BY LAYER
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * 
 * Infrastructure Layer (Concrete):
 *   ✓ Config           - Configuration key-value store
 *   ✓ EventBus         - Pub/sub messaging
 *   ✓ MetricsManager   - Stats collection
 *   ✓ Logger           - Logging I/O (may use ILogger interface)
 *   ✓ DashboardManager - UI presentation
 * 
 * Business Logic Layer (Interface Required):
 *   ✓ IDiagnosticManager → DiagnosticManager
 *   ✓ IAlertManager      → AlertManager
 *   ✓ IUserService       → UserService
 *   ✓ IOrderService      → OrderService
 *   ✓ IPaymentService    → StripePaymentService
 * 
 * Data Layer (Scoped, Concrete):
 *   ✓ RequestContext - Request metadata
 *   ✓ UserContext    - Current user info
 *   ✓ AuthContext    - Auth state
 * 
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 */

} // namespace mm_rec
