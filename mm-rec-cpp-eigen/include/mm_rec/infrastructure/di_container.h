#pragma once

#include <memory>
#include <map>
#include <typeindex>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>

namespace mm_rec {

// --- Constructor Parameter Detection (must be before DIContainer) ---

/**
 * Trait to declare constructor parameters for auto-wiring.
 * Use REGISTER_CTOR macro to specialize for your types.
 */
template<typename T>
struct ctor_args {
    using type = std::tuple<>; // Default: no-arg constructor
};

/**
 * Lightweight Dependency Injection Container
 * 
 * Zero-dependency DI with:
 * - Service registration (bind<Interface, Implementation>)
 * - Lifetime management (Singleton, Transient)
 * - Auto-wiring via constructor injection
 * - Thread-safe resolution
 * 
 * Usage:
 *   DIContainer container;
 *   container.bind_singleton<ILogger, ConsoleLogger>();
 *   auto logger = container.resolve<ILogger>();
 */
class DIContainer {
public:
    enum Lifetime {
        Singleton,  // One instance for lifetime of container
        Transient,  // New instance on every resolve()
        Scoped      // One instance per scope (e.g., HTTP request)
    };

private:
    struct ServiceBinding {
        Lifetime lifetime;
        std::function<std::shared_ptr<void>()> factory;
        std::shared_ptr<void> instance; // For singletons only
    };

    std::map<std::type_index, ServiceBinding> bindings_;
    mutable std::mutex mutex_;

public:
    DIContainer() = default;
    ~DIContainer() = default;

    // Non-copyable, non-movable (for now)
    DIContainer(const DIContainer&) = delete;
    DIContainer& operator=(const DIContainer&) = delete;

    // --- Registration API ---

    /**
     * Bind interface to implementation as singleton.
     * One instance shared across all resolve() calls.
     */
    template<typename Interface, typename Implementation>
    void bind_singleton() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        bindings_[std::type_index(typeid(Interface))] = {
            Singleton,
            [this]() { return std::static_pointer_cast<void>(create<Implementation>()); },
            nullptr
        };
        
        // Validate AFTER binding (so dependencies can be checked)
        validate_lifetime_rules<Implementation>(Singleton);
    }

    /**
     * Bind interface to factory function as singleton.
     */
    template<typename Interface>
    void bind_singleton(std::function<std::shared_ptr<Interface>()> factory) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        bindings_[std::type_index(typeid(Interface))] = {
            Singleton,
            [factory]() { return std::static_pointer_cast<void>(factory()); },
            nullptr
        };
        
        // Can't validate factory-based bindings (don't know dependencies)
    }

    /**
     * Bind interface to factory function (with DIContainer access) as singleton.
     */
    template<typename Interface>
    void bind_singleton(std::function<std::shared_ptr<Interface>(DIContainer&)> factory) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        bindings_[std::type_index(typeid(Interface))] = {
            Singleton,
            [this, factory]() { return std::static_pointer_cast<void>(factory(*this)); },
            nullptr
        };
    }

    /**
     * Bind interface to implementation as transient.
     * New instance created on every resolve() call.
     */
    template<typename Interface, typename Implementation>
    void bind_transient() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        bindings_[std::type_index(typeid(Interface))] = {
            Transient,
            [this]() { return std::static_pointer_cast<void>(create<Implementation>()); },
            nullptr
        };
        
        validate_lifetime_rules<Implementation>(Transient);
    }

    /**
     * Bind interface to factory function as transient.
     */
    template<typename Interface>
    void bind_transient(std::function<std::shared_ptr<Interface>()> factory) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        bindings_[std::type_index(typeid(Interface))] = {
            Transient,
            [factory]() { return std::static_pointer_cast<void>(factory()); },
            nullptr
        };
    }

    /**
     * Bind interface to implementation as scoped.
     * One instance per Scope object (e.g., per HTTP request).
     */
    template<typename Interface, typename Implementation>
    void bind_scoped() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        bindings_[std::type_index(typeid(Interface))] = {
            Scoped,
            [this]() { return std::static_pointer_cast<void>(create<Implementation>()); },
            nullptr
        };
        
        validate_lifetime_rules<Implementation>(Scoped);
    }

    /**
     * Bind interface to factory function as scoped.
     */
    template<typename Interface>
    void bind_scoped(std::function<std::shared_ptr<Interface>()> factory) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        bindings_[std::type_index(typeid(Interface))] = {
            Scoped,
            [factory]() { return std::static_pointer_cast<void>(factory()); },
            nullptr
        };
    }

    /**
     * Bind concrete type to itself (no interface).
     */
    template<typename T>
    void bind_singleton() {
        bind_singleton<T, T>();
    }

    template<typename T>
    void bind_transient() {
        bind_transient<T, T>();
    }
    
    template<typename T>
    void bind_scoped() {
        bind_scoped<T, T>();
    }

    // --- Resolution API ---

    /**
     * Resolve service by type.
     * Returns shared_ptr to instance.
     * Throws if service not registered.
     */
    template<typename T>
    std::shared_ptr<T> resolve() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        auto it = bindings_.find(std::type_index(typeid(T)));
        if (it == bindings_.end()) {
            throw std::runtime_error(
                "Service not registered: " + std::string(typeid(T).name()) +
               "\nDid you forget to call bind_singleton<" + std::string(typeid(T).name()) + ">()"
            );
        }

        auto& binding = it->second;

        if (binding.lifetime == Singleton) {
            if (!binding.instance) {
                // Copy factory to call outside lock (avoid deadlock during auto-wiring)
                auto factory = binding.factory;
                lock.unlock();
                
                auto new_instance = factory();
                
                lock.lock();
                // Double-check: another thread might have created it
                if (!binding.instance) {
                    binding.instance = new_instance;
                }
            }
            return std::static_pointer_cast<T>(binding.instance);
        } else {
            // Transient: create new instance each time
            auto factory = binding.factory;
            lock.unlock();
            return std::static_pointer_cast<T>(factory());
        }
    }

    /**
     * Try to resolve service, returns nullptr if not registered.
     */
    template<typename T>
    std::shared_ptr<T> try_resolve() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        auto it = bindings_.find(std::type_index(typeid(T)));
        if (it == bindings_.end()) {
            return nullptr;
        }

        auto& binding = it->second;

        if (binding.lifetime == Singleton) {
            if (!binding.instance) {
                auto factory = binding.factory;
                lock.unlock();
                
                auto new_instance = factory();
                
                lock.lock();
                if (!binding.instance) {
                    binding.instance = new_instance;
                }
            }
            return std::static_pointer_cast<T>(binding.instance);
        } else {
            auto factory = binding.factory;
            lock.unlock();
            return std::static_pointer_cast<T>(factory());
        }
    }

    /**
     * Check if service is registered.
     */
    template<typename T>
    bool is_registered() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return bindings_.find(std::type_index(typeid(T))) != bindings_.end();
    }

    /**
     * Clear all bindings.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        bindings_.clear();
    }
    
    /**
     * Get lifetime of a registered service.
     */
    template<typename T>
    Lifetime get_lifetime() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return get_lifetime_internal<T>();
    }

private:
    /**
     * Internal get_lifetime (no lock, for use when already locked).
     */
    template<typename T>
    Lifetime get_lifetime_internal() const {
        auto it = bindings_.find(std::type_index(typeid(T)));
        if (it == bindings_.end()) {
            throw std::runtime_error("Service not registered");
        }
        return it->second.lifetime;
    }

    /**
     * Validate captive dependency rules.
     * Singleton/Scoped cannot depend on shorter-lived services.
     */
    template<typename T>
    void validate_lifetime_rules(Lifetime parent_lifetime) {
        // Extract dependencies from ctor_args
        using deps = typename ctor_args<T>::type;
        validate_dependencies<T>(parent_lifetime, deps{});
    }
    
    // Base case: no dependencies
    template<typename T>
    void validate_dependencies(Lifetime, std::tuple<>) {
        // No dependencies, nothing to validate
    }
    
    // Unpack dependencies and validate each
    template<typename T, typename... Args>
    void validate_dependencies(Lifetime parent_lifetime, std::tuple<std::shared_ptr<Args>...>) {
        // Validate each dependency
        int dummy[] = { (validate_single_dependency<Args>(parent_lifetime), 0)... };
        (void)dummy; // Silence unused warning
    }
    
    // Validate single dependency
    template<typename Dep>
    void validate_single_dependency(Lifetime parent_lifetime) {
        auto it = bindings_.find(std::type_index(typeid(Dep)));
        if (it == bindings_.end()) {
            return; // Dependency not yet registered, can't validate
        }
        
        Lifetime child_lifetime = it->second.lifetime;
        
        // Captive dependency check
        if (parent_lifetime == Singleton && child_lifetime != Singleton) {
            throw std::runtime_error(
                std::string("Captive Dependency: Singleton service depends on ") +
                lifetime_name(child_lifetime) + " service: " + std::string(typeid(Dep).name()) +
                "\nSingleton services can only depend on other Singleton services."
            );
        }
        
        if (parent_lifetime == Scoped && child_lifetime == Transient) {
            // This is actually OK, but warn in debug mode
            // Scoped will get a new Transient each time it's created
        }
    }
    
    const char* lifetime_name(Lifetime lt) const {
        switch (lt) {
            case Singleton: return "Singleton";
            case Transient: return "Transient";
            case Scoped: return "Scoped";
            default: return "Unknown";
        }
    }

    /**
     * Create instance with auto-wiring (constructor injection).
     * Uses ctor_args trait to detect constructor parameters.
     */
    template<typename T>
    std::shared_ptr<T> create() {
        return create_impl<T>(typename ctor_args<T>::type{});
    }

    /**
     * Create instance by resolving constructor dependencies.
     * Args are dependency types (e.g., ILogger), auto-wrapped in std::shared_ptr.
     */
    template<typename T, typename... Args>
    std::shared_ptr<T> create_impl(std::tuple<std::shared_ptr<Args>...>) {
        return std::make_shared<T>(resolve<Args>()...);
    }
    
    /**
     * Create instance by resolving bare types (deprecated, use shared_ptr).
     */
    template<typename T, typename... Args>
    std::shared_ptr<T> create_impl(std::tuple<Args...>) {
        return std::make_shared<T>(resolve<Args>()...);
    }
};

/**
 * Scoped lifetime manager.
 * Creates a scope (e.g., per HTTP request) where scoped services
 * are cached and reused within that scope only.
 * 
 * Usage:
 *   DIContainer container;
 *   container.bind_scoped<RequestContext>();
 *   
 *   Scope scope(container);
 *   auto ctx1 = scope.resolve<RequestContext>();
 *   auto ctx2 = scope.resolve<RequestContext>();
 *   // ctx1 == ctx2 (same instance within scope)
 */
class Scope {
private:
    DIContainer& container_;
    std::map<std::type_index, std::shared_ptr<void>> scoped_instances_;
    std::mutex mutex_;

public:
    explicit Scope(DIContainer& container) : container_(container) {}
    
    ~Scope() {
        // Scoped instances automatically destroyed
        scoped_instances_.clear();
    }
    
    template<typename T>
    std::shared_ptr<T> resolve() {
        auto lifetime = container_.get_lifetime<T>();
        
        if (lifetime == DIContainer::Scoped) {
            std::lock_guard<std::mutex> lock(mutex_);
            
            auto it = scoped_instances_.find(std::type_index(typeid(T)));
            if (it != scoped_instances_.end()) {
                return std::static_pointer_cast<T>(it->second);
            }
            
            // Create new scoped instance
            auto instance = container_.resolve<T>();
            scoped_instances_[std::type_index(typeid(T))] = instance;
            return instance;
        } else {
            // Singleton/Transient: delegate to container
            return container_.resolve<T>();
        }
    }
};

/**
 * Macro to register constructor parameters for DI.
 * Dependencies should be specified wrapped in std::shared_ptr.
 * 
 * Usage:
 *   class MyService {
 *       MyService(std::shared_ptr<Dep1> d1, std::shared_ptr<Dep2> d2);
 *   };
 *   
 *   REGISTER_CTOR(MyService, std::shared_ptr<Dep1>, std::shared_ptr<Dep2>);
 */
#define REGISTER_CTOR(Type, ...) \
    namespace mm_rec { \
        template<> struct ctor_args<Type> { \
            using type = std::tuple<__VA_ARGS__>; \
        }; \
    }

} // namespace mm_rec
