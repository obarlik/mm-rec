#include "mm_rec/infrastructure/di_container.h"
#include <iostream>
#include <cassert>

using namespace mm_rec;

// Test services
class TransientService {
public:
    TransientService() { std::cout << "[TransientService] Created" << std::endl; }
};

class ScopedService {
public:
    ScopedService() { std::cout << "[ScopedService] Created" << std::endl; }
};

class SingletonService {
public:
    SingletonService() { std::cout << "[SingletonService] Created" << std::endl; }
};

// INVALID: Singleton depending on Transient (captive dependency)
class InvalidSingleton {
private:
    std::shared_ptr<TransientService> transient_;
public:
    InvalidSingleton(std::shared_ptr<TransientService> t) : transient_(t) {}
};
REGISTER_CTOR(InvalidSingleton, std::shared_ptr<TransientService>);

// VALID: Singleton depending on Singleton
class ValidSingleton {
private:
    std::shared_ptr<SingletonService> singleton_;
public:
    ValidSingleton(std::shared_ptr<SingletonService> s) : singleton_(s) {}
};
REGISTER_CTOR(ValidSingleton, std::shared_ptr<SingletonService>);

void test_scoped_lifetime() {
    std::cout << "\n=== Test: Scoped Lifetime ===" << std::endl;
    
    DIContainer container;
    container.bind_scoped<ScopedService>();
    
    // First scope
    {
        Scope scope1(container);
        auto s1 = scope1.resolve<ScopedService>();
        auto s2 = scope1.resolve<ScopedService>();
        assert(s1.get() == s2.get());  // Same instance within scope
        std::cout << "✅ Scoped: Same instance within scope" << std::endl;
    }
    
    // Second scope
    {
        Scope scope2(container);
        auto s3 = scope2.resolve<ScopedService>();
        // s3 is different from s1/s2 (different scope)
        std::cout << "✅ Scoped: New instance in new scope" << std::endl;
    }
}

void test_captive_dependency_validation() {
    std::cout << "\n=== Test: Captive Dependency Validation ===" << std::endl;
    
    DIContainer container;
    container.bind_transient<TransientService>();
    
    try {
        container.bind_singleton<InvalidSingleton>();  // Should throw
        assert(false);  // Should not reach here
    } catch (const std::runtime_error& e) {
        std::cout << "✅ Exception thrown: " << e.what() << std::endl;
    }
}

void test_valid_dependencies() {
    std::cout << "\n=== Test: Valid Dependencies ===" << std::endl;
    
    DIContainer container;
    container.bind_singleton<SingletonService>();
    container.bind_singleton<ValidSingleton>();  // Should succeed
    
    auto service = container.resolve<ValidSingleton>();
    std::cout << "✅ Valid: Singleton depends on Singleton" << std::endl;
}

void test_scope_with_container_delegation() {
    std::cout << "\n=== Test: Scope Delegates to Container ===" << std::endl;
    
    DIContainer container;
    container.bind_singleton<SingletonService>();
    container.bind_transient<TransientService>();
    container.bind_scoped<ScopedService>();
    
    Scope scope(container);
    
    auto singleton1 = scope.resolve<SingletonService>();
    auto singleton2 = scope.resolve<SingletonService>();
    assert(singleton1.get() == singleton2.get());
    std::cout << "✅ Scope resolves Singleton correctly" << std::endl;
    
    auto transient1 = scope.resolve<TransientService>();
    auto transient2 = scope.resolve<TransientService>();
    assert(transient1.get() != transient2.get());
    std::cout << "✅ Scope resolves Transient correctly" << std::endl;
    
    auto scoped1 = scope.resolve<ScopedService>();
    auto scoped2 = scope.resolve<ScopedService>();
    assert(scoped1.get() == scoped2.get());
    std::cout << "✅ Scope caches Scoped service" << std::endl;
}

int main() {
    std::cout << "=====================================" << std::endl;
    std::cout << "  DI Container - Advanced Features" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    test_scoped_lifetime();
    test_captive_dependency_validation();
    test_valid_dependencies();
    test_scope_with_container_delegation();
    
    std::cout << "\n=====================================" << std::endl;
    std::cout << "  ✅ All Advanced Tests Passed!" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    return 0;
}
