#include "mm_rec/infrastructure/di_container.h"
#include <iostream>
#include <cassert>

using namespace mm_rec;

// --- Test Interfaces and Implementations ---

class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(const std::string& msg) = 0;
};

class ConsoleLogger : public ILogger {
public:
    void log(const std::string& msg) override {
        std::cout << "[LOG] " << msg << std::endl;
    }
};

class IDatabase {
public:
    virtual ~IDatabase() = default;
    virtual void query(const std::string& sql) = 0;
};

class SqliteDatabase : public IDatabase {
private:
    std::shared_ptr<ILogger> logger_;
    
public:
    SqliteDatabase(std::shared_ptr<ILogger> logger) : logger_(logger) {
        logger_->log("SqliteDatabase created");
    }
    
    void query(const std::string& sql) override {
        logger_->log("Executing: " + sql);
    }
};

// Register constructor dependencies (wrapped in shared_ptr)
REGISTER_CTOR(SqliteDatabase, std::shared_ptr<ILogger>);

class SimpleService {
public:
    SimpleService() {
        std::cout << "[SimpleService] Created (no dependencies)" << std::endl;
    }
    
    void do_something() {
        std::cout << "[SimpleService] Doing something..." << std::endl;
    }
};

// --- Tests ---

void test_singleton_lifetime() {
    std::cout << "\n=== Test: Singleton Lifetime ===" << std::endl;
    
    DIContainer container;
    container.bind_singleton<ILogger, ConsoleLogger>();
    
    auto logger1 = container.resolve<ILogger>();
    auto logger2 = container.resolve<ILogger>();
    
    assert(logger1.get() == logger2.get());  // Same instance
    std::cout << "✅ Singleton: Same instance returned" << std::endl;
}

void test_transient_lifetime() {
    std::cout << "\n=== Test: Transient Lifetime ===" << std::endl;
    
    DIContainer container;
    container.bind_transient<ILogger, ConsoleLogger>();
    
    auto logger1 = container.resolve<ILogger>();
    auto logger2 = container.resolve<ILogger>();
    
    assert(logger1.get() != logger2.get());  // Different instances
    std::cout << "✅ Transient: Different instances returned" << std::endl;
}

void test_auto_wiring() {
    std::cout << "\n=== Test: Auto-wiring (Constructor Injection) ===" << std::endl;
    
    DIContainer container;
    container.bind_singleton<ILogger, ConsoleLogger>();
    container.bind_singleton<IDatabase, SqliteDatabase>();
    
    auto db = container.resolve<IDatabase>();
    db->query("SELECT * FROM users");
    
    std::cout << "✅ Auto-wiring: Dependencies injected successfully" << std::endl;
}

void test_concrete_type() {
    std::cout << "\n=== Test: Concrete Type (No Interface) ===" << std::endl;
    
    DIContainer container;
    container.bind_singleton<SimpleService>();
    
    auto service = container.resolve<SimpleService>();
    service->do_something();
    
    std::cout << "✅ Concrete type: Service resolved" << std::endl;
}

void test_factory_binding() {
    std::cout << "\n=== Test: Factory Binding ===" << std::endl;
    
    DIContainer container;
    
    int counter = 0;
    container.bind_singleton<ILogger>([&counter]() -> std::shared_ptr<ILogger> {
        counter++;
        std::cout << "[Factory] Creating logger #" << counter << std::endl;
        return std::make_shared<ConsoleLogger>();
    });
    
    auto logger1 = container.resolve<ILogger>();
    auto logger2 = container.resolve<ILogger>();
    
    assert(counter == 1);  // Factory called once for singleton
    std::cout << "✅ Factory: Called once for singleton" << std::endl;
}

void test_not_registered() {
    std::cout << "\n=== Test: Service Not Registered ===" << std::endl;
    
    DIContainer container;
    
    try {
        auto db = container.resolve<IDatabase>();
        assert(false);  // Should throw
    } catch (const std::runtime_error& e) {
        std::cout << "✅ Exception thrown: " << e.what() << std::endl;
    }
}

void test_is_registered() {
    std::cout << "\n=== Test: is_registered() ===" << std::endl;
    
    DIContainer container;
    container.bind_singleton<ILogger, ConsoleLogger>();
    
    assert(container.is_registered<ILogger>());
    assert(!container.is_registered<IDatabase>());
    
    std::cout << "✅ is_registered: Correct detection" << std::endl;
}

// --- Main ---

int main() {
    std::cout << "=====================================" << std::endl;
    std::cout << "  DI Container Test Suite" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    test_singleton_lifetime();
    test_transient_lifetime();
    test_auto_wiring();
    test_concrete_type();
    test_factory_binding();
    test_not_registered();
    test_is_registered();
    
    std::cout << "\n=====================================" << std::endl;
    std::cout << "  ✅ All Tests Passed!" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    return 0;
}
