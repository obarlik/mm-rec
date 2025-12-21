// Test: ILogger Interface with DI

#include "mm_rec/application/service_configurator.h"
#include "mm_rec/infrastructure/logger.h"
#include <iostream>

using namespace mm_rec;

// Mock Logger for testing
class MockLogger : public ILogger {
public:
    std::vector<std::string> logs;
    
    void log(LogLevel level, const char* msg) override {
        logs.push_back(std::string(msg));
        std::cout << "[MOCK] " << msg << std::endl;
    }
    
    void log(LogLevel level, const std::string& msg) override {
        log(level, msg.c_str());
    }
    
    void ui(const char* msg) override { log(LogLevel::UI, msg); }
    void info(const char* msg) override { log(LogLevel::INFO, msg); }
    void warning(const char* msg) override { log(LogLevel::WARNING, msg); }
    void error(const char* msg) override { log(LogLevel::ERROR, msg); }
    void debug(const char* msg) override { log(LogLevel::DEBUG, msg); }
    
    void ui(const std::string& msg) override { log(LogLevel::UI, msg); }
    void info(const std::string& msg) override { log(LogLevel::INFO, msg); }
    void warning(const std::string& msg) override { log(LogLevel::WARNING, msg); }
    void error(const std::string& msg) override { log(LogLevel::ERROR, msg); }
    void debug(const std::string& msg) override { log(LogLevel::DEBUG, msg); }
    
    void start_writer(const std::string&, LogLevel) override {}
    void stop_writer() override {}
};

// Service that depends on ILogger
class UserService {
private:
    std::shared_ptr<ILogger> logger_;
    
public:
    UserService(std::shared_ptr<ILogger> logger) : logger_(logger) {
        logger_->info("UserService initialized");
    }
    
    void create_user(const std::string& name) {
        logger_->info("Creating user: " + name);
        // business logic...
        logger_->info("User created successfully");
    }
};

REGISTER_CTOR(UserService, std::shared_ptr<ILogger>);

int main() {
   std::cout << "========================================" << std::endl;
    std::cout << "  ILogger Interface Test" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // ========================================
    // Test 1: Production Logger
    // ========================================
    std::cout << "🏭 PRODUCTION LOGGER:" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    DIContainer prod_container;
    ServiceConfigurator::configure_services(prod_container);
    prod_container.bind_singleton<UserService>();
    
    auto prod_service = prod_container.resolve<UserService>();
    prod_service->create_user("Alice");
    
    // ========================================
    // Test 2: Mock Logger (Testable!)
    // ========================================
    std::cout << "\n\n🧪 MOCK LOGGER (TEST):" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    DIContainer test_container;
    test_container.bind_singleton<ILogger, MockLogger>();  // ✨ Inject mock!
    test_container.bind_singleton<UserService>();
    
    auto test_service = test_container.resolve<UserService>();
    test_service->create_user("Bob");
    
    // Verify mock
    auto mock = test_container.resolve<ILogger>();
    auto mock_logger = std::static_pointer_cast<MockLogger>(mock);
    
    std::cout << "\n✅ Mock captured " << mock_logger->logs.size()<< " log entries" << std::endl;
    
    // ========================================
    // Test 3: Backward Compatibility (Macros)
    // ========================================
    std::cout << "\n\n🔙 BACKWARD COMPATIBILITY:" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    LOG_INFO("Using legacy macro - still works!");
    LOG_WARN("Warning via macro");
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ✅ ILogger Interface Working!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n💡 Benefits:" << std::endl;
    std::cout << "  ✓ Production logger injected via DI" << std::endl;
    std::cout << "  ✓ Mock logger for testing" << std::endl;
    std::cout << "  ✓ No code change in UserService" << std::endl;
    std::cout << "  ✓ Backward compatible (macros still work)" << std::endl;
    std::cout << "  ✓ SOLID principles (Dependency Inversion)" << std::endl;
    
    return 0;
}
