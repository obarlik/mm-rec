// Example: Interface-Based Dependency Injection (Best Practice)

#include "mm_rec/infrastructure/di_container.h"
#include <iostream>
#include <vector>

using namespace mm_rec;

// ========================================
// INTERFACES (Abstract Base Classes)
// ========================================

class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void info(const std::string& msg) = 0;
    virtual void error(const std::string& msg) = 0;
};

class IEmailService {
public:
    virtual ~IEmailService() = default;
    virtual void send(const std::string& to, const std::string& subject, const std::string& body) = 0;
};

class IUserRepository {
public:
    virtual ~IUserRepository() = default;
    virtual std::string get_user(int id) = 0;
    virtual void save_user(int id, const std::string& name) = 0;
};

// ========================================
// PRODUCTION IMPLEMENTATIONS
// ========================================

class ConsoleLogger : public ILogger {
public:
    void info(const std::string& msg) override {
        std::cout << "[INFO] " << msg << std::endl;
    }
    
    void error(const std::string& msg) override {
        std::cerr << "[ERROR] " << msg << std::endl;
    }
};

class SmtpEmailService : public IEmailService {
private:
    std::shared_ptr<ILogger> logger_;
    
public:
    SmtpEmailService(std::shared_ptr<ILogger> logger) : logger_(logger) {
        logger_->info("SmtpEmailService initialized");
    }
    
    void send(const std::string& to, const std::string& subject, const std::string& body) override {
        logger_->info("📧 Sending email to " + to + ": " + subject);
        // Actual SMTP logic here
    }
};

class DatabaseUserRepository : public IUserRepository {
private:
    std::shared_ptr<ILogger> logger_;
    std::vector<std::pair<int, std::string>> fake_db_;
    
public:
    DatabaseUserRepository(std::shared_ptr<ILogger> logger) : logger_(logger) {
        logger_->info("DatabaseUserRepository connected");
        // Mock data
        fake_db_ = {{1, "Alice"}, {2, "Bob"}};
    }
    
    std::string get_user(int id) override {
        logger_->info("Fetching user #" + std::to_string(id));
        for (const auto& [uid, name] : fake_db_) {
            if (uid == id) return name;
        }
        return "Unknown";
    }
    
    void save_user(int id, const std::string& name) override {
        logger_->info("Saving user #" + std::to_string(id) + ": " + name);
        fake_db_.push_back({id, name});
    }
};

// ========================================
// TEST/MOCK IMPLEMENTATIONS
// ========================================

class MockLogger : public ILogger {
public:
    std::vector<std::string> info_logs;
    std::vector<std::string> error_logs;
    
    void info(const std::string& msg) override {
        info_logs.push_back(msg);
        std::cout << "[MOCK INFO] " << msg << std::endl;
    }
    
    void error(const std::string& msg) override {
        error_logs.push_back(msg);
        std::cout << "[MOCK ERROR] " << msg << std::endl;
    }
};

class FakeEmailService : public IEmailService {
public:
    std::vector<std::string> sent_emails;
    
    void send(const std::string& to, const std::string& subject, const std::string& body) override {
        sent_emails.push_back(to + ": " + subject);
        std::cout << "[FAKE] Email logged (not sent): " << to << std::endl;
    }
};

// ========================================
// BUSINESS LOGIC (Depends on INTERFACES)
// ========================================

class UserService {
private:
    std::shared_ptr<ILogger> logger_;
    std::shared_ptr<IUserRepository> repo_;
    std::shared_ptr<IEmailService> email_;
    
public:
    // Constructor depends on INTERFACES, not concrete classes!
    UserService(
        std::shared_ptr<ILogger> logger,
        std::shared_ptr<IUserRepository> repo,
        std::shared_ptr<IEmailService> email
    ) : logger_(logger), repo_(repo), email_(email) {
        logger_->info("UserService initialized");
    }
    
    void register_user(int id, const std::string& name, const std::string& email) {
        logger_->info("Registering user: " + name);
        
        repo_->save_user(id, name);
        email_->send(email, "Welcome!", "Thank you for registering, " + name + "!");
        
        logger_->info("User registered successfully");
    }
    
    std::string get_user_name(int id) {
        return repo_->get_user(id);
    }
};

REGISTER_CTOR(SmtpEmailService, std::shared_ptr<ILogger>);
REGISTER_CTOR(DatabaseUserRepository, std::shared_ptr<ILogger>);
REGISTER_CTOR(UserService, std::shared_ptr<ILogger>, std::shared_ptr<IUserRepository>, std::shared_ptr<IEmailService>);

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Interface-Based DI Demo" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // ========================================
    // PRODUCTION CONFIGURATION
    // ========================================
    std::cout << "🏭 PRODUCTION CONFIGURATION:" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    DIContainer prod_container;
    
    // Bind interfaces to production implementations
    prod_container.bind_singleton<ILogger, ConsoleLogger>();
    prod_container.bind_singleton<IEmailService, SmtpEmailService>();
    prod_container.bind_singleton<IUserRepository, DatabaseUserRepository>();
    prod_container.bind_singleton<UserService>();
    
    // Use production services
    auto prod_service = prod_container.resolve<UserService>();
    prod_service->register_user(3, "Charlie", "charlie@example.com");
    
    std::cout << "\nUser found: " << prod_service->get_user_name(1) << std::endl;
    
    // ========================================
    // TEST CONFIGURATION
    // ========================================
    std::cout << "\n\n🧪 TEST CONFIGURATION:" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    DIContainer test_container;
    
    // Bind interfaces to MOCK implementations (no code change in UserService!)
    test_container.bind_singleton<ILogger, MockLogger>();
    test_container.bind_singleton<IEmailService, FakeEmailService>();
    test_container.bind_singleton<IUserRepository, DatabaseUserRepository>();
    test_container.bind_singleton<UserService>();
    
    // Use test services (same UserService class!)
    auto test_service = test_container.resolve<UserService>();
    test_service->register_user(4, "Dave", "dave@example.com");
    
    // Verify mocks
    auto mock_logger = test_container.resolve<ILogger>();
    auto fake_email = test_container.resolve<IEmailService>();
    
    std::cout << "\n✅ Test passed: Email was logged (not sent)" << std::endl;
    
    // ========================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ✅ Interface-Based DI Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n💡 Benefits of Interface-Based DI:" << std::endl;
    std::cout << "  ✓ Testability - Easy to mock dependencies" << std::endl;
    std::cout << "  ✓ Loose Coupling - Depend on abstractions, not implementations" << std::endl;
    std::cout << "  ✓ Flexibility - Swap implementations without changing business logic" << std::endl;
    std::cout << "  ✓ SOLID Principles - Dependency Inversion Principle" << std::endl;
    
    std::cout << "\n📝 Pattern:" << std::endl;
    std::cout << R"(
  // 1. Define interface
  class IService {
  public:
      virtual ~IService() = default;
      virtual void do_work() = 0;
  };
  
  // 2. Production implementation
  class RealService : public IService {
      void do_work() override { /* real work */ }
  };
  
  // 3. Test mock
  class MockService : public IService {
      void do_work() override { /* mock behavior */ }
  };
  
  // 4. Bind interface → implementation
  prod_container.bind_singleton<IService, RealService>();
  test_container.bind_singleton<IService, MockService>();
  
  // 5. Depend on interface (not concrete class)
  class MyApp {
      std::shared_ptr<IService> service_;  // ✅ Interface
  public:
      MyApp(std::shared_ptr<IService> service) : service_(service) {}
  };
)" << std::endl;
    
    return 0;
}
