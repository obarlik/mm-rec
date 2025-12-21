// Example: Custom Constructor with Factory Pattern

#include "mm_rec/utils/di_container.h"
#include <iostream>
#include <string>

using namespace mm_rec;

// Service with complex constructor
class DatabaseConnection {
private:
    std::string host_;
    int port_;
    std::string db_name_;
    bool ssl_enabled_;
    
public:
    DatabaseConnection(const std::string& host, int port, const std::string& db_name, bool ssl = true)
        : host_(host), port_(port), db_name_(db_name), ssl_enabled_(ssl) {
        std::cout << "✅ DatabaseConnection created:" << std::endl;
        std::cout << "   Host: " << host_ << ":" << port_ << std::endl;
        std::cout << "   Database: " << db_name_ << std::endl;
        std::cout << "   SSL: " << (ssl_enabled_ ? "enabled" : "disabled") << std::endl;
    }
    
    void query(const std::string& sql) {
        std::cout << "📡 [" << db_name_ << "] Query: " << sql << std::endl;
    }
};

// Service with environment-based configuration
class EmailService {
private:
    std::string smtp_server_;
    std::string from_email_;
    
public:
    EmailService(const std::string& server, const std::string& from)
        : smtp_server_(server), from_email_(from) {
        std::cout << "✅ EmailService created:" << std::endl;
        std::cout << "   SMTP: " << smtp_server_ << std::endl;
        std::cout << "   From: " << from_email_ << std::endl;
    }
    
    void send(const std::string& to, const std::string& subject) {
        std::cout << "📧 [" << from_email_ << " -> " << to << "] " << subject << std::endl;
    }
};

// Service with dependency + custom config
class UserRepository {
private:
    std::shared_ptr<DatabaseConnection> db_;
    std::string table_name_;
    
public:
    UserRepository(std::shared_ptr<DatabaseConnection> db, const std::string& table = "users")
        : db_(db), table_name_(table) {
        std::cout << "✅ UserRepository created (table: " << table_name_ << ")" << std::endl;
    }
    
    void find_user(int id) {
        db_->query("SELECT * FROM " + table_name_ + " WHERE id = " + std::to_string(id));
    }
};

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Custom Constructor with Factories" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    DIContainer container;
    
    // === 1. Factory with Custom Parameters (Singleton) ===
    std::cout << "📦 Registering DatabaseConnection with custom factory..." << std::endl;
    container.bind_singleton<DatabaseConnection>([]() {
        // Custom parameters from environment/config
        return std::make_shared<DatabaseConnection>(
            "localhost",      // host
            5432,            // port
            "production_db", // database name
            true             // SSL enabled
        );
    });
    std::cout << std::endl;
    
    // === 2. Factory with Different Config (Singleton) ===
    std::cout << "📦 Registering EmailService with custom factory..." << std::endl;
    container.bind_singleton<EmailService>([]() {
        // Read from environment or config file
        const char* env = std::getenv("ENVIRONMENT");
        std::string smtp = (env && std::string(env) == "production") 
            ? "smtp.gmail.com" 
            : "localhost";
        
        return std::make_shared<EmailService>(smtp, "noreply@myapp.com");
    });
    std::cout << std::endl;
    
    // === 3. Factory with Dependency Injection (Scoped) ===
    std::cout << "📦 Registering UserRepository with custom factory (scoped)..." << std::endl;
    container.bind_scoped<UserRepository>([&container]() {
        // Resolve dependency
        auto db = container.resolve<DatabaseConnection>();
        
        // Custom table name
        return std::make_shared<UserRepository>(db, "customers");
    });
    std::cout << std::endl;
    
    // === Use Services ===
    std::cout << "=== Resolving Services ===" << std::endl;
    
    // Resolve DatabaseConnection (factory called once, cached as singleton)
    std::cout << "\n1️⃣ Resolving DatabaseConnection:" << std::endl;
    auto db = container.resolve<DatabaseConnection>();
    db->query("SELECT version()");
    
    // Resolve EmailService
    std::cout << "\n2️⃣ Resolving EmailService:" << std::endl;
    auto email = container.resolve<EmailService>();
    email->send("user@example.com", "Welcome!");
    
    // Resolve UserRepository in a scope (factory called per scope)
    std::cout << "\n3️⃣ Resolving UserRepository (Scoped):" << std::endl;
    {
        Scope scope1(container);
        auto repo1 = scope1.resolve<UserRepository>();
        repo1->find_user(123);
    }
    
    std::cout << "\n4️⃣ Resolving UserRepository in another scope:" << std::endl;
    {
        Scope scope2(container);
        auto repo2 = scope2.resolve<UserRepository>();
        repo2->find_user(456);
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ✅ Factory Pattern Demo Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n💡 Factory Benefits:" << std::endl;
    std::cout << "  ✓ Custom constructor parameters" << std::endl;
    std::cout << "  ✓ Environment-based config" << std::endl;
    std::cout << "  ✓ Complex initialization logic" << std::endl;
    std::cout << "  ✓ Manual dependency resolution" << std::endl;
    std::cout << "  ✓ Runtime configuration" << std::endl;
    
    std::cout << "\n📝 Usage Pattern:" << std::endl;
    std::cout << R"(
  // Singleton with custom params:
  container.bind_singleton<IService>([]() {
      return std::make_shared<MyService>("param1", 42, true);
  });
  
  // Scoped with dependency injection:
  container.bind_scoped<IRepo>([&container]() {
      auto db = container.resolve<IDatabase>();
      return std::make_shared<MyRepo>(db, "custom_table");
  });
  
  // Transient with config:
  container.bind_transient<IWorker>([]() {
      auto config = load_config();
      return std::make_shared<Worker>(config.threads);
  });
)" << std::endl;
    
    return 0;
}
