// Demo: Automatic Function Tracing with Compiler Macros

#include "mm_rec/infrastructure/request_context.h"
#include "mm_rec/application/service_configurator.h"
#include <iostream>

using namespace mm_rec;
using namespace mm_rec::net;

// Global context for demo (in production, use DI)
thread_local RequestContext* g_current_ctx = nullptr;

// Define Logger's thread_local static member
thread_local mm_rec::net::RequestContext* mm_rec::Logger::current_request_context_ = nullptr;

// Implement get_request_context for demo
mm_rec::net::RequestContext* get_request_context() {
    return g_current_ctx;
}

// ========================================
// Simulate Service Layers with Auto-Trace
// ========================================

class DatabaseService {
public:
    void connect() {
        TRACE_FUNC();  // ✨ Compiler auto-inserts "DatabaseService::connect"
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    void query(const std::string& sql) {
        TRACE_FUNC();  // ✨ "DatabaseService::query"
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    
    void disconnect() {
        TRACE_FUNC();  // ✨ "DatabaseService::disconnect"
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
};

class AuthService {
public:
    void validate_token(const std::string& token) {
        TRACE_FUNC();  // ✨ "AuthService::validate_token"
        
        DatabaseService db;
        db.connect();  // Nested scope!
        db.query("SELECT * FROM sessions WHERE token = '" + token + "'");
        db.disconnect();
    }
};

class UserService {
public:
    void load_user(int user_id) {
        TRACE_FUNC();  // ✨ "UserService::load_user"
        
        DatabaseService db;
        db.connect();
        db.query("SELECT * FROM users WHERE id = " + std::to_string(user_id));
        db.disconnect();
    }
    
    void update_profile(int user_id) {
        TRACE_FUNC();  // ✨ "UserService::update_profile"
        
        // Manual scope for specific operation
        {
            TRACE_SCOPE("Profile Validation");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        DatabaseService db;
        db.connect();
        db.query("UPDATE users SET last_login = NOW()");
        db.disconnect();
    }
};

class APIGateway {
public:
    void handle_request(const std::string& path) {
        TRACE_FUNC();  // ✨ "APIGateway::handle_request"
        
        AuthService auth;
        auth.validate_token("abc123");
        
        UserService users;
        users.load_user(42);
        users.update_profile(42);
    }
};

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Automatic Function Tracing Demo" << std::endl;
    std::cout << "   (Compiler-Provided Names)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    ServiceConfigurator::initialize();
    
    // ========================================
    // Test: Automatic Hierarchical Tracing
    // ========================================
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        
        // Set global context for demo
        g_current_ctx = ctx.get();
        
        ctx->method = "POST";
        ctx->path = "/api/users/42/profile";
        
        std::cout << "📡 Processing API Request...\n" << std::endl;
        
        // Call API gateway - ALL function calls auto-traced!
        APIGateway gateway;
        gateway.handle_request("/api/users/42/profile");
        
        std::cout << "\n✅ Request Complete!\n" << std::endl;
        
        // Show full trace with indentation
        std::cout << ctx->flush_trace() << std::endl;
        
        g_current_ctx = nullptr;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ✅ Automatic Tracing Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n💡 Features Demonstrated:" << std::endl;
    std::cout << "  ✓ TRACE_FUNC() - Compiler auto-inserts function name\n";
    std::cout << "  ✓ Hierarchical Indentation - Shows call tree\n";
    std::cout << "  ✓ RAII Scope Guard - Auto enter/exit\n";
    std::cout << "  ✓ Timing - Per-function elapsed time\n";
    std::cout << "  ✓ Manual scopes - TRACE_SCOPE(\"name\")\n";
    std::cout << "  ✓ Zero boilerplate - Just add TRACE_FUNC()\n";
    
    std::cout << "\n📝 Usage:" << std::endl;
    std::cout << R"(
  // 1. Add to any method (compiler gets name!):
  void MyClass::my_method() {
      TRACE_FUNC();  // ✨ Magic! Compiler inserts "MyClass::my_method"
      // ... work ...
  }
  
  // 2. Manual scope for blocks:
  {
      TRACE_SCOPE("Database Transaction");
      // ... work ...
  } // Auto-exit
  
  // 3. In production, traces only written on ERROR:
  if (result.is_err()) {
      LOG_ERROR(ctx->flush_trace());  // FULL call tree!
  }
)" << std::endl;
    
    std::cout << "\n🎯 Benefits:" << std::endl;
    std::cout << "  • See EXACT call chain on errors\n";
    std::cout << "  • No manual logging in every method\n";
    std::cout << "  • Compiler provides accurate names\n";
    std::cout << "  • RAII ensures no leaks\n";
    std::cout << "  • Indented output = easy to read\n";
    
    return 0;
}
