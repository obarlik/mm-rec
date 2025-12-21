#include "mm_rec/infrastructure/di_container.h"
#include <iostream>
#include <memory>

using namespace mm_rec;

// Mock Request/Response for simple demo
struct SimpleRequest {
    mm_rec::Scope* scope = nullptr;
    
    template<typename T>
    std::shared_ptr<T> get() const {
        if (!scope) {
            throw std::runtime_error("DI Scope not available");
        }
        return scope->template resolve<T>();
    }
};

// Services
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

class RequestContext {
public:
    std::string request_id;
    
    RequestContext() {
        request_id = "REQ-" + std::to_string(rand() % 10000);
        std::cout << "  [RequestContext] Created: " << request_id << std::endl;
    }
    
    ~RequestContext() {
        std::cout << "  [RequestContext] Destroyed: " << request_id << std::endl;
    }
};

class UserService {
private:
    std::shared_ptr<ILogger> logger_;
    std::shared_ptr<RequestContext> ctx_;
    
public:
    UserService(std::shared_ptr<ILogger> logger, std::shared_ptr<RequestContext> ctx)
        : logger_(logger), ctx_(ctx) {
        logger_->log("UserService created for " + ctx_->request_id);
    }
    
    std::string get_users() {
        logger_->log("Fetching users for " + ctx_->request_id);
        return "{\"users\": [\"Alice\", \"Bob\"], \"request_id\": \"" + ctx_->request_id + "\"}";
    }
};

REGISTER_CTOR(UserService, std::shared_ptr<ILogger>, std::shared_ptr<RequestContext>);

void handle_request(DIContainer& container, const std::string& path) {
    std::cout << "\n🌐 HTTP Request: GET " << path << std::endl;
    
    // Middleware: Create scope for request
    mm_rec::Scope request_scope(container);
    
    // Create request with scope
    SimpleRequest req;
    req.scope = &request_scope;
    
    // Handler: Inject dependencies via req.get<T>()
    auto userService = req.get<UserService>();
    auto ctx = req.get<RequestContext>();
    
    std::string users = userService->get_users();
    std::cout << "  📤 Response: " << users << std::endl;
    
    // Scope destroyed here -> RequestContext cleaned up
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Request-Scoped DI Demo" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Setup DI Container
    DIContainer container;
    container.bind_singleton<ILogger, ConsoleLogger>();
    container.bind_scoped<RequestContext>();  // Per-request
    container.bind_scoped<UserService>();     // Per-request
    
    // Simulate 3 HTTP requests
    handle_request(container, "/api/users");
    handle_request(container, "/api/users");
    handle_request(container, "/api/users");
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ✅ Demo Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n💡 Key Points:" << std::endl;
    std::cout << "  - Each request creates NEW RequestContext" << std::endl;
    std::cout << "  - Services injected via req.get<T>()" << std::endl;
    std::cout << "  - Scoped instances auto-destroyed after request" << std::endl;
    std::cout << "  - Logger is Singleton (shared across requests)" << std::endl;
    
    return 0;
}
