#include "mm_rec/infrastructure/http_server.h"
#include "mm_rec/infrastructure/di_container.h"
#include <iostream>

using namespace mm_rec;
using namespace mm_rec::net;

// Example services
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
    std::chrono::steady_clock::time_point start_time;
    
    RequestContext() {
        request_id = "REQ-" + std::to_string(rand() % 10000);
        start_time = std::chrono::steady_clock::now();
        std::cout << "[RequestContext] Created: " << request_id << std::endl;
    }
    
    ~RequestContext() {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time
        ).count();
        std::cout << "[RequestContext] Destroyed: " << request_id << " (took " << elapsed << "ms)" << std::endl;
    }
};

class UserService {
private:
    std::shared_ptr<ILogger> logger_;
    std::shared_ptr<RequestContext> ctx_;
    
public:
    UserService(std::shared_ptr<ILogger> logger, std::shared_ptr<RequestContext> ctx)
        : logger_(logger), ctx_(ctx) {
        logger_->log("UserService created for request: " + ctx_->request_id);
    }
    
    std::string get_users() {
        logger_->log("Fetching users for request: " + ctx_->request_id);
        return "{\"users\": [\"Alice\", \"Bob\"], \"request_id\": \"" + ctx_->request_id + "\"}";
    }
};

REGISTER_CTOR(UserService, std::shared_ptr<ILogger>, std::shared_ptr<RequestContext>);

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  DI Container - Request Scope Demo" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Setup DI Container
    DIContainer container;
    container.bind_singleton<ILogger, ConsoleLogger>();
    container.bind_scoped<RequestContext>();  // Per-request
    container.bind_scoped<UserService>();     // Per-request
    
    // Create HTTP server
    HttpServer server(8080);
    
    // Middleware: Create Scope for each request
    server.use([&container](const Request& req, std::shared_ptr<Response> res, auto next) {
        std::cout << "\n[Middleware] Request started: " << req.method << " " << req.path << std::endl;
        
        // Create scope for this request
        mm_rec::Scope request_scope(container);
        
        // Attach scope to request (non-const cast needed)
        const_cast<Request&>(req).scope = &request_scope;
        
        // Call handler
        next(req, res);
        
        std::cout << "[Middleware] Request completed" << std::endl;
        // Scope destroyed here -> RequestContext destroyed
    });
    
    // Handler with DI
    server.register_handler("/api/users", [](const Request& req, std::shared_ptr<Response> res) {
        std::cout << "[Handler] Processing /api/users" << std::endl;
        
        // Inject dependencies from request scope!
        auto userService = req.get<UserService>();
        auto ctx = req.get<RequestContext>();
        
        std::string users = userService->get_users();
        
        res->set_header("Content-Type", "application/json");
        res->send(users);
    });
    
    // Simulate 3 requests
    std::cout << "\n=== Simulating 3 HTTP Requests ===" << std::endl;
    
    for (int i = 1; i <= 3; i++) {
        std::cout << "\n--- Request #" << i << " ---" << std::endl;
        
        Request req;
        req.method = "GET";
        req.path = "/api/users";
        req.is_connected = []() { return true; };
        
        auto res = std::make_shared<Response>(0);  // Dummy socket
        
        // Simulate request handling (manually call middleware chain)
        server.use([&](const Request& r, std::shared_ptr<Response> response, auto next) {
            mm_rec::Scope request_scope(container);
            const_cast<Request&>(r).scope = &request_scope;
            
            // Manually call handler
            auto userService = r.get<UserService>();
            auto users = userService->get_users();
            std::cout << "[Response] " << users << std::endl;
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ✅ Demo Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nNotice:" << std::endl;
    std::cout << "- Each request gets its own RequestContext" << std::endl;
    std::cout << "- Services are injected via req.get<T>()" << std::endl;
    std::cout << "- Scoped instances destroyed after each request" << std::endl;
    
    return 0;
}
