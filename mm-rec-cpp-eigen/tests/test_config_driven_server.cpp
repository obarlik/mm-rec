#include <iostream>
#include <cassert>
#include <memory>
#include "mm_rec/infrastructure/di_container.h"
#include "mm_rec/application/service_configurator.h"
#include "mm_rec/infrastructure/config.h"
#include "mm_rec/infrastructure/http_server.h"

using namespace mm_rec;
using namespace mm_rec::net;

void test_config_driven_server() {
    std::cout << "TEST: Config-Driven Server Initialization..." << std::endl;

    // 1. Setup Container and Config
    DIContainer container;
    
    // Bind Config with specific test values
    container.bind_singleton<Config>([]() {
        auto cfg = std::make_shared<Config>();
        cfg->set("server.port", 9999);
        cfg->set("server.threads", 2);
        cfg->set("server.max_connections", 50);
        return cfg;
    });

    // 2. Configure Services (Manually bind HttpServer using the same logic as ServiceConfigurator)
    // We emulate ServiceConfigurator logic here to avoid linking unrelated heavy dependencies (Dashboard, Model, etc.)
    container.bind_singleton<HttpServer>([](DIContainer& c) {
        auto cfg = c.resolve<Config>();
        
        // Exact logic from ServiceConfigurator:
        int port = cfg->get_int("server.port", 8085);
        int threads = cfg->get_int("server.threads", 4);
        int timeout = cfg->get_int("server.timeout", 3);
        int max_conn = cfg->get_int("server.max_connections", 100);
        int rate_limit = cfg->get_int("server.rate_limit", 1000);
        
        HttpServerConfig config;
        config.port = port;
        config.threads = threads;
        config.timeout_sec = timeout;
        config.max_connections = max_conn;
        config.max_req_per_min = rate_limit;
        
        return std::make_shared<HttpServer>(config);
    });
    
    // (Skipping ServiceConfigurator::configure_services(container) to avoid heavy dependencies)

    // 3. Resolve HttpServer
    std::cout << "Resolving HttpServer..." << std::endl;
    auto server = container.resolve<HttpServer>();

    // 4. Verification
    assert(server != nullptr);
    
    // Check Config Values were applied
    std::cout << "Checking Port: " << server->port() << std::endl;
    assert(server->port() == 9999);
    
    // Initial state check
    assert(server->is_running() == false);

    // Check fluent API manually (since we can't inspect internal config easily without getters)
    server->set_timeout(10);
    // Note: We can't verify timeout easily unless we add a getter, but ensuring it doesn't crash is good.

    std::cout << "✅ TEST PASSED: Config-driven server initialized correctly." << std::endl;
}

int main() {
    try {
        test_config_driven_server();
    } catch (const std::exception& e) {
        std::cerr << "❌ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
