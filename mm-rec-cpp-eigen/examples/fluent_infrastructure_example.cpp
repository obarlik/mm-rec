// Example: Fluent Infrastructure API Design

#pragma once

#include <string>
#include <memory>

namespace mm_rec {
namespace net {

/**
 * Fluent Interface Example - HttpServer
 * 
 * Infrastructure provides fluent API for elegant configuration.
 * Methods return `this` to enable chaining.
 */
class HttpServer {
public:
    HttpServer(int port) : port_(port) {}
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // FLUENT API - All methods return this*
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    HttpServer* set_timeout(int seconds) {
        timeout_ = seconds;
        return this;  // ✅ Enable chaining
    }
    
    HttpServer* set_max_connections(int max) {
        max_connections_ = max;
        return this;
    }
    
    HttpServer* enable_ssl(const std::string& cert, const std::string& key) {
        ssl_enabled_ = true;
        ssl_cert_ = cert;
        ssl_key_ = key;
        return this;
    }
    
    HttpServer* enable_compression() {
        compression_enabled_ = true;
        return this;
    }
    
    HttpServer* add_middleware(const std::string& name) {
        // ... add middleware
        return this;
    }
    
    // Terminal methods (don't return this)
    void start() {
        // Start server
    }
    
    void stop() {
        // Stop server
    }
    
private:
    int port_;
    int timeout_ = 30;
    int max_connections_ = 100;
    bool ssl_enabled_ = false;
    bool compression_enabled_ = false;
    std::string ssl_cert_;
    std::string ssl_key_;
};

} // namespace net
} // namespace mm_rec

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// USAGE EXAMPLE (Domain Layer)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void example_fluent_usage() {
    using namespace mm_rec::net;
    
    // ✅ Fluent API - Elegant chaining!
    auto server = std::make_shared<HttpServer>(8085);
    
    server->set_timeout(30)
          ->set_max_connections(100)
          ->enable_ssl("/path/to/cert.pem", "/path/to/key.pem")
          ->enable_compression()
          ->add_middleware("auth")
          ->add_middleware("logging")
          ->start();  // Terminal method
    
    // vs. Non-fluent (verbose):
    /*
    server->set_timeout(30);
    server->set_max_connections(100);
    server->enable_ssl("/path/to/cert.pem", "/path/to/key.pem");
    server->enable_compression();
    server->add_middleware("auth");
    server->add_middleware("logging");
    server->start();
    */
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DOMAIN USAGE IN SERVICE CONFIGURATOR
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/*
container.bind_singleton<HttpServer>([](DIContainer& c) {
    auto cfg = c.resolve<Config>();
    
    auto server = std::make_shared<HttpServer>(
        cfg->get_int("server.port", 8085)
    );
    
    // Fluent configuration from config!
    server->set_timeout(cfg->get_int("server.timeout", 30))
          ->set_max_connections(cfg->get_int("server.max_connections", 100));
    
    // Conditional features (fluent)
    if (cfg->get_bool("server.ssl.enabled", false)) {
        server->enable_ssl(
            cfg->get("server.ssl.cert"),
            cfg->get("server.ssl.key")
        );
    }
    
    if (cfg->get_bool("server.compression", true)) {
        server->enable_compression();
    }
    
    return server;
});
*/
