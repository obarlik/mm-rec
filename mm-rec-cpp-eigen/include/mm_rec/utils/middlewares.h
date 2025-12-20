#pragma once

#include "mm_rec/utils/http_server.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/metrics.h"
#include <chrono>

namespace mm_rec {
namespace net {

class Middlewares {
public:
    // 1. Logger Middleware
    // Logs: [METHOD] path -> status code (duration ms)
    static std::string Logger(const Request& req, HttpServer::NextFn next) {
        auto start = std::chrono::steady_clock::now();
        
        // Process request
        std::string response = next(req);
        
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Extract Status Code (Naive parsing for log, ideally HttpServer returns Response object)
        std::string status = "???";
        if (response.size() > 12) {
            status = response.substr(9, 3);
        }

        LOG_INFO("[" + req.method + "] " + req.path + " -> " + status + " (" + std::to_string(duration) + "ms)");
        
        return response;
    }

    // 2. Security Middleware (CORS + Headers)
    static std::string Security(const Request& req, HttpServer::NextFn next) {
        std::string response = next(req);

        // Security Headers injection if not present
        // (Note: This naive string insertion works because our server builds simple responses. 
        //  A proper HttpParser would be better, but we stick to simplicity.)
        
        std::string headers_to_add = 
            "X-Content-Type-Options: nosniff\r\n"
            "X-Frame-Options: DENY\r\n"
            "X-XSS-Protection: 1; mode=block\r\n";
            
        // Find end of headers (\r\n\r\n)
        size_t header_end = response.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            response.insert(header_end, headers_to_add);
        }

        return response;
    }

    // 3. Metrics Middleware
    static std::string Metrics(const Request& req, HttpServer::NextFn next) {
        auto start = std::chrono::steady_clock::now();
        std::string response = next(req);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Record API Latency (Value1 = Duration ms)
        MetricsManager::record(MetricType::API_LATENCY, static_cast<float>(duration));
        
        return response;
    }
};

} // namespace net
} // namespace mm_rec
