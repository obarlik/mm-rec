#pragma once

#include "mm_rec/infrastructure/http_server.h"
#include "mm_rec/infrastructure/logger.h"
#include "mm_rec/business/metrics.h"
#include <chrono>

namespace mm_rec {
namespace net {

class Middlewares {
public:
    // 1. Logger Middleware
    // Logs: [METHOD] path -> status code (duration ms)
    static void Logger(const Request& req, std::shared_ptr<Response> res, HttpServer::NextFn next) {
        auto start = std::chrono::steady_clock::now();
        
        // Process request
        next(req, res);
        
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        LOG_INFO("[" + req.method + "] " + req.path + " -> " + std::to_string(res->get_status()) + " (" + std::to_string(duration) + "ms)");
    }

    // 2. Security Middleware (CORS + Headers)
    static void Security(const Request& req, std::shared_ptr<Response> res, HttpServer::NextFn next) {
        // Pre-process: Set headers BEFORE next() potentially sends data
        res->set_header("X-Content-Type-Options", "nosniff");
        res->set_header("X-Frame-Options", "DENY");
        res->set_header("X-XSS-Protection", "1; mode=block");
        
        // CORS (Already default in Response constructor, but enforcing here if needed)
        // res->set_header("Access-Control-Allow-Origin", "*"); 

        next(req, res);
    }

    // 3. Metrics Middleware
    static void Metrics(const Request& req, std::shared_ptr<Response> res, HttpServer::NextFn next) {
        auto start = std::chrono::steady_clock::now();
        next(req, res);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Record API Latency (Value1 = Duration ms)
        MetricsManager::record(MetricType::API_LATENCY, static_cast<float>(duration));
    }
};

} // namespace net
} // namespace mm_rec
