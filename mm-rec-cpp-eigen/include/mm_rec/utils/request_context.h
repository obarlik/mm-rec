#pragma once

#include <string>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>

namespace mm_rec {
namespace net {

/**
 * Request Context - Per-request metadata and utilities
 * 
 * Scoped to a single HTTP request, provides:
 * - Unique request ID for tracing
 * - Timing information
 * - Client metadata (IP, User-Agent)
 * - Request path and method
 */
class RequestContext {
public:
    std::string request_id;
    std::chrono::steady_clock::time_point start_time;
    std::string ip_address;
    std::string user_agent;
    std::string method;
    std::string path;
    
    RequestContext() {
        request_id = generate_request_id();
        start_time = std::chrono::steady_clock::now();
    }
    
    /**
     * Get elapsed time since request started (in milliseconds).
     */
    int64_t elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start_time
        ).count();
    }
    
    /**
     * Get elapsed time since request started (in microseconds).
     */
    int64_t elapsed_us() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            now - start_time
        ).count();
    }
    
    /**
     * Format as log prefix: [REQ-abc123] 
     */
    std::string log_prefix() const {
        return "[" + request_id + "] ";
    }

private:
    /**
     * Generate unique request ID (REQ-{timestamp}-{random})
     */
    static std::string generate_request_id() {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()
        ).count();
        
        // Random 4-digit suffix
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(1000, 9999);
        int random_suffix = dis(gen);
        
        std::ostringstream oss;
        oss << "REQ-" << std::hex << (ms & 0xFFFFFF) << "-" << std::dec << random_suffix;
        return oss.str();
    }
};

} // namespace net
} // namespace mm_rec
