#pragma once

#include "mm_rec/utils/logger.h"
#include <string>
#include <mutex>
#include <map>
#include <chrono>
#include <thread>

namespace mm_rec {
namespace net {

struct ClientStats {
    int request_count = 0;
    std::chrono::steady_clock::time_point window_start;
};

// Traffic Manager for Rate Limiting and Throttling
class TrafficManager {
public:
    TrafficManager(int max_req_per_min = 600, int throttle_delay_ms = 0) 
        : max_req_per_min_(max_req_per_min), throttle_delay_ms_(throttle_delay_ms) {}

    bool allowed(const std::string& ip) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        
        auto& stats = clients_[ip];
        
        // Reset window if passed
        if (std::chrono::duration_cast<std::chrono::minutes>(now - stats.window_start).count() >= 1) {
            stats.window_start = now;
            stats.request_count = 0;
        }

        if (stats.request_count >= max_req_per_min_) {
             LOG_WARN("Rate limit exceeded for " + ip);
             return false;
        }

        stats.request_count++;
        return true;
    }

    void apply_throttling() {
        if (throttle_delay_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(throttle_delay_ms_));
        }
    }

private:
    int max_req_per_min_;
    int throttle_delay_ms_;
    std::map<std::string, ClientStats> clients_;
    std::mutex mutex_;
};

} // namespace net
} // namespace mm_rec
