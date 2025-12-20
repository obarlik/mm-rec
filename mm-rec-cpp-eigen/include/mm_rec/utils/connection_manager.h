#pragma once

#include "mm_rec/utils/logger.h"
#include <string>
#include <mutex>
#include <set>

namespace mm_rec {
namespace net {

// Connection Manager to track and limit active connections
class ConnectionManager {
public:
    ConnectionManager(size_t max_conns = 1000) : max_connections_(max_conns) {}

    bool accept_connection(const std::string& ip_address) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_count_ >= max_connections_) {
            LOG_WARN("Connection limit reached (" + std::to_string(max_connections_) + "). Rejecting " + ip_address);
            return false;
        }
        active_connections_.insert(ip_address); 
        current_count_++;
        return true;
    }

    void close_connection(const std::string& ip_address) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_count_ > 0) current_count_--;
        // active_connections_.erase(ip_address); // Keep set logic simple for now, just counting
        (void)ip_address; // Suppress unused warning
    }

    size_t get_active_count() const {
        return current_count_;
    }

private:
    size_t max_connections_;
    size_t current_count_ = 0;
    std::set<std::string> active_connections_;
    std::mutex mutex_;
};

} // namespace net
} // namespace mm_rec
