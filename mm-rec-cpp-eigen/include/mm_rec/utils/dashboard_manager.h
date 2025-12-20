#pragma once

#include "mm_rec/utils/http_server.h"
#include <atomic>
#include <mutex>
#include <deque>
#include <string>
#include <vector>
#include <memory>

namespace mm_rec {

struct DashboardStats {
    std::atomic<float> current_loss{0.0f};
    std::atomic<float> current_speed{0.0f}; // tokens/sec
    std::atomic<int>   current_step{0};
    std::atomic<int>   total_steps{0};
    std::atomic<float> current_lr{0.0f};
    std::atomic<float> current_flux_scale{0.0f};
    
    // System stats
    std::atomic<size_t> memory_usage_mb{0};
    
    // Control flags
    std::atomic<bool> should_stop{false};
};

class DashboardManager {
public:
    static DashboardManager& instance();

    // Start the background server
    bool start(int port = 8085);
    void stop();
    
    // Update methods for workers
    void update_training_stats(float loss, float lr, float speed, int step);
    void update_system_stats(size_t mem_mb);
    
    // Accessors
    bool should_stop() const { return stats_.should_stop; }
    void reset_stop_signal() { stats_.should_stop = false; }
    
    // Raw access if needed
    DashboardStats& stats() { return stats_; }

private:
    DashboardManager();
    ~DashboardManager();

    // Non-copyable
    DashboardManager(const DashboardManager&) = delete;
    DashboardManager& operator=(const DashboardManager&) = delete;

    void register_routes();

    std::unique_ptr<mm_rec::net::HttpServer> server_;
    DashboardStats stats_;
    
    // History for graphs
    std::mutex history_mtx_;
    std::deque<float> loss_history_;
    const size_t max_history_size_ = 500;
    std::ofstream history_file_;
};

} // namespace mm_rec
