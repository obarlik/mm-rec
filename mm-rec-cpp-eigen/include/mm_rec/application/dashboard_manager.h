#pragma once

#include "mm_rec/infrastructure/http_server.h"
#include "mm_rec/infrastructure/di_container.h"
#include "mm_rec/application/i_training_monitor.h" // Interface
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
    
    // Context
    std::string current_run_name; // Not atomic, but set once at start
};

class DashboardManager : public ITrainingMonitor { // ITrainingMonitor Implementation
public:
    void on_step_complete(const TrainingStats& stats) override;
    bool should_stop() override;

    // Singleton Access (Legacy support)
    static DashboardManager& instance() {
        static std::shared_ptr<DashboardManager> inst = std::make_shared<DashboardManager>();
        return *inst;
    }

    // Start the dashboard server
    bool start(int port = 8085) {
        net::HttpServerConfig config;
        config.port = port;
        return start(config);
    }

    bool start(const net::HttpServerConfig& config);
    void stop();
    
    // Update methods for workers
    void set_history_path(const std::string& path);
    void set_active_run(const std::string& run_name) {
        stats_.current_run_name = run_name;
    }
    void clear_active_run() {
         stats_.current_run_name.clear();
    }
    
    void update_training_stats(const TrainingStats& stats);
    void update_system_stats(size_t mem_mb);
    
    // Accessors
    bool should_stop() const { return stats_.should_stop; }
    void reset_stop_signal() { stats_.should_stop = false; }
    
    // Raw access if needed
    DashboardStats& stats() { return stats_; }

    // DI-friendly: Constructor with dependencies
    DashboardManager(std::shared_ptr<mm_rec::net::HttpServer> server);
    
    // Legacy/Default constructor
    DashboardManager();
    
    ~DashboardManager();

private:
    // Non-copyable
    DashboardManager(const DashboardManager&) = delete;
    DashboardManager& operator=(const DashboardManager&) = delete;

    void register_routes();

    // Injected dependencies (not owned!)
    std::shared_ptr<mm_rec::net::HttpServer> server_;
    DashboardStats stats_;
    
    // History for graphs
    std::mutex history_mtx_;
    const size_t max_history_size_ = 10000;
    // Trainer had 10000. If we want full history, we should increase.
    // However, DashboardManager logic implies it's for live view.
    // Let's match Trainer's logic or use a reasonable buffer. 10000 is safer for long runs.
    
    std::deque<float> loss_history_;
    std::deque<float> avg_loss_history_;
    std::deque<float> grad_norm_history_;
    std::deque<float> lr_history_;
    std::deque<float> data_stall_history_;
    std::deque<float> moe_loss_history_;
    std::deque<float> mem_history_;
    std::ofstream history_file_;
};

} // namespace mm_rec
