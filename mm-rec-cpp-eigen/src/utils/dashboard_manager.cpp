#include "mm_rec/utils/dashboard_manager.h"
#include "mm_rec/utils/dashboard_html.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/ui.h" // [NEW] for console feedback
#include <sstream>
#include <iostream>

namespace mm_rec {

DashboardManager& DashboardManager::instance() {
    static DashboardManager instance;
    return instance;
}

DashboardManager::DashboardManager() {
    // Initialize standard/safe values
}

DashboardManager::~DashboardManager() {
    stop();
}

bool DashboardManager::start(int base_port) {
    if (server_) return true; // Already running

    int max_retries = 10;
    bool success = false;

    for (int i = 0; i < max_retries; ++i) {
        int port = base_port + i;
        server_ = std::make_unique<mm_rec::net::HttpServer>(port);
        register_routes();
        
        if (server_->start()) {
            std::string msg = "Global Dashboard started on port " + std::to_string(port);
            LOG_INFO(msg);
            mm_rec::ui::success(msg);
            
            if (port != base_port) {
                std::string fallback_msg = "(Port " + std::to_string(base_port) + " was busy)";
                LOG_INFO(fallback_msg);
                mm_rec::ui::warning(fallback_msg);
            }
            success = true;
            break;
        }
        
        // Failed, try next
        server_.reset(); 
    }

    if (!success) {
        LOG_ERROR("Failed to start Global Dashboard on any port " + 
                 std::to_string(base_port) + "-" + std::to_string(base_port + max_retries - 1));
    }
    return success;
}

void DashboardManager::stop() {
    if (server_) {
        server_->stop();
        server_.reset();
        LOG_INFO("Global Dashboard stopped.");
    }
}

void DashboardManager::update_training_stats(float loss, float lr, float speed, int step) {
    stats_.current_loss = loss;
    stats_.current_lr = lr;
    stats_.current_speed = speed;
    stats_.current_step = step;
    
    std::lock_guard<std::mutex> lock(history_mtx_);
    loss_history_.push_back(loss);
    if (loss_history_.size() > max_history_size_) {
        loss_history_.pop_front();
    }
}

void DashboardManager::update_system_stats(size_t mem_mb) {
    stats_.memory_usage_mb = mem_mb;
}

void DashboardManager::register_routes() {
    if (!server_) return;

    // Home
    server_->register_handler("/", [](const std::string&) -> std::string {
        // Use the embedded HTML from dashboard_html.h
        // Assuming mm_rec::ui::DASHBOARD_HTML is available
        return mm_rec::net::HttpServer::build_response(200, "text/html", mm_rec::ui::DASHBOARD_HTML);
    });

    // API Stats
    server_->register_handler("/api/stats", [this](const std::string&) -> std::string {
        std::stringstream ss;
        ss << "{";
        ss << "\"loss\": " << stats_.current_loss.load() << ",";
        ss << "\"step\": " << stats_.current_step.load() << ",";
        ss << "\"total_steps\": " << stats_.total_steps.load() << ",";
        ss << "\"lr\": " << stats_.current_lr.load() << ",";
        ss << "\"speed\": " << stats_.current_speed.load() << ",";
        ss << "\"mem\": " << stats_.memory_usage_mb.load() << ",";
        ss << "\"epoch\": 1,"; // simplified
        
        std::lock_guard<std::mutex> lock(history_mtx_);
        ss << "\"history\": [";
        for(size_t i=0; i<loss_history_.size(); ++i) {
            ss << loss_history_[i] << (i < loss_history_.size()-1 ? "," : "");
        }
        ss << "],";
        ss << "\"avg_history\": []";
        ss << "}";
        return mm_rec::net::HttpServer::build_response(200, "application/json", ss.str());
    });
    
    // API Hardware (Mock/Real)
    server_->register_handler("/api/hardware", [](const std::string&) -> std::string {
        // We could hook this into SystemOptimizer or similar later
        std::string json = R"JSON({
            "cpu_model": "Intel Core (Production)",
            "compute_device": "Hybrid (CPU+Vulkan)"
        })JSON";
        return mm_rec::net::HttpServer::build_response(200, "application/json", json);
    });
    
    // Stop Signal
    server_->register_handler("/api/stop", [this](const std::string&) -> std::string {
        stats_.should_stop = true;
        LOG_INFO("Stop signal received from Dashboard.");
        return mm_rec::net::HttpServer::build_response(200, "text/plain", "Stopping...");
    });
}

} // namespace mm_rec
