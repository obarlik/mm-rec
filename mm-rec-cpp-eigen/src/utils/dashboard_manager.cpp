#include "mm_rec/utils/dashboard_manager.h"
#include "mm_rec/utils/dashboard_html.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/ui.h" // [NEW] for console feedback
#include "mm_rec/core/vulkan_backend.h"
#include <sstream>
#include <iostream>

namespace mm_rec {

DashboardManager& DashboardManager::instance() {
    static DashboardManager instance;
    return instance;
}

DashboardManager::DashboardManager() {
    // Initialize standard/safe values
    // Open history file in append mode
    history_file_.open("dashboard_history.csv", std::ios::app);
}

DashboardManager::~DashboardManager() {
    if (history_file_.is_open()) history_file_.close();
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
    
    // Persist to CSV: step,loss,lr,speed
    if (history_file_.is_open()) {
        history_file_ << step << "," << loss << "," << lr << "," << speed << "\n";
        history_file_.flush();
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
    
    // API Hardware (Real)
    server_->register_handler("/api/hardware", [](const std::string&) -> std::string {
        auto& vk = mm_rec::VulkanBackend::get();
        std::string gpu_name = vk.is_ready() ? vk.get_device_name() : "Vulkan Not Ready";
        size_t vram_mb = vk.is_ready() ? (vk.get_total_vram() / 1024 / 1024) : 0;
        
        std::stringstream json;
        json << "{";
        json << "  \"cpu_model\": \"Host Processor\","; 
        json << "  \"compute_device\": \"" << gpu_name << "\",";
        json << "  \"mem_total_mb\": " << vram_mb << ","; 
        json << "  \"arch\": \"x86_64 / SPIR-V\",";
        json << "  \"cores_logical\": \"N/A\","; 
        json << "  \"simd\": \"AVX2 / Int8\"";
        json << "}";
        return mm_rec::net::HttpServer::build_response(200, "application/json", json.str());
    });
    
    // Stop Signal
    server_->register_handler("/api/stop", [this](const std::string&) -> std::string {
        stats_.should_stop = true;
        LOG_INFO("Stop signal received from Dashboard.");
        return mm_rec::net::HttpServer::build_response(200, "text/plain", "Stopping...");
    });

    // History API
    server_->register_handler("/api/history", [](const std::string&) -> std::string {
        std::ifstream f("dashboard_history.csv");
        std::vector<float> losses;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string segment;
            std::vector<std::string> seglist;
            while(std::getline(ss, segment, ',')) {
               seglist.push_back(segment);
            }
            if (seglist.size() >= 2) {
                try {
                    losses.push_back(std::stof(seglist[1])); // Loss is 2nd column
                } catch (...) {}
            }
        }
        
        std::stringstream json;
        json << "{ \"loss_history\": [";
        for (size_t i = 0; i < losses.size(); ++i) {
             json << losses[i] << (i < losses.size() - 1 ? "," : "");
        }
        json << "] }";
        
        return mm_rec::net::HttpServer::build_response(200, "application/json", json.str());
    });
}

} // namespace mm_rec
