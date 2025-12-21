#pragma once

#include "mm_rec/business/metrics.h"
#include "mm_rec/infrastructure/i_metrics_exporter.h"
#include <thread>
#include <atomic>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <mutex>

namespace mm_rec {
namespace infrastructure {

namespace fs = std::filesystem;

/**
 * Metrics Exporter
 * 
 * Responsibilities:
 * - Periodically wakes up
 * - Drains data from Business::MetricsManager
 * - Writes binary data to disk
 * - Handles file rotation
 */
class MetricsExporter : public IMetricsExporter {
public:
    MetricsExporter() : manager_(&MetricsManager::instance()) {}
    
    // DI Constructor
    explicit MetricsExporter(MetricsManager& manager) : manager_(&manager) {}
    
    ~MetricsExporter() override {
        stop();
    }
    
    // Start the background writer thread
    void start(const std::string& output_path, 
               const MetricsSamplingConfig& sampling = MetricsSamplingConfig()) override {
        
        if (running_.load(std::memory_order_acquire)) return;
        
        output_path_ = output_path;
        
        // Apply sampling config to the manager
        manager_->set_sampling_config(sampling);
        
        // File Rotation Logic
        rotate_if_needed(output_path);
        
        running_.store(true, std::memory_order_release);
        worker_thread_ = std::make_unique<std::thread>(&MetricsExporter::writer_loop, this);
    }
    
    // Stop the writer
    void stop() override {
        if (!running_.exchange(false)) return;
        
        if (worker_thread_ && worker_thread_->joinable()) {
            worker_thread_->join();
        }
        
        // Final flush
        flush();
    }
    
private:
    void rotate_if_needed(const std::string& path_str) {
        try {
            fs::path path(path_str);
            size_t max_bytes = 50 * 1024 * 1024; // 50MB
            int max_backups = 2;
            
            if (fs::exists(path) && fs::file_size(path) >= max_bytes) {
                fs::path oldest = path.parent_path() / (path.stem().string() + "." + std::to_string(max_backups) + path.extension().string());
                if (fs::exists(oldest)) fs::remove(oldest);
                
                for (int i = max_backups - 1; i >= 1; --i) {
                    fs::path src = path.parent_path() / (path.stem().string() + "." + std::to_string(i) + path.extension().string());
                    fs::path dst = path.parent_path() / (path.stem().string() + "." + std::to_string(i + 1) + path.extension().string());
                    if (fs::exists(src)) fs::rename(src, dst);
                }
                fs::path first = path.parent_path() / (path.stem().string() + ".1" + path.extension().string());
                fs::rename(path, first);
            }
        } catch (...) {
            // Log error?
        }
    }

    void writer_loop() {
        std::ofstream ofs(output_path_, std::ios::binary | std::ios::out | std::ios::app);
        if (!ofs) return;
        
        write_header_if_new(ofs);
        
        std::vector<MetricEvent> batch;
        batch.reserve(2048);
        std::vector<char> byte_buffer;
        byte_buffer.reserve(2048 * sizeof(MetricEvent));
        
        while (running_.load(std::memory_order_acquire)) {
            // 1. Collect from Business Layer
            manager_->collect_all(batch, 1024);
            
            // 2. Write to Infra Layer (Disk)
            if (!batch.empty()) {
                byte_buffer.clear();
                for (const auto& e : batch) {
                    const char* data = reinterpret_cast<const char*>(&e);
                    byte_buffer.insert(byte_buffer.end(), data, data + sizeof(MetricEvent));
                }
                
                ofs.write(byte_buffer.data(), byte_buffer.size());
                ofs.flush();
                batch.clear();
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    void flush() {
        std::ofstream ofs(output_path_, std::ios::binary | std::ios::out | std::ios::app);
        if (!ofs) return;
        
        std::vector<MetricEvent> batch;
        manager_->collect_all(batch, 100000); // Drain everything
        
        if (!batch.empty()) {
            for (const auto& e : batch) {
                ofs.write(reinterpret_cast<const char*>(&e), sizeof(MetricEvent));
            }
        }
    }

    void write_header_if_new(std::ofstream& ofs) {
        ofs.seekp(0, std::ios::end);
        if (ofs.tellp() == 0) {
            const char magic[4] = {'M', 'M', 'R', 'C'};
            const uint32_t version = 1;
            const uint32_t reserved = 0;
            ofs.write(magic, 4);
            ofs.write(reinterpret_cast<const char*>(&version), 4);
            ofs.write(reinterpret_cast<const char*>(&reserved), 4);
        }
    }

    std::string output_path_;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::thread> worker_thread_;
    MetricsManager* manager_; // Injected dependency
};

} // namespace infrastructure
} // namespace mm_rec
