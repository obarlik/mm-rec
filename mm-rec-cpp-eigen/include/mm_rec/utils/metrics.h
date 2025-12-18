/**
 * Zero-Overhead Metrics Collection System
 * 
 * Design principles:
 * - Thread-local storage (no locks)
 * - Compile-time toggling (#ifdef ENABLE_METRICS)
 * - Async background writer
 * - Ring buffer (no dynamic allocation during training)
 */

#pragma once

#include <atomic>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace mm_rec {

// Compile-time flag (set via CMake or -DENABLE_METRICS)
#ifndef ENABLE_METRICS
#define ENABLE_METRICS 1  // Default: enabled for development
#endif

/**
 * Metric Event Types
 */
enum class MetricType {
    TRAINING_STEP,
    INFERENCE_STEP,
    FORWARD_PASS,
    BACKWARD_PASS,
    OPTIMIZER_STEP,
    CHECKPOINT_SAVE,
    MEMORY_USAGE,
    FLUX_BRAKE,
    CUSTOM
};

/**
 * Single metric event (32 bytes, cache-friendly)
 */
struct MetricEvent {
    MetricType type;
    uint64_t timestamp_us;  // Microseconds since epoch
    float value1;
    float value2;
    uint32_t extra;
    char label[8];  // Short identifier
    
    MetricEvent() = default;
    
    MetricEvent(MetricType t, float v1 = 0.0f, float v2 = 0.0f, 
                uint32_t ex = 0, const char* lbl = "")
        : type(t), value1(v1), value2(v2), extra(ex) {
        auto now = std::chrono::high_resolution_clock::now();
        timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
        
        // Safe string copy
        int i = 0;
        while (i < 7 && lbl[i] != '\0') {
            label[i] = lbl[i];
            ++i;
        }
        label[i] = '\0';
    }
};

/**
 * Lock-Free Ring Buffer (Thread-Local per worker)
 */
class MetricsBuffer {
public:
    static constexpr size_t BUFFER_SIZE = 16384;  // 16K events ~512KB
    
    MetricsBuffer() : write_idx_(0), read_idx_(0) {
        buffer_.resize(BUFFER_SIZE);
    }
    
    // Fast path: push event (lock-free if single producer)
    bool push(const MetricEvent& event) {
#if ENABLE_METRICS
        size_t current = write_idx_.load(std::memory_order_relaxed);
        size_t next = (current + 1) % BUFFER_SIZE;
        
        // Check if buffer is full
        if (next == read_idx_.load(std::memory_order_acquire)) {
            return false;  // Full, drop event
        }
        
        buffer_[current] = event;
        write_idx_.store(next, std::memory_order_release);
        return true;
#else
        return true;  // No-op if metrics disabled
#endif
    }
    
    // Slow path: consume events (called by background thread)
    size_t consume(std::vector<MetricEvent>& out, size_t max_count) {
        size_t count = 0;
        size_t current = read_idx_.load(std::memory_order_relaxed);
        size_t end = write_idx_.load(std::memory_order_acquire);
        
        while (current != end && count < max_count) {
            out.push_back(buffer_[current]);
            current = (current + 1) % BUFFER_SIZE;
            ++count;
        }
        
        read_idx_.store(current, std::memory_order_release);
        return count;
    }
    
private:
    std::vector<MetricEvent> buffer_;
    std::atomic<size_t> write_idx_;
    std::atomic<size_t> read_idx_;
};

/**
 * Global Metrics Manager
 */
class MetricsManager {
public:
    static MetricsManager& instance() {
        static MetricsManager inst;
        return inst;
    }
    
    // Thread-local buffer accessor with auto-registration
    static MetricsBuffer& get_local_buffer() {
        static thread_local MetricsBuffer buffer;
        static thread_local bool registered = false;
        
        if (!registered) {
            instance().register_buffer(&buffer);
            registered = true;
        }
        
        return buffer;
    }
    
    // Fast recording (inline, essentially free)
    static inline void record(MetricType type, float v1 = 0.0f, 
                              float v2 = 0.0f, uint32_t ex = 0, 
                              const char* lbl = "") {
#if ENABLE_METRICS
        MetricEvent event(type, v1, v2, ex, lbl);
        get_local_buffer().push(event);
#endif
    }
    
    // Start background writer
    void start_writer(const std::string& output_path) {
#if ENABLE_METRICS
        if (writer_running_) return;
        
        output_path_ = output_path;
        writer_running_ = true;
        writer_thread_ = std::thread(&MetricsManager::writer_loop, this);
#endif
    }
    
    // Stop background writer
    void stop_writer() {
#if ENABLE_METRICS
        if (!writer_running_) return;
        
        writer_running_ = false;
        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
        
        // Flush remaining events from all threads
        flush_all();
#endif
    }
    
    ~MetricsManager() {
        stop_writer();
    }
    
private:
    MetricsManager() : writer_running_(false) {}
    
    void register_buffer(MetricsBuffer* buf) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        buffers_.push_back(buf);
    }
    
    void writer_loop() {
        std::ofstream ofs(output_path_, std::ios::out);
        if (!ofs) return;
        
        std::vector<MetricEvent> batch;
        batch.reserve(1024);
        
        while (writer_running_) {
            // Collect from all registered thread-local buffers
            {
                std::lock_guard<std::mutex> lock(registry_mutex_);
                for (auto* buf : buffers_) {
                    buf->consume(batch, 1024);
                }
            }
            
            // Write to disk
            for (const auto& e : batch) {
                write_event_json(ofs, e);
            }
            batch.clear();
            ofs.flush();
            
            // Sleep to avoid busy-wait
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    void flush_all() {
        std::vector<MetricEvent> events;
        
        // Flush from all thread buffers
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            for (auto* buf : buffers_) {
                buf->consume(events, 100000);
            }
        }
        
        if (!events.empty() && !output_path_.empty()) {
            std::ofstream ofs(output_path_, std::ios::app);
            for (const auto& e : events) {
                write_event_json(ofs, e);
            }
        }
    }
    
    void write_event_json(std::ofstream& ofs, const MetricEvent& e) {
        ofs << "{\"type\":" << static_cast<int>(e.type) 
            << ",\"ts\":" << e.timestamp_us
            << ",\"v1\":" << e.value1
            << ",\"v2\":" << e.value2
            << ",\"extra\":" << e.extra
            << ",\"label\":\"" << e.label << "\"}\n";
    }
    
    std::string output_path_;
    std::atomic<bool> writer_running_;
    std::thread writer_thread_;
    
    // Thread buffer registry
    std::mutex registry_mutex_;
    std::vector<MetricsBuffer*> buffers_;
};

// Convenience macros
#define METRIC_RECORD(type, v1, v2, extra, label) \
    mm_rec::MetricsManager::record(mm_rec::MetricType::type, v1, v2, extra, label)

#define METRIC_TRAINING_STEP(loss, lr) \
    METRIC_RECORD(TRAINING_STEP, loss, lr, 0, "")

#define METRIC_INFERENCE(latency_ms, tokens) \
    METRIC_RECORD(INFERENCE_STEP, latency_ms, tokens, 0, "")

#define METRIC_FLUX_BRAKE(count) \
    METRIC_RECORD(FLUX_BRAKE, count, 0, 0, "")

} // namespace mm_rec
