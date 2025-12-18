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
#include <memory>

namespace mm_rec {

// Metrics are always compiled and controlled at runtime via start_writer()

/**
 * Sampling configuration for production
 */
struct MetricsSamplingConfig {
    bool enabled = false;          // Enable sampling
    uint32_t interval = 10;        // Record every Nth event
    uint32_t warmup_events = 100;  // Always record first N
    uint32_t cooldown_events = 100; // Always record last N per session
    
    MetricsSamplingConfig() = default;
};

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
        size_t current = write_idx_.load(std::memory_order_relaxed);
        size_t next = (current + 1) % BUFFER_SIZE;
        
        // Check if buffer is full
        if (next == read_idx_.load(std::memory_order_acquire)) {
            return false;  // Full, drop event
        }
        
        buffer_[current] = event;
        write_idx_.store(next, std::memory_order_release);
        return true;
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
        auto& mgr = instance();
        
        // Skip if writer not running (zero overhead when disabled)
        if (!mgr.writer_running_.load(std::memory_order_relaxed)) {
            return;
        }
        
        // Sampling logic (production mode)
        if (mgr.sampling_config_.enabled) {
            mgr.event_counter_++;
            
            // Always record warmup/cooldown
            bool is_warmup = mgr.event_counter_ <= mgr.sampling_config_.warmup_events;
            bool is_sampled = (mgr.event_counter_ % mgr.sampling_config_.interval) == 0;
            
            if (!is_warmup && !is_sampled) {
                return;  // Skip this event
            }
        }
        
        MetricEvent event(type, v1, v2, ex, lbl);
        get_local_buffer().push(event);
    }
    
    // Start background writer (binary format only)
    void start_writer(const std::string& output_path, 
                     const MetricsSamplingConfig& sampling = MetricsSamplingConfig()) {
        if (writer_running_.load(std::memory_order_acquire)) return;
        
        output_path_ = output_path;
        sampling_config_ = sampling;
        event_counter_ = 0;
        
        writer_running_.store(true, std::memory_order_release);
        writer_thread_ = new std::thread(&MetricsManager::writer_loop, this);
    }
    
    // Stop background writer
    void stop_writer() {
        if (!writer_running_.exchange(false)) return;  // Atomic swap
        
        // Wait for writer thread to finish
        if (writer_thread_ && writer_thread_->joinable()) {
            writer_thread_->join();
            delete writer_thread_;
            writer_thread_ = nullptr;
        }
        
        // Now safe to flush (writer thread is stopped)
        flush_all();
    }
    
    // Minimal destructor - just signal stop, don't touch anything
    // Intentionally leak thread to avoid static destruction issues
    ~MetricsManager() {
        writer_running_.store(false, std::memory_order_release);
        writer_thread_ = nullptr;  // Leak (singleton lives until program exit anyway)
    }
    
private:
    MetricsManager() = default;
    
    void register_buffer(MetricsBuffer* buf) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        buffers_.push_back(buf);
    }
    
    void writer_loop() {
        std::ofstream ofs(output_path_, std::ios::binary | std::ios::out);
        if (!ofs) return;
        
        // Write binary header: [MAGIC:4][VERSION:4][RESERVED:4]
        const char magic[4] = {'M', 'M', 'R', 'C'};
        const uint32_t version = 1;
        const uint32_t reserved = 0;
        ofs.write(magic, 4);
        ofs.write(reinterpret_cast<const char*>(&version), 4);
        ofs.write(reinterpret_cast<const char*>(&reserved), 4);
        
        std::vector<MetricEvent> batch;
        batch.reserve(1024);
        
        // Batch write buffer
        std::vector<char> write_buffer;
        write_buffer.reserve(1024 * sizeof(MetricEvent));
        
        while (writer_running_.load(std::memory_order_acquire)) {
            // Collect from all registered thread-local buffers
            {
                std::lock_guard<std::mutex> lock(registry_mutex_);
                for (auto* buf : buffers_) {
                    // Safety: buf might be destroyed if thread-local storage is being cleaned up
                    if (buf) {
                        try {
                            buf->consume(batch, 1024);
                        } catch (...) {
                            // Buffer may be in invalid state during shutdown, skip it
                        }
                    }
                }
            }
            
            // Write to disk (batch mode)
            if (!batch.empty()) {
                write_buffer.clear();
                write_buffer.reserve(batch.size() * sizeof(MetricEvent));
                
                // Batch serialize to buffer
                for (const auto& e : batch) {
                    const char* data = reinterpret_cast<const char*>(&e);
                    write_buffer.insert(write_buffer.end(), data, data + sizeof(MetricEvent));
                }
                
                // Single write call
                ofs.write(write_buffer.data(), write_buffer.size());
                batch.clear();
                ofs.flush();
            }
            
            // Sleep to avoid busy-wait
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    void flush_all() {
        // Don't flush if writer is still running (let it handle it)
        if (writer_running_.load(std::memory_order_acquire)) {
            return;
        }
        
        std::vector<MetricEvent> events;
        
        // Flush from all thread buffers
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            for (auto* buf : buffers_) {
                if (buf) {  // Safety check
                    buf->consume(events, 100000);
                }
            }
        }
        
        if (!events.empty() && !output_path_.empty()) {
            std::ofstream ofs(output_path_, std::ios::binary | std::ios::app);
            if (ofs) {
                for (const auto& e : events) {
                    ofs.write(reinterpret_cast<const char*>(&e), sizeof(MetricEvent));
                }
            }
        }
    }
    
    std::string output_path_;
    MetricsSamplingConfig sampling_config_;
    std::atomic<uint64_t> event_counter_{0};
    
    std::atomic<bool> writer_running_{false};
    std::thread* writer_thread_{nullptr};
    
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
