#pragma once

#include "mm_rec/business/metric_types.h"
#include <atomic>
#include <vector>
#include <mutex>
#include <thread>
#include <memory>
// No fstream or file I/O here!

namespace mm_rec {

class MetricsManager;

/**
 * Lock-Free Ring Buffer (Thread-Local per worker)
 */
class MetricsBuffer {
public:
    static constexpr size_t BUFFER_SIZE = 16384;  // 16K events ~512KB
    
    MetricsBuffer() : write_idx_(0), read_idx_(0) {
        buffer_.resize(BUFFER_SIZE);
    }
    
    ~MetricsBuffer();

    bool push(const MetricEvent& event) {
        size_t current = write_idx_.load(std::memory_order_relaxed);
        size_t next = (current + 1) % BUFFER_SIZE;
        
        if (next == read_idx_.load(std::memory_order_acquire)) {
            return false;  // Full, drop event
        }
        
        buffer_[current] = event;
        write_idx_.store(next, std::memory_order_release);
        return true;
    }
    
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
 * Scoped Timer for Tree-Like Tracing (RAII)
 */
class ScopedMetric {
public:
    inline ScopedMetric(MetricType type, const char* label = "");
    inline ~ScopedMetric();

private:
    MetricType type_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    char label_[8];
};

/**
 * Global Metrics Manager (In-Memory Only)
 */
class MetricsManager {
    friend class MetricsBuffer;
public:
    static MetricsManager& instance() {
        static MetricsManager inst;
        return inst;
    }
    
    static MetricsBuffer& get_local_buffer() {
        static thread_local MetricsBuffer buffer;
        static thread_local bool registered = false;
        
        if (!registered) {
            instance().register_buffer(&buffer);
            registered = true;
        }
        
        return buffer;
    }
    
    // Fast recording
    static inline void record(MetricType type, float v1 = 0.0f, 
                              float v2 = 0.0f, uint32_t ex = 0, 
                              const char* lbl = "") {
        auto& mgr = instance();
        
        // Sampling logic
        if (mgr.sampling_config_.enabled) {
            mgr.event_counter_++;
            bool is_warmup = mgr.event_counter_ <= mgr.sampling_config_.warmup_events;
            bool is_sampled = (mgr.event_counter_ % mgr.sampling_config_.interval) == 0;
            
            if (!is_warmup && !is_sampled) {
                return;
            }
        }
        
        MetricEvent event(type, v1, v2, ex, lbl);
        get_local_buffer().push(event);
    }
    
    // Configuration
    void set_sampling_config(const MetricsSamplingConfig& config) {
        sampling_config_ = config;
    }
    
    // Access for Exporter (Consumer)
    // Drains all thread buffers into a single batch vector
    void collect_all(std::vector<MetricEvent>& out_batch, size_t limit_per_buffer = 1024) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        for (auto* buf : buffers_) {
            if (buf) {
                try {
                    buf->consume(out_batch, limit_per_buffer);
                } catch (...) {}
            }
        }
    }

    MetricsManager() = default;
    ~MetricsManager() = default;

private:
    void register_buffer(MetricsBuffer* buf) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        buffers_.push_back(buf);
    }

    void unregister_buffer(MetricsBuffer* buf) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
            if (*it == buf) {
                buffers_.erase(it);
                return;
            }
        }
    }
    
    MetricsSamplingConfig sampling_config_;
    std::atomic<uint64_t> event_counter_{0};
    
    std::mutex registry_mutex_;
    std::vector<MetricsBuffer*> buffers_;
};

// --- Inline Implementations ---

inline ScopedMetric::ScopedMetric(MetricType type, const char* label) 
    : type_(type), start_time_(std::chrono::high_resolution_clock::now()) {
    int i = 0;
    while (i < 7 && label[i] != '\0') {
        label_[i] = label[i];
        ++i;
    }
    label_[i] = '\0';
}

inline ScopedMetric::~ScopedMetric() {
    auto end_time = std::chrono::high_resolution_clock::now();
    float duration_ms = std::chrono::duration<float, std::milli>(end_time - start_time_).count();
    MetricsManager::record(type_, duration_ms, 0.0f, 0, label_);
}

inline MetricsBuffer::~MetricsBuffer() {
    MetricsManager::instance().unregister_buffer(this);
}

// Macros
#define METRIC_RECORD(type, v1, v2, extra, label) \
    mm_rec::MetricsManager::record(mm_rec::MetricType::type, v1, v2, extra, label)

#define METRIC_TRAINING_STEP(loss, lr) \
    METRIC_RECORD(TRAINING_STEP, loss, lr, 0, "")

#define METRIC_SCOPE(type, label) \
    mm_rec::ScopedMetric _scoped_metric_##__LINE__(mm_rec::MetricType::type, label)

#define METRIC_INFERENCE(latency_ms, tokens) \
    METRIC_RECORD(INFERENCE_STEP, latency_ms, tokens, 0, "")

#define METRIC_FLUX_BRAKE(count) \
    METRIC_RECORD(FLUX_BRAKE, count, 0, 0, "")

} // namespace mm_rec
