#pragma once

#include <chrono>
#include <cstdint>

namespace mm_rec {

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
    HYBRID_PERF,
    API_LATENCY,
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
 * Sampling configuration
 */
struct MetricsSamplingConfig {
    bool enabled = false;          // Enable sampling
    uint32_t interval = 10;        // Record every Nth event
    uint32_t warmup_events = 100;  // Always record first N
    uint32_t cooldown_events = 100; // Always record last N per session
    
    MetricsSamplingConfig() = default;
};

} // namespace mm_rec
