#pragma once

#include <atomic>
#include <mutex>

namespace mm_rec {

/**
 * Dynamic Load Balancer for Hybrid Execution
 * 
 * Monitors synchronization time between CPU and GPU tasks.
 * Adjusts the workload ratio to minimize idle time (Total Wait minimization).
 */
class DynamicBalancer {
public:
    // Get the current optimal GPU ratio (0.0 - 1.0)
    static float get_gpu_ratio();
    
    // Get the current time difference (CPU - GPU) in ms
    // Near 0 means perfect balance. Positive means CPU is slow.
    static double get_sync_diff();

    // Updates the ratio based on measured execution times
    // cpu_ms: Duration of CPU portion
    // gpu_ms: Duration of GPU portion
    static void report_metrics(double cpu_ms, double gpu_ms);
    
    // Reset to defaults
    static void reset();

private:
    static std::atomic<float> current_ratio;
    static std::atomic<int> sample_count;
    static std::atomic<double> avg_diff; 
    
    // Config
    static constexpr int SAMPLE_WINDOW = 50; // Update every 50 calls
    static constexpr float LEARNING_RATE = 0.05f; // Adjustment step size
};

} // namespace mm_rec
