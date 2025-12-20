#include "mm_rec/core/dynamic_balancer.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace mm_rec {

std::atomic<float> DynamicBalancer::current_ratio{0.5f}; // Start neutral, let it converge
// In linear.cpp code was: gpu_batch = batch * 0.8; So 0.8 was GPU ratio.
// But AutoTuner said "Ratio: 0.6" -> 60% CPU?
// Let's check linear.cpp existing code:
// int64_t gpu_batch = (int64_t)(batch * 0.8);
// So 0.8 is GPU ratio default.
// AutoTuner log said: "Launch... Ratio: 0.6" might be confusing.
// Let's initialize safely at 0.5 and let it converge.
std::atomic<int> DynamicBalancer::sample_count{0};
std::atomic<double> DynamicBalancer::avg_diff{0.0};

float DynamicBalancer::get_gpu_ratio() {
    return current_ratio.load(std::memory_order_relaxed);
}

double DynamicBalancer::get_sync_diff() {
    return avg_diff.load(std::memory_order_relaxed);
}

void DynamicBalancer::report_metrics(double cpu_ms, double gpu_ms) {
    // We want cpu_ms == gpu_ms
    // If cpu_ms > gpu_ms, CPU is bottleneck -> Reduce CPU load (Increase GPU ratio)
    // If gpu_ms > cpu_ms, GPU is bottleneck -> Reduce GPU load (Decrease GPU ratio)
    
    // Metric: (cpu - gpu)
    // Positive -> Increase Ratio
    // Negative -> Decrease Ratio
    
    // Update simple moving average diff
    // Not strictly thread safe atomic usage for avg_diff (RMW), but fine for stats
    double diff = cpu_ms - gpu_ms;
    
    // Using simple accumulation for window
    // Spin lock or just race is fine for statistics
    // Let's verify synchronization requirement.
    // Since we update periodically, we can accumulate.
    
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
     
    avg_diff = avg_diff.load() + diff;
    
    int count = ++sample_count;
    
    if (count >= SAMPLE_WINDOW) {
        double mean_diff = avg_diff.load() / SAMPLE_WINDOW;
        float ratio = current_ratio.load();
        
        // Control Logic
        if (mean_diff > 1.0) { // CPU significantly slower (>1ms)
            ratio += LEARNING_RATE;
        } else if (mean_diff < -1.0) { // GPU significantly slower
            ratio -= LEARNING_RATE;
        }
        
        // Clamp
        ratio = std::max(0.1f, std::min(0.9f, ratio));
        
        // Apply
        if (std::abs(ratio - current_ratio.load()) > 0.001f) {
            // std::cout << "⚖️  Dynamic Balance: Ratio -> " << ratio << " (Diff: " << mean_diff << "ms)" << std::endl;
            current_ratio.store(ratio);
        }
        
        // Reset
        sample_count = 0;
        avg_diff = 0.0;
    }
}

void DynamicBalancer::reset() {
    current_ratio = 0.5f;
    sample_count = 0;
}

} // namespace mm_rec
