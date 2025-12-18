#include "mm_rec/utils/metrics.h"
#include <chrono>
#include <iostream>

using namespace mm_rec;

int main() {
    const int N = 1000000;  // 1M events
    
    // Test 1: With metrics ENABLED
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        METRIC_TRAINING_STEP(1.234f, 0.001f);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dt_enabled = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    // Test 2: Baseline (no metrics, just function call)
    auto t3 = std::chrono::high_resolution_clock::now();
    volatile float dummy = 0.0f;
    for (int i = 0; i < N; ++i) {
        dummy += 1.234f + 0.001f;  // Simulate minimal work
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    auto dt_baseline = std::chrono::duration<double, std::milli>(t4 - t3).count();
    
    std::cout << "=== Metrics Overhead Benchmark ===" << std::endl;
    std::cout << "Events: " << N << std::endl;
    std::cout << "With Metrics: " << dt_enabled << " ms" << std::endl;
    std::cout << "Baseline: " << dt_baseline << " ms" << std::endl;
    std::cout << "Overhead: " << (dt_enabled - dt_baseline) << " ms" << std::endl;
    std::cout << "Per-event cost: " << (dt_enabled - dt_baseline) * 1000.0 / N << " ns" << std::endl;
    
    // Use _Exit to avoid static destruction order issues
    _Exit(0);
}
