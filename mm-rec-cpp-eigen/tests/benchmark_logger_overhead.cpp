#include "mm_rec/utils/logger_v2.h"
#include <iostream>
#include <chrono>

using namespace mm_rec;

int main() {
    const int N = 1000000;
    
    std::cout << "=== Logger Overhead Benchmark ===" << std::endl;
    std::cout << "Events: " << N << std::endl;
    
    // Test 1: With Logger (enabled)
    Logger::instance().start_writer("bench_logger.log", LogLevel::DEBUG);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        LOG_DEBUG("Benchmark test message");
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto dt_enabled = std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    Logger::instance().stop_writer();
    
    // Test 2: Baseline (disabled)
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        // No logging
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    auto dt_baseline = std::chrono::duration<double, std::milli>(t4 - t3).count();
    
    std::cout << "With Logger: " << dt_enabled << " ms" << std::endl;
    std::cout << "Baseline: " << dt_baseline << " ms" << std::endl;
    std::cout << "Overhead: " << (dt_enabled - dt_baseline) << " ms" << std::endl;
    std::cout << "Per-event cost: " << (dt_enabled - dt_baseline) * 1000.0 / N << " ns" << std::endl;
    
    return 0;
}
