#include "mm_rec/core/vulkan_compute.h"
#include "mm_rec/infrastructure/system_optimizer.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <thread>
#include <future>
#include <Eigen/Dense>

#include "mm_rec/core/vulkan_backend.h"
#include "mm_rec/core/vulkan_compute.h"
#include "mm_rec/infrastructure/system_optimizer.h"
#include "mm_rec/core/auto_tuner.h" // New AutoTuner API

using namespace mm_rec;

// Simple CPU Benchmark for Comparison
void benchmark_cpu(int M, int N, int K) {
    std::cout << "   [CPU] Allocating..." << std::flush;
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, K);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(K, N);
    Eigen::MatrixXf C(M, N);
    std::cout << " Done." << std::endl;
    
    // Warmup
    C.noalias() = A * B;
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<3; ++i) { // Run 3 times
       C.noalias() = A * B;
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    double avg_time = diff.count() / 3.0;
    
    double gflops = (2.0 * (double)M * N * K) / avg_time / 1e9;
    std::cout << "   [CPU] Time: " << avg_time << "s | Throughput: " << gflops << " GFLOPS" << std::endl;
}

int main() {
    std::cout << "=== Universal Hardware Benchmark & Auto-Tuner ===" << std::endl;
    
    // 1. Setup
    SystemOptimizer::optimize_runtime();
    VulkanBackend::get().init();
    
    int check_size = 4096; // Use large matrix for sustaining max load
    
    std::cout << "\nTest Size: " << check_size << "^3" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    
    // 2. Baseline CPU
    benchmark_cpu(check_size, check_size, check_size);
    std::cout << "-----------------------------------" << std::endl;

    // 3. Run Auto-Tuner
    // This will perform:
    //  - Shader Race (4x4 vs 8x8 vs Naive)
    //  - Coarse Scan (0, 20, 40... 100%)
    //  - Fine Scan (e.g. 43%, 44%, 45%...)
    TuningResult result = AutoTuner::tune_system(check_size, true); // true = Precision Mode
    
    std::cout << "\n✅ FINAL RESULT FOR THIS MACHINE:" << std::endl;
    std::cout << "   Strategy: " << result.best_shader << " + Hybrid (" 
              << (int)(result.best_cpu_ratio * 100) << "% CPU)" << std::endl;
    std::cout << "   Peak GFLOPS: " << result.peak_gflops << std::endl;
    
    return 0;
}
