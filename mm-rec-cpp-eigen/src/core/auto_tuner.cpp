#include "mm_rec/core/auto_tuner.h"
#include "mm_rec/core/vulkan_compute.h"
#include "mm_rec/utils/system_optimizer.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <future>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

namespace mm_rec {

// Helper: Calculate GFLOPS
static double get_gflops(int M, int N, int K, double seconds) {
    if (seconds <= 1e-9) return 0.0;
    double ops = 2.0 * (double)M * (double)N * (double)K;
    return (ops / seconds) / 1e9;
}

TuningResult AutoTuner::tune_system(int check_size, bool precision_mode) {
    SystemOptimizer::optimize_runtime();
    VulkanBackend::get().init();

    std::cout << "\n🛡️  AutoTuner: Starting Calibration (Size=" << check_size << ")..." << std::endl;

    // 1. Find Best Shader
    std::string best_shader = find_best_shader(check_size, check_size, check_size);
    std::cout << "✅ AutoTuner: Best Shader Selected -> " << best_shader << std::endl;

    // 2. Find Optimal Hybrid Ratio
    float best_ratio = find_optimal_ratio(check_size, check_size, check_size, best_shader, precision_mode);
    
    // 3. Final Verification
    double peak_gflops = measure_throughput(check_size, check_size, check_size, best_shader, best_ratio);
    
    std::cout << "🏆 AutoTuner: Calibration Complete!" << std::endl;
    std::cout << "   Best Shader: " << best_shader << std::endl;
    std::cout << "   Best Ratio:  " << std::fixed << std::setprecision(2) << best_ratio * 100.0f << "% CPU" << std::endl;
    std::cout << "   Peak Perf:   " << peak_gflops << " GFLOPS" << std::endl;

    TuningResult res;
    res.best_shader = best_shader;
    res.best_cpu_ratio = best_ratio;
    res.peak_gflops = peak_gflops;
    res.hardware_summary = "Calibrated";
    return res;
}

std::string AutoTuner::find_best_shader(int M, int N, int K) {
    std::vector<std::string> candidates = {
        "matmul_4x4.spv", 
        "matmul.spv",       // Naive implementation
        "matmul_16x16.spv"  // Tiled implementation
    };
    
    std::string best_name = candidates[0];
    double best_score = 0.0;
    
    // Allocate GPU buffers once
    std::vector<float> A(M*K, 1.0f);
    std::vector<float> B(K*N, 1.0f);
    std::vector<float> C(M*N, 0.0f);
    
    // Warmup
    try {
        VulkanCompute::matmul(A.data(), B.data(), C.data(), 64, 64, 64, candidates[0]);
    } catch(const std::exception& e) {
        std::cerr << "⚠️ Shader Warmup Failed: " << e.what() << std::endl;
    }

    for (const auto& shader : candidates) {
        // Try running
        auto start = std::chrono::high_resolution_clock::now();
        bool success = false;
        try {
            success = VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, shader);
        } catch (...) {
            success = false;
        }
        auto end = std::chrono::high_resolution_clock::now(); // Move this outside try/catch to valid logic if needed? 
        // Actually if it throws, end time is irrelevant.
        
        if (success) {
            double gflops = get_gflops(M, N, K, std::chrono::duration<double>(end - start).count());
            // std::cout << "   Shader Bench: " << shader << " -> " << gflops << " GFLOPS" << std::endl;
            if (gflops > best_score) {
                best_score = gflops;
                best_name = shader;
            }
        }
    }
    return best_name;
}

double AutoTuner::measure_throughput(int M, int N, int K, const std::string& shader, float cpu_ratio) {
    if (cpu_ratio < 0.0f) cpu_ratio = 0.0f;
    if (cpu_ratio > 1.0f) cpu_ratio = 1.0f;

    int M_cpu = (int)(M * cpu_ratio);
    int M_gpu = M - M_cpu;

    // Allocate & Prep
    Eigen::MatrixXf A_cpu, B_cpu, C_cpu;
    if (M_cpu > 0) {
        A_cpu = Eigen::MatrixXf::Random(M_cpu, K);
        B_cpu = Eigen::MatrixXf::Random(K, N);
        C_cpu.resize(M_cpu, N);
    }
    
    std::vector<float> A_gpu, B_gpu, C_gpu;
    if (M_gpu > 0) {
        A_gpu.resize(M_gpu * K, 1.0f);
        B_gpu.resize(K * N, 1.0f);
        C_gpu.resize(M_gpu * N, 0.0f);
    }

    // Warmup GPU
    if (M_gpu > 0) VulkanCompute::matmul(A_gpu.data(), B_gpu.data(), C_gpu.data(), 64, 64, 64, shader);

    auto start = std::chrono::high_resolution_clock::now();
    
    std::future<bool> gpu_future;
    if (M_gpu > 0) {
        gpu_future = std::async(std::launch::async, [&]() {
            return VulkanCompute::matmul(A_gpu.data(), B_gpu.data(), C_gpu.data(), M_gpu, N, K, shader);
        });
    }

    if (M_cpu > 0) {
        C_cpu.noalias() = A_cpu * B_cpu;
    }

    bool gpu_ok = true;
    if (M_gpu > 0) gpu_ok = gpu_future.get();
    
    auto end = std::chrono::high_resolution_clock::now();
    
    if (!gpu_ok) return 0.0;
    
    return get_gflops(M, N, K, std::chrono::duration<double>(end - start).count());
}

float AutoTuner::find_optimal_ratio(int M, int N, int K, const std::string& shader, bool precision) {
    std::cout << "⚖️  AutoTuner: Tuning Hybrid Ratio (Precision Mode: " << (precision ? "ON" : "OFF") << ")..." << std::endl;
    
    float best_r = 0.0f;
    double max_gflops = 0.0;
    
    // Phase 1: Coarse Sweep (0, 0.2, 0.4, 0.6, 0.8, 1.0)
    float start_r = 0.0f, end_r = 1.0f;
    float step = 0.2f;
    
    // If precision is requested, we do a multi-stage search
    int stages = precision ? 2 : 1;
    
    for(int stage=0; stage<stages; ++stage) {
        float peak_r_in_stage = start_r;
        double max_gflops_in_stage = 0.0;
        
        // std::cout << "   Stage " << stage+1 << ": Scanning [" << start_r << " - " << end_r << "] step " << step << std::endl;
        
        for (float r = start_r; r <= end_r + 0.001f; r += step) {
            double gflops = measure_throughput(M, N, K, shader, r);
            if (gflops > max_gflops_in_stage) {
                max_gflops_in_stage = gflops;
                peak_r_in_stage = r;
            }
            
            // Track global best
            if (gflops > max_gflops) {
                max_gflops = gflops;
                best_r = r;
            }
        }
        
        // Narrow down search range for next stage
        // e.g. if winner is 0.4, search [0.3, 0.5] with step 0.02
        if (precision && stage == 0) {
            start_r = std::max(0.0f, peak_r_in_stage - step);
            end_r   = std::min(1.0f, peak_r_in_stage + step);
            step    = 0.02f; // 2% increments
        }
    }
    
    return best_r;
}

} // namespace mm_rec
