#include "mm_rec/core/vulkan_compute.h"
#include "mm_rec/utils/system_optimizer.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>

using namespace mm_rec;

double get_gflops(long M, long N, long K, double seconds) {
    // 2 * M * N * K operations (multiply + add)
    double ops = 2.0 * (double)M * (double)N * (double)K;
    return (ops / seconds) / 1e9;
}

void benchmark_cpu(int M, int N, int K) {
    std::cout << "   [CPU] Allocating..." << std::flush;
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, K);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(K, N);
    Eigen::MatrixXf C(M, N);
    std::cout << " Done." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    
    // Eigen is highly optimized AVX2/AVX512 internally
    C.noalias() = A * B;
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    std::cout << "   [CPU] Time: " << std::fixed << std::setprecision(4) << diff.count() << "s | Throughput: " << get_gflops(M, N, K, diff.count()) << " GFLOPS" << std::endl;
}

#include <future>

// ...

void benchmark_gpu(int M, int N, int K) {
    if (!VulkanCompute::is_ready()) {
        std::cout << "   [GPU] ❌ Vulkan not ready." << std::endl;
        return;
    }

    std::vector<float> A(M*K, 1.0f);
    std::vector<float> B(K*N, 1.0f);
    std::vector<float> C(M*N, 0.0f);

    // Warmup & Use best shader
    std::string best_shader = "matmul_4x4.spv";
    VulkanCompute::matmul(A.data(), B.data(), C.data(), 64, 64, 64, best_shader);

    std::cout << "   [GPU] Running (Best Shader)..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    bool result = VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, best_shader);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (result) {
        std::cout << " Done." << std::endl;
        std::cout << "   [GPU] Time: " << std::fixed << std::setprecision(4) << diff.count() << "s | Throughput: " << get_gflops(M, N, K, diff.count()) << " GFLOPS" << std::endl;
    } else {
        std::cout << " ❌ Failed." << std::endl;
    }
}

void benchmark_hybrid(int M, int N, int K) {
    std::cout << "   [HYBRID] Initializing (CPU + GPU Parallel)..." << std::endl;
    // Split: CPU (25%), GPU (75%) based on ~110 vs ~330 GFLOPS capacity
    int M_cpu = M / 4;
    int M_gpu = M - M_cpu;
    
    // CPU Data
    Eigen::MatrixXf A_cpu = Eigen::MatrixXf::Random(M_cpu, K);
    Eigen::MatrixXf B_cpu = Eigen::MatrixXf::Random(K, N); 
    Eigen::MatrixXf C_cpu(M_cpu, N);
    
    // GPU Data
    std::vector<float> A_gpu(M_gpu * K, 1.0f);
    std::vector<float> B_gpu(K * N, 1.0f);
    std::vector<float> C_gpu(M_gpu * N, 0.0f);
    
    // Warmup GPU
    VulkanCompute::matmul(A_gpu.data(), B_gpu.data(), C_gpu.data(), 64, 64, 64, "matmul_4x4.spv");
    
    std::cout << "   [HYBRID] Dispatched..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    // 1. Launch GPU (Async Thread)
    auto gpu_future = std::async(std::launch::async, [&]() {
        return VulkanCompute::matmul(A_gpu.data(), B_gpu.data(), C_gpu.data(), M_gpu, N, K, "matmul_4x4.spv");
    });
    
    // 2. Run CPU (Main Thread)
    C_cpu.noalias() = A_cpu * B_cpu;
    
    // 3. Sync
    bool gpu_success = gpu_future.get();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    if (gpu_success) {
        double ops_cpu = 2.0 * (double)M_cpu * N * K;
        double ops_gpu = 2.0 * (double)M_gpu * N * K;
        double total_ops = ops_cpu + ops_gpu;
        double gflops = (total_ops / diff.count()) / 1e9;
        
        std::cout << " Done." << std::endl;
        // std::cout << "   [DEBUG] Time: " << diff.count() << "s | Total Ops: " << total_ops << std::endl;
        std::cout << "   [HYBRID] Throughput: " << gflops << " GFLOPS" << std::endl;
    } else {
        std::cout << " ❌ Hybrid Failed." << std::endl;
    }
}

int main() {
    std::cout << "=== Hardware Throughput Benchmark ===" << std::endl;
    
    // 1. Setup
    SystemOptimizer::optimize_runtime();
    VulkanBackend::get().init();
    
    int M = 4096; // Increase size for better hybrid parallelism
    int N = 4096;
    int K = 4096;
    
    std::cout << "\nTest Size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    
    // 2. CPU
    benchmark_cpu(M, N, K);
    
    // 3. GPU
    benchmark_gpu(M, N, K);
    
    // 4. Hybrid
    benchmark_hybrid(M, N, K);
    
    std::cout << "-----------------------------------" << std::endl;
    return 0;
}
