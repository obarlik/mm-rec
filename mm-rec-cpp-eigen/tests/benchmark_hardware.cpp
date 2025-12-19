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

void benchmark_gpu(int M, int N, int K) {
    if (!VulkanCompute::is_ready()) {
        std::cout << "   [GPU] ❌ Vulkan not ready." << std::endl;
        return;
    }

    std::vector<float> A(M*K, 1.0f);
    std::vector<float> B(K*N, 1.0f);
    std::vector<float> C(M*N, 0.0f);

    // Warmup (Driver Initialization overhead)
    // Shader is deployed to src/shaders/matmul.spv in build dir
    VulkanCompute::matmul(A.data(), B.data(), C.data(), 64, 64, 64);

    std::cout << "   [GPU] Running..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    bool result = VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (result) {
        // Note: This currently measures Upload + Compute + Download
        // Ideally we only measure Compute for peak throughput, but this is "Real World" throughput.
        std::cout << " Done." << std::endl;
        std::cout << "   [GPU] Time: " << std::fixed << std::setprecision(4) << diff.count() << "s | Throughput: " << get_gflops(M, N, K, diff.count()) << " GFLOPS" << std::endl;
    } else {
        std::cout << " ❌ Failed." << std::endl;
    }
}

int main() {
    std::cout << "=== Hardware Throughput Benchmark ===" << std::endl;
    
    // 1. Setup
    SystemOptimizer::optimize_runtime();
    VulkanBackend::get().init();
    
    int M = 2048;
    int N = 2048;
    int K = 2048;
    
    std::cout << "\nTest Size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    
    // 2. CPU
    benchmark_cpu(M, N, K);
    
    // 3. GPU
    benchmark_gpu(M, N, K);
    
    std::cout << "-----------------------------------" << std::endl;
    return 0;
}
