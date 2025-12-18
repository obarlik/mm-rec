/**
 * Benchmark: Shader Optimization
 */

#include "mm_rec/core/vulkan_compute.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace mm_rec;

int main() {
    std::cout << "=== Shader Optimization Benchmark ===" << std::endl;
    
    VulkanBackend::get().init();
    
    // Size: 2048 (Multiple of 64, good for vec4)
    int M = 2048;
    int N = 2048;
    int K = 2048;
    
    std::cout << "Matrix Size: " << M << "x" << N << "x" << K << std::endl;
    
    // Data
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 0.5f);
    std::vector<float> C(M * N, 0.0f);
    
    double ops = 2.0 * M * N * K;
    
    // 1. Scalar Benchmark
    std::cout << "\n--- [Scalar Shader] ---" << std::endl;
    // Warmup
    VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul.spv");
    
    auto t0 = std::chrono::high_resolution_clock::now();
    VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul.spv");
    auto t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dt_scalar = t1 - t0;
    double gflops_scalar = (ops / dt_scalar.count()) / 1e9;
    std::cout << "Time: " << dt_scalar.count() << "s | GFLOPS: " << gflops_scalar << std::endl;
    
    // 2. Vectorized Benchmark
    std::cout << "\n--- [Vectorized vec4 Shader] ---" << std::endl;
    // Warmup
    VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_vec4.spv");
    
    t0 = std::chrono::high_resolution_clock::now();
    int ITER = 3;
    for(int i=0; i<ITER; ++i)
        VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_vec4.spv");
    t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dt_vec = (t1 - t0) / ITER;
    double gflops_vec = (ops / dt_vec.count()) / 1e9;
    std::cout << "Time: " << dt_vec.count() << "s | GFLOPS: " << gflops_vec << std::endl;
    
    // 3. FP16 Benchmark
    std::cout << "\n--- [FP16 Mixed Precision Shader] ---" << std::endl;
    // Warmup
    VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_fp16.spv");
    
    t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITER; ++i)
        VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_fp16.spv");
    t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dt_fp16 = (t1 - t0) / ITER;
    double gflops_fp16 = (ops / dt_fp16.count()) / 1e9;
    std::cout << "Time: " << dt_fp16.count() << "s | GFLOPS: " << gflops_fp16 << std::endl;

    // Conclusion
    std::cout << "\n>>> Scalar GFLOPS: " << gflops_scalar << std::endl;
    std::cout << ">>> FP16 GFLOPS:   " << gflops_fp16 << " (Speedup: " << std::fixed << std::setprecision(2) << gflops_fp16/gflops_scalar << "x)" << std::endl;

    // 4. Register Blocking Benchmark
    std::cout << "\n--- [Register Blocked (32x32) Shader] ---" << std::endl;
    VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_regblock.spv");
    
    t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITER; ++i)
        VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_regblock.spv");
    t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dt_reg = (t1 - t0) / ITER;
    double gflops_reg = (ops / dt_reg.count()) / 1e9;
    std::cout << "Time: " << dt_reg.count() << "s | GFLOPS: " << gflops_reg << std::endl;
    std::cout << ">>> RegBlock GFLOPS: " << gflops_reg << " (Speedup: " << std::fixed << std::setprecision(2) << gflops_reg/gflops_scalar << "x)" << std::endl;
    
    return 0;
}
