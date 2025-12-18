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

    // 5. Unrolled FP16 Shader
    std::cout << "\n--- [Unrolled FP16 Shader] ---" << std::endl;
    VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_subgroup.spv");
    
    t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITER; ++i)
        VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_subgroup.spv");
    t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dt_unroll = (t1 - t0) / ITER;
    double gflops_unroll = (ops / dt_unroll.count()) / 1e9;
    std::cout << "Time: " << dt_unroll.count() << "s | GFLOPS: " << gflops_unroll << std::endl;
    std::cout << ">>> Unrolled GFLOPS: " << gflops_unroll << " (Speedup: " << std::fixed << std::setprecision(2) << gflops_unroll/gflops_scalar << "x)" << std::endl;

    // 6. Packed Memory Layout Benchmark
    std::cout << "\n--- [Packed (Unit Stride) Shader] ---" << std::endl;
    // Creates B_packed = B^T.
    std::vector<float> B_packed(K * N);
    for(int n=0; n<N; ++n) {
        for(int k=0; k<K; ++k) {
            B_packed[n*K + k] = B[k*N + n];
        }
    }
    
    VulkanCompute::matmul(A.data(), B_packed.data(), C.data(), M, N, K, "src/shaders/matmul_packed.spv");
    
    t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITER; ++i)
        VulkanCompute::matmul(A.data(), B_packed.data(), C.data(), M, N, K, "src/shaders/matmul_packed.spv");
    t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dt_pack = (t1 - t0) / ITER;
    double gflops_pack = (ops / dt_pack.count()) / 1e9;
    std::cout << "Time: " << dt_pack.count() << "s | GFLOPS: " << gflops_pack << std::endl;
    std::cout << ">>> Packed GFLOPS: " << gflops_pack << " (Speedup: " << std::fixed << std::setprecision(2) << gflops_pack/gflops_scalar << "x)" << std::endl;

    // 7. Prefetch (Double Buffering) Benchmark
    std::cout << "\n--- [Prefetch (Double Buffer) Shader] ---" << std::endl;
    // Uses standard matmul call
    VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_prefetch.spv");
    
    t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITER; ++i)
        VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_prefetch.spv");
    t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dt_pre = (t1 - t0) / ITER;
    double gflops_pre = (ops / dt_pre.count()) / 1e9;
    std::cout << "Time: " << dt_pre.count() << "s | GFLOPS: " << gflops_pre << std::endl;
    std::cout << ">>> Prefetch GFLOPS: " << gflops_pre << " (Speedup: " << std::fixed << std::setprecision(2) << gflops_pre/gflops_scalar << "x)" << std::endl;

    // 8. Super-Tiled (32x32) Benchmark
    std::cout << "\n--- [Super-Tiled (32x32) 2x2 Block] ---" << std::endl;
    // Uses standard matmul call
    VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_32x32.spv");
    
    t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITER; ++i)
        VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_32x32.spv");
    t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dt_super = (t1 - t0) / ITER;
    double gflops_super = (ops / dt_super.count()) / 1e9;
    std::cout << "Time: " << dt_super.count() << "s | GFLOPS: " << gflops_super << std::endl;
    std::cout << ">>> Super-Tiled GFLOPS: " << gflops_super << " (Speedup: " << std::fixed << std::setprecision(2) << gflops_super/gflops_scalar << "x)" << std::endl;

    // 9. Small-Tile (8x8) Benchmark
    std::cout << "\n--- [Small-Tiled (8x8)] ---" << std::endl;
    // Uses standard matmul call
    VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_8x8.spv");
    
    t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITER; ++i)
        VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_8x8.spv");
    t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dt_small = (t1 - t0) / ITER;
    double gflops_small = (ops / dt_small.count()) / 1e9;
    std::cout << "Time: " << dt_small.count() << "s | GFLOPS: " << gflops_small << std::endl;
    std::cout << ">>> Small-Tiled GFLOPS: " << gflops_small << " (Speedup: " << std::fixed << std::setprecision(2) << gflops_small/gflops_scalar << "x)" << std::endl;

    // 10. Micro-Tile (4x4) Benchmark
    std::cout << "\n--- [Micro-Tiled (4x4)] ---" << std::endl;
    VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_4x4.spv");
    
    t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITER; ++i)
        VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_4x4.spv");
    t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dt_micro = (t1 - t0) / ITER;
    double gflops_micro = (ops / dt_micro.count()) / 1e9;
    std::cout << "Time: " << dt_micro.count() << "s | GFLOPS: " << gflops_micro << std::endl;
    std::cout << ">>> Micro-Tiled GFLOPS: " << gflops_micro << " (Speedup: " << std::fixed << std::setprecision(2) << gflops_micro/gflops_scalar << "x)" << std::endl;

    // 11. Nano-Tile (2x2) Benchmark
    std::cout << "\n--- [Nano-Tiled (2x2)] ---" << std::endl;
    VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_2x2.spv");
    
    t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITER; ++i)
        VulkanCompute::matmul(A.data(), B.data(), C.data(), M, N, K, "src/shaders/matmul_2x2.spv");
    t1 = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> dt_nano = (t1 - t0) / ITER;
    double gflops_nano = (ops / dt_nano.count()) / 1e9;
    std::cout << "Time: " << dt_nano.count() << "s | GFLOPS: " << gflops_nano << std::endl;
    std::cout << ">>> Nano-Tiled GFLOPS: " << gflops_nano << " (Speedup: " << std::fixed << std::setprecision(2) << gflops_nano/gflops_scalar << "x)" << std::endl;
    
    return 0;
}
