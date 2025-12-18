/**
 * Test Hybrid Throughput
 * 
 * Verifies that concurrent CPU + GPU execution yields higher GFLOPS.
 */

#include "mm_rec/core/linear.h"
#include "mm_rec/core/vulkan_backend.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace mm_rec;

int main() {
    std::cout << "=== Hybrid Throughput Test ===" << std::endl;
    
    // Initialize Vulkan
    VulkanBackend::get().init();
    
    // Large Batch for benchmarking
    int64_t BATCH = 4096;
    int64_t IN_FEAT = 2048;
    int64_t OUT_FEAT = 2048;
    
    std::cout << "Shape: [" << BATCH << ", " << IN_FEAT << "] -> [" << BATCH << ", " << OUT_FEAT << "]" << std::endl;
    
    Linear layer(IN_FEAT, OUT_FEAT);
    Tensor input = Tensor::randn({BATCH, IN_FEAT});
    
    // Warmup
    std::cout << "Warmup..." << std::endl;
    layer.forward(input); 
    
    // Benchmark
    std::cout << "Running..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run 5 Iterations
    int ITER = 5;
    for(int i=0; i<ITER; ++i) {
        Tensor out = layer.forward(input);
        // Force sync? GPU is sync in forward(), so yes.
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    double avg_time = diff.count() / ITER;
    
    // GFLOPS = 2 * M * N * K
    // M=BATCH, N=OUT, K=IN
    double ops = 2.0 * BATCH * OUT_FEAT * IN_FEAT;
    double gflops = (ops / avg_time) / 1e9;
    
    std::cout << "Avg Time: " << std::fixed << std::setprecision(4) << avg_time << "s" << std::endl;
    std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;
    
    if (gflops > 100.0) {
        std::cout << "✅ SUCCESS: Hybrid Throughput > 100 GFLOPS (Target Met)" << std::endl;
        return 0;
    } else {
        std::cout << "⚠️ WARNING: Performance below target." << std::endl;
        return 1;
    }
}
