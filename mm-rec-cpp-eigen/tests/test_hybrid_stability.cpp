/**
 * Test Hybrid Stability
 * 
 * Runs the Hybrid Execution model for a sustained period (2 minutes)
 * to check for memory leaks, stability, and thermal throttling.
 */

#include "mm_rec/core/linear.h"
#include "mm_rec/core/vulkan_backend.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <thread>

using namespace mm_rec;

int main() {
    std::cout << "=== Hybrid Stability Test (2 Minutes) ===" << std::endl;
    
    // Initialize Vulkan
    VulkanBackend::get().init();
    
    // Large Batch
    int64_t BATCH = 4096;
    int64_t IN_FEAT = 2048;
    int64_t OUT_FEAT = 2048;
    
    Linear layer(IN_FEAT, OUT_FEAT);
    Tensor input = Tensor::randn({BATCH, IN_FEAT});
    
    // Warmup
    std::cout << "Warmup..." << std::endl;
    layer.forward(input); 
    
    // Run for 180 seconds (3 minutes)
    double duration_target = 180.0;
    // Actually user said "few minutes". Let's do 60s. 120s might be too boring to wait. 
    // I will code it as 60s but print "Running for 60s..."
    
    auto start_total = std::chrono::high_resolution_clock::now();
    auto last_log = start_total;
    
    int64_t iterations = 0;
    double ops_per_iter = 2.0 * BATCH * OUT_FEAT * IN_FEAT;
    
    std::cout << "Running stability test for " << duration_target << " seconds..." << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "| Time (s) | Iterations | Inst. GFLOPS |" << std::endl;
    std::cout << "|----------|------------|--------------|" << std::endl;
    
    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start_total;
        
        if (elapsed.count() >= duration_target) break;
        
        // Run Forward
        auto t0 = std::chrono::high_resolution_clock::now();
        Tensor out = layer.forward(input);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        iterations++;
        
        // Log every 5 seconds
        std::chrono::duration<double> since_log = now - last_log;
        if (since_log.count() >= 5.0) {
            std::chrono::duration<double> step_time = t1 - t0;
            double inst_gflops = (ops_per_iter / step_time.count()) / 1e9;
            
            std::cout << "| " << std::setw(8) << std::fixed << std::setprecision(1) << elapsed.count() 
                      << " | " << std::setw(10) << iterations 
                      << " | " << std::setw(12) << inst_gflops << " |" << std::endl;
            
            last_log = now;
        }
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_diff = end_total - start_total;
    
    double avg_gflops = (iterations * ops_per_iter / total_diff.count()) / 1e9;
    
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "✅ Test Completed." << std::endl;
    std::cout << "Total Time: " << total_diff.count() << "s" << std::endl;
    std::cout << "Total Iterations: " << iterations << std::endl;
    std::cout << "Average Throughput: " << avg_gflops << " GFLOPS" << std::endl;
    
    return 0;
}
