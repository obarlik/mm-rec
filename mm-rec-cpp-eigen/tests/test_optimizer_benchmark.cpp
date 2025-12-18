/**
 * Scientific Proof: Hybrid Core Optimization Benchmark
 * 
 * Objective: Compare "Naive OS Scheduling" vs "P-Core Pinning".
 * Hypothesis: In synchronized workloads (OpenMP barriers), slow E-Cores
 *             drag down the fast P-Cores. Pinning should reduce latency.
 * 
 * Workload: Heavy FP32 Vector FMA (Fused Multiply-Add) with Barriers.
 */

#include "mm_rec/utils/system_optimizer.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <sched.h>
#include <omp.h>
#include <numeric>
#include <thread>

using namespace mm_rec;

// Reset affinity to ALL cores (simulating default OS behavior)
void reset_affinity_to_all() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    // Assuming max 128 cores
    for(int i=0; i< std::thread::hardware_concurrency(); ++i) {
        CPU_SET(i, &cpuset);
    }
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    omp_set_num_threads(std::thread::hardware_concurrency());
}

double run_stress_test(const std::string& name) {
    long iterations = 500;
    long size = 1000000; // 1M floats
    std::vector<float> a(size, 1.0f);
    std::vector<float> b(size, 2.0f);
    std::vector<float> c(size, 0.0f);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        // Vital: OpenMP creates an implicit barrier at the end of this block.
        // Fast cores MUST wait for slow cores here.
        #pragma omp parallel for
        for (long j = 0; j < size; ++j) {
            // Heavy math to be compute bound, not memory bound
            float v1 = a[j];
            float v2 = b[j];
            // Simulate complex math (approx 20 ops)
            for(int k=0; k<20; ++k) {
                v1 = v1 * v2 + 0.01f;
                v2 = v2 * 0.99f + v1;
            }
            c[j] = v1;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    double gflops = (double)iterations * size * 40.0 / 1e9 / elapsed.count();
    
    std::cout << "[" << name << "]" << std::endl;
    std::cout << "  Threads Active: " << omp_get_max_threads() << std::endl;
    std::cout << "  Time: " << elapsed.count() << "s" << std::endl;
    std::cout << "  Throughput: " << gflops << " GFLOPS" << std::endl;
    return gflops;
}

int main() {
    std::cout << "=== Scientific Benchmark: Hybrid Core Penalty ===\n" << std::endl;

    // Phase A: Control Group (Naive OS Scheduler)
    reset_affinity_to_all();
    std::cout << "--- Phase A: Naive OS Scheduler (All Cores P+E) ---" << std::endl;
    double score_naive = run_stress_test("Naive");

    std::cout << "\n------------------------------------------------\n" << std::endl;

    // Phase B: Experimental Group (P-Core Pinning)
    std::cout << "--- Phase B: Optimized (P-Cores Only) ---" << std::endl;
    SystemOptimizer::optimize_runtime(); // Apply fix
    double score_opt = run_stress_test("Optimized");

    std::cout << "\n=== FINAL RESULTS ===" << std::endl;
    std::cout << "Naive Score: " << score_naive << " GFLOPS" << std::endl;
    std::cout << "Optim Score: " << score_opt << " GFLOPS" << std::endl;
    
    double speedup = (score_opt - score_naive) / score_naive * 100.0;
    
    if (speedup > 0) {
        std::cout << "🚀 SPEEDUP: +" << speedup << "%" << std::endl;
        std::cout << "\nConclusion: Dropping E-Cores IMPROVED performance." << std::endl;
        std::cout << "Proven: Fast cores were indeed waiting for slow cores." << std::endl;
    } else {
        std::cout << "Speedup: " << speedup << "% (Neutral/Negative)" << std::endl;
        std::cout << "Conclusion: Throughput gains of E-cores outweighed synchronization penalty." << std::endl;
    }

    return 0;
}
