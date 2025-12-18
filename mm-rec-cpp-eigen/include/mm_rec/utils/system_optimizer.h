/**
 * System Optimizer
 * 
 * Automatically detects Hybrid CPU Topology (P-Core vs E-Core)
 * and pins OpenMP threads to Performance Cores for maximum AVX2 throughput.
 * 
 * Target: Intel hybrid CPUs (Alder/Raptor/Meteor Lake)
 */

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <sched.h>
#include <cstring>

namespace mm_rec {

class SystemOptimizer {
public:
    static void optimize_runtime() {
        std::cout << "🚀 SystemOptimizer: Analyzing CPU Topology..." << std::endl;
        
        std::vector<int> p_cores = detect_p_cores();
        
        if (p_cores.empty()) {
            std::cout << "⚠️  SystemOptimizer: Could not detect P-Cores. Using default scheduler." << std::endl;
            return;
        }

        std::cout << "✅ SystemOptimizer: Detected " << p_cores.size() << " Performance Cores (High Freq)." << std::endl;
        std::cout << "   Targeting Cores: ";
        for(int c : p_cores) std::cout << c << " ";
        std::cout << std::endl;

        // 1. Set OpenMP Num Threads matches P-Core count
        omp_set_num_threads(p_cores.size());
        
        // 2. Pin Threads (Processor Affinity)
        // This is tricky with OpenMP internal pool, but we can try setting the process mask
        // or using KMP_AFFINITY env logic programmatically.
        // A simpler way for a single process is sched_setaffinity.
        
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for(int c : p_cores) CPU_SET(c, &cpuset);

        int result = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        if (result == 0) {
            std::cout << "🔒 SystemOptimizer: Process PINNED to P-Cores only!" << std::endl;
            std::cout << "   E-Cores (Background noise) are ignored." << std::endl;
        } else {
            std::cout << "⚠️  SystemOptimizer: Failed to set affinity: " << std::strerror(errno) << std::endl;
        }
    }

private:
    struct CpuInfo {
        int id;
        int max_freq;
    };

    static std::vector<int> detect_p_cores() {
        std::vector<CpuInfo> cpus;
        
        // Scan up to 128 cores
        for (int i = 0; i < 128; ++i) {
            std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/scaling_max_freq";
            std::ifstream file(path);
            if (!file.is_open()) break; // End of CPUs
            
            int freq;
            file >> freq;
            cpus.push_back({i, freq});
        }
        
        if (cpus.empty()) return {};
        
        // Find max frequency
        int global_max = 0;
        for(auto& c : cpus) global_max = std::max(global_max, c.max_freq);
        
        // Filter cores that are within 10% of max freq (P-Cores)
        std::vector<int> p_cores;
        int threshold = global_max * 0.9;
        
        for(auto& c : cpus) {
            if(c.max_freq >= threshold) {
                p_cores.push_back(c.id);
            }
        }
        
        return p_cores;
    }
};

} // namespace mm_rec
