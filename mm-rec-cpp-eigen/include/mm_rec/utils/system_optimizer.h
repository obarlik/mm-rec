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
    static void optimize_runtime(bool include_ecores = false) {
        std::cout << "🚀 SystemOptimizer: Analyzing CPU Topology..." << std::endl;
        
        std::vector<int> target_cores = detect_cores(include_ecores);
        
        if (target_cores.empty()) {
            std::cout << "⚠️  SystemOptimizer: Could not detect cores. Using default scheduler." << std::endl;
            return;
        }

        // Print CPU Model Name
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line, model_name = "Unknown CPU";
        int total_logical = 0;
        
        while(std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos && model_name == "Unknown CPU") {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    model_name = line.substr(pos + 2); 
                }
            }
            if (line.find("processor") == 0) total_logical++;
        }

        std::cout << "   CPU: " << model_name << " (Total Cores: " << total_logical << ")" << std::endl;
        
        if (include_ecores) {
             std::cout << "✅ SystemOptimizer: Utilizing ALL " << target_cores.size() << " Cores (P+E) as requested." << std::endl;
        } else {
             std::cout << "✅ SystemOptimizer: Selected " << target_cores.size() << " Performance Cores for Compute." << std::endl;
        }

        std::cout << "   Targeting Cores: ";
        for(int c : target_cores) std::cout << c << " ";
        std::cout << std::endl;

        // 1. Set OpenMP Num Threads matches Core count
        omp_set_num_threads(target_cores.size());
        
        // 2. Pin Threads (Processor Affinity)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for(int c : target_cores) CPU_SET(c, &cpuset);

        int result = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        if (result == 0) {
            std::cout << "🔒 SystemOptimizer: Process PINNED to Selected Cores." << std::endl;
        } else {
            std::cout << "⚠️  SystemOptimizer: Failed to set affinity: " << std::strerror(errno) << std::endl;
        }
    }

private:
    struct CpuInfo {
        int id;
        int max_freq;
    };

    static std::vector<int> detect_cores(bool include_ecores) {
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
        
        if (include_ecores) {
            // Return ALL detected cores
            std::vector<int> all_cores;
            for(auto& c : cpus) all_cores.push_back(c.id);
            return all_cores;
        }

        // P-Core Logic (High Freq Only)
        int global_max = 0;
        for(auto& c : cpus) global_max = std::max(global_max, c.max_freq);
        
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
