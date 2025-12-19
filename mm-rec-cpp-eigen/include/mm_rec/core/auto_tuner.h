#pragma once

#include <string>
#include <vector>
#include <map>

namespace mm_rec {

struct TuningResult {
    std::string best_shader;      // e.g., "matmul_4x4.spv"
    float best_cpu_ratio;         // e.g., 0.45
    double peak_gflops;           // e.g., 340.5
    std::string hardware_summary; // e.g., "CPU: 200, GPU: 260"
};

class AutoTuner {
public:
    /**
     * Run full system calibration.
     * @param check_size Matrix dimension for testing (default 2048 or 4096)
     * @param precision_mode If true, runs Phase 2 fine-grained sweep.
     */
    static TuningResult tune_system(int check_size = 4096, bool precision_mode = true);

private:
    static std::string find_best_shader(int M, int N, int K);
    static float find_optimal_ratio(int M, int N, int K, const std::string& shader, bool precision);
    static double measure_throughput(int M, int N, int K, const std::string& shader, float cpu_ratio);
};

} // namespace mm_rec
