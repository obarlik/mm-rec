/**
 * Linear Layer Implementation
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "mm_rec/core/linear.h"
#include <cmath>
#include <cstdlib>
#include <future>
#include "mm_rec/core/vulkan_compute.h"
#include "mm_rec/core/vulkan_op.h"
#include "mm_rec/core/dynamic_balancer.h"
#include <chrono>
#include <thread>

namespace mm_rec {

Linear::Linear(int64_t in_features, int64_t out_features)
    : in_features_(in_features), out_features_(out_features) {
    
    // Xavier/Glorot initialization
    float std = std::sqrt(2.0f / (in_features + out_features));
    
    weight_ = Tensor::randn({out_features, in_features}, 0.0f, std);
    bias_ = Tensor::zeros({out_features});
}



Tensor Linear::forward(const Tensor& input) {
    // input: [batch, in_features]
    int64_t batch = input.size(0);
    // int64_t workload = batch * in_features_ * out_features_;
    
    // Threshold for Hybrid Execution (e.g., 10M FLOPs)
    // const int64_t GPU_THRESHOLD = 0; // User request: No threshold

    // If Vulkan not ready, use CPU
    if (!VulkanCompute::is_ready()) {
         // ... (existing CPU fallback logic)
         Tensor output = input.matmul(weight_.transpose());
         // Bias add (CPU)
         int64_t out_f = out_features_;
         float* out_data = output.data();
         const float* bias_data = bias_.data();
         #pragma omp parallel for if(batch > 1)
         for (int64_t b = 0; b < batch; ++b) {
             for (int64_t f = 0; f < out_f; ++f) out_data[b * out_f + f] += bias_data[f];
         }
         return output;
    }
    
    // --- Hybrid Execution (Dynamic Load Balancing) ---
    static bool force_cpu = (std::getenv("MM_REC_FORCE_CPU") != nullptr);

    // 1. Get Dynamic Split Ratio (Runtime Tuned)
    float gpu_ratio = DynamicBalancer::get_gpu_ratio();
    
    // Calculate Partition
    int64_t gpu_batch = (int64_t)(batch * gpu_ratio);
    int64_t cpu_batch = batch - gpu_batch;

    // Safety Guard: If batch is too small to split (gpu_batch=0), run all on CPU.
    if (force_cpu || gpu_batch == 0) {
         // Pure CPU Path
         Tensor result = Tensor::zeros({batch, out_features_});
         #pragma omp parallel for collapse(2)
         for(int64_t b=0; b<batch; ++b) {
             for(int64_t f=0; f<out_features_; ++f) {
                 float sum = bias_.data()[f];
                 for(int64_t k=0; k<in_features_; ++k) {
                     sum += input.data()[b * in_features_ + k] * weight_.data()[f * in_features_ + k];
                 }
                 result.data()[b * out_features_ + f] = sum;
             }
         }
         return result;
    }
    
    // Normal Hybrid Path (gpu_batch > 0)
    Tensor input_cpu = input.slice(0, 0, cpu_batch);
    Tensor input_gpu = input.slice(0, cpu_batch, gpu_batch);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // 1. Launch GPU Async
    auto gpu_future = std::async(std::launch::async, [this, &input_gpu, gpu_batch]() {
        static bool once_gpu = false;
        if (!once_gpu) { 
            std::cout << "🚀 Hybrid Execution: GPU Thread Activated! (GPU Batch: " << gpu_batch << ")" << std::endl; 
            once_gpu = true; 
        }

        Tensor result_gpu = Tensor::zeros({gpu_batch, out_features_});
        Tensor W_T = weight_.transpose();
        
        // --- Use Shader Selected by Global AutoTuner ---
        static std::string shaderPath = []() -> std::string {
            // 1. Check User Override
            if(const char* env_kernel = std::getenv("MM_REC_GPU_KERNEL")) {
                std::string mode(env_kernel);
                if (mode == "16x16") return "src/shaders/matmul_16x16.spv";
                if (mode == "8x8")   return "src/shaders/matmul_8x8.spv";
                if (mode == "4x4")   return "src/shaders/matmul_4x4.spv";
                if (mode == "2x2")   return "src/shaders/matmul_2x2.spv";
                if (mode == "fp16")  return "src/shaders/matmul_fp16.spv";
                return mode; // Return as-is if it's a full path
            }
            
            // 2. Use AutoTuner's Selection (set during calibration)
            if(const char* tuned = std::getenv("MM_REC_TUNED_SHADER")) {
                return std::string(tuned);
            }
            
            // 3. Fallback (if AutoTuner didn't run for some reason)
            return "src/shaders/matmul_4x4.spv";
        }();

        // Use stateless, robust dispatch (Allocates per frame, but stable 148 GFLOPS)
        // Retry Loop for transient GPU OOM (Error -2)
        bool ok = false;
        // Increase patience: 10 retries * 50ms = 500ms max wait
        for(int retries=0; retries<10; ++retries) {
             ok = VulkanCompute::matmul(
                input_gpu.data(), 
                W_T.data(), 
                result_gpu.data(), 
                gpu_batch, 
                out_features_, 
                in_features_, 
                shaderPath
            );
            
            if(ok) break;
            
            // Wait 50ms and retry (Drain queue)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if(retries > 0) std::cout << "⚠️ GPU Busy (Retry " << retries << "/10)..." << std::endl;
        }
        
        if (!ok) throw std::runtime_error("GPU Dispatch Failed (After 10 Retries - System Overloaded)");
        
        // Add bias (CPU) - doing it here allows parallel bias add
        float* out_data = result_gpu.data();
        const float* bias_data = bias_.data();
        for (int64_t b = 0; b < gpu_batch; ++b) {
             for (int64_t f = 0; f < out_features_; ++f) out_data[b * out_features_ + f] += bias_data[f];
        }
        
        return result_gpu;
    });
    
    // 2. Run CPU Sync
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    Tensor output_cpu = input_cpu.matmul(weight_.transpose());
    {
         int64_t out_f = out_features_;
         float* out_data = output_cpu.data();
         const float* bias_data = bias_.data();
         #pragma omp parallel for if(cpu_batch > 1)
         for (int64_t b = 0; b < cpu_batch; ++b) {
             for (int64_t f = 0; f < out_f; ++f) out_data[b * out_f + f] += bias_data[f];
         }
    }
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    
    // 3. Join (Wait for GPU)
    Tensor output_gpu = gpu_future.get();
    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    
    // 4. Report Metrics (CPU vs GPU Time)
    // CPU time is pure computation
    // GPU time is total wait time (since we started async immediately) minus CPU time?
    // No, GPU duration is t_gpu_end - t0.
    // CPU duration is t_cpu_end - t_cpu_start (approx t_cpu_end - t0).
    // Note: CPU starts slightly after GPU launch overhead.
    // We want to balance Total Latency.
    // If CPU finishes at T+10 and GPU at T+15, GPU is bottleneck.
    // If CPU finishes at T+15 and GPU at T+10, CPU is bottleneck.
    // So we compare finish times relative to start?
    // Or durations?
    // We want `t_cpu_end` ~= `t_gpu_end`.
    
    // Let's use durations relative to t0.
    double cpu_ms = std::chrono::duration<double, std::milli>(t_cpu_end - t0).count();
    double gpu_ms = std::chrono::duration<double, std::milli>(t_gpu_end - t0).count();
    
    DynamicBalancer::report_metrics(cpu_ms, gpu_ms);
    
    // 5. Concat (Existing code handles return)
    // Note: The original code continued here.

    // Need a dummy instance to call member cat? or make cat static?
    // Current design is member.
    return input.cat({output_cpu, output_gpu}, 0);
}

} // namespace mm_rec
