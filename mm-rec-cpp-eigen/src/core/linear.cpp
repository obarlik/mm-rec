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
    
    // --- Hybrid Execution ---
    static bool force_cpu = (std::getenv("MM_REC_FORCE_CPU") != nullptr);

    // Calculate Partition
    int64_t gpu_batch = (int64_t)(batch * 0.8);
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
    
    // 1. Launch GPU Async
    auto gpu_future = std::async(std::launch::async, [this, &input_gpu, gpu_batch]() {
        static bool once_gpu = false;
        if (!once_gpu) { std::cout << "🚀 Executing on GPU! Workload>" << (gpu_batch * in_features_ * out_features_) << std::endl; once_gpu = true; }

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
        bool ok = VulkanCompute::matmul(
            input_gpu.data(), 
            W_T.data(), 
            result_gpu.data(), 
            gpu_batch, 
            out_features_, 
            in_features_, 
            shaderPath
        );
        
        if (!ok) throw std::runtime_error("GPU Dispatch Failed");
        
        // Add bias (CPU)
        float* out_data = result_gpu.data();
        const float* bias_data = bias_.data();
        for (int64_t b = 0; b < gpu_batch; ++b) {
             for (int64_t f = 0; f < out_features_; ++f) out_data[b * out_features_ + f] += bias_data[f];
        }
        
        return result_gpu;
    });
    
    // 2. Run CPU Sync
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
    
    // 3. Join
    Tensor output_gpu = gpu_future.get();
    
    // 4. Concat
    // Need a dummy instance to call member cat? or make cat static?
    // Current design is member.
    return input.cat({output_cpu, output_gpu}, 0);
}

} // namespace mm_rec
