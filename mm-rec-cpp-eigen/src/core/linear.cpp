/**
 * Linear Layer Implementation
 */

#include "mm_rec/core/linear.h"
#include <cmath>
#include <cstdlib>
#include <future>
#include "mm_rec/core/vulkan_compute.h"

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
    int64_t workload = batch * in_features_ * out_features_;
    
    // Threshold for Hybrid Execution (e.g., 10M FLOPs)
    const int64_t GPU_THRESHOLD = 512 * 512 * 64; // Tunable
    
    // If small batch or not available, use CPU
    if (workload < GPU_THRESHOLD || !VulkanCompute::is_ready()) {
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
    // Split Ratio: 45% CPU, 55% GPU
    int64_t cpu_batch = batch * 0.45;
    int64_t gpu_batch = batch - cpu_batch; // Remainder to GPU
    
    // Slices (dim 0)
    Tensor input_cpu = input.slice(0, 0, cpu_batch);
    Tensor input_gpu = input.slice(0, cpu_batch, gpu_batch);
    
    // 1. Launch GPU Async
    auto gpu_future = std::async(std::launch::async, [this, &input_gpu, gpu_batch]() {
        Tensor result_gpu = Tensor::zeros({gpu_batch, out_features_});
        // We need weight transpose for MatMul: (B x I) * (I x O) -> (B x O) ?
        // Wait, input.matmul(weight.T) : (B x I) * (O x I).T = (B x I) * (I x O) = (B x O).
        // Vulkan matmul expects A, B, C pointers.
        // A = input_gpu (RowMajor) [gpu_batch, in_features]
        // B = weight.T (RowMajor) [in_features, out_features]
        // C = result (RowMajor)
        
        // Note: transpose() creates a copy. That's fine for now (can cache if weight fixed).
        Tensor W_T = weight_.transpose(); 
        
        bool ok = VulkanCompute::matmul(
            input_gpu.data(), 
            W_T.data(), 
            result_gpu.data(), 
            gpu_batch, 
            out_features_, // N
            in_features_,  // K (common dim)
            (std::getenv("MM_REC_USE_FP16") ? "src/shaders/matmul_fp16.spv" : "src/shaders/matmul.spv")
        );
        
        if (!ok) throw std::runtime_error("GPU Dispatch Failed");
        
        // Add bias on CPU after (or implement bias shader later).
        // Let's add bias here quickly (CPU is fast enough for vector add).
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
