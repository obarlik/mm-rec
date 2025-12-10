/**
 * Associative Scan (Exponential Product) - CPU Optimized Implementation
 * 
 * Modern CPU optimizations:
 * - SIMD (SSE/AVX) for vectorized operations
 * - OpenMP for parallel processing
 * - Work-efficient parallel scan (Blelloch algorithm)
 */

#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <immintrin.h>  // AVX/SSE intrinsics
#ifdef _OPENMP
#include <omp.h>
#endif

// Include optimized SIMD functions and Blelloch scan
// Forward declarations - implementations in separate files
extern void blelloch_scan_parallel_log_sum_exp(
    float* input,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int dim,
    int num_threads
);

// Forward declarations for SIMD conversion functions
#ifdef __AVX2__
extern void vectorized_log_conversion_avx2(
    const float* input, float* output, int n,
    float epsilon, float min_clamp, float max_clamp
);
extern void vectorized_exp_conversion_avx2(
    const float* input, float* output, int n,
    float min_clamp, float max_clamp, float output_max
);
#endif

// Log-Sum-Exp helper function (numerically stable)
inline float stable_log_sum_exp(float a, float b) {
    float max_val = std::max(a, b);
    float diff = std::abs(a - b);
    float diff_clamped = std::min(diff, 20.0f);
    return max_val + std::log1p(std::exp(-diff_clamped));
}

/**
 * Work-efficient parallel scan using Blelloch algorithm
 * Optimized for modern CPUs with SIMD and OpenMP
 */
torch::Tensor associative_scan_exponential_cpu(
    torch::Tensor log_gamma  // [batch, heads, seq_len, head_dim] - already in log-space
) {
    // Get tensor info
    auto sizes = log_gamma.sizes();
    int64_t batch_size = sizes[0];
    int64_t num_heads = sizes[1];
    int64_t seq_len = sizes[2];
    int64_t head_dim = sizes[3];
    
    // Get contiguous tensor data pointers
    auto log_gamma_contiguous = log_gamma.contiguous();
    float* log_gamma_ptr = log_gamma_contiguous.data_ptr<float>();
    
    auto log_cumsum = torch::zeros_like(log_gamma);
    float* log_cumsum_ptr = log_cumsum.data_ptr<float>();
    
    // Use optimized Blelloch parallel scan with PyTorch-style thread management
    // Thread count will be auto-optimized based on problem size
    blelloch_scan_parallel_log_sum_exp(
        log_gamma_ptr,
        log_cumsum_ptr,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        0  // Auto-optimize thread count (PyTorch-style: 4-8 optimal)
    );
    
    return log_cumsum;
}

/**
 * Full associative scan exponential product (CPU optimized)
 * Input: gamma in [0, 1] range
 * Output: cumulative product
 * 
 * Adaptive strategy:
 * - Small tensors: Use PyTorch cumprod (lower overhead)
 * - Medium/Large tensors: Use C++ SIMD scan (faster)
 */
torch::Tensor associative_scan_exponential_cpu_full(
    torch::Tensor gamma  // [batch, heads, seq_len, head_dim]
) {
    // Get tensor info
    auto sizes = gamma.sizes();
    int64_t batch_size = sizes[0];
    int64_t num_heads = sizes[1];
    int64_t seq_len = sizes[2];
    int64_t head_dim = sizes[3];
    int64_t total_elements = gamma.numel();
    
    // ADAPTIVE STRATEGY: For very small tensors, PyTorch is faster
    // Threshold: ~10K elements (empirically determined)
    const int64_t SMALL_TENSOR_THRESHOLD = 10000;
    
    if (total_elements < SMALL_TENSOR_THRESHOLD) {
        // Small tensor: Use PyTorch cumprod (lower overhead)
        return torch::cumprod(gamma, 2);
    }
    
    // Medium/Large tensor: Use optimized C++ SIMD scan
    // Convert to FP32 for numerical stability
    auto gamma_fp32 = gamma.to(torch::kFloat32);
    
    // Convert to log-space (use PyTorch for accuracy)
    const float epsilon = 1e-8f;
    auto log_gamma = torch::log(gamma_fp32 + epsilon);
    
    // Clamp log values to [-50, 0] range
    log_gamma = torch::clamp(log_gamma, -50.0f, 0.0f);
    
    // Perform parallel scan (log-space addition for cumulative product)
    auto log_cumsum = associative_scan_exponential_cpu(log_gamma);
    
    // CRITICAL: No max normalization needed for cumulative product!
    // Direct exp: exp(log_cumsum) = cumulative_product
    // Clamp log values before exp to prevent overflow
    log_cumsum = torch::clamp(log_cumsum, -50.0f, 0.0f);
    auto cumulative_product = torch::exp(log_cumsum);
    
    // Final clamp to prevent NaN/Inf
    cumulative_product = torch::clamp(cumulative_product, 0.0f, 1e10f);
    
    // Convert back to original dtype
    return cumulative_product.to(gamma.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("associative_scan_exponential_cpu", &associative_scan_exponential_cpu_full,
          "Associative scan exponential product (CPU optimized with SIMD/OpenMP)");
}

