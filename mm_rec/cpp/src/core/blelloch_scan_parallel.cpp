/**
 * SIMD-Optimized Sequential Scan (Correct and Fast)
 * 
 * Uses sequential scan with SIMD vectorization for correctness and performance.
 * Parallelized across batch and heads, SIMD within each sequence.
 * 
 * This is simpler and more reliable than Blelloch parallel scan for prefix scan.
 */

#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>  // For std::min, std::max

// Forward declarations for SIMD functions (implemented in exp_log_simd.cpp)
#if defined(__AVX512F__) && defined(__AVX512DQ__)
__m512 vectorized_log_sum_exp_avx512(__m512 a, __m512 b);
#elif defined(__AVX2__)
__m256 vectorized_log_sum_exp_avx2(__m256 a, __m256 b);
#endif

// Scalar fallback
inline float stable_log_sum_exp_scalar(float a, float b) {
    float max_val = std::max(a, b);
    float diff = std::abs(a - b);
    float diff_clamped = std::min(diff, 20.0f);
    return max_val + std::log1p(std::exp(-diff_clamped));
}

/**
 * SIMD-Optimized Sequential Scan for Log-Sum-Exp
 * 
 * @param input: Input array [batch, heads, seq_len, dim] in log-space
 * @param output: Output array [batch, heads, seq_len, dim] (cumulative log-sum)
 * @param batch_size: Batch dimension
 * @param num_heads: Number of heads
 * @param seq_len: Sequence length
 * @param dim: Head dimension
 * @param num_threads: Number of OpenMP threads (0 = auto)
 */
void blelloch_scan_parallel_log_sum_exp(
    float* input,           // Input: log_gamma [batch, heads, seq_len, dim]
    float* output,          // Output: log_cumsum [batch, heads, seq_len, dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int dim,
    int num_threads = 0
) {
    // OPTIMIZED: PyTorch-style thread management
    // Optimal thread count: 4-8 (based on PyTorch analysis)
    if (num_threads <= 0) {
        // Auto-detect optimal thread count
        int total_elements = batch_size * num_heads * seq_len * dim;
        int cpu_cores = omp_get_max_threads();
        
        // For small problems, use fewer threads
        if (total_elements < 50000) {
            num_threads = std::min(2, cpu_cores);
        }
        // For medium problems, use 4-8 threads (PyTorch optimal)
        else if (total_elements < 500000) {
            num_threads = std::min(8, std::max(4, cpu_cores / 2));
        }
        // For large problems, use more threads but cap at 8
        else {
            num_threads = std::min(8, cpu_cores);
        }
    }
    
    omp_set_num_threads(num_threads);
    
    // OPTIMIZED: Only parallelize if problem is large enough
    // Small problems: OpenMP overhead > benefit
    const int MIN_ELEMENTS_FOR_PARALLEL = 10000;  // ~batch*heads*seq_len*dim
    int total_elements = batch_size * num_heads * seq_len * dim;
    
    if (total_elements >= MIN_ELEMENTS_FOR_PARALLEL) {
        // Large problem: Use OpenMP
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < num_heads; h++) {
                int base_offset = (b * num_heads + h) * seq_len * dim;
                
                // Initialize first element
                for (int d = 0; d < dim; d++) {
                    output[base_offset + d] = input[base_offset + d];
                }
                
                // Sequential scan with SIMD vectorization + cache prefetching
                for (int t = 1; t < seq_len; t++) {
                    int prev_offset = base_offset + (t - 1) * dim;
                    int curr_offset = base_offset + t * dim;
                    
                    // Prefetch next iteration's data (Quake3 style)
                    if (t + 1 < seq_len) {
                        int next_offset = base_offset + (t + 1) * dim;
                        __builtin_prefetch(&input[next_offset], 0, 3);  // Read, high temporal
                        __builtin_prefetch(&output[next_offset], 1, 3);  // Write, high temporal
                    }
                
                // CRITICAL FIX: For cumulative PRODUCT, we need ADDITION in log-space, not log-sum-exp!
                // log(exp(a) * exp(b)) = a + b (not log(exp(a) + exp(b)))
                #if defined(__AVX512F__) && defined(__AVX512DQ__)
                // AVX-512: 16 floats at once
                for (int d = 0; d < dim - 15; d += 16) {
                    __m512 vprev = _mm512_loadu_ps(&output[prev_offset + d]);
                    __m512 vcurr = _mm512_loadu_ps(&input[curr_offset + d]);
                    __m512 vresult = _mm512_add_ps(vprev, vcurr);  // Simple addition!
                    _mm512_storeu_ps(&output[curr_offset + d], vresult);
                }
                // Handle remaining elements
                for (int d = (dim / 16) * 16; d < dim; d++) {
                    output[curr_offset + d] = output[prev_offset + d] + input[curr_offset + d];
                }
                #elif defined(__AVX2__)
                // AVX2: 8 floats at once with prefetching
                for (int d = 0; d < dim - 7; d += 8) {
                    // Prefetch next cache line
                    if (d + 8 < dim) {
                        __builtin_prefetch(&input[curr_offset + d + 8], 0, 2);
                        __builtin_prefetch(&output[prev_offset + d + 8], 0, 2);
                    }
                    
                    __m256 vprev = _mm256_loadu_ps(&output[prev_offset + d]);
                    __m256 vcurr = _mm256_loadu_ps(&input[curr_offset + d]);
                    __m256 vresult = _mm256_add_ps(vprev, vcurr);  // Simple addition!
                    _mm256_storeu_ps(&output[curr_offset + d], vresult);
                }
                // Handle remaining elements
                for (int d = (dim / 8) * 8; d < dim; d++) {
                    output[curr_offset + d] = output[prev_offset + d] + input[curr_offset + d];
                }
                #else
                // Scalar fallback with prefetching
                for (int d = 0; d < dim; d++) {
                    if (d + 16 < dim) {
                        __builtin_prefetch(&input[curr_offset + d + 16], 0, 2);
                    }
                    output[curr_offset + d] = output[prev_offset + d] + input[curr_offset + d];
                }
                #endif
                }
            }
        }
    } else {
        // Small problem: Sequential (no OpenMP overhead)
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < num_heads; h++) {
                int base_offset = (b * num_heads + h) * seq_len * dim;
                
                // Initialize first element
                for (int d = 0; d < dim; d++) {
                    output[base_offset + d] = input[base_offset + d];
                }
                
                // Sequential scan with SIMD vectorization
                for (int t = 1; t < seq_len; t++) {
                    int prev_offset = base_offset + (t - 1) * dim;
                    int curr_offset = base_offset + t * dim;
                    
                    // CRITICAL FIX: For cumulative PRODUCT, we need ADDITION in log-space, not log-sum-exp!
                    // log(exp(a) * exp(b)) = a + b (not log(exp(a) + exp(b)))
                    #if defined(__AVX512F__) && defined(__AVX512DQ__)
                    // AVX-512: 16 floats at once
                    for (int d = 0; d < dim - 15; d += 16) {
                        __m512 vprev = _mm512_loadu_ps(&output[prev_offset + d]);
                        __m512 vcurr = _mm512_loadu_ps(&input[curr_offset + d]);
                        __m512 vresult = _mm512_add_ps(vprev, vcurr);  // Simple addition!
                        _mm512_storeu_ps(&output[curr_offset + d], vresult);
                    }
                    // Handle remaining elements
                    for (int d = (dim / 16) * 16; d < dim; d++) {
                        output[curr_offset + d] = output[prev_offset + d] + input[curr_offset + d];
                    }
                    #elif defined(__AVX2__)
                    // AVX2: 8 floats at once
                    for (int d = 0; d < dim - 7; d += 8) {
                        __m256 vprev = _mm256_loadu_ps(&output[prev_offset + d]);
                        __m256 vcurr = _mm256_loadu_ps(&input[curr_offset + d]);
                        __m256 vresult = _mm256_add_ps(vprev, vcurr);  // Simple addition!
                        _mm256_storeu_ps(&output[curr_offset + d], vresult);
                    }
                    // Handle remaining elements
                    for (int d = (dim / 8) * 8; d < dim; d++) {
                        output[curr_offset + d] = output[prev_offset + d] + input[curr_offset + d];
                    }
                    #else
                    // Scalar fallback
                    for (int d = 0; d < dim; d++) {
                        output[curr_offset + d] = output[prev_offset + d] + input[curr_offset + d];
                    }
                    #endif
                }
            }
        }
    }
}
