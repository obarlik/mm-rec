/**
 * Blelloch Parallel Scan (Work-Efficient) - FIXED VERSION
 * 
 * Correct implementation for prefix scan:
 * - Up-sweep: Build reduction tree (cumulative values)
 * - Down-sweep: Propagate prefixes correctly
 * 
 * For prefix scan with log-sum-exp operator:
 * - Identity: 0.0 (in log-space, which is 1.0 in linear)
 * - Operator: log_sum_exp(a, b) = max(a, b) + log1p(exp(-abs(a-b)))
 */

#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include "exp_log_simd.h"

// Forward declarations for SIMD functions
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
 * Work-efficient parallel scan using Blelloch algorithm
 * CORRECTED VERSION for prefix scan
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
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Parallelize across batch and heads
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            int base_offset = (b * num_heads + h) * seq_len * dim;
            
            // ====================================================================
            // UP-SWEEP PHASE: Build reduction tree
            // ====================================================================
            // Copy input to output first
            for (int t = 0; t < seq_len; t++) {
                for (int d = 0; d < dim; d++) {
                    output[base_offset + t * dim + d] = input[base_offset + t * dim + d];
                }
            }
            
            // Build reduction tree: for each level (stride = 1, 2, 4, 8, ...)
            for (int stride = 1; stride < seq_len; stride *= 2) {
                #pragma omp parallel for
                for (int t = stride; t < seq_len; t += 2 * stride) {
                    int left_idx = t - stride;
                    int right_idx = t;
                    
                    int left_offset = base_offset + left_idx * dim;
                    int right_offset = base_offset + right_idx * dim;
                    
                    // Combine: output[right] = output[left] + output[right]
                    // (log-sum-exp operator)
                    #if defined(__AVX512F__) && defined(__AVX512DQ__)
                    for (int d = 0; d < dim - 15; d += 16) {
                        __m512 vleft = _mm512_loadu_ps(&output[left_offset + d]);
                        __m512 vright = _mm512_loadu_ps(&output[right_offset + d]);
                        __m512 vresult = vectorized_log_sum_exp_avx512(vleft, vright);
                        _mm512_storeu_ps(&output[right_offset + d], vresult);
                    }
                    for (int d = (dim / 16) * 16; d < dim; d++) {
                        output[right_offset + d] = stable_log_sum_exp_scalar(
                            output[left_offset + d],
                            output[right_offset + d]
                        );
                    }
                    #elif defined(__AVX2__)
                    for (int d = 0; d < dim - 7; d += 8) {
                        __m256 vleft = _mm256_loadu_ps(&output[left_offset + d]);
                        __m256 vright = _mm256_loadu_ps(&output[right_offset + d]);
                        __m256 vresult = vectorized_log_sum_exp_avx2(vleft, vright);
                        _mm256_storeu_ps(&output[right_offset + d], vresult);
                    }
                    for (int d = (dim / 8) * 8; d < dim; d++) {
                        output[right_offset + d] = stable_log_sum_exp_scalar(
                            output[left_offset + d],
                            output[right_offset + d]
                        );
                    }
                    #else
                    for (int d = 0; d < dim; d++) {
                        output[right_offset + d] = stable_log_sum_exp_scalar(
                            output[left_offset + d],
                            output[right_offset + d]
                        );
                    }
                    #endif
                }
            }
            
            // ====================================================================
            // DOWN-SWEEP PHASE: Propagate prefixes
            // ====================================================================
            // Set last element to identity (0 in log-space = 1 in linear)
            for (int d = 0; d < dim; d++) {
                int last_idx = base_offset + (seq_len - 1) * dim + d;
                output[last_idx] = 0.0f;  // Identity for log-sum-exp
            }
            
            // Propagate prefixes from root to leaves
            for (int stride = seq_len / 2; stride > 0; stride /= 2) {
                #pragma omp parallel for
                for (int t = stride; t < seq_len; t += 2 * stride) {
                    int left_idx = t - stride;
                    int right_idx = t;
                    
                    int left_offset = base_offset + left_idx * dim;
                    int right_offset = base_offset + right_idx * dim;
                    
                    // CORRECT: Save right value, then update right with left+right, then update left with saved
                    // For prefix scan: right = left + old_right, left = old_right
                    float* temp = new float[dim];
                    for (int d = 0; d < dim; d++) {
                        temp[d] = output[right_offset + d];
                    }
                    
                    // right = left + old_right (log-sum-exp)
                    #if defined(__AVX512F__) && defined(__AVX512DQ__)
                    for (int d = 0; d < dim - 15; d += 16) {
                        __m512 vleft = _mm512_loadu_ps(&output[left_offset + d]);
                        __m512 vold_right = _mm512_loadu_ps(&temp[d]);
                        __m512 vresult = vectorized_log_sum_exp_avx512(vleft, vold_right);
                        _mm512_storeu_ps(&output[right_offset + d], vresult);
                    }
                    for (int d = (dim / 16) * 16; d < dim; d++) {
                        output[right_offset + d] = stable_log_sum_exp_scalar(
                            output[left_offset + d],
                            temp[d]
                        );
                    }
                    #elif defined(__AVX2__)
                    for (int d = 0; d < dim - 7; d += 8) {
                        __m256 vleft = _mm256_loadu_ps(&output[left_offset + d]);
                        __m256 vold_right = _mm256_loadu_ps(&temp[d]);
                        __m256 vresult = vectorized_log_sum_exp_avx2(vleft, vold_right);
                        _mm256_storeu_ps(&output[right_offset + d], vresult);
                    }
                    for (int d = (dim / 8) * 8; d < dim; d++) {
                        output[right_offset + d] = stable_log_sum_exp_scalar(
                            output[left_offset + d],
                            temp[d]
                        );
                    }
                    #else
                    for (int d = 0; d < dim; d++) {
                        output[right_offset + d] = stable_log_sum_exp_scalar(
                            output[left_offset + d],
                            temp[d]
                        );
                    }
                    #endif
                    
                    // left = old_right (prefix propagation)
                    for (int d = 0; d < dim; d++) {
                        output[left_offset + d] = temp[d];
                    }
                    
                    delete[] temp;
                }
            }
        }
    }
}
