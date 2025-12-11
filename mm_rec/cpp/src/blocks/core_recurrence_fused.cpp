/**
 * Core Recurrence Formula - Fused Kernel (CPU Optimized)
 * 
 * Implements: h_t = z_t ⊙ σ(W_g @ h_{t-1}) + γ ⊙ h_{t-1}
 * 
 * Optimizations:
 * - Fused operations in single kernel
 * - SIMD for element-wise operations
 * - SIMD-accelerated BLAS for matrix-vector multiply
 * - OpenMP for parallelization (only for large problems)
 */

#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <algorithm>

// PyTorch headers for MKL access
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cpu/Loops.h>

// Include BLAS wrapper header
#include "core/blas_wrapper.h"

// OPTIMIZED Manual BLAS with SIMD and cache optimization (fallback)
// Computes: y = alpha * A @ x + beta * y (row-major A)
inline void manual_sgemv_rowmajor(
    int m, int n, float alpha,
    const float* A, int lda,
    const float* x, int incx,
    float beta, float* y, int incy
) {
    // Use SIMD for inner loop when possible
    #ifdef __AVX2__
    // Process 8 elements at once with AVX2
    for (int i = 0; i < m; i++) {
        __m256 sum_vec = _mm256_setzero_ps();
        int j = 0;
        
        // Vectorized inner loop
        for (; j < n - 7; j += 8) {
            // Prefetch next cache line
            if (j + 16 < n) {
                __builtin_prefetch(&A[i * lda + j + 16], 0, 1);
                __builtin_prefetch(&x[(j + 16) * incx], 0, 1);
            }
            
            __m256 a_vec = _mm256_loadu_ps(&A[i * lda + j]);
            __m256 x_vec = _mm256_loadu_ps(&x[j * incx]);
            sum_vec = _mm256_fmadd_ps(a_vec, x_vec, sum_vec);
        }
        
        // Horizontal sum of vector
        float sum = 0.0f;
        float sum_arr[8];
        _mm256_storeu_ps(sum_arr, sum_vec);
        for (int k = 0; k < 8; k++) {
            sum += sum_arr[k];
        }
        
        // Handle remaining elements
        for (; j < n; j++) {
            sum += A[i * lda + j] * x[j * incx];
        }
        
        y[i * incy] = alpha * sum + beta * y[i * incy];
    }
    #else
    // Scalar fallback with cache optimization
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        // Prefetch next row
        if (i + 1 < m) {
            __builtin_prefetch(&A[(i + 1) * lda], 0, 1);
        }
        
        for (int j = 0; j < n; j++) {
            // Prefetch next elements
            if (j + 16 < n) {
                __builtin_prefetch(&A[i * lda + j + 16], 0, 1);
                __builtin_prefetch(&x[(j + 16) * incx], 0, 1);
            }
            sum += A[i * lda + j] * x[j * incx];
        }
        y[i * incy] = alpha * sum + beta * y[i * incy];
    }
    #endif
}

// Forward declarations for SIMD exp functions (from exp_log_simd.cpp)
#if defined(__AVX512F__) && defined(__AVX512DQ__)
__m512 vectorized_exp_avx512(__m512 x);
#elif defined(__AVX2__)
__m256 vectorized_exp_avx2(__m256 x);
#endif

// ============================================================================
// Core Recurrence Fused Kernel
// ============================================================================

/**
 * Fused kernel for core recurrence formula
 * h_t = z_t ⊙ σ(W_g @ h_{t-1}) + γ ⊙ h_{t-1}
 * 
 * @param z_t: Input [batch, seq_len, hidden_dim]
 * @param h_prev: Previous state [batch, seq_len, hidden_dim]
 * @param W_g: Gating weights [hidden_dim, hidden_dim]
 * @param gamma: Decay coefficients [batch, seq_len, hidden_dim]
 * @param h_t: Output [batch, seq_len, hidden_dim]
 * @param batch_size: Batch size
 * @param seq_len: Sequence length
 * @param hidden_dim: Hidden dimension
 * @param num_threads: OpenMP threads (0 = auto)
 */
void core_recurrence_fused_cpu(
    const float* z_t,        // [batch, seq_len, hidden_dim]
    const float* h_prev,     // [batch, seq_len, hidden_dim]
    const float* W_g,        // [hidden_dim, hidden_dim]
    const float* gamma,      // [batch, seq_len, hidden_dim]
    float* h_t,              // Output [batch, seq_len, hidden_dim]
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_threads = 0
) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // OPTIMIZED: Only parallelize if problem is large enough
    // Small problems: OpenMP overhead > benefit
    if (batch_size * seq_len * hidden_dim > 100000) {
        // Large problem: Use OpenMP
        #pragma omp parallel
        {
            // Thread-local gate buffer (avoid allocation in hot loop)
            thread_local static float* gate_buffer = nullptr;
            thread_local static int gate_buffer_size = 0;
            
            if (gate_buffer_size < hidden_dim) {
                if (gate_buffer) delete[] gate_buffer;
                gate_buffer = new float[hidden_dim];
                gate_buffer_size = hidden_dim;
            }
            float* gate = gate_buffer;
            
            #pragma omp for collapse(2)
            for (int b = 0; b < batch_size; b++) {
                for (int t = 0; t < seq_len; t++) {
                    int base_idx = b * seq_len * hidden_dim + t * hidden_dim;
                    
                    // 1. Matrix-vector multiply: g = h_prev @ W_g.t()
                    // Use PyTorch's optimized MKL-backed matmul for maximum performance
                    // Create temporary tensors and use PyTorch's matmul
                    at::Tensor h_prev_tensor = at::from_blob(
                        const_cast<float*>(&h_prev[base_idx]), 
                        {hidden_dim}, 
                        at::kFloat
                    );
                    at::Tensor W_g_tensor = at::from_blob(
                        const_cast<float*>(W_g), 
                        {hidden_dim, hidden_dim}, 
                        at::kFloat
                    );
                    at::Tensor gate_tensor = at::from_blob(
                        gate, 
                        {hidden_dim}, 
                        at::kFloat
                    );
                    
                    // h_prev @ W_g.t() using PyTorch's optimized MKL matmul
                    gate_tensor.copy_(h_prev_tensor.matmul(W_g_tensor.t()));
                    
                    // 2. Vectorized sigmoid: σ(g) = 1 / (1 + exp(-g))
                    #if defined(__AVX512F__) && defined(__AVX512DQ__)
                    for (int d = 0; d < hidden_dim - 15; d += 16) {
                        __m512 vg = _mm512_loadu_ps(&gate[d]);
                        __m512 vneg = _mm512_mul_ps(vg, _mm512_set1_ps(-1.0f));
                        __m512 vexp = vectorized_exp_avx512(vneg);
                        __m512 vone = _mm512_set1_ps(1.0f);
                        __m512 vsigmoid = _mm512_div_ps(vone, _mm512_add_ps(vone, vexp));
                        _mm512_storeu_ps(&gate[d], vsigmoid);
                    }
                    for (int d = (hidden_dim / 16) * 16; d < hidden_dim; d++) {
                        gate[d] = 1.0f / (1.0f + std::exp(-gate[d]));
                    }
                    #elif defined(__AVX2__)
                    // AVX2: Use std::exp for accuracy (polynomial approximation is inaccurate for small values)
                    // Performance: std::exp is fast enough for sigmoid, and accuracy is critical
                    for (int d = 0; d < hidden_dim; d++) {
                        // Stable sigmoid: use std::exp for accuracy
                        if (gate[d] > 0.0f) {
                            gate[d] = 1.0f / (1.0f + std::exp(-gate[d]));
                        } else {
                            float exp_g = std::exp(gate[d]);
                            gate[d] = exp_g / (1.0f + exp_g);
                        }
                    }
                    #else
                    for (int d = 0; d < hidden_dim; d++) {
                        gate[d] = 1.0f / (1.0f + std::exp(-gate[d]));
                    }
                    #endif
                    
                    // 3. Fused element-wise: z_t ⊙ σ(g) + γ ⊙ h_prev
                    #if defined(__AVX512F__) && defined(__AVX512DQ__)
                    for (int d = 0; d < hidden_dim - 15; d += 16) {
                        __m512 vz = _mm512_loadu_ps(&z_t[base_idx + d]);
                        __m512 vg = _mm512_loadu_ps(&gate[d]);
                        __m512 vh = _mm512_loadu_ps(&h_prev[base_idx + d]);
                        __m512 vgamma = _mm512_loadu_ps(&gamma[base_idx + d]);
                        
                        __m512 vzg = _mm512_mul_ps(vz, vg);
                        __m512 vgh = _mm512_mul_ps(vgamma, vh);
                        __m512 vht = _mm512_add_ps(vzg, vgh);
                        
                        _mm512_storeu_ps(&h_t[base_idx + d], vht);
                    }
                    for (int d = (hidden_dim / 16) * 16; d < hidden_dim; d++) {
                        h_t[base_idx + d] = z_t[base_idx + d] * gate[d] + 
                                            gamma[base_idx + d] * h_prev[base_idx + d];
                    }
                    #elif defined(__AVX2__)
                    // AVX2 processes 8 elements at once
                    // Fix: Use <= instead of < to handle hidden_dim=8 correctly
                    for (int d = 0; d <= hidden_dim - 8; d += 8) {
                        __m256 vz = _mm256_loadu_ps(&z_t[base_idx + d]);
                        __m256 vg = _mm256_loadu_ps(&gate[d]);
                        __m256 vh = _mm256_loadu_ps(&h_prev[base_idx + d]);
                        __m256 vgamma = _mm256_loadu_ps(&gamma[base_idx + d]);
                        
                        __m256 vzg = _mm256_mul_ps(vz, vg);
                        __m256 vgh = _mm256_mul_ps(vgamma, vh);
                        __m256 vht = _mm256_add_ps(vzg, vgh);
                        
                        _mm256_storeu_ps(&h_t[base_idx + d], vht);
                    }
                    for (int d = (hidden_dim / 8) * 8; d < hidden_dim; d++) {
                        h_t[base_idx + d] = z_t[base_idx + d] * gate[d] + 
                                            gamma[base_idx + d] * h_prev[base_idx + d];
                    }
                    #else
                    for (int d = 0; d < hidden_dim; d++) {
                        h_t[base_idx + d] = z_t[base_idx + d] * gate[d] + 
                                            gamma[base_idx + d] * h_prev[base_idx + d];
                    }
                    #endif
                }
            }
        }
    } else {
        // Small problem: Sequential (no OpenMP overhead)
        float* gate = new float[hidden_dim];
        
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                int base_idx = b * seq_len * hidden_dim + t * hidden_dim;
                
                // 1. Matrix-vector multiply: g = h_prev @ W_g.t()
                // Use optimized BLAS (MKL/OpenBLAS) if available, otherwise PyTorch matmul
                #ifdef BLAS_AVAILABLE
                // Use BLAS: g = W_g.t() @ h_prev
                optimized_sgemv(
                    hidden_dim, hidden_dim,  // m, n
                    1.0f,                    // alpha
                    W_g, hidden_dim,         // A, lda
                    &h_prev[base_idx], 1,    // x, incx
                    0.0f,                    // beta
                    gate, 1                  // y, incy (output)
                );
                #else
                // Fallback: Use PyTorch's optimized MKL-backed matmul
                at::Tensor h_prev_tensor = at::from_blob(
                    const_cast<float*>(&h_prev[base_idx]), 
                    {hidden_dim}, 
                    at::kFloat
                );
                at::Tensor W_g_tensor = at::from_blob(
                    const_cast<float*>(W_g), 
                    {hidden_dim, hidden_dim}, 
                    at::kFloat
                );
                at::Tensor gate_tensor = at::from_blob(
                    gate, 
                    {hidden_dim}, 
                    at::kFloat
                );
                
                // h_prev @ W_g.t() using PyTorch's optimized MKL matmul
                gate_tensor.copy_(h_prev_tensor.matmul(W_g_tensor.t()));
                #endif
                
                // 2. Vectorized sigmoid: σ(g) = 1 / (1 + exp(-g))
                #if defined(__AVX512F__) && defined(__AVX512DQ__)
                for (int d = 0; d < hidden_dim - 15; d += 16) {
                    __m512 vg = _mm512_loadu_ps(&gate[d]);
                    __m512 vneg = _mm512_mul_ps(vg, _mm512_set1_ps(-1.0f));
                    __m512 vexp = vectorized_exp_avx512(vneg);
                    __m512 vone = _mm512_set1_ps(1.0f);
                    __m512 vsigmoid = _mm512_div_ps(vone, _mm512_add_ps(vone, vexp));
                    _mm512_storeu_ps(&gate[d], vsigmoid);
                }
                for (int d = (hidden_dim / 16) * 16; d < hidden_dim; d++) {
                    gate[d] = 1.0f / (1.0f + std::exp(-gate[d]));
                }
                    #elif defined(__AVX2__)
                    // AVX2: Use std::exp for accuracy (polynomial approximation is inaccurate for small values)
                    // Performance: std::exp is fast enough for sigmoid, and accuracy is critical
                    for (int d = 0; d < hidden_dim; d++) {
                        // Stable sigmoid: use std::exp for accuracy
                        if (gate[d] > 0.0f) {
                            gate[d] = 1.0f / (1.0f + std::exp(-gate[d]));
                        } else {
                            float exp_g = std::exp(gate[d]);
                            gate[d] = exp_g / (1.0f + exp_g);
                        }
                    }
                #else
                // Scalar: Use std::exp for accuracy
                for (int d = 0; d < hidden_dim; d++) {
                    // Stable sigmoid: use std::exp for accuracy
                    if (gate[d] > 0.0f) {
                        gate[d] = 1.0f / (1.0f + std::exp(-gate[d]));
                    } else {
                        float exp_g = std::exp(gate[d]);
                        gate[d] = exp_g / (1.0f + exp_g);
                    }
                }
                #endif
                
                // 3. Fused element-wise: z_t ⊙ σ(g) + γ ⊙ h_prev
                #if defined(__AVX512F__) && defined(__AVX512DQ__)
                for (int d = 0; d < hidden_dim - 15; d += 16) {
                    __m512 vz = _mm512_loadu_ps(&z_t[base_idx + d]);
                    __m512 vg = _mm512_loadu_ps(&gate[d]);
                    __m512 vh = _mm512_loadu_ps(&h_prev[base_idx + d]);
                    __m512 vgamma = _mm512_loadu_ps(&gamma[base_idx + d]);
                    
                    __m512 vzg = _mm512_mul_ps(vz, vg);
                    __m512 vgh = _mm512_mul_ps(vgamma, vh);
                    __m512 vht = _mm512_add_ps(vzg, vgh);
                    
                    _mm512_storeu_ps(&h_t[base_idx + d], vht);
                }
                for (int d = (hidden_dim / 16) * 16; d < hidden_dim; d++) {
                    h_t[base_idx + d] = z_t[base_idx + d] * gate[d] + 
                                        gamma[base_idx + d] * h_prev[base_idx + d];
                }
                #elif defined(__AVX2__)
                // AVX2 processes 8 elements at once
                // Fix: Use <= instead of < to handle hidden_dim=8 correctly
                for (int d = 0; d <= hidden_dim - 8; d += 8) {
                    __m256 vz = _mm256_loadu_ps(&z_t[base_idx + d]);
                    __m256 vg = _mm256_loadu_ps(&gate[d]);
                    __m256 vh = _mm256_loadu_ps(&h_prev[base_idx + d]);
                    __m256 vgamma = _mm256_loadu_ps(&gamma[base_idx + d]);
                    
                    __m256 vzg = _mm256_mul_ps(vz, vg);
                    __m256 vgh = _mm256_mul_ps(vgamma, vh);
                    __m256 vht = _mm256_add_ps(vzg, vgh);
                    
                    _mm256_storeu_ps(&h_t[base_idx + d], vht);
                }
                for (int d = (hidden_dim / 8) * 8; d < hidden_dim; d++) {
                    h_t[base_idx + d] = z_t[base_idx + d] * gate[d] + 
                                        gamma[base_idx + d] * h_prev[base_idx + d];
                }
                #else
                for (int d = 0; d < hidden_dim; d++) {
                    h_t[base_idx + d] = z_t[base_idx + d] * gate[d] + 
                                        gamma[base_idx + d] * h_prev[base_idx + d];
                }
                #endif
            }
        }
        
        delete[] gate;
    }
}
