/**
 * SIMD-Optimized Log and Exp Conversion
 * 
 * Optimizes the log-space conversion and exp conversion for associative scan
 */

#include <immintrin.h>
#include <cmath>
#include <algorithm>

// Forward declarations for vectorized exp (from exp_log_simd.cpp)
#if defined(__AVX512F__) && defined(__AVX512DQ__)
__m512 vectorized_exp_avx512(__m512 x);
#elif defined(__AVX2__)
__m256 vectorized_exp_avx2(__m256 x);
#endif

// ============================================================================
// Vectorized Log Conversion (SIMD)
// ============================================================================

#ifdef __AVX2__
/**
 * AVX2 vectorized log conversion: log(x + epsilon) with clamping
 */
void vectorized_log_conversion_avx2(
    const float* input,
    float* output,
    int n,
    float epsilon = 1e-8f,
    float min_clamp = -50.0f,
    float max_clamp = 0.0f
) {
    __m256 eps_vec = _mm256_set1_ps(epsilon);
    __m256 min_clamp_vec = _mm256_set1_ps(min_clamp);
    __m256 max_clamp_vec = _mm256_set1_ps(max_clamp);
    
    int i = 0;
    for (; i < n - 7; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 x_plus_eps = _mm256_add_ps(x, eps_vec);
        
        // Compute log using scalar (std::log is accurate)
        // CRITICAL: Process each element individually for accuracy
        float x_arr[8], log_arr[8];
        _mm256_storeu_ps(x_arr, x_plus_eps);
        for (int j = 0; j < 8; j++) {
            float val = std::log(x_arr[j]);
            log_arr[j] = std::clamp(val, min_clamp, max_clamp);
        }
        __m256 log_result = _mm256_loadu_ps(log_arr);
        
        _mm256_storeu_ps(&output[i], log_result);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        float val = std::log(input[i] + epsilon);
        output[i] = std::clamp(val, min_clamp, max_clamp);
    }
}
#endif

// ============================================================================
// Vectorized Exp Conversion (SIMD)
// ============================================================================

#ifdef __AVX2__
/**
 * AVX2 vectorized exp conversion: exp(x) with clamping
 */
void vectorized_exp_conversion_avx2(
    const float* input,
    float* output,
    int n,
    float min_clamp = -50.0f,
    float max_clamp = 0.0f,
    float output_max = 1e10f
) {
    __m256 min_clamp_vec = _mm256_set1_ps(min_clamp);
    __m256 max_clamp_vec = _mm256_set1_ps(max_clamp);
    __m256 output_max_vec = _mm256_set1_ps(output_max);
    __m256 zero_vec = _mm256_setzero_ps();
    
    int i = 0;
    for (; i < n - 7; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        
        // Clamp input
        x = _mm256_max_ps(x, min_clamp_vec);
        x = _mm256_min_ps(x, max_clamp_vec);
        
        // Compute exp using optimized vectorized exp
        __m256 exp_result = vectorized_exp_avx2(x);
        
        // Clamp output
        exp_result = _mm256_min_ps(exp_result, output_max_vec);
        exp_result = _mm256_max_ps(exp_result, zero_vec);
        
        _mm256_storeu_ps(&output[i], exp_result);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        float val = std::clamp(input[i], min_clamp, max_clamp);
        float exp_val = std::exp(val);
        output[i] = std::clamp(exp_val, 0.0f, output_max);
    }
}
#endif
