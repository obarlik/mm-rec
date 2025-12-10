/**
 * Vectorized Exp/Log Functions (SIMD)
 * 
 * High-performance SIMD implementations for exponential and logarithmic functions.
 * Optimized for Log-Sum-Exp operations in Associative Scan.
 * 
 * Supports:
 * - AVX-512 (16 floats) - Best performance
 * - AVX2 (8 floats) - Good performance
 * - SSE4.2 (4 floats) - Fallback
 * - Scalar - Last resort
 */

#include <immintrin.h>
#include <cmath>
#include <algorithm>

// ============================================================================
// Fast Exp Approximation (Polynomial)
// Optimized for range [-20, 0] (Log-Sum-Exp use case)
// ============================================================================

// AVX-512 support - check at compile time
#if defined(__AVX512F__) && defined(__AVX512DQ__)
/**
 * AVX-512 vectorized exp (16 floats at once)
 * Fast polynomial approximation optimized for [-20, 0] range
 */
__m512 vectorized_exp_avx512(__m512 x) {
    // Clamp to [-20, 0] for Log-Sum-Exp stability
    __m512 x_clamped = _mm512_max_ps(x, _mm512_set1_ps(-20.0f));
    x_clamped = _mm512_min_ps(x_clamped, _mm512_set1_ps(0.0f));
    
    // Fast polynomial approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
    // More accurate: Remez polynomial or Chebyshev approximation
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 x2 = _mm512_mul_ps(x_clamped, x_clamped);
    __m512 x3 = _mm512_mul_ps(x2, x_clamped);
    __m512 x4 = _mm512_mul_ps(x3, x_clamped);
    
    // Horner's method for better numerical stability
    // exp(x) ≈ 1 + x*(1 + x/2*(1 + x/3*(1 + x/4)))
    __m512 result = _mm512_fmadd_ps(
        x4, _mm512_set1_ps(1.0f/24.0f),
        one
    );
    result = _mm512_fmadd_ps(
        x3, _mm512_set1_ps(1.0f/6.0f),
        result
    );
    result = _mm512_fmadd_ps(
        x2, _mm512_set1_ps(0.5f),
        result
    );
    result = _mm512_fmadd_ps(
        x_clamped, one,
        result
    );
    
    return result;
}

/**
 * AVX-512 vectorized log1p (16 floats at once)
 * log1p(x) = log(1 + x)
 * Optimized for small x values
 */
__m512 vectorized_log1p_avx512(__m512 x) {
    // For small x: log1p(x) ≈ x - x²/2 + x³/3
    // For larger x: use standard log(1+x)
    __m512 small_threshold = _mm512_set1_ps(0.1f);
    __mmask16 is_small = _mm512_cmp_ps_mask(x, small_threshold, _CMP_LT_OQ);
    
    // Small x approximation
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 x3 = _mm512_mul_ps(x2, x);
    __m512 approx = _mm512_sub_ps(x, _mm512_mul_ps(x2, _mm512_set1_ps(0.5f)));
    approx = _mm512_add_ps(approx, _mm512_mul_ps(x3, _mm512_set1_ps(1.0f/3.0f)));
    
    // For larger values, use standard log (scalar for now, can be optimized)
    // Note: AVX-512 doesn't have log, so we'd need approximation or scalar
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 one_plus_x = _mm512_add_ps(one, x);
    
    // For now, blend: use approximation for small, scalar log for large
    // In production, use vectorized log approximation
    float x_arr[16], log_arr[16];
    _mm512_storeu_ps(x_arr, one_plus_x);
    for (int i = 0; i < 16; i++) {
        log_arr[i] = std::log(x_arr[i]);
    }
    __m512 standard = _mm512_loadu_ps(log_arr);
    
    return _mm512_mask_blend_ps(is_small, standard, approx);
}
#endif

#ifdef __AVX2__
/**
 * AVX2 vectorized exp (8 floats at once)
 * Similar to AVX-512 but for 8 floats
 * Export for linking
 */
__attribute__((visibility("default"))) __m256 vectorized_exp_avx2(__m256 x) {
    // Clamp to [-20, 0]
    __m256 x_clamped = _mm256_max_ps(x, _mm256_set1_ps(-20.0f));
    x_clamped = _mm256_min_ps(x_clamped, _mm256_set1_ps(0.0f));
    
    // Polynomial approximation
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 x2 = _mm256_mul_ps(x_clamped, x_clamped);
    __m256 x3 = _mm256_mul_ps(x2, x_clamped);
    __m256 x4 = _mm256_mul_ps(x3, x_clamped);
    
    __m256 result = _mm256_fmadd_ps(
        x4, _mm256_set1_ps(1.0f/24.0f),
        one
    );
    result = _mm256_fmadd_ps(
        x3, _mm256_set1_ps(1.0f/6.0f),
        result
    );
    result = _mm256_fmadd_ps(
        x2, _mm256_set1_ps(0.5f),
        result
    );
    result = _mm256_fmadd_ps(
        x_clamped, one,
        result
    );
    
    return result;
}

__attribute__((visibility("default"))) __m256 vectorized_log1p_avx2(__m256 x) {
    __m256 small_threshold = _mm256_set1_ps(0.1f);
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    __m256 approx = _mm256_sub_ps(x, _mm256_mul_ps(x2, _mm256_set1_ps(0.5f)));
    approx = _mm256_add_ps(approx, _mm256_mul_ps(x3, _mm256_set1_ps(1.0f/3.0f)));
    
    // For larger values, use scalar log
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 one_plus_x = _mm256_add_ps(one, x);
    float x_arr[8], log_arr[8];
    _mm256_storeu_ps(x_arr, one_plus_x);
    for (int i = 0; i < 8; i++) {
        log_arr[i] = std::log(x_arr[i]);
    }
    __m256 standard = _mm256_loadu_ps(log_arr);
    
    // Blend based on threshold
    __m256 mask = _mm256_cmp_ps(x, small_threshold, _CMP_LT_OQ);
    return _mm256_blendv_ps(standard, approx, mask);
}
#endif

// ============================================================================
// Vectorized Log-Sum-Exp (SIMD)
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512DQ__)
/**
 * AVX-512 vectorized log-sum-exp
 * stable_log_sum_exp(a, b) = max(a, b) + log1p(exp(-abs(a - b)))
 */
__m512 vectorized_log_sum_exp_avx512(__m512 a, __m512 b) {
    // max(a, b)
    __m512 max_val = _mm512_max_ps(a, b);
    
    // abs(a - b)
    __m512 diff = _mm512_sub_ps(a, b);
    __m512 abs_diff = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), diff);
    
    // Clamp to [0, 20]
    __m512 clamped = _mm512_min_ps(abs_diff, _mm512_set1_ps(20.0f));
    
    // exp(-clamped) - vectorized!
    __m512 neg_clamped = _mm512_mul_ps(clamped, _mm512_set1_ps(-1.0f));
    __m512 exp_neg = vectorized_exp_avx512(neg_clamped);
    
    // log1p(exp(-clamped)) - vectorized!
    __m512 log1p_exp = vectorized_log1p_avx512(exp_neg);
    
    // max + log1p
    return _mm512_add_ps(max_val, log1p_exp);
}
#endif

#ifdef __AVX2__
/**
 * AVX2 vectorized log-sum-exp (8 floats)
 * Export for linking
 */
__attribute__((visibility("default"))) __m256 vectorized_log_sum_exp_avx2(__m256 a, __m256 b) {
    // max(a, b) - single instruction
    __m256 max_val = _mm256_max_ps(a, b);
    
    // abs(a - b) - bit manipulation (faster than conditional)
    __m256 diff = _mm256_sub_ps(a, b);
    __m256 abs_diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff);
    
    // Clamp to [0, 20] for numerical stability (IMPROVED)
    __m256 clamped = _mm256_min_ps(abs_diff, _mm256_set1_ps(20.0f));
    clamped = _mm256_max_ps(clamped, _mm256_set1_ps(0.0f));  // Ensure non-negative
    
    // Fast exp(-clamped) using optimized exp
    __m256 neg_clamped = _mm256_mul_ps(clamped, _mm256_set1_ps(-1.0f));
    __m256 exp_neg = vectorized_exp_avx2(neg_clamped);
    
    // CRITICAL FIX: Use std::log1p for accuracy (Google XNNPACK style)
    // For numerical stability, we need accurate log1p, not approximation
    // Convert to scalar for accurate log1p computation
    float exp_neg_arr[8];
    float log1p_arr[8];
    _mm256_storeu_ps(exp_neg_arr, exp_neg);
    
    // Use accurate std::log1p for each element
    for (int i = 0; i < 8; i++) {
        log1p_arr[i] = std::log1p(exp_neg_arr[i]);
    }
    
    __m256 log1p_approx = _mm256_loadu_ps(log1p_arr);
    
    // max + log1p
    return _mm256_add_ps(max_val, log1p_approx);
}
#endif

// ============================================================================
// Scalar Fallback (for remaining elements)
// ============================================================================

inline float stable_log_sum_exp_scalar(float a, float b) {
    float max_val = std::max(a, b);
    float diff = std::abs(a - b);
    float diff_clamped = std::min(diff, 20.0f);
    return max_val + std::log1p(std::exp(-diff_clamped));
}
