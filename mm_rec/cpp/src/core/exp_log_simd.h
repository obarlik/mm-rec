/**
 * Vectorized Exp/Log Functions (SIMD) - Header
 */

#ifndef MM_REC_EXP_LOG_SIMD_H
#define MM_REC_EXP_LOG_SIMD_H

#include <immintrin.h>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
// AVX-512 functions (16 floats) - only if fully supported
__m512 vectorized_exp_avx512(__m512 x);
__m512 vectorized_log1p_avx512(__m512 x);
__m512 vectorized_log_sum_exp_avx512(__m512 a, __m512 b);
#endif

#ifdef __AVX2__
// AVX2 functions (8 floats)
__m256 vectorized_exp_avx2(__m256 x);
__m256 vectorized_log1p_avx2(__m256 x);
__m256 vectorized_log_sum_exp_avx2(__m256 a, __m256 b);
#endif

// Scalar fallback
inline float stable_log_sum_exp_scalar(float a, float b) {
    float max_val = std::max(a, b);
    float diff = std::abs(a - b);
    float diff_clamped = std::min(diff, 20.0f);
    return max_val + std::log1p(std::exp(-diff_clamped));
}

#endif // MM_REC_EXP_LOG_SIMD_H
