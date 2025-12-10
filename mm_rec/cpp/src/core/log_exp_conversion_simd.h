/**
 * SIMD-Optimized Log and Exp Conversion Header
 */

#ifndef LOG_EXP_CONVERSION_SIMD_H
#define LOG_EXP_CONVERSION_SIMD_H

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
);

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
);
#endif

#endif // LOG_EXP_CONVERSION_SIMD_H
