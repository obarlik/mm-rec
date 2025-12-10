/**
 * Fast Math Functions - Assembly Optimized (Header)
 */

#ifndef MM_REC_FAST_MATH_ASM_H
#define MM_REC_FAST_MATH_ASM_H

#include <immintrin.h>

// Fast exp approximations
float fast_exp_asm(float x);
float fast_exp_lut(float x);
void init_exp_lut();

// Fast log1p
float fast_log1p_asm(float x);

// Fast log-sum-exp
float fast_log_sum_exp_asm(float a, float b);

// SIMD versions
#ifdef __AVX2__
__m256 fast_exp_avx2_asm(__m256 x);
__m256 fast_log_sum_exp_avx2_asm(__m256 a, __m256 b);
#endif

// Memory prefetching
void prefetch_next(const void* addr);
void prefetch_read(const void* addr);

// Branch prediction hints
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

// Cache alignment
#define CACHE_LINE_SIZE 64
#define ALIGN_CACHE_LINE __attribute__((aligned(CACHE_LINE_SIZE)))

// CPU feature detection
bool has_avx2();
bool has_fma();
bool has_avx512f();

#endif // MM_REC_FAST_MATH_ASM_H
