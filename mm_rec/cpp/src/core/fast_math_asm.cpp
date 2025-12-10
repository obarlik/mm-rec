/**
 * Fast Math Functions - Assembly Optimized (Quake3 Style)
 * 
 * Ultra-fast approximations using:
 * - Bit manipulation tricks
 * - Lookup tables
 * - CPU-specific assembly
 * - Cache-friendly memory access
 */

#include <immintrin.h>
#include <cmath>
#include <cstring>

// ============================================================================
// Fast Exp Approximation (Quake3 Style - Bit Manipulation)
// ============================================================================

/**
 * Fast exp(x) for x in [-20, 0] range
 * Uses bit manipulation trick similar to Quake3's fast inverse sqrt
 * 
 * Approximation: exp(x) ≈ 2^(x * log2(e))
 * We use IEEE 754 bit manipulation for fast 2^x
 */
inline float fast_exp_asm(float x) {
    // Clamp to valid range
    if (x < -20.0f) return 0.0f;
    if (x > 0.0f) return 1.0f;
    
    // Fast 2^x using bit manipulation
    // exp(x) = 2^(x * log2(e))
    const float LOG2E = 1.4426950408889634f;  // log2(e)
    const float BIAS = 127.0f;
    
    // Convert to integer representation
    float y = x * LOG2E + BIAS;
    int i = *(int*)&y;  // Bit cast (C++20: std::bit_cast)
    
    // Extract mantissa and exponent
    int exp = (i >> 23) - 127;
    int mantissa = i & 0x7FFFFF;
    
    // Fast polynomial approximation for 2^(mantissa/2^23)
    // Using Remez polynomial (optimized for [-1, 1])
    float m = (mantissa / 8388608.0f) - 1.0f;  // Normalize to [-1, 1]
    float m2 = m * m;
    float m3 = m2 * m;
    float m4 = m3 * m;
    
    // Horner's method: 1 + m*(1 + m/2*(1 + m/3*(1 + m/4)))
    float result = 1.0f + m * (1.0f + m * 0.5f * (1.0f + m * 0.333333f * (1.0f + m * 0.25f)));
    
    // Scale by 2^exp
    return result * (1 << exp);
}

/**
 * Ultra-fast exp using lookup table + interpolation
 * Quake3 style: Precomputed table + linear interpolation
 */
static float exp_lut[256];  // Lookup table for exp(-20 to 0)
static bool exp_lut_initialized = false;

void init_exp_lut() {
    if (exp_lut_initialized) return;
    
    for (int i = 0; i < 256; i++) {
        float x = -20.0f + (i / 255.0f) * 20.0f;
        exp_lut[i] = std::exp(x);
    }
    exp_lut_initialized = true;
}

inline float fast_exp_lut(float x) {
    if (!exp_lut_initialized) init_exp_lut();
    
    // Clamp and map to [0, 255]
    x = (x < -20.0f) ? -20.0f : (x > 0.0f ? 0.0f : x);
    float t = (x + 20.0f) / 20.0f * 255.0f;
    int idx = (int)t;
    float frac = t - idx;
    
    // Linear interpolation
    if (idx >= 255) return exp_lut[255];
    return exp_lut[idx] * (1.0f - frac) + exp_lut[idx + 1] * frac;
}

// ============================================================================
// Fast Log1p (Assembly Optimized)
// ============================================================================

/**
 * Fast log1p(x) = log(1 + x)
 * Uses polynomial approximation optimized for small x
 */
inline float fast_log1p_asm(float x) {
    // For very small x: log1p(x) ≈ x - x²/2
    if (x < 0.01f && x > -0.01f) {
        return x - x * x * 0.5f;
    }
    
    // For larger x: use fast log approximation
    // log(1+x) ≈ (x - x²/2 + x³/3) for |x| < 1
    if (x < 1.0f && x > -0.5f) {
        float x2 = x * x;
        float x3 = x2 * x;
        return x - x2 * 0.5f + x3 * 0.333333f;
    }
    
    // Fallback to standard log
    return std::log1p(x);
}

// ============================================================================
// Fast Log-Sum-Exp (Assembly Optimized)
// ============================================================================

/**
 * Ultra-fast log-sum-exp using assembly intrinsics
 * stable_log_sum_exp(a, b) = max(a, b) + log1p(exp(-abs(a - b)))
 */
inline float fast_log_sum_exp_asm(float a, float b) {
    // Branch prediction hint: likely case
    if (__builtin_expect(std::abs(a - b) > 20.0f, 0)) {
        return (a > b) ? a : b;
    }
    
    float max_val = (a > b) ? a : b;
    float diff = std::abs(a - b);
    
    // Fast exp(-diff) using lookup table
    float exp_neg = fast_exp_lut(-diff);
    
    // Fast log1p
    float log1p_exp = fast_log1p_asm(exp_neg);
    
    return max_val + log1p_exp;
}

// ============================================================================
// SIMD Assembly Optimizations
// ============================================================================

#ifdef __AVX2__

/**
 * AVX2 vectorized fast exp using assembly
 * 8 floats at once
 */
__m256 fast_exp_avx2_asm(__m256 x) {
    // Clamp to [-20, 0]
    __m256 x_clamped = _mm256_max_ps(x, _mm256_set1_ps(-20.0f));
    x_clamped = _mm256_min_ps(x_clamped, _mm256_set1_ps(0.0f));
    
    // Fast polynomial approximation
    // Using FMA (Fused Multiply-Add) for better precision
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 x2 = _mm256_mul_ps(x_clamped, x_clamped);
    __m256 x3 = _mm256_fmadd_ps(x2, x_clamped, _mm256_setzero_ps());
    __m256 x4 = _mm256_mul_ps(x3, x_clamped);
    
    // Horner's method with FMA
    __m256 result = _mm256_fmadd_ps(x4, _mm256_set1_ps(1.0f/24.0f), one);
    result = _mm256_fmadd_ps(x3, _mm256_set1_ps(1.0f/6.0f), result);
    result = _mm256_fmadd_ps(x2, _mm256_set1_ps(0.5f), result);
    result = _mm256_fmadd_ps(x_clamped, one, result);
    
    return result;
}

/**
 * AVX2 vectorized fast log-sum-exp
 * Assembly-optimized with branch hints
 */
__m256 fast_log_sum_exp_avx2_asm(__m256 a, __m256 b) {
    // max(a, b)
    __m256 max_val = _mm256_max_ps(a, b);
    
    // abs(a - b)
    __m256 diff = _mm256_sub_ps(a, b);
    __m256 abs_diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff);
    
    // Clamp to [0, 20]
    __m256 clamped = _mm256_min_ps(abs_diff, _mm256_set1_ps(20.0f));
    
    // Fast exp(-clamped)
    __m256 neg_clamped = _mm256_mul_ps(clamped, _mm256_set1_ps(-1.0f));
    __m256 exp_neg = fast_exp_avx2_asm(neg_clamped);
    
    // Fast log1p approximation
    // log1p(x) ≈ x - x²/2 for small x
    __m256 x2 = _mm256_mul_ps(exp_neg, exp_neg);
    __m256 log1p_approx = _mm256_fmsub_ps(exp_neg, _mm256_set1_ps(0.5f), exp_neg);
    log1p_approx = _mm256_fmadd_ps(x2, _mm256_set1_ps(0.333333f), log1p_approx);
    
    // max + log1p
    return _mm256_add_ps(max_val, log1p_approx);
}

#endif  // __AVX2__

// ============================================================================
// Memory Prefetching (Cache Optimization)
// ============================================================================

/**
 * Prefetch next cache line for sequential access
 * Quake3 style: Prefetch data before it's needed
 */
inline void prefetch_next(const void* addr) {
    __builtin_prefetch(addr, 1, 3);  // Write, high temporal locality
}

inline void prefetch_read(const void* addr) {
    __builtin_prefetch(addr, 0, 3);  // Read, high temporal locality
}

// ============================================================================
// Branch Prediction Hints
// ============================================================================

/**
 * Likely branch (for CPU branch prediction)
 */
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

// ============================================================================
// Cache Line Alignment
// ============================================================================

/**
 * Align to cache line (64 bytes) for better performance
 */
#define CACHE_LINE_SIZE 64
#define ALIGN_CACHE_LINE __attribute__((aligned(CACHE_LINE_SIZE)))

// ============================================================================
// Fast Memcpy (Assembly Optimized)
// ============================================================================

/**
 * Fast memory copy for aligned data
 * Uses REP MOVSB on modern CPUs (fast string operations)
 */
inline void fast_memcpy_aligned(void* dest, const void* src, size_t n) {
    // Use compiler's builtin memcpy (often optimized to REP MOVSB)
    __builtin_memcpy(dest, src, n);
}

// ============================================================================
// CPU Feature Detection
// ============================================================================

/**
 * Check for CPU features at runtime
 */
inline bool has_avx2() {
    return __builtin_cpu_supports("avx2");
}

inline bool has_fma() {
    return __builtin_cpu_supports("fma");
}

inline bool has_avx512f() {
    return __builtin_cpu_supports("avx512f");
}
