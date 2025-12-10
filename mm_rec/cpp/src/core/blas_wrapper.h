/**
 * BLAS Wrapper - MKL/OpenBLAS Support
 * 
 * Provides unified interface for BLAS operations:
 * - MKL (Intel Math Kernel Library) - if available
 * - OpenBLAS - if available
 * - Manual SIMD fallback - if neither available
 */

#ifndef BLAS_WRAPPER_H
#define BLAS_WRAPPER_H

#include <cstddef>

// Try to use MKL first, then OpenBLAS, then fallback
#ifdef USE_MKL
    #include <mkl.h>
    #define BLAS_AVAILABLE 1
    #define BLAS_TYPE_MKL 1
#elif defined(USE_OPENBLAS)
    extern "C" {
        #include <cblas.h>
    }
    #define BLAS_AVAILABLE 1
    #define BLAS_TYPE_OPENBLAS 1
#else
    #define BLAS_AVAILABLE 0
    #define BLAS_TYPE_MANUAL 1
#endif

/**
 * Optimized sgemv (single-precision matrix-vector multiply)
 * y = alpha * A * x + beta * y
 * 
 * Uses MKL/OpenBLAS if available, otherwise falls back to manual SIMD
 */
void optimized_sgemv(
    int m, int n,
    float alpha,
    const float* A, int lda,
    const float* x, int incx,
    float beta,
    float* y, int incy
);

#endif // BLAS_WRAPPER_H
