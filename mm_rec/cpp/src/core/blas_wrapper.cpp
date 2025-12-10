/**
 * BLAS Wrapper Implementation
 */

#include "blas_wrapper.h"
#include <immintrin.h>
#include <algorithm>

// Export function for linking
__attribute__((visibility("default")))

#if BLAS_AVAILABLE
    #if BLAS_TYPE_MKL
        // MKL implementation
        void optimized_sgemv(
            int m, int n,
            float alpha,
            const float* A, int lda,
            const float* x, int incx,
            float beta,
            float* y, int incy
        ) {
            cblas_sgemv(
                CblasRowMajor,
                CblasNoTrans,
                m, n,
                alpha,
                A, lda,
                x, incx,
                beta,
                y, incy
            );
        }
    #elif BLAS_TYPE_OPENBLAS
        // OpenBLAS implementation
        void optimized_sgemv(
            int m, int n,
            float alpha,
            const float* A, int lda,
            const float* x, int incx,
            float beta,
            float* y, int incy
        ) {
            cblas_sgemv(
                CblasRowMajor,
                CblasNoTrans,
                m, n,
                alpha,
                A, lda,
                x, incx,
                beta,
                y, incy
            );
        }
    #endif
#else
    // Manual SIMD fallback (from core_recurrence_fused.cpp)
    inline void manual_sgemv_rowmajor(
        int m, int n, float alpha,
        const float* A, int lda,
        const float* x, int incx,
        float beta, float* y, int incy
    ) {
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
        // Scalar fallback
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += A[i * lda + j] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
        #endif
    }
    
    void optimized_sgemv(
        int m, int n,
        float alpha,
        const float* A, int lda,
        const float* x, int incx,
        float beta,
        float* y, int incy
    ) {
        manual_sgemv_rowmajor(m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
#endif
