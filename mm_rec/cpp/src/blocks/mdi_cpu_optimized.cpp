/**
 * MDI (Memory Decay/Integration) - CPU Optimized
 * 
 * Optimized MDI operations with SIMD
 * - Gated integration
 * - Decay computation
 * - Element-wise operations
 */

#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// MDI Update Fused (SIMD)
// ============================================================================

/**
 * Fused MDI update with SIMD
 * h_updated = gate ⊙ h_new + (1 - gate) ⊙ h_old + γ ⊙ h_old
 * 
 * @param h_new: New memory state [batch, seq_len, model_dim]
 * @param h_old: Old memory state [batch, seq_len, model_dim]
 * @param gamma: Decay coefficients [batch, seq_len, model_dim]
 * @param gate: Integration gate [batch, seq_len, model_dim]
 * @param h_updated: Output [batch, seq_len, model_dim]
 * @param batch_size: Batch size
 * @param seq_len: Sequence length
 * @param model_dim: Model dimension
 * @param num_threads: OpenMP threads (0 = auto)
 */
void mdi_update_fused_simd(
    const float* h_new,
    const float* h_old,
    const float* gamma,
    const float* gate,
    float* h_updated,
    int batch_size,
    int seq_len,
    int model_dim,
    int num_threads = 0
) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Parallelize across batch and sequence
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int base_idx = b * seq_len * model_dim + t * model_dim;
            
            // SIMD element-wise operations
            #if defined(__AVX512F__) && defined(__AVX512DQ__)
            // AVX-512: 16 floats at once (if fully supported)
            for (int d = 0; d < model_dim - 15; d += 16) {
                __m512 vnew = _mm512_loadu_ps(&h_new[base_idx + d]);
                __m512 vold = _mm512_loadu_ps(&h_old[base_idx + d]);
                __m512 vgate = _mm512_loadu_ps(&gate[base_idx + d]);
                __m512 vgamma = _mm512_loadu_ps(&gamma[base_idx + d]);
                
                // gate ⊙ h_new
                __m512 vgnew = _mm512_mul_ps(vgate, vnew);
                
                // (1 - gate) ⊙ h_old
                __m512 vone = _mm512_set1_ps(1.0f);
                __m512 voneg = _mm512_sub_ps(vone, vgate);
                __m512 voneg_old = _mm512_mul_ps(voneg, vold);
                
                // γ ⊙ h_old
                __m512 vg_old = _mm512_mul_ps(vgamma, vold);
                
                // Sum: gate ⊙ h_new + (1 - gate) ⊙ h_old + γ ⊙ h_old
                __m512 vresult = _mm512_add_ps(vgnew, voneg_old);
                vresult = _mm512_add_ps(vresult, vg_old);
                
                _mm512_storeu_ps(&h_updated[base_idx + d], vresult);
            }
            // Handle remaining
            for (int d = (model_dim / 16) * 16; d < model_dim; d++) {
                h_updated[base_idx + d] = 
                    gate[base_idx + d] * h_new[base_idx + d] +
                    (1.0f - gate[base_idx + d]) * h_old[base_idx + d] +
                    gamma[base_idx + d] * h_old[base_idx + d];
            }
            #elif defined(__AVX2__)
            // AVX2: 8 floats at once
            for (int d = 0; d < model_dim - 7; d += 8) {
                __m256 vnew = _mm256_loadu_ps(&h_new[base_idx + d]);
                __m256 vold = _mm256_loadu_ps(&h_old[base_idx + d]);
                __m256 vgate = _mm256_loadu_ps(&gate[base_idx + d]);
                __m256 vgamma = _mm256_loadu_ps(&gamma[base_idx + d]);
                
                __m256 vgnew = _mm256_mul_ps(vgate, vnew);
                __m256 vone = _mm256_set1_ps(1.0f);
                __m256 voneg = _mm256_sub_ps(vone, vgate);
                __m256 voneg_old = _mm256_mul_ps(voneg, vold);
                __m256 vg_old = _mm256_mul_ps(vgamma, vold);
                
                __m256 vresult = _mm256_add_ps(vgnew, voneg_old);
                vresult = _mm256_add_ps(vresult, vg_old);
                
                _mm256_storeu_ps(&h_updated[base_idx + d], vresult);
            }
            for (int d = (model_dim / 8) * 8; d < model_dim; d++) {
                h_updated[base_idx + d] = 
                    gate[base_idx + d] * h_new[base_idx + d] +
                    (1.0f - gate[base_idx + d]) * h_old[base_idx + d] +
                    gamma[base_idx + d] * h_old[base_idx + d];
            }
            #else
            // Scalar fallback
            for (int d = 0; d < model_dim; d++) {
                h_updated[base_idx + d] = 
                    gate[base_idx + d] * h_new[base_idx + d] +
                    (1.0f - gate[base_idx + d]) * h_old[base_idx + d] +
                    gamma[base_idx + d] * h_old[base_idx + d];
            }
            #endif
        }
    }
}
