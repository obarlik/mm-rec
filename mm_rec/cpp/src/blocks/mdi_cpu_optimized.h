/**
 * MDI CPU Optimized - Header
 */

#ifndef MM_REC_MDI_CPU_H
#define MM_REC_MDI_CPU_H

/**
 * Fused MDI update with SIMD
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
);

#endif // MM_REC_MDI_CPU_H
