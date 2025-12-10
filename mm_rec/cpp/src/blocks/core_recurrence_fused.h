/**
 * Core Recurrence Fused Kernel - Header
 */

#ifndef MM_REC_CORE_RECURRENCE_FUSED_H
#define MM_REC_CORE_RECURRENCE_FUSED_H

/**
 * Fused kernel for core recurrence formula
 * h_t = z_t ⊙ σ(W_g @ h_{t-1}) + γ ⊙ h_{t-1}
 */
void core_recurrence_fused_cpu(
    const float* z_t,
    const float* h_prev,
    const float* W_g,
    const float* gamma,
    float* h_t,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_threads = 0
);

#endif // MM_REC_CORE_RECURRENCE_FUSED_H
