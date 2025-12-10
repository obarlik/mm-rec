/**
 * Blelloch Parallel Scan - Header
 */

#ifndef MM_REC_BLELLOCH_SCAN_H
#define MM_REC_BLELLOCH_SCAN_H

/**
 * Work-efficient parallel scan using Blelloch algorithm
 * Operator: Log-Sum-Exp (for exponential product)
 */
void blelloch_scan_parallel_log_sum_exp(
    float* input,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int dim,
    int num_threads = 0
);

#endif // MM_REC_BLELLOCH_SCAN_H
