/**
 * UBOO Loss Computation
 * 
 * Universal Basis for Output Optimization
 * Every layer predicts, weighted combination
 */

#pragma once

#include "mm_rec/core/tensor.h"

namespace mm_rec {

/**
 * Compute UBOO loss from all layer predictions
 * 
 * Loss = 0.5 * final_loss + 0.5 * mean(auxiliary_losses)
 * 
 * @param all_layer_logits [num_layers, batch, seq, vocab]
 * @param targets [batch, seq] - target token IDs
 * @return Total weighted loss
 */
Tensor compute_uboo_loss(const Tensor& all_layer_logits, const Tensor& targets);

/**
 * Cross-entropy loss for a single layer
 * 
 * @param logits [batch, seq, vocab]
 * @param targets [batch, seq]
 * @return Scalar loss
 */
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);

} // namespace mm_rec
