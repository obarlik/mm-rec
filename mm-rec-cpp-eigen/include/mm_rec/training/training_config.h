#pragma once

#include <string>
#include <vector>
#include "mm_rec/core/tensor.h"

namespace mm_rec {

/**
 * Simple dataset interface (placeholder for now)
 */
struct TrainingBatch {
    Tensor input_ids;   // [batch, seq]
    Tensor targets;     // [batch, seq]
    Tensor loss_mask;   // [batch, seq] - optional, 1=compute loss, 0=ignore
};

/**
 * Training configuration
 */
struct TrainingConfig {
    float learning_rate = 1e-4f;
    int batch_size = 16;
    int num_epochs = 10;
    int validate_every = 100;
    float grad_clip_norm = 1.0f;
    std::string checkpoint_dir = "checkpoints/";
    int warmup_steps = 100;

    int total_steps = 10000;
    std::string optimizer_type = "sgd"; // sgd, adam, adamw
    float weight_decay = 0.01f;
    
    // Adaptive Flux
    float easy_threshold = 1.0f;
    float hard_threshold = 100.0f;
};

} // namespace mm_rec
