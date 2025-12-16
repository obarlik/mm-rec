/**
 * Trainer Implementation
 */

#include "mm_rec/training/trainer.h"
#include "mm_rec/training/uboo_loss.h"
#include "mm_rec/training/gradient_utils.h"
#include "mm_rec/training/gradient_accum.h"
#include "mm_rec/training/forward_cache.h"
#include "mm_rec/training/optimizer.h"
#include <iostream>

namespace mm_rec {

Trainer::Trainer(MMRecModel& model, const TrainingConfig& config)
    : model_(model),
      config_(config),
      step_(0)
{
    // Create learning rate scheduler
    scheduler_ = std::make_unique<CosineScheduler>(
        config.learning_rate,
        config.warmup_steps,
        config.total_steps,
        config.learning_rate * 0.01f  // min_lr
    );
    
    // Create optimizer
    optimizer_ = std::make_unique<SGD>(config.learning_rate);
}

    // 44-50 original
float Trainer::train_step(const TrainingBatch& batch) {
    // CRITICAL: Process each sequence INDEPENDENTLY for true isolation
    // This is essential for instruction tuning where samples must not influence each other
    
    int64_t batch_size = batch.input_ids.size(0);
    int64_t seq_len = batch.input_ids.size(1);
    
    // Accumulate gradients across batch
    ModelGradients accumulated_grads;
    float total_loss = 0.0f;
    int valid_samples = 0;
    
    // Process each sequence independently
    for (int64_t b = 0; b < batch_size; ++b) {
        // Extract single sequence
        Tensor single_input = Tensor::zeros({1, seq_len});
        Tensor single_target = Tensor::zeros({1, seq_len});
        Tensor single_mask = Tensor::zeros({1, seq_len});
        
        for (int64_t s = 0; s < seq_len; ++s) {
            single_input.data()[s] = batch.input_ids.data()[b * seq_len + s];
            single_target.data()[s] = batch.targets.data()[b * seq_len + s];
            if (batch.loss_mask.numel() > 0) {
                single_mask.data()[s] = batch.loss_mask.data()[b * seq_len + s];
            }
        }
        
        // RESET MEMORY for this sequence (complete isolation!)
        model_.reset_memory(1);
        
        // Forward pass for this sequence
        ForwardCache cache;
        Tensor logits = model_.forward(single_input, &cache);
        
        // Compute loss
        Tensor loss_tensor = compute_uboo_loss(logits, single_target, single_mask);
        float loss = loss_tensor.item();
        total_loss += loss;
        
        // Backward pass
        ModelGradients grads = model_.backward(single_target, cache);
        
        // Gradient clipping per sample
        clip_gradients_by_norm(grads, config_.grad_clip_norm);
        
        // Accumulate gradients
        if (b == 0) {
            accumulated_grads = grads;
        } else {
            accumulate_gradients(accumulated_grads, grads);
        }
        
        valid_samples++;
    }
    
    // Average loss and gradients
    float avg_loss = total_loss / valid_samples;
    scale_gradients(accumulated_grads, 1.0f / valid_samples);
    
    // Update parameters with accumulated gradients
    float current_lr = scheduler_->get_lr(step_);
    optimizer_->set_lr(current_lr);
    model_.update_parameters(*optimizer_, accumulated_grads);
    
    step_++;
    
    return avg_loss;
}

float Trainer::validate_step(const TrainingBatch& batch) {
    // No gradient computation in validation
    Tensor logits = model_.forward(batch.input_ids);
    Tensor loss_tensor = compute_uboo_loss(logits, batch.targets);
    
    return loss_tensor.item();
}

float Trainer::get_current_lr() const {
    return scheduler_->get_lr(step_);
}

} // namespace mm_rec
