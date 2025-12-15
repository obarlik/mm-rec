/**
 * Trainer Implementation
 */

#include "mm_rec/training/trainer.h"
#include "mm_rec/training/uboo_loss.h"
#include "mm_rec/training/uboo_loss.h"
#include "mm_rec/training/gradient_utils.h" // Might not be needed if using direct backward
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
    // 0. Prepare cache
    // We reuse a thread-local or member cache to avoid reallocating
    // But for simplicity, let's create one on stack/heap
    // Better: Make cache a member of Trainer?
    // For now, create locally to ensure correctness.
    ForwardCache cache;
    
    // 1. Forward pass
    // Populates cache with activations
    Tensor logits = model_.forward(batch.input_ids, &cache);
    
    // 2. Compute loss (UBOO deep supervision)
    Tensor loss_tensor = compute_uboo_loss(logits, batch.targets);
    float loss = loss_tensor.item();
    
    // 3. Backward pass
    // Returns gradients for all parameters
    ModelGradients grads = model_.backward(batch.targets, cache);
    
    // 4. Gradient clipping (TODO: Implement clipping)
    // clip_gradients(grads, config_.grad_clip_norm);
    
    // 5. Update parameters
    float current_lr = scheduler_->get_lr(step_);
    optimizer_->set_lr(current_lr);
    
    model_.update_parameters(*optimizer_, grads);
    
    // 6. Step management
    step_++;
    
    return loss;
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
