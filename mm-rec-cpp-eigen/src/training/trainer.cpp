/**
 * Trainer Implementation
 */

#include "mm_rec/training/trainer.h"
#include "mm_rec/training/uboo_loss.h"
#include "mm_rec/training/gradient_utils.h"
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
        config.learning_rate * 0.01f  // min_lr = 1% of base
    );
}

float Trainer::train_step(const TrainingBatch& batch) {
    // 1. Forward pass
    Tensor logits = model_.forward(batch.input_ids);
    
    // 2. Compute loss (UBOO deep supervision)
    Tensor loss_tensor = compute_uboo_loss(logits, batch.targets);
    float loss = loss_tensor.item();  // Extract scalar
    
    // 3. Backward pass (STUB - no actual gradients yet)
    // std::vector<Tensor> gradients = compute_gradients(loss);
    
    // 4. Gradient clipping (STUB - would clip real gradients)
    // clip_gradients_by_norm(gradients, config_.grad_clip_norm);
    
    // 5. Optimizer step (STUB - would update weights)
    // float lr = get_current_lr();
    // optimizer.step(gradients, lr);
    
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
