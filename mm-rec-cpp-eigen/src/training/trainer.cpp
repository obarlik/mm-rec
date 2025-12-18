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
#include <cstring>  // memcpy
#include <cmath>
#include "mm_rec/core/memory_manager.h"
#include <atomic> // Used for total_loss
#include <omp.h> // Good practice

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
    if (config.optimizer_type == "adamw") {
        std::cout << "Creating AdamW Optimizer (LR=" << config.learning_rate << ", WD=" << config.weight_decay << ")" << std::endl;
        optimizer_ = std::make_unique<AdamW>(config.learning_rate, 0.9f, 0.999f, 1e-8f, config.weight_decay);
    } else if (config.optimizer_type == "adam") {
         std::cout << "Creating Adam Optimizer (LR=" << config.learning_rate << ")" << std::endl;
        optimizer_ = std::make_unique<Adam>(config.learning_rate);
    } else {
         std::cout << "Creating SGD Optimizer (LR=" << config.learning_rate << ")" << std::endl;
        optimizer_ = std::make_unique<SGD>(config.learning_rate);
    }
}

    // 44-50 original
// Vectorized training step (Full Batch GPU Offload)
float Trainer::train_step(const TrainingBatch& batch) {
    int64_t batch_size = batch.input_ids.size(0);
    int64_t seq_len = batch.input_ids.size(1);
    
    // 1. Mark persistent memory start
    // We want gradients to survive until update_parameters
    MemoryManager::instance().mark_persistent();
    
    // 2. Forward Pass (Full Batch)
    // The Linear layer will see [batch, in_features] and dispatch to GPU if batch is large enough.
    // With batch=8, it goes to GPU (since Threshold=0).
    ForwardCache cache;
    Tensor logits = model_.forward(batch.input_ids, &cache);
    
    // 3. Loss Calculation (Full Batch)
    // UBOO loss supports batching naturally
    Tensor loss_tensor = compute_uboo_loss(logits, batch.targets, batch.loss_mask);
    float loss = 0.0f;
    for(int i=0; i<loss_tensor.size(0); ++i) loss += loss_tensor.data()[i];
    if(loss_tensor.size(0) > 0) loss /= loss_tensor.size(0);
    
    // Add Aux Loss from MoE
    float total_step_loss = loss + cache.total_aux_loss * 0.01f;
    
    // 4. Backward Pass (Full Batch)
    // Gradients are computed for the whole batch at once
    ModelGradients grads = model_.backward(batch.targets, cache);
    
    // 5. Gradient Clipping
    clip_gradients_by_norm(grads, config_.grad_clip_norm);
    
    // 6. Update Parameters
    // Scale gradients by 1.0 (already averaged in backward/loss)?
    // Usually loss is average, so gradients are average.
    // Check compute_uboo_loss: returns average.
    // Check linear_backward: db sums over batch. dW sums over batch.
    // If loss is avg, then dL/dy is 1/N.
    // So gradients are 1/N * sum(grads). Correct.
    
    float current_lr = scheduler_->get_lr(step_);
    optimizer_->set_lr(current_lr);
    model_.update_parameters(*optimizer_, grads);
    
    step_++;
    
    // 7. Cleanup
    // Free everything allocated during this step (logits, cache, grads)
    // EXCEPT parameters which are in a deeper scope or marked differently?
    // No, parameters are members of Model, not in Arena.
    // Intermediates are in Arena.
    // We marked persistent at start? No, mark_persistent saves STATE.
    // reset_arena rewinds to that state.
    // Wait, mark_persistent establishes a "save point".
    // If we call reset_arena(), it wipes everything AFTER mark_persistent.
    // But we just updated parameters (which are safe).
    // So we can wipe everything.
    
    // Unmark to clear the save point?
    MemoryManager::instance().clear_persistent();
    MemoryManager::instance().reset_arena();

    return total_step_loss;
}

float Trainer::forward_only(const TrainingBatch& batch) {
    // Forward pass only for difficulty assessment (no backward!)
    int64_t batch_size = batch.input_ids.size(0);
    int64_t seq_len = batch.input_ids.size(1);
    
    float total_loss = 0.0f;
    int valid_samples = 0;
    
    // Pre-allocate buffers
    Tensor single_input = Tensor::zeros({1, seq_len});
    Tensor single_target = Tensor::zeros({1, seq_len});
    Tensor single_mask = Tensor::zeros({1, seq_len});
    size_t copy_size = seq_len * sizeof(float);
    
    // Process each sequence independently (same as train_step but no backward)
    for (int64_t b = 0; b < batch_size; ++b) {
        // Fast copy
        std::memcpy(single_input.data(), batch.input_ids.data() + b * seq_len, copy_size);
        std::memcpy(single_target.data(), batch.targets.data() + b * seq_len, copy_size);
        
        if (batch.loss_mask.numel() > 0) {
            std::memcpy(single_mask.data(), batch.loss_mask.data() + b * seq_len, copy_size);
        }
        
        // Reset memory for isolation
        model_.reset_memory(1);
        
        // Forward pass only
        ForwardCache cache;
        Tensor logits = model_.forward(single_input, &cache);
        
        // Compute loss
        Tensor loss_tensor = compute_uboo_loss(logits, single_target, single_mask);
        float loss = loss_tensor.item();
        total_loss += loss;
        valid_samples++;

        // Fix: Reset Arena to prevent accumulation
        MemoryManager::instance().reset_arena();
    }
    
    return total_loss / valid_samples;
}

Tensor Trainer::forward_vectorized(const TrainingBatch& batch) {
    // Forward pass only for difficulty assessment (no backward!)
    // Returns [Batch] loss tensor
    
    // Reset Memory just in case
    MemoryManager::instance().reset_arena();
    
    ForwardCache cache;
    Tensor logits = model_.forward(batch.input_ids, &cache);
    
    // Compute loss per sample
    Tensor loss_tensor = compute_uboo_loss(logits, batch.targets, batch.loss_mask);
    
    // We must return a copy because reset_arena might wipe it if not careful?
    // The caller is responsible for MemoryManager if they want to keep it?
    // Actually, cmd_train calls reset_arena after decision.
    // So returning this Tensor (allocated in Arena) is fine as long as caller uses it before reset.
    
    return loss_tensor;
}

float Trainer::validate_step(const TrainingBatch& batch) {
    // No gradient computation in validation
    Tensor logits = model_.forward(batch.input_ids);
    Tensor loss_tensor = compute_uboo_loss(logits, batch.targets);
    
    float avg_loss = 0.0f;
    for(int i=0; i<loss_tensor.size(0); ++i) avg_loss += loss_tensor.data()[i];
    if(loss_tensor.size(0) > 0) avg_loss /= loss_tensor.size(0);
    
    return avg_loss;
}

float Trainer::get_current_lr() const {
    return scheduler_->get_lr(step_);
}

} // namespace mm_rec
