/**
 * Trainer
 * 
 * Training loop orchestration
 */

#pragma once

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/training/scheduler.h"
#include "mm_rec/training/optimizer.h"
#include <string>
#include <memory>
#include <deque>
#include <atomic>
#include <mutex>
#include "mm_rec/application/i_training_monitor.h"

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

/**
 * Trainer orchestrates training loop
 */
class Trainer {
public:
    Trainer(MMRecModel& model, const TrainingConfig& config, ITrainingMonitor& monitor);
    ~Trainer();
    
    /**
     * Train for one batch
     * Returns average loss for the batch
     */
    float train_step(const TrainingBatch& batch, float data_stall_ms = 0.0f, float speed_tps = 0.0f, float mem_mb = 0.0f);
    
    /**
     * Forward pass only (no backward, for difficulty assessment)
     * Returns loss value for sample categorization
     */
    float forward_only(const TrainingBatch& batch);
    
    /**
     * Vectorized forward pass for difficulty probing
     * Returns [Batch] tensor of loss values
     */
    Tensor forward_vectorized(const TrainingBatch& batch);
    
    /**
     * Validate on one batch
     * Returns loss value
     */
    float validate_step(const TrainingBatch& batch);
    
    /**
     * Get current learning rate
     */
    float get_current_lr() const;
    
    /**
     * Step counter for scheduler
     */
    void increment_step() { step_++; }
    int get_step() const { return step_; }
    
    
    Optimizer* get_optimizer() { return optimizer_.get(); }
    
    // Update base learning rate (needed for Auto-Rollback)
    void update_learning_rate(float new_lr) {
        if (scheduler_) scheduler_->set_base_lr(new_lr);
        if (optimizer_) optimizer_->set_lr(new_lr); // Immediate effect
    }
    
    // Dashboard control (Proxy to Monitor)
    bool should_stop() const;
    
private:
    MMRecModel& model_;
    TrainingConfig config_;
    ITrainingMonitor& training_monitor_; // Injected Dependency
    
    std::unique_ptr<LRScheduler> scheduler_;
    std::unique_ptr<Optimizer> optimizer_;
    int step_;
};

} // namespace mm_rec
