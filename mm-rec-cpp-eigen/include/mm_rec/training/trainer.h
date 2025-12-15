/**
 * Trainer
 * 
 * Training loop orchestration
 */

#pragma once

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/training/scheduler.h"
#include <string>
#include <memory>

namespace mm_rec {

/**
 * Simple dataset interface (placeholder for now)
 */
struct TrainingBatch {
    Tensor input_ids;   // [batch, seq]
    Tensor targets;     // [batch, seq]
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
};

/**
 * Trainer orchestrates training loop
 */
class Trainer {
public:
    Trainer(MMRecModel& model, const TrainingConfig& config);
    
    /**
     * Train for one batch
     * Returns loss value
     */
    float train_step(const TrainingBatch& batch);
    
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
    
private:
    MMRecModel& model_;
    TrainingConfig config_;
    std::unique_ptr<LRScheduler> scheduler_;
    int step_;
};

} // namespace mm_rec
