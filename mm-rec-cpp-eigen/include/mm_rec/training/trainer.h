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
#include "mm_rec/infrastructure/http_server.h"

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
    Trainer(MMRecModel& model, const TrainingConfig& config);
    ~Trainer();
    
    /**
     * Train for one batch
     * Returns average loss for the batch
     */
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
    
    // Dashboard control
    bool should_stop() const { return stop_requested_; }
    void update_stats(float loss, float speed, float grad_norm, float lr, float data_stall_ms, float moe_loss, float mem_mb);
    
private:
    MMRecModel& model_;
    TrainingConfig config_;
    std::unique_ptr<LRScheduler> scheduler_;
    std::unique_ptr<Optimizer> optimizer_;
    int step_;
    
    std::unique_ptr<net::HttpServer> dashboard_server_;
    std::atomic<bool> stop_requested_{false};
    void setup_dashboard_handlers();
    
    // Stats for API (Protected by mutex)
    mutable std::mutex stats_mutex_;
    std::deque<float> loss_history_;
    std::deque<float> avg_loss_history_; // EMA History
    std::deque<float> grad_norm_history_;
    std::deque<float> lr_history_;
    std::deque<float> data_stall_history_;
    std::deque<float> moe_loss_history_;
    std::deque<float> mem_history_;
    
    float current_speed_ = 0.0f;
    float current_ema_ = 0.0f; // Helper for calculation
};

} // namespace mm_rec
