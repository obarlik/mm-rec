/**
 * Learning Rate Scheduler
 * 
 * Manages learning rate during training
 */

#pragma once

#include <cstdint>

namespace mm_rec {

/**
 * Base LR Scheduler interface
 */
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float get_lr(int step) const = 0;
};

/**
 * Cosine Annealing with Linear Warmup
 * 
 * Schedule:
 *   step < warmup: lr = base_lr * (step / warmup)
 *   step >= warmup: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
 *   
 *   where progress = (step - warmup) / (total - warmup)
 */
class CosineScheduler : public LRScheduler {
public:
    CosineScheduler(
        float base_lr,
        int warmup_steps,
        int total_steps,
        float min_lr = 0.0f
    );
    
    float get_lr(int step) const override;
    
private:
    float base_lr_;
    float min_lr_;
    int warmup_steps_;
    int total_steps_;
};

} // namespace mm_rec
