/**
 * Learning Rate Scheduler Implementation
 */

#include "mm_rec/training/scheduler.h"
#include <cmath>
#include <algorithm>

namespace mm_rec {

CosineScheduler::CosineScheduler(
    float base_lr,
    int warmup_steps,
    int total_steps,
    float min_lr
) : base_lr_(base_lr),
    min_lr_(min_lr),
    warmup_steps_(warmup_steps),
    total_steps_(total_steps)
{
}

float CosineScheduler::get_lr(int step) const {
    // Linear warmup
    if (step < warmup_steps_) {
        return base_lr_ * (static_cast<float>(step) / warmup_steps_);
    }
    
    // Cosine decay
    float progress = static_cast<float>(step - warmup_steps_) / 
                     (total_steps_ - warmup_steps_);
    progress = std::min(progress, 1.0f);  // Cap at 1.0
    
    const float pi = 3.14159265359f;
    float cosine_factor = 0.5f * (1.0f + std::cos(pi * progress));
    
    return min_lr_ + (base_lr_ - min_lr_) * cosine_factor;
}

} // namespace mm_rec
