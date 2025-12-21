#pragma once

#include <vector>

namespace mm_rec {

/**
 * Training Statistics Snapshot
 * (Pure Data derived from Domain)
 */
struct TrainingStats {
    int step = 0;
    int epoch = 0;
    int total_steps = 0;
    
    float loss = 0.0f;
    float current_lr = 0.0f;
    float speed_tps = 0.0f;
    float grad_norm = 0.0f;
    float data_stall_ms = 0.0f;
    float moe_loss = 0.0f;
    float mem_mb = 0.0f;
};

/**
 * Interface: Training Monitor
 * 
 * Allows the Domain Layer (Trainer) to report progress and receive
 * commands (stop signal) from the Application Layer without depending
 * on specific infrastructure (HttpServer, Console, etc.).
 */
class ITrainingMonitor {
public:
    virtual ~ITrainingMonitor() = default;

    /**
     * Report progress after a training step
     */
    virtual void on_step_complete(const TrainingStats& stats) = 0;
    
    /**
     * Check if the training should be aborted (User request)
     */
    virtual bool should_stop() = 0;
};

} // namespace mm_rec
