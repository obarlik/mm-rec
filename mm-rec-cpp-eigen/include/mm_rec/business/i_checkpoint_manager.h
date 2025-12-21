#pragma once

#include "mm_rec/model/mm_rec_model.h"
#include <string>

namespace mm_rec {

struct CheckpointMetadata {
    int epoch = 0;
    int64_t batch_idx = 0; // Steps within epoch
    float loss = 0.0f;
    float learning_rate = 0.0f;
};

class ICheckpointManager {
public:
    virtual ~ICheckpointManager() = default;

    /**
     * Save model checkpoint to binary file
     */
    virtual void save_checkpoint(
        const std::string& path,
        const MMRecModel& model,
        const CheckpointMetadata& metadata
    ) = 0;
    
    /**
     * Load model checkpoint from binary file
     */
    virtual void load_checkpoint(
        const std::string& path,
        MMRecModel& model,
        CheckpointMetadata& metadata
    ) = 0;
};

} // namespace mm_rec
