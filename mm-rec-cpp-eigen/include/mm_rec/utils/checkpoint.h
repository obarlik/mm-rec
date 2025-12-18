/**
 * Checkpoint Manager
 * 
 * Save/load model weights for training continuity
 */

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

class CheckpointManager {
public:
    /**
     * Save model checkpoint to binary file
     * 
     * Format:
     *   [MAGIC: "MMRC"] [VERSION: 4 bytes]
     *   [METADATA: epoch, loss, lr]
     *   [CONFIG: vocab, hidden, mem, ffn, layers]
     *   [EMBEDDING WEIGHTS]
     *   For each layer: [BLOCK WEIGHTS]
     * 
     * @param path Output file path
     * @param model MM-Rec model to save
     * @param metadata Training metadata
     */
    static void save_checkpoint(
        const std::string& path,
        const MMRecModel& model,
        const CheckpointMetadata& metadata
    );
    
    /**
     * Load model checkpoint from binary file
     * 
     * @param path Input file path
     * @param model MM-Rec model to load into (must have correct config)
     * @param metadata Training metadata output
     */
    static void load_checkpoint(
        const std::string& path,
        MMRecModel& model,
        CheckpointMetadata& metadata
    );
};

} // namespace mm_rec
