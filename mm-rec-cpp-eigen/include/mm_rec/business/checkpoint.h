/**
 * Checkpoint Manager
 * 
 * Save/load model weights for training continuity
 */

#pragma once

#include "mm_rec/business/i_checkpoint_manager.h"
#include "mm_rec/model/mm_rec_model.h"
#include <string>

namespace mm_rec {

// CheckpointMetadata is defined in i_checkpoint_manager.h

class CheckpointManager : public ICheckpointManager {
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
    void save_checkpoint(
        const std::string& path,
        const MMRecModel& model,
        const CheckpointMetadata& metadata
    ) override;
    
    /**
     * Load model checkpoint from binary file
     * 
     * @param path Input file path
     * @param model MM-Rec model to load into (must have correct config)
     * @param metadata Training metadata output
     */
    void load_checkpoint(
        const std::string& path,
        MMRecModel& model,
        CheckpointMetadata& metadata
    ) override;
};

} // namespace mm_rec
