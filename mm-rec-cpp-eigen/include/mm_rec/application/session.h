/**
 * Session Manager
 * 
 * Save/load memory states for session continuity
 */

#pragma once

#include "mm_rec/core/tensor.h"
#include <vector>
#include <string>

namespace mm_rec {

class SessionManager {
public:
    /**
     * Save memory states to binary file
     * 
     * Format:
     *   [MAGIC: "MMRS"] [VERSION: 4 bytes] [NUM_LAYERS: 4 bytes]
     *   For each layer:
     *     [NDIM: 4 bytes] [SHAPE: 8*ndim bytes] [DATA: 4*numel bytes]
     * 
     * @param path Output file path
     * @param memory_states Vector of memory tensors (one per layer)
     */
    static void save_session(
        const std::string& path,
        const std::vector<Tensor>& memory_states
    );
    
    /**
     * Load memory states from binary file
     * 
     * @param path Input file path
     * @return Vector of memory tensors
     */
    static std::vector<Tensor> load_session(const std::string& path);
};

} // namespace mm_rec
