/**
 * Session Manager Implementation
 */

#include "mm_rec/application/session.h"
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace mm_rec {

// Magic header for session files
static const char MAGIC[4] = {'M', 'M', 'R', 'S'};
static const uint32_t VERSION = 1;

void SessionManager::save_session(
    const std::string& path,
    const std::vector<Tensor>& memory_states
) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open session file for writing: " + path);
    }
    
    // Write magic and version
    file.write(MAGIC, 4);
    file.write(reinterpret_cast<const char*>(&VERSION), sizeof(uint32_t));
    
    // Write number of layers
    uint32_t num_layers = static_cast<uint32_t>(memory_states.size());
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(uint32_t));
    
    // Write each layer's memory
    for (const auto& memory : memory_states) {
        // Write ndim
        uint32_t ndim = static_cast<uint32_t>(memory.ndim());
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));
        
        // Write shape
        auto shape = memory.sizes();
        for (auto dim : shape) {
            file.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
        }
        
        // Write data
        int64_t numel = memory.numel();
        file.write(
            reinterpret_cast<const char*>(memory.data()),
            numel * sizeof(float)
        );
    }
    
    file.close();
}

std::vector<Tensor> SessionManager::load_session(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open session file for reading: " + path);
    }
    
    // Read and verify magic
    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, MAGIC, 4) != 0) {
        throw std::runtime_error("Invalid session file (bad magic): " + path);
    }
    
    // Read and verify version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != VERSION) {
        throw std::runtime_error("Unsupported session version: " + std::to_string(version));
    }
    
    // Read number of layers
    uint32_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));
    
    std::vector<Tensor> memory_states;
    memory_states.reserve(num_layers);
    
    // Read each layer's memory
    for (uint32_t i = 0; i < num_layers; ++i) {
        // Read ndim
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));
        
        // Read shape
        std::vector<int64_t> shape(ndim);
        for (uint32_t j = 0; j < ndim; ++j) {
            file.read(reinterpret_cast<char*>(&shape[j]), sizeof(int64_t));
        }
        
        // Create tensor
        Tensor memory = Tensor::zeros(shape);
        
        // Read data
        int64_t numel = memory.numel();
        file.read(
            reinterpret_cast<char*>(memory.data()),
            numel * sizeof(float)
        );
        
        memory_states.push_back(memory);
    }
    
    file.close();
    return memory_states;
}

} // namespace mm_rec
