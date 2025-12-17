/**
 * Checkpoint Manager Implementation
 */

#include "mm_rec/utils/checkpoint.h"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cstdio> // std::rename

namespace mm_rec {

// Magic header for checkpoint files
static const char CHECKPOINT_MAGIC[4] = {'M', 'M', 'R', 'C'};
static const uint32_t CHECKPOINT_VERSION = 1;

void CheckpointManager::save_checkpoint(
    const std::string& path,
    const MMRecModel& model,
    const CheckpointMetadata& metadata
) {
    std::string tmp_path = path + ".tmp";
    std::ofstream file(tmp_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open temp checkpoint file for writing: " + tmp_path);
    }
    
    // Write magic and version
    file.write(CHECKPOINT_MAGIC, 4);
    file.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(uint32_t));
    
    // Write metadata
    int32_t epoch = metadata.epoch;
    file.write(reinterpret_cast<const char*>(&epoch), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&metadata.loss), sizeof(float));
    file.write(reinterpret_cast<const char*>(&metadata.learning_rate), sizeof(float));
    
    // Write config
    const auto& config = model.get_config();
    file.write(reinterpret_cast<const char*>(&config.vocab_size), sizeof(int64_t));
    file.write(reinterpret_cast<const char*>(&config.hidden_dim), sizeof(int64_t));
    file.write(reinterpret_cast<const char*>(&config.mem_dim), sizeof(int64_t));
    file.write(reinterpret_cast<const char*>(&config.ffn_dim), sizeof(int64_t));
    file.write(reinterpret_cast<const char*>(&config.num_layers), sizeof(int64_t));
    
    // Write embedding weights
    const auto& embedding = model.get_embedding_weights();
    int64_t emb_numel = embedding.numel();
    file.write(reinterpret_cast<const char*>(&emb_numel), sizeof(int64_t));
    file.write(
        reinterpret_cast<const char*>(embedding.data()),
        emb_numel * sizeof(float)
    );
    
    // Flush and close to ensure data is on disk (soft guarantee)
    file.flush();
    file.close();
    if (file.fail()) {
        throw std::runtime_error("Failed to write/close temp checkpoint: " + tmp_path);
    }
    
    // Atomic Rename
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("Failed to atomically rename checkpoint: " + tmp_path + " -> " + path);
    }
}

void CheckpointManager::load_checkpoint(
    const std::string& path,
    MMRecModel& model,
    CheckpointMetadata& metadata
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open checkpoint file for reading: " + path);
    }
    
    // Read and verify magic
    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, CHECKPOINT_MAGIC, 4) != 0) {
        throw std::runtime_error("Invalid checkpoint file (bad magic): " + path);
    }
    
    // Read and verify version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != CHECKPOINT_VERSION) {
        throw std::runtime_error("Unsupported checkpoint version: " + std::to_string(version));
    }
    
    // Read metadata
    int32_t epoch;
    file.read(reinterpret_cast<char*>(&epoch), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&metadata.loss), sizeof(float));
    file.read(reinterpret_cast<char*>(&metadata.learning_rate), sizeof(float));
    metadata.epoch = epoch;
    
    // Read config (verify matches)
    MMRecModelConfig file_config;
    file.read(reinterpret_cast<char*>(&file_config.vocab_size), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&file_config.hidden_dim), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&file_config.mem_dim), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&file_config.ffn_dim), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&file_config.num_layers), sizeof(int64_t));
    
    const auto& model_config = model.get_config();
    if (file_config.vocab_size != model_config.vocab_size ||
        file_config.hidden_dim != model_config.hidden_dim ||
        file_config.mem_dim != model_config.mem_dim ||
        file_config.ffn_dim != model_config.ffn_dim ||
        file_config.num_layers != model_config.num_layers) {
        throw std::runtime_error("Config mismatch between checkpoint and model");
    }
    
    // Read embedding weights
    int64_t emb_numel;
    file.read(reinterpret_cast<char*>(&emb_numel), sizeof(int64_t));
    
    auto& embedding = model.get_embedding_weights();
    if (emb_numel != embedding.numel()) {
        throw std::runtime_error("Embedding size mismatch");
    }
    
    file.read(
        reinterpret_cast<char*>(embedding.data()),
        emb_numel * sizeof(float)
    );
    
    // Note: Would load block weights here if we had access
    
    file.close();
}

} // namespace mm_rec
