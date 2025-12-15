/**
 * Model Configuration
 * 
 * Centralized configuration for MM-Rec model
 */

#pragma once

#include <cstdint>
#include <string>

namespace mm_rec {

struct ModelConfig {
    // Architecture
    int64_t vocab_size = 50000;
    int64_t hidden_dim = 512;
    int64_t mem_dim = 512;
    int64_t ffn_dim = 2048;
    int64_t num_layers = 12;
    
    // Training
    float learning_rate = 1e-4f;
    int64_t batch_size = 16;
    int64_t max_seq_len = 512;
    
    // UBOO
    float uboo_weight = 0.5f;  // 0.5 final + 0.5 auxiliary
    
    // Paths
    std::string checkpoint_path = "checkpoint.bin";
    std::string vocab_path = "vocab.txt";
    
    // Load from file (simple key=value format)
    static ModelConfig from_file(const std::string& path);
    
    // Save to file
    void save(const std::string& path) const;
    
    // Print configuration
    void print() const;
};

} // namespace mm_rec
