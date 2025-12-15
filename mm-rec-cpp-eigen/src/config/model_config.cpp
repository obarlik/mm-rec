/**
 * Model Configuration Implementation
 */

#include "mm_rec/config/model_config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace mm_rec {

ModelConfig ModelConfig::from_file(const std::string& path) {
    ModelConfig config;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        // Parse key=value
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Set config values
        if (key == "vocab_size") config.vocab_size = std::stoll(value);
        else if (key == "hidden_dim") config.hidden_dim = std::stoll(value);
        else if (key == "mem_dim") config.mem_dim = std::stoll(value);
        else if (key == "ffn_dim") config.ffn_dim = std::stoll(value);
        else if (key == "num_layers") config.num_layers = std::stoll(value);
        else if (key == "learning_rate") config.learning_rate = std::stof(value);
        else if (key == "num_experts") config.num_experts = std::stoll(value);
        else if (key == "top_k") config.top_k = std::stoll(value);
        else if (key == "batch_size") config.batch_size = std::stoll(value);
        else if (key == "max_seq_len") config.max_seq_len = std::stoll(value);
        else if (key == "uboo_weight") config.uboo_weight = std::stof(value);
        else if (key == "checkpoint_path") config.checkpoint_path = value;
        else if (key == "vocab_path") config.vocab_path = value;
    }
    
    return config;
}

void ModelConfig::save(const std::string& path) const {
    std::ofstream file(path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file for writing: " + path);
    }
    
    file << "# MM-Rec Model Configuration\n\n";
    file << "# Architecture\n";
    file << "vocab_size=" << vocab_size << "\n";
    file << "hidden_dim=" << hidden_dim << "\n";
    file << "mem_dim=" << mem_dim << "\n";
    file << "ffn_dim=" << ffn_dim << "\n";
    file << "num_layers=" << num_layers << "\n";
    file << "num_experts=" << num_experts << "\n";
    file << "top_k=" << top_k << "\n\n";
    
    file << "# Training\n";
    file << "learning_rate=" << learning_rate << "\n";
    file << "batch_size=" << batch_size << "\n";
    file << "max_seq_len=" << max_seq_len << "\n\n";
    
    file << "# UBOO\n";
    file << "uboo_weight=" << uboo_weight << "\n\n";
    
    file << "# Paths\n";
    file << "checkpoint_path=" << checkpoint_path << "\n";
    file << "vocab_path=" << vocab_path << "\n";
}

void ModelConfig::print() const {
    std::cout << "=== MM-Rec Model Configuration ===" << std::endl;
    std::cout << "Architecture:" << std::endl;
    std::cout << "  vocab_size: " << vocab_size << std::endl;
    std::cout << "  hidden_dim: " << hidden_dim << std::endl;
    std::cout << "  mem_dim: " << mem_dim << std::endl;
    std::cout << "  ffn_dim: " << ffn_dim << std::endl;
    std::cout << "  num_layers: " << num_layers << std::endl;
    std::cout << "  num_experts: " << num_experts << std::endl;
    std::cout << "  top_k: " << top_k << std::endl;
    std::cout << "\nTraining:" << std::endl;
    std::cout << "  learning_rate: " << learning_rate << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  max_seq_len: " << max_seq_len << std::endl;
    std::cout << "\nUBOO:" << std::endl;
    std::cout << "  uboo_weight: " << uboo_weight << std::endl;
    std::cout << "===================================" << std::endl;
}

} // namespace mm_rec
