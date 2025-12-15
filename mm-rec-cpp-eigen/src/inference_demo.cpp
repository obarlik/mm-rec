/**
 * Basic Inference Demo
 * 
 * Standalone executable for MM-Rec inference
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/config/model_config.h"
#include <iostream>
#include <string>

using namespace mm_rec;

int main(int argc, char** argv) {
    std::cout << "=== MM-Rec Inference Demo ===" << std::endl;
    
    // Load config
    std::string config_path = "config.txt";
    if (argc > 1) {
        config_path = argv[1];
    }
    
    std::cout << "\nLoading config from: " << config_path << std::endl;
    ModelConfig config;
    try {
        config = ModelConfig::from_file(config_path);
        config.print();
    } catch (const std::exception& e) {
        std::cout << "Using default config (file not found)" << std::endl;
        config = ModelConfig();
        config.vocab_size = 1000;
        config.hidden_dim = 64;
        config.mem_dim = 64;
        config.ffn_dim = 128;
        config.num_layers = 3;
    }
    
    // Create model
    std::cout << "\nCreating model..." << std::endl;
    MMRecModelConfig model_config{
        config.vocab_size,
        config.hidden_dim,
        config.mem_dim,
        config.ffn_dim,
        config.num_layers
    };
    
    MMRecModel model(model_config);
    std::cout << "✅ Model created successfully" << std::endl;
    
    // Dummy inference
    std::cout << "\nRunning inference..." << std::endl;
    int64_t batch = 2;
    int64_t seq = 8;
    
    Tensor input_ids = Tensor::zeros({batch, seq});
    // Dummy token IDs
    for (int64_t i = 0; i < batch * seq; ++i) {
        input_ids.data()[i] = static_cast<float>(i % config.vocab_size);
    }
    
    std::cout << "  Input: [" << batch << ", " << seq << "]" << std::endl;
    
    // Forward pass
    Tensor logits = model.forward(input_ids);
    
    std::cout << "✅ Inference complete!" << std::endl;
    
    // Check output validity
    if (logits.ndim() != 4) {
        std::cerr << "Error: Expected 4D output, got " << logits.ndim() << "D" << std::endl;
        return 1;
    }
    
    std::cout << "  Output shape: [" << logits.size(0) << ", " 
              << logits.size(1) << ", " << logits.size(2) << ", "
              << logits.size(3) << "]" << std::endl;
    
    // Next token prediction (argmax of last token)
    std::cout << "\nNext token predictions:" << std::endl;
    for (int64_t b = 0; b < batch; ++b) {
        // Get logits for last token of this batch item
        // From final layer
        int64_t final_layer = config.num_layers - 1;
        float max_logit = -1e9f;
        int64_t max_idx = 0;
        
        for (int64_t v = 0; v < config.vocab_size; ++v) {
            int64_t idx = final_layer * batch * seq * config.vocab_size +
                         b * seq * config.vocab_size +
                         (seq - 1) * config.vocab_size +
                         v;
            if (logits.data()[idx] > max_logit) {
                max_logit = logits.data()[idx];
                max_idx = v;
            }
        }
        
        std::cout << "  Batch " << b << ": token " << max_idx 
                  << " (logit: " << max_logit << ")" << std::endl;
    }
    
    std::cout << "\n=== Inference Demo Complete ===" << std::endl;
    
    return 0;
}
