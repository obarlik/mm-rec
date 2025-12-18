/**
 * Inference Command - Generate text from trained model
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/config/model_config.h"
#include "mm_rec/data/tokenizer.h"
#include "mm_rec/utils/checkpoint.h"
#include "commands.h"

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

using namespace mm_rec;

int cmd_infer(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: mm_rec infer <config_file> <model_path> <vocab_file> [prompt]" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string model_path = argv[2];
    std::string vocab_path = argv[3];
    std::string prompt = (argc > 4) ? argv[4] : "The";

    // 1. Load Config
    std::cout << "📄 Loading config from: " << config_path << std::endl;
    // Use ModelConfig::from_file instead of non-existent loader
    ModelConfig cfg = ModelConfig::from_file(config_path);
    
    // Convert to MMRecModelConfig (Internal struct vs Config struct mismatch?)
    // Actually ModelConfig IS the config struct used by MMRecModel? 
    // Wait, MMRecModel takes `MMRecModelConfig`. Let's check model.h
    // Assuming we need to map ModelConfig (YAML/File) to MMRecModelConfig (Internal)
    
    MMRecModelConfig config;
    config.vocab_size = cfg.vocab_size;
    config.hidden_dim = cfg.hidden_dim;
    config.mem_dim = cfg.mem_dim;
    config.ffn_dim = cfg.ffn_dim;
    config.num_layers = cfg.num_layers;
    config.num_experts = cfg.num_experts;
    config.top_k = cfg.top_k;
    
    std::cout << "   Vocab: " << config.vocab_size 
              << " | Hidden: " << config.hidden_dim 
              << " | Layers: " << config.num_layers << std::endl;

    // 2. Load Vocab (BPE Model)
    std::cout << "📚 Loading tokenizer from: " << vocab_path << std::endl;
    Tokenizer tokenizer;
    // Infers merges.txt path (e.g. vocab.json -> merges.txt in same dir)
    // Assuming vocab_path is the directory or full path to vocab.json
    // Logic: if passed "vocab.txt" (legacy), try usage.
    // If passed "dir/vocab.json", look for "dir/merges.txt".
    std::string merges_path = "merges.txt"; // Default
    size_t last_slash = vocab_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        merges_path = vocab_path.substr(0, last_slash + 1) + "merges.txt";
    }
    // Also handle if user passed just the file name
    
    tokenizer.load_model(vocab_path, merges_path);
    
    if (tokenizer.vocab_size() == 0) {
         std::cerr << "⚠️  Vocab empty. BPE loading failed?" << std::endl;
    }

    // 3. Initialize Model
    std::cout << "🧠 Initializing model..." << std::endl;
    MMRecModel model(config);

    // 4. Load Checkpoint
    std::cout << "📥 Loading weights from: " << model_path << std::endl;
    CheckpointMetadata metadata;
    // Use CheckpointManager::load_checkpoint
    try {
        CheckpointManager::load_checkpoint(model_path, model, metadata);
        std::cout << "   Resumed from Epoch: " << metadata.epoch << ", Loss: " << metadata.loss << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to load checkpoint: " << e.what() << std::endl;
        return 1;
    }

    // 5. Tokenize Prompt
    std::vector<int32_t> tokens = tokenizer.encode(prompt);
    if (tokens.empty()) {
        tokens.push_back(tokenizer.bos_id()); // Default to BOS
    }
    
    std::cout << "\n📝 Prompt: \"" << prompt << "\" (" << tokens.size() << " tokens)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << prompt << std::flush;

    // 6. Generation Loop
    int64_t batch_size = 1;
    model.reset_memory(batch_size);

    // Prime the model with prompt (except last token)
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        Tensor input = Tensor::zeros({1, 1});
        input.data()[0] = (float)tokens[i];
        model.forward(input); // Just update memory state
    }

    // Start generating from last token of prompt
    int32_t current_token = tokens.back();
    int generated_count = 0;
    int max_generate = 50;

    while (generated_count < max_generate) {
        Tensor input = Tensor::zeros({1, 1});
        input.data()[0] = (float)current_token;

        // Forward
        Tensor logits = model.forward(input); 
        // logits: [layers, batch, seq, vocab]
        
        // Extract final layer logits
        int64_t last_layer = config.num_layers - 1;
        float* vocab_logits = logits.data() + (last_layer * 1 * 1 * config.vocab_size);

        // Greedy Algo (Argmax)
        int32_t best_token = 0;
        float max_val = -1e9;
        
        for(int v=0; v<config.vocab_size; ++v) {
            if (vocab_logits[v] > max_val) {
                max_val = vocab_logits[v];
                best_token = v;
            }
        }

        // Decode and Print
        std::string word = tokenizer.decode({best_token});
        
        // Simple spacing heuristic (add space if not punctuation)
        if (word.length() > 0 && isalnum(word[0])) std::cout << " ";
        std::cout << word << std::flush;
        
        // Stop conditions
        if (best_token == tokenizer.eos_id()) break;
        
        // Next step
        current_token = best_token;
        generated_count++;
        
        // Sleep for effect
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << "\n\n----------------------------------------" << std::endl;
    return 0;
}
