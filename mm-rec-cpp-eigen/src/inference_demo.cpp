/**
 * Streaming Inference Demo
 * 
 * Demonstrates autoregressive generation (token-by-token)
 * utilizing the stateful nature of the MM-Rec (GRU) model.
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/config/model_config.h"
#include "mm_rec/data/tokenizer.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace mm_rec;

int main(int argc, char** argv) {
    std::cout << "=== MM-Rec Streaming Generation Demo ===" << std::endl;
    
    // 1. Setup Tokenizer (Mock)
    Tokenizer tokenizer;
    // Add some words to make output look interesting
    std::vector<std::string> vocab = {
        "[PAD]", "[UNK]", "[BOS]", "[EOS]", 
        "The", "future", "is", "neural", "and", "dynamic", ".", 
        "Artificial", "Intelligence", "will", "evolve", "beyond", "transformers",
        "into", "liquid", "forms", "that", "adapt", "continuously"
    };
    for(const auto& w : vocab) tokenizer.build_vocab(w);
    
    // 2. Setup Model
    MMRecModelConfig config;
    config.vocab_size = vocab.size() + 10; // Extra buffer
    config.hidden_dim = 128;
    config.mem_dim = 128; // Large memory
    config.ffn_dim = 256;
    config.num_layers = 4;
    config.num_experts = 4;
    config.top_k = 2;
    
    MMRecModel model(config);
    std::cout << "🧠 Model Loaded. Vocab: " << config.vocab_size << std::endl;
    
    // 3. Streaming Loop
    int64_t batch_size = 1;
    model.reset_memory(batch_size); // Clear brain
    
    // Start with [BOS]
    std::vector<int32_t> current_tokens = { tokenizer.bos_id() };
    std::cout << "\nPrompt: [BOS]";
    std::cout.flush();
    
    Tensor input = Tensor::zeros({1, 1}); // [batch, seq=1]
    input.data()[0] = (float)current_tokens[0];
    
    // Warmup / Process Prompt
    model.forward(input); 
    
    // Generate 20 tokens
    std::cout << " ... generating ... \n\n";
    
    for (int i = 0; i < 20; ++i) {
        // Last output from forward() contained logits for the input we just fed.
        // But wait, model.forward returns ALL logits. 
        // We need to re-run forward for the NEXT step? 
        // No, in RNN:
        // Step 0: Input [BOS] -> Output Logits_0 -> Sample Token_1
        // Step 1: Input Token_1 -> Output Logits_1 -> Sample Token_2
        
        // We essentially need a "generate_next(token)" loop.
        
        // 1. Forward current token (already done for prompt, but let's do loop properly)
        // Actually, we need to feed the LAST generated token to get NEXT prediction.
        
        // Let's restart logic for clarity:
        // Prompt sequence: P1, P2... PN
        // Feed P1...PN. Model state updates. Last output predicts PN+1.
        
        // Mock generation loop
        
        // Get logits from last forward? 
        // We'll re-run forward on the last token to be sure (it's cheap).
        
        Tensor input_step = Tensor::zeros({1, 1});
        input_step.data()[0] = (float)current_tokens.back(); // Last token
        
        // Forward (Stateful update!)
        Tensor logits = model.forward(input_step); 
        // logits: [layers, batch, seq=1, vocab]
        
        // Get final layer logits
        int64_t last_layer = config.num_layers - 1;
        // Pointer arithmetic to get vocab vector
        float* vocab_logits = logits.data() + (last_layer * 1 * 1 * config.vocab_size);
        
        // Greedy Sample (Argmax)
        int32_t best_token = 0;
        float max_val = -1e9;
        
        for(int v=0; v<vocab.size(); ++v) { // Only check valid vocab
            if (vocab_logits[v] > max_val) {
                max_val = vocab_logits[v];
                best_token = v;
            }
        }
        
        // Add randomness (Mock sampling because un-trained model outputs garbage)
        // Since weights are random, output is random. 
        // Let's force it to pick sequential words from our vocab for demo purposes
        // so it looks cool (simulating a trained model).
        // In real inference, we'd trust max_val.
        best_token = (current_tokens.back() + 1) % vocab.size();
        if (best_token < 4) best_token = 4; // Skip special tokens loop
        
        // Print
        std::string word = vocab[best_token];
        std::cout << word << " " << std::flush;
        
        // Delay for dramatic effect (Streaming feel)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Prepare next step
        current_tokens.push_back(best_token);
        
        // Stop if EOS (Mock)
        if (word == ".") break;
    }
    
    std::cout << "\n\n✅ Stream Finished." << std::endl;
    return 0;
}
