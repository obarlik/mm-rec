/**
 * Inference Command - Generate text from trained model
 * Consolidated for single binary.
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/config/model_config.h"
#include "mm_rec/data/tokenizer.h"
#include "mm_rec/business/checkpoint.h"
#include "mm_rec/business/i_checkpoint_manager.h" // Interface
#include "mm_rec/business/metric_types.h"
#include "mm_rec/business/metrics.h"  // Zero-overhead metrics
#include "mm_rec/infrastructure/i_metrics_exporter.h"
#include "mm_rec/application/service_configurator.h" // For DI
#include "mm_rec/infrastructure/logger.h"
#include "mm_rec/utils/ui.h"
#include "commands.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

using namespace mm_rec;
using namespace mm_rec::ui;

int cmd_infer(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: mm_rec infer <config_file> <model_path> <vocab_file> [prompt] [--metrics]" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string model_path = argv[2];
    std::string vocab_path = argv[3];
    std::string prompt = (argc > 4) ? argv[4] : "The";
    
    // Metrics enabled by default (--no-metrics to disable)
    bool enable_metrics = true;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-metrics") {
            enable_metrics = false;
            break;
        }
    }
    
    // Start Logger
    Logger::instance().start_writer("inference.log", LogLevel::INFO);
    ui::print_header("MM-Rec Inference Engine");
    
    // Create exporter in scope
    auto metrics_exporter = mm_rec::ServiceConfigurator::container().resolve<mm_rec::infrastructure::IMetricsExporter>();
    
    // Resolve CheckpointManager
    auto checkpoint_manager = mm_rec::ServiceConfigurator::container().resolve<mm_rec::ICheckpointManager>();

    if (enable_metrics) {
        metrics_exporter->start("inference_metrics.jsonl");
        LOG_INFO("Metrics enabled → inference_metrics.jsonl");
    }

    // 1. Load Config
    ui::info("Loading config from: " + config_path);
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
    
    LOG_INFO("Vocab: " + std::to_string(config.vocab_size) + 
             " | Hidden: " + std::to_string(config.hidden_dim) + 
             " | Layers: " + std::to_string(config.num_layers));

    // 2. Load Vocab (BPE Model)
    ui::info("Loading tokenizer from: " + vocab_path);
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
         ui::warning("Vocab empty. BPE loading failed?");
    }

    // 3. Initialize Model
    ui::info("Initializing model...");
    MMRecModel model(config);

    // 4. Load Checkpoint
    ui::info("Loading weights from: " + model_path);
    CheckpointMetadata metadata;
    
    auto load_start = std::chrono::high_resolution_clock::now();
    try {
        checkpoint_manager->load_checkpoint(model_path, model, metadata);
        auto load_end = std::chrono::high_resolution_clock::now();
        float load_time_ms = std::chrono::duration<float, std::milli>(load_end - load_start).count();
        
        std::cout << "   Resumed from Epoch: " << metadata.epoch << ", Loss: " << metadata.loss << std::endl;
        METRIC_RECORD(CHECKPOINT_SAVE, load_time_ms, metadata.epoch, 0, "load");
    } catch (const std::exception& e) {
        ui::error("Failed to load checkpoint: " + std::string(e.what()));
        return 1;
    }

    // 5. Tokenize Prompt
    std::vector<int32_t> tokens = tokenizer.encode(prompt);
    if (tokens.empty()) {
        tokens.push_back(tokenizer.bos_id()); // Default to BOS
    }
    
    std::cout << "\n";
    ui::print_header("Generation Start", 40);
    std::cout << color::BOLD << "Prompt:" << color::RESET << " \"" << prompt << "\" (" << tokens.size() << " tokens)\n" << std::endl;
    
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
    
    auto gen_start = std::chrono::high_resolution_clock::now();
    float total_token_latency_ms = 0.0f;

    while (generated_count < max_generate) {
        Tensor input = Tensor::zeros({1, 1});
        input.data()[0] = (float)current_token;

        // Forward (timed)
        auto token_start = std::chrono::high_resolution_clock::now();
        Tensor logits = model.forward(input); 
        auto token_end = std::chrono::high_resolution_clock::now();
        
        float token_latency_ms = std::chrono::duration<float, std::milli>(token_end - token_start).count();
        total_token_latency_ms += token_latency_ms;
        
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
        
        // Record per-token metrics
        METRIC_RECORD(FORWARD_PASS, token_latency_ms, generated_count, best_token, "");

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
        
        // Sleep for effect (optional visual)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    auto gen_end = std::chrono::high_resolution_clock::now();
    float total_time_s = std::chrono::duration<float>(gen_end - gen_start).count();
    float tokens_per_sec = generated_count / std::max(0.001f, total_time_s);
    float avg_latency_ms = generated_count > 0 ? total_token_latency_ms / generated_count : 0.0f;
    
    // Record total stats
    METRIC_INFERENCE(total_time_s * 1000, generated_count);
    METRIC_RECORD(CUSTOM, tokens_per_sec, total_time_s, 0, "tps");

    std::cout << "\n\n";
    
    Table stats({"Metric", "Value", "Unit"}, 20);
    stats.add_row({"Tokens Generated", std::to_string(generated_count), ""});
    
    std::stringstream time_ss; time_ss << std::fixed << std::setprecision(3) << total_time_s;
    stats.add_row({"Total Time", time_ss.str(), "s"});
    
    std::stringstream speed_ss; speed_ss << std::fixed << std::setprecision(2) << tokens_per_sec;
    stats.add_row({"Speed", speed_ss.str(), "tok/s"});
    
    std::stringstream lat_ss; lat_ss << std::fixed << std::setprecision(2) << avg_latency_ms;
    stats.add_row({"Avg Latency", lat_ss.str(), "ms/tok"});
    
    stats.finish();
    
    if (enable_metrics) {
        metrics_exporter->stop();
    }
    
    Logger::instance().stop_writer();
    
    return 0;
}
