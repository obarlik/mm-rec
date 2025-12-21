#include "mm_rec/jobs/job_training.h"
#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/config/model_config.h"
#include "mm_rec/training/trainer.h"
#include "mm_rec/training/gradient_utils.h"
#include "mm_rec/business/checkpoint.h"
#include "mm_rec/business/i_checkpoint_manager.h" // Interface
#include "mm_rec/core/memory_manager.h"
#include "mm_rec/infrastructure/logger.h"
#include "mm_rec/utils/ui.h"
#include "mm_rec/business/metrics.h"
#include "mm_rec/core/vulkan_backend.h"
#include "mm_rec/core/i_compute_backend.h" // Interface
#include "mm_rec/core/auto_tuner.h"
#include "mm_rec/data/data_loader.h"
#include "mm_rec/data/i_data_loader.h" // Interface
#include "mm_rec/data/dataset.h"
#include "mm_rec/application/dashboard_manager.h"
#include "mm_rec/application/dashboard_manager.h"
#include "mm_rec/infrastructure/i_metrics_exporter.h" // Interface
#include "mm_rec/application/service_configurator.h" // For DI resolution

#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>

namespace mm_rec {

namespace fs = std::filesystem;

JobTraining::JobTraining() {}

JobTraining::~JobTraining() {
    // Ensure thread is cleaned up to prevent zombie processes
    stop();
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool JobTraining::start(const TrainingJobConfig& config) {
    if (running_) {
        ui::error("Job already running!");
        return false;
    }
    
    stop_signal_ = false;
    running_ = true;
    
    // Launch worker thread
    worker_ = std::thread(&JobTraining::run_internal, this, config);
    return true;
}

void JobTraining::stop() {
    if (running_) {
        stop_signal_ = true;
        // Do NOT stop the dashboard manager here. It is global.
        // DashboardManager::instance().stop(); 
    }
}

void JobTraining::join() {
    if (worker_.joinable()) {
        worker_.join();
    }
    running_ = false;
}

void JobTraining::run_internal(TrainingJobConfig config) {
    // Declared here to be visible for cleanup in finally/end of function
    // DI: Resolve from container (Transient = new instance)
    std::shared_ptr<infrastructure::IMetricsExporter> metrics_exporter = 
        ServiceConfigurator::container().resolve<infrastructure::IMetricsExporter>();
    auto checkpoint_manager = ServiceConfigurator::container().resolve<ICheckpointManager>();
    auto data_loader_factory = ServiceConfigurator::container().resolve<IDataLoaderFactory>();
    auto compute_backend = ServiceConfigurator::container().resolve<IComputeBackend>();

    try {
        // --- Setup Run Directory ---
        std::string runs_dir = "runs";
        std::string run_dir = runs_dir + "/" + config.run_name;
        
        if (!fs::exists(runs_dir)) fs::create_directory(runs_dir);
        if (!fs::exists(run_dir)) fs::create_directory(run_dir);
        
        // --- Login ---
        Logger::instance().start_writer(run_dir + "/train.log", LogLevel::DEBUG);
        ui::info("Job Started: " + config.run_name);
        
        // --- Backend ---
        if (compute_backend->init()) {
             ui::success("GPU Backend Active");
        } else {
             ui::warning("Using CPU Backend");
        }
        
        // --- Dashboard ---
        DashboardManager::instance().reset_stop_signal();
        DashboardManager::instance().set_active_run(config.run_name);
        DashboardManager::instance().set_history_path(run_dir + "/dashboard_history.csv");

        // --- Load Config ---
        MMRecModelConfig model_config;

        // Default values
        float learning_rate = 0.01f;
        std::string optimizer_type = "sgd";
        float weight_decay = 0.01f;
        float grad_clip_norm = 1.0f; 
        int batch_size = 8;
        int max_seq_len = 128;
        float easy_threshold = 50.0f;
        float hard_threshold = 500.0f;
        int max_iterations = 15;
        int warmup_steps = 100;
        
        std::ifstream cfg_file(config.config_path);
        if (!cfg_file.good()) {
            ui::error("Config file not found: " + config.config_path);
            running_ = false;
            return;
        }
        
        std::string line;
        while (std::getline(cfg_file, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            
            auto comment_pos = val.find('#');
            if (comment_pos != std::string::npos) val = val.substr(0, comment_pos);
            
            while (!val.empty() && std::isspace(val.back())) val.pop_back();
            while (!val.empty() && std::isspace(val.front())) val.erase(0, 1);

            if (key == "vocab_size") model_config.vocab_size = std::stoi(val);
            else if (key == "hidden_dim") model_config.hidden_dim = std::stoi(val);
            else if (key == "mem_dim") model_config.mem_dim = std::stoi(val);
            else if (key == "ffn_dim") model_config.ffn_dim = std::stoi(val);
            else if (key == "num_layers") model_config.num_layers = std::stoi(val);
            else if (key == "num_experts") model_config.num_experts = std::stoi(val);
            else if (key == "top_k") model_config.top_k = std::stoi(val);
            else if (key == "learning_rate") learning_rate = std::stof(val);
            else if (key == "optimizer_type") optimizer_type = val;
            else if (key == "weight_decay") weight_decay = std::stof(val);
            else if (key == "batch_size") batch_size = std::stoi(val);
            else if (key == "max_seq_len") max_seq_len = std::stoi(val);
            else if (key == "easy_threshold") easy_threshold = std::stof(val);
            else if (key == "hard_threshold") hard_threshold = std::stof(val);
            else if (key == "grad_clip_norm") grad_clip_norm = std::stof(val);
            else if (key == "max_iterations") max_iterations = std::stoi(val);
            else if (key == "warmup_steps") warmup_steps = std::stoi(val);
            else if (key == "uboo_weight") model_config.uboo_weight = std::stof(val);
            else if (key == "max_memory_mb") {
                 size_t mb = std::stoul(val);
                 MemoryManager::set_global_memory_limit(mb * 1024 * 1024);
            }
            else if (key == "vram_reservation_mb") {
                 size_t mb = std::stoul(val);
                 compute_backend->set_reservation(mb);
            }
        }
        
        // --- Data ---
        if (!fs::exists(config.data_path)) {
            ui::error("Data file not found: " + config.data_path);
            running_ = false;
            return;
        }

        auto dataset = std::make_shared<Dataset>(config.data_path);
        // data_loader_factory->create_loader returns std::unique_ptr<IDataLoader>
        auto loader = data_loader_factory->create_loader(dataset, batch_size, max_seq_len, true, 4);
        
        int64_t total_batches = loader->total_batches(); // arrow syntax
        DashboardManager::instance().stats().total_steps = total_batches * max_iterations;

        // --- Model ---
        MMRecModel model(model_config);
        
        TrainingConfig train_config;
        train_config.learning_rate = learning_rate;
        train_config.batch_size = batch_size;
        train_config.optimizer_type = optimizer_type;
        train_config.weight_decay = weight_decay;
        train_config.warmup_steps = warmup_steps;
        train_config.grad_clip_norm = grad_clip_norm;
        train_config.easy_threshold = easy_threshold;
        train_config.hard_threshold = hard_threshold;
        
        Trainer trainer(model, train_config, DashboardManager::instance());
        
        MemoryManager::instance().mark_persistent();

        // --- Checkpoint Resume ---
        std::string latest_ckpt_path = run_dir + "/checkpoint_latest.bin";
        std::string best_ckpt_path = run_dir + "/checkpoint_best.bin";
        int start_epoch = 1;
        float best_loss = 1e9;
        
        try {
            if (std::ifstream(latest_ckpt_path).good()) {
                CheckpointMetadata loaded_meta;
                checkpoint_manager->load_checkpoint(latest_ckpt_path, model, loaded_meta);
                start_epoch = loaded_meta.epoch + 1;
                best_loss = loaded_meta.loss;
                ui::success("Resumed from Epoch " + std::to_string(loaded_meta.epoch) + " (Loss: " + std::to_string(loaded_meta.loss) + ")");
            }
        } catch (...) { ui::warning("Starting fresh."); }
        
        if (start_epoch == 1) {
            CheckpointMetadata init_meta;
            init_meta.epoch = 0;
            init_meta.learning_rate = learning_rate;
            checkpoint_manager->save_checkpoint(latest_ckpt_path, model, init_meta);
        }
        
        // --- Metrics Exporter (Infra) ---
        // infrastructure::MetricsExporter metrics_exporter; // Moved to function scope
        if (config.enable_metrics) {
            MetricsSamplingConfig sampling;
            sampling.enabled = true;
            sampling.interval = 10;
            metrics_exporter->start(run_dir + "/training_metrics.bin", sampling);
        }

        // --- Loop ---
        int64_t global_step = 0;
        
        for (int iteration = start_epoch; iteration <= max_iterations; ++iteration) {
            if (stop_signal_ || DashboardManager::instance().should_stop()) break;
            
            ui::print_header("Epoch " + std::to_string(iteration) + "/" + std::to_string(max_iterations), 40);
            
            float epoch_loss = 0.0f;
            int epoch_step_count = 0;
            auto epoch_start = std::chrono::steady_clock::now();
            auto step_end_time = std::chrono::high_resolution_clock::now();
            
            TrainingBatch batch;
            loader->reset(); // arrow syntax
            int64_t batch_idx = 0;
            
            while(loader->next(batch)) { // arrow syntax
                if (stop_signal_ || DashboardManager::instance().should_stop()) break;
                
                auto step_start_time = std::chrono::high_resolution_clock::now();
                double stall_ms = std::chrono::duration<double, std::milli>(step_start_time - step_end_time).count();
                
                size_t mem_bytes = MemoryManager::get_global_memory_usage();
                double mem_mb = mem_bytes / (1024.0 * 1024.0);
                
                auto now = std::chrono::steady_clock::now();
                double total_time = std::chrono::duration<double>(now - epoch_start).count();
                float speed_tps = (epoch_step_count * batch_size * max_seq_len) / (total_time + 1e-9);

                float loss = trainer.train_step(batch, (float)stall_ms, speed_tps, (float)mem_mb);
                
                // Auto Rolback
                if (!std::isfinite(loss) || loss > 100.0f) {
                    ui::error("Explosion! Rolling back...");
                    CheckpointMetadata meta;
                    checkpoint_manager->load_checkpoint(latest_ckpt_path, model, meta);
                    learning_rate *= 0.5f;
                    trainer.update_learning_rate(learning_rate);
                    MemoryManager::instance().reset_arena();
                    continue;
                }

                epoch_loss += loss;
                epoch_step_count++;
                global_step++;
                
                // DashboardManager::instance().update_training_stats(...); // Handled by Trainer
                DashboardManager::instance().update_system_stats((int)mem_mb);

                if (batch_idx % 10 == 0) {
                     std::stringstream ss;
                     ss << "\r[" << iteration << "] Step " << global_step 
                        << " | Loss: " << std::fixed << std::setprecision(4) << loss
                        << " | Mem: " << (int)mem_mb << "MB";
                     std::cout << ui::color::CYAN << ss.str() << ui::color::RESET << std::flush;
                }
                
                METRIC_TRAINING_STEP(loss, trainer.get_current_lr());
                step_end_time = std::chrono::high_resolution_clock::now();
                MemoryManager::instance().reset_arena();
                batch_idx++;
            }
            std::cout << "\n";
            
            if (stop_signal_) break;

            CheckpointMetadata meta;
            meta.epoch = iteration;
            meta.loss = (epoch_step_count > 0) ? epoch_loss / epoch_step_count : 0.0f;
            meta.learning_rate = learning_rate;
            checkpoint_manager->save_checkpoint(latest_ckpt_path, model, meta);
            
            if (meta.loss < best_loss) {
                best_loss = meta.loss;
                checkpoint_manager->save_checkpoint(best_ckpt_path, model, meta);
            }
        }
        
        // Final
        if (!stop_signal_) {
            CheckpointMetadata final_meta;
            final_meta.epoch = max_iterations;
            final_meta.learning_rate = learning_rate;
            checkpoint_manager->save_checkpoint(run_dir + "/kernel_adaptive_final.bin", model, final_meta);
            ui::success("Training Completed");
        } else {
            ui::warning("Training Aborted by User");
        }
    } catch (const std::exception& e) {
        ui::error("CRITICAL: Training Thread Crashed: " + std::string(e.what()));
    } catch (...) {
        ui::error("CRITICAL: Training Thread Crashed with unknown error");
    }

    // === CLEANUP: Prevent State Leakage Between Runs ===
    // === CLEANUP: Prevent State Leakage Between Runs ===
    if (config.enable_metrics) {
        metrics_exporter->stop();
    }
    Logger::instance().stop_writer();
    
    // Force memory cleanup (clear all arenas, release blocks to global pool)
    MemoryManager::instance().clear_persistent();
    MemoryManager::instance().reset_arena();
    
    // Note: VulkanBackend is a singleton. We don't destroy it (shared across runs).
    // GPU resources are managed per-compute call, no persistent state between runs.
    
    running_ = false;
}

} // namespace mm_rec
