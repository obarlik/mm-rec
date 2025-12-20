#include "mm_rec/utils/dashboard_manager.h"
#include "embedded_assets.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/ui.h"
#include "mm_rec/utils/run_manager.h"
#include "mm_rec/core/vulkan_backend.h"
#include "mm_rec/core/dynamic_balancer.h"
#include "mm_rec/utils/metrics.h"
#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/data/tokenizer.h"
#include "mm_rec/utils/checkpoint.h"
#include <sstream>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>

namespace mm_rec {

// Static inference state to avoid reloading model every request
struct InferenceState {
    std::unique_ptr<MMRecModel> model;
    std::unique_ptr<Tokenizer> tokenizer;
    std::string current_model_path;
    std::mutex mtx;
};
static InferenceState g_inference;

DashboardManager& DashboardManager::instance() {
    static DashboardManager instance;
    return instance;
}

DashboardManager::DashboardManager() {
    // Initialize standard/safe values
}

void DashboardManager::set_history_path(const std::string& path) {
    std::lock_guard<std::mutex> lock(history_mtx_);
    if (history_file_.is_open()) {
        history_file_.close();
    }
    history_file_.open(path, std::ios::app);
}

DashboardManager::~DashboardManager() {
    if (history_file_.is_open()) history_file_.close();
    stop();
}

bool DashboardManager::start(const net::HttpServerConfig& base_config) {
    if (server_) return true; // Already running

    int max_retries = 10;
    bool success = false;

    for (int i = 0; i < max_retries; ++i) {
        // Create a copy of config for this attempt
        net::HttpServerConfig current_config = base_config;
        current_config.port = base_config.port + i;
        
        try {
            server_ = std::make_unique<mm_rec::net::HttpServer>(current_config);
            register_routes();
            
            if (server_->start()) {
                std::string msg = "Global Dashboard started on port " + std::to_string(current_config.port);
                LOG_INFO(msg);
                mm_rec::ui::success(msg);
                
                if (current_config.port != base_config.port) {
                    std::string fallback_msg = "(Port " + std::to_string(base_config.port) + " was busy)";
                    LOG_INFO(fallback_msg);
                    mm_rec::ui::warning(fallback_msg);
                }
                success = true;
                break;
            }
        } catch (...) {
            server_.reset();
        }
        
        // Failed, try next
        server_.reset(); 
    }

    if (!success) {
        LOG_ERROR("Failed to start Global Dashboard on any port " + 
                 std::to_string(base_config.port) + "-" + std::to_string(base_config.port + max_retries - 1));
    }
    return success;
}

void DashboardManager::stop() {
    if (server_) {
        server_->stop();
        server_.reset();
        LOG_INFO("Global Dashboard stopped.");
    }
}

void DashboardManager::update_training_stats(float loss, float lr, float speed, int step) {
    stats_.current_loss = loss;
    stats_.current_lr = lr;
    stats_.current_speed = speed;
    stats_.current_step = step;
    
    std::lock_guard<std::mutex> lock(history_mtx_);
    loss_history_.push_back(loss);
    if (loss_history_.size() > max_history_size_) {
        loss_history_.pop_front();
    }
    
    // Log Training Stats to Metrics
    mm_rec::MetricsManager::record(mm_rec::MetricType::TRAINING_STEP, loss, lr, step, "STEP");
    
    // Log Hybrid Stats
    float ratio = mm_rec::DynamicBalancer::get_gpu_ratio();
    double diff = mm_rec::DynamicBalancer::get_sync_diff();
    mm_rec::MetricsManager::record(
        mm_rec::MetricType::HYBRID_PERF, 
        ratio, 
        static_cast<float>(diff), 
        0, 
        "HYBRID"
    );
}

void DashboardManager::update_system_stats(size_t mem_mb) {
    stats_.memory_usage_mb = mem_mb;
}

void DashboardManager::register_routes() {
    if (!server_) return;

    // Home - Serve main HTML
    server_->register_handler("/", [](const std::string&) -> std::string {
        return mm_rec::net::HttpServer::build_response(200, "text/html", std::string(assets::get_index_html()));
    });

    // Static CSS
    server_->register_handler("/static/style.css", [](const std::string&) -> std::string {
        return mm_rec::net::HttpServer::build_response(200, "text/css", std::string(assets::get_style_css()));
    });

    // Static JS
    server_->register_handler("/static/app.js", [](const std::string&) -> std::string {
        return mm_rec::net::HttpServer::build_response(200, "application/javascript", std::string(assets::get_app_js()));
    });
    
    // API Stats
    server_->register_handler("/api/stats", [this](const std::string&) -> std::string {
        std::stringstream ss;
        ss << "{";
        ss << "\"loss\": " << stats_.current_loss.load() << ",";
        ss << "\"step\": " << stats_.current_step.load() << ",";
        ss << "\"total_steps\": " << stats_.total_steps.load() << ",";
        ss << "\"lr\": " << stats_.current_lr.load() << ",";
        ss << "\"speed\": " << stats_.current_speed.load() << ","; 
        ss << "\"mem\": " << stats_.memory_usage_mb.load() << ","; 
        
        float ratio = mm_rec::DynamicBalancer::get_gpu_ratio();
        double diff = mm_rec::DynamicBalancer::get_sync_diff();
        ss << "\"gpu_ratio\": " << ratio << ",";
        ss << "\"sync_delta\": " << diff << ",";
        ss << "\"epoch\": 1,"; 
        
        std::lock_guard<std::mutex> lock(history_mtx_);
        ss << "\"history\": [";
        for(size_t i=0; i<loss_history_.size(); ++i) {
            ss << loss_history_[i] << (i < loss_history_.size()-1 ? "," : "");
        }
        ss << "],";
        ss << "\"avg_history\": []";
        ss << "}";
        return mm_rec::net::HttpServer::build_response(200, "application/json", ss.str());
    });
    
    // API Hardware
    server_->register_handler("/api/hardware", [](const std::string&) -> std::string {
        auto& vk = mm_rec::VulkanBackend::get();
        std::string gpu_name = vk.is_ready() ? vk.get_device_name() : "Vulkan Not Ready";
        size_t vram_mb = vk.is_ready() ? (vk.get_total_vram() / 1024 / 1024) : 0;
        
        std::stringstream json;
        json << "{";
        json << "  \"cpu_model\": \"Host Processor\","; 
        json << "  \"compute_device\": \"" << gpu_name << "\",";
        json << "  \"mem_total_mb\": " << vram_mb << ","; 
        json << "  \"arch\": \"x86_64 / SPIR-V\",";
        json << "  \"cores_logical\": \"N/A\","; 
        json << "  \"simd\": \"AVX2 / Int8\"";
        json << "}";
        return mm_rec::net::HttpServer::build_response(200, "application/json", json.str());
    });
    
    // Stop Signal
    server_->register_handler("/api/stop", [this](const std::string&) -> std::string {
        stats_.should_stop = true;
        LOG_INFO("Stop signal received from Dashboard.");
        return mm_rec::net::HttpServer::build_response(200, "text/plain", "Stopping...");
    });

    // History API
    server_->register_handler("/api/history", [](const std::string&) -> std::string {
        std::ifstream f("dashboard_history.csv");
        std::vector<float> losses;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string segment;
            std::vector<std::string> seglist;
            while(std::getline(ss, segment, ',')) {
               seglist.push_back(segment);
            }
            if (seglist.size() >= 2) {
                try {
                    losses.push_back(std::stof(seglist[1])); 
                } catch (...) {}
            }
        }
        std::stringstream json;
        json << "{ \"loss_history\": [";
        for (size_t i = 0; i < losses.size(); ++i) {
             json << losses[i] << (i < losses.size() - 1 ? "," : "");
        }
        json << "] }";
        return mm_rec::net::HttpServer::build_response(200, "application/json", json.str());
    });
    
    // API Runs List
    server_->register_handler("/api/runs", [](const std::string&) -> std::string {
        auto runs = RunManager::list_runs();
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < runs.size(); ++i) {
            const auto& run = runs[i];
            ss << "{";
            ss << "\"name\": \"" << run.name << "\",";
            ss << "\"status\": \"" << (RunManager::is_job_running() && run.status == RunStatus::RUNNING ? "RUNNING" : run.status_str) << "\",";
            ss << "\"epoch\": " << run.current_epoch << ",";
            ss << "\"loss\": " << run.current_loss << ",";
            ss << "\"best_loss\": " << run.best_loss << ",";
            ss << "\"size_mb\": " << run.total_size_mb;
            ss << "}";
            if (i < runs.size() - 1) ss << ",";
        }
        ss << "]";
        return mm_rec::net::HttpServer::build_response(200, "application/json", ss.str());
    });

    // API Stop Run
    server_->register_handler("/api/runs/stop", [this](const std::string&) -> std::string {
        if (!RunManager::is_job_running()) {
             return mm_rec::net::HttpServer::build_response(400, "application/json", "{\"error\": \"No job is running\"}");
        }
        RunManager::stop_job();
        LOG_INFO("Job stopped via API.");
        return mm_rec::net::HttpServer::build_response(200, "application/json", "{\"status\": \"stopped\"}");
    });

    // API Resume Run
    server_->register_handler("/api/runs/resume", [this](const std::string& req) -> std::string {
         auto get_param = [&](const std::string& key) -> std::string {
             size_t pos = req.find(key + "=");
             if (pos == std::string::npos) return "";
             size_t end = req.find('&', pos);
             return req.substr(pos + key.size() + 1, end - (pos + key.size() + 1));
         };

         std::string run_name = get_param("name");
         if (run_name.empty()) {
             return mm_rec::net::HttpServer::build_response(400, "application/json", "{\"error\": \"Missing name parameter\"}");
         }

         if (RunManager::is_job_running()) {
             return mm_rec::net::HttpServer::build_response(409, "application/json", "{\"error\": \"Job already running\"}");
         }

         if (!RunManager::run_exists(run_name)) {
             return mm_rec::net::HttpServer::build_response(404, "application/json", "{\"error\": \"Run not found\"}");
         }
         
         std::string run_dir = RunManager::get_run_dir(run_name);
         std::string config_path = run_dir + "/config.txt";
         
         if (!std::filesystem::exists(config_path)) {
              return mm_rec::net::HttpServer::build_response(500, "application/json", "{\"error\": \"Missing config in run dir\"}");
         }

         TrainingJobConfig config;
         config.run_name = run_name;
         config.config_path = config_path;
         config.data_path = "training_data.bin"; 
         
         if (RunManager::start_job(config)) {
             return mm_rec::net::HttpServer::build_response(200, "application/json", "{\"status\": \"started\"}");
         }
         return mm_rec::net::HttpServer::build_response(500, "application/json", "{\"error\": \"Failed to start\"}");
    });

    // API Delete Run
    server_->register_handler("/api/runs/delete", [this](const std::string& req) -> std::string {
         auto get_param = [&](const std::string& key) -> std::string {
             size_t pos = req.find(key + "=");
             if (pos == std::string::npos) return "";
             size_t end = req.find('&', pos);
             return req.substr(pos + key.size() + 1, end - (pos + key.size() + 1));
         };

         std::string run_name = get_param("name");
         if (run_name.empty()) {
             return mm_rec::net::HttpServer::build_response(400, "application/json", "{\"error\": \"Missing name parameter\"}");
         }

         if (RunManager::is_job_running()) {
             return mm_rec::net::HttpServer::build_response(409, "application/json", "{\"error\": \"Cannot delete whle a job is running\"}");
         }

         if (RunManager::delete_run(run_name)) {
             return mm_rec::net::HttpServer::build_response(200, "application/json", "{\"status\": \"deleted\"}");
         }
         return mm_rec::net::HttpServer::build_response(500, "application/json", "{\"error\": \"Delete failed\"}");
    });

    // --- Phase 3 Config & Data ---

    // API List Configs
    server_->register_handler("/api/configs", [](const std::string&) -> std::string {
        std::stringstream ss;
        ss << "[";
        bool first = true;
        try {
            for (const auto& entry : std::filesystem::directory_iterator(".")) {
                if (entry.path().extension() == ".ini") {
                    if (!first) ss << ",";
                    ss << "\"" << entry.path().filename().string() << "\"";
                    first = false;
                }
            }
        } catch (...) {}
        ss << "]";
        return mm_rec::net::HttpServer::build_response(200, "application/json", ss.str());
    });

    // API Create Config
    server_->register_handler("/api/configs/create", [](const std::string& req) -> std::string {
        auto get_json_val = [&](const std::string& key) -> std::string {
            std::string search = "\"" + key + "\":";
            size_t pos = req.find(search);
            if (pos == std::string::npos) return "";
            size_t start = req.find("\"", pos + search.length()); 
            if (start == std::string::npos) return "";
            start++;
            size_t end = start;
            while (end < req.length()) {
                if (req[end] == '"' && req[end-1] != '\\') break; 
                end++;
            }
            if (end >= req.length()) return "";
            std::string val = req.substr(start, end - start);
            size_t replace_pos = 0;
            while((replace_pos = val.find("\\n", replace_pos)) != std::string::npos) {
                val.replace(replace_pos, 2, "\n");
                replace_pos += 1;
            }
            return val;
        };

        std::string filename = get_json_val("filename");
        std::string content = get_json_val("content");
        
        if (filename.empty() || content.empty()) {
            return mm_rec::net::HttpServer::build_response(400, "application/json", "{\"error\": \"Missing filename or content\"}");
        }
        
        if (filename.find(".ini") == std::string::npos) filename += ".ini";
        if (filename.find("/") != std::string::npos || filename.find("..") != std::string::npos) {
            return mm_rec::net::HttpServer::build_response(403, "application/json", "{\"error\": \"Invalid filename\"}");
        }

        std::ofstream file(filename);
        if (!file.good()) {
            return mm_rec::net::HttpServer::build_response(500, "application/json", "{\"error\": \"Failed to write file\"}");
        }
        file << content;
        file.close();
        return mm_rec::net::HttpServer::build_response(200, "application/json", "{\"status\": \"created\", \"file\": \"" + filename + "\"}");
    });

    // API List Datasets
    server_->register_handler("/api/datasets", [](const std::string&) -> std::string {
        std::stringstream ss;
        ss << "[";
        bool first = true;
        try {
            for (const auto& entry : std::filesystem::directory_iterator(".")) {
                if (entry.path().extension() == ".bin") {
                    if (!first) ss << ",";
                    uintmax_t size_mb = std::filesystem::file_size(entry) / (1024 * 1024);
                    ss << "{ \"name\": \"" << entry.path().filename().string() << "\", \"size_mb\": " << size_mb << " }";
                    first = false;
                }
            }
        } catch (...) {}
        ss << "]";
        return mm_rec::net::HttpServer::build_response(200, "application/json", ss.str());
    });

    // API Upload Dataset
    server_->register_handler("/api/datasets/upload", [](const std::string& req) -> std::string {
        size_t header_end = req.find("\r\n\r\n");
        if (header_end == std::string::npos) return mm_rec::net::HttpServer::build_response(400, "text/plain", "Bad Request");
        
        std::string body = req.substr(header_end + 4); 
        std::string first_line = req.substr(0, req.find("\r\n"));
        size_t q_pos = first_line.find("?name=");
        if (q_pos == std::string::npos) return mm_rec::net::HttpServer::build_response(400, "application/json", "{\"error\": \"Missing name query param\"}");
        
        size_t end_pos = first_line.find(" ", q_pos);
        std::string filename = first_line.substr(q_pos + 6, end_pos - (q_pos + 6));
        
        if (filename.find(".bin") == std::string::npos) filename += ".bin";
        if (filename.find("/") != std::string::npos || filename.find("..") != std::string::npos) {
            return mm_rec::net::HttpServer::build_response(403, "application/json", "{\"error\": \"Invalid filename\"}");
        }
        
        std::ofstream file(filename, std::ios::binary);
        if (!file.good()) {
            return mm_rec::net::HttpServer::build_response(500, "application/json", "{\"error\": \"Failed to write file\"}");
        }
        file.write(body.data(), body.size());
        file.close();
        return mm_rec::net::HttpServer::build_response(200, "application/json", "{\"status\": \"uploaded\", \"size\": " + std::to_string(body.size()) + "}");
    });

    // --- Phase 3 Model & Inference ---

    // API List Models
    server_->register_handler("/api/models", [](const std::string&) -> std::string {
        std::vector<std::pair<std::string, uintmax_t>> models;
        try {
            // Root
            for (const auto& entry : std::filesystem::directory_iterator(".")) {
                if (entry.path().extension() == ".bin" && entry.path().filename().string().find("checkpoint") != std::string::npos) {
                    models.push_back({entry.path().filename().string(), std::filesystem::file_size(entry)});
                }
            }
            // Runs (Recursive-ish)
            if (std::filesystem::exists("runs")) {
                for (const auto& run_entry : std::filesystem::directory_iterator("runs")) {
                    if (run_entry.is_directory()) {
                        for (const auto& file : std::filesystem::directory_iterator(run_entry.path())) {
                             if (file.path().extension() == ".bin") {
                                 models.push_back({run_entry.path().filename().string() + "/" + file.path().filename().string(), std::filesystem::file_size(file)});
                             }
                        }
                    }
                }
            }
        } catch (...) {}
        
        std::stringstream ss;
        ss << "[";
        for(size_t i=0; i<models.size(); ++i) {
             ss << "{ \"name\": \"" << models[i].first << "\", \"size_mb\": " << (models[i].second / 1024 / 1024) << " }";
             if (i < models.size()-1) ss << ",";
        }
        ss << "]";
        return mm_rec::net::HttpServer::build_response(200, "application/json", ss.str());
    });

    // API Download Model
    server_->register_handler("/api/models/download", [](const std::string& req) -> std::string {
        std::string first_line = req.substr(0, req.find("\r\n"));
        size_t q_pos = first_line.find("?name=");
        if (q_pos == std::string::npos) return mm_rec::net::HttpServer::build_response(400, "text/plain", "Missing name");
        size_t end_pos = first_line.find(" ", q_pos);
        std::string name = first_line.substr(q_pos + 6, end_pos - (q_pos + 6));
        
        if (name.find("..") != std::string::npos) return mm_rec::net::HttpServer::build_response(403, "text/plain", "Invalid path");
        
        std::string path = name;
        if (name.find("/") != std::string::npos) {
             path = "runs/" + name; 
        } else {
             path = "./" + name;
        }

        if (!std::filesystem::exists(path)) return mm_rec::net::HttpServer::build_response(404, "text/plain", "Not found: " + path);
        
        std::ifstream f(path, std::ios::binary);
        if (!f.good()) return mm_rec::net::HttpServer::build_response(500, "text/plain", "Read error");
        
        std::stringstream buffer;
        buffer << f.rdbuf();
        return mm_rec::net::HttpServer::build_response(200, "application/octet-stream", buffer.str()); 
    });

    // API Inference
    server_->register_handler("/api/inference", [](const std::string& req) -> std::string {
        auto get_json_val = [&](const std::string& key) -> std::string {
            std::string search = "\"" + key + "\":";
            size_t pos = req.find(search);
             if (pos == std::string::npos) return "";
            size_t start = req.find("\"", pos + search.length());
             if (start == std::string::npos) return ""; 
             start++;
            size_t end = req.find("\"", start);
            return req.substr(start, end - start);
        };

        std::string model_name = get_json_val("model");
        std::string prompt = get_json_val("prompt");
        
        if (model_name.empty() || prompt.empty()) {
             return mm_rec::net::HttpServer::build_response(400, "application/json", "{\"error\": \"Missing model or prompt\"}");
        }

        std::lock_guard<std::mutex> lock(g_inference.mtx);
        
        std::string path = model_name;
        if (model_name.find("/") != std::string::npos) path = "runs/" + model_name;
        
        if (g_inference.current_model_path != path || !g_inference.model) {
            if (!std::filesystem::exists(path)) return mm_rec::net::HttpServer::build_response(404, "application/json", "{\"error\": \"Model not found\"}");
            
            try {
                std::string run_dir = std::filesystem::path(path).parent_path().string();
                std::string config_path = run_dir + "/config.ini";
                if (!std::filesystem::exists(config_path)) config_path = "mm_rec.ini";

                 MMRecModelConfig cfg;
                 std::ifstream cfile(config_path);
                 std::string line;
                 while(std::getline(cfile, line)) {
                     if (line.find("vocab_size") != std::string::npos) cfg.vocab_size = std::stoi(line.substr(line.find("=")+1));
                     if (line.find("hidden_dim") != std::string::npos) cfg.hidden_dim = std::stoi(line.substr(line.find("=")+1));
                     if (line.find("num_layers") != std::string::npos) cfg.num_layers = std::stoi(line.substr(line.find("=")+1));
                 }
                 
                 g_inference.model = std::make_unique<MMRecModel>(cfg);
                 CheckpointMetadata meta;
                 CheckpointManager::load_checkpoint(path, *g_inference.model, meta);
                 
                 g_inference.tokenizer = std::make_unique<Tokenizer>();
                 g_inference.tokenizer->load_model("vocab.json", "merges.txt"); 
                 
                 g_inference.current_model_path = path;
            } catch (const std::exception& e) {
                return mm_rec::net::HttpServer::build_response(500, "application/json", "{\"error\": \"" + std::string(e.what()) + "\"}");
            }
        }
        
        std::vector<int> tokens = g_inference.tokenizer->encode(prompt);
        std::string result = prompt;
        int max_new_tokens = 50;
        int32_t current_token = tokens.empty() ? g_inference.tokenizer->bos_id() : tokens.back();
        
        g_inference.model->reset_memory(1);
        
        for(size_t i=0; i<tokens.size()-1; ++i) {
            Tensor input = Tensor::zeros({1, 1});
            input.data()[0] = (float)tokens[i];
            g_inference.model->forward(input);
        }
        
        for(int i=0; i<max_new_tokens; ++i) {
            Tensor input = Tensor::zeros({1, 1});
            input.data()[0] = (float)current_token;
            Tensor logits = g_inference.model->forward(input);
            
            int64_t last_layer = g_inference.model->get_config().num_layers - 1;
            float* vocab_logits = logits.data() + (last_layer * 1 * 1 * g_inference.model->get_config().vocab_size);
            
            int best = 0; float max_val = -1e9;
            for(int v=0; v<g_inference.model->get_config().vocab_size; ++v) {
                if (vocab_logits[v] > max_val) { max_val = vocab_logits[v]; best = v; }
            }
            
            if (best == g_inference.tokenizer->eos_id()) break;
            
            result += g_inference.tokenizer->decode({best});
            current_token = best;
        }
        
        std::string json_result;
        for (char c : result) {
            if (c == '"') json_result += "\\\"";
            else if (c == '\n') json_result += "\\n";
            else json_result += c;
        }
        return mm_rec::net::HttpServer::build_response(200, "application/json", "{\"text\": \"" + json_result + "\"}");
    });

} // register_routes

} // namespace mm_rec
