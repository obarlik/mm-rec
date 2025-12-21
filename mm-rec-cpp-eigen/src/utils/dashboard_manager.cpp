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
#include "mm_rec/utils/middlewares.h" // New include
#include "mm_rec/utils/event_bus.h"    // EventBus for SSE
#include "mm_rec/utils/service_configurator.h" // DI Configuration
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

    // User requested range: 8085-8089 (5 ports)
    int max_retries = 5; 
    bool success = false;

    for (int i = 0; i < max_retries; ++i) {
        // Create a copy of config for this attempt
        net::HttpServerConfig current_config = base_config;
        current_config.port = base_config.port + i;

        try {
            server_ = std::make_unique<mm_rec::net::HttpServer>(current_config);
            
            // === Dependency Injection Setup ===
            // Initialize global DI container (once, at app startup)
            static bool di_initialized = false;
            if (!di_initialized) {
                mm_rec::ServiceConfigurator::initialize();
                
                // Register DashboardManager in production (not in demos to avoid linking issues)
                mm_rec::ServiceConfigurator::container().bind_singleton<DashboardManager>();
                
                di_initialized = true;
            }
            
            // === Middleware Registration ===
            // (Order: Last added is Outermost)
            
            // 0. DI Scope Middleware (Outermost - creates scope for request)
            server_->use([](const mm_rec::net::Request& req, std::shared_ptr<mm_rec::net::Response> res, auto next) {
                // Create DI scope for this request (using global container)
                mm_rec::Scope request_scope(mm_rec::ServiceConfigurator::container());
                
                // Attach scope to request for handlers to use
                const_cast<mm_rec::net::Request&>(req).scope = &request_scope;
                
                // Call next middleware/handler
                next(req, res);
                
                // Scope destroyed here (RAII) - scoped services cleaned up
            });
            
            // 1. Security (Modifies response headers)
            server_->use(mm_rec::net::Middlewares::Security);
            // 2. Metrics (Records latency)
            server_->use(mm_rec::net::Middlewares::Metrics);
            // 3. Logger (Logs final result and timing)
            server_->use(mm_rec::net::Middlewares::Logger);
            
            // Connect EventBus to SSE broadcasting
            mm_rec::EventBus::instance().on("training.step", [this](const mm_rec::EventData& data) {
                server_->broadcast_sse("training.step", mm_rec::to_json(data));
            });
            
            mm_rec::EventBus::instance().on("training.started", [this](const mm_rec::EventData& data) {
                server_->broadcast_sse("training.started", mm_rec::to_json(data));
            });
            
            mm_rec::EventBus::instance().on("training.stopped", [this](const mm_rec::EventData& data) {
                server_->broadcast_sse("training.stopped", mm_rec::to_json(data));
            });
            
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

// --- Helper for Resumable File Serving ---
// --- Helper for Resumable File Serving ---
static void serve_file_with_range(const std::string& path, const mm_rec::net::Request& req, std::shared_ptr<mm_rec::net::Response> res) {
    if (!std::filesystem::exists(path)) {
        res->status(404);
        res->send("Not found");
        return;
    }
    
    uintmax_t file_size = std::filesystem::file_size(path);
    size_t range_start = 0;
    size_t range_end = file_size - 1;
    bool is_range = false;

    // Parse Range Header
    size_t range_header_pos = req.body.find("Range: bytes=");
    if (range_header_pos != std::string::npos) {
         is_range = true;
         size_t start_val = range_header_pos + 13;
         size_t dash_pos = req.body.find("-", start_val);
         size_t end_line = req.body.find("\r\n", start_val);
         
         if (dash_pos != std::string::npos && dash_pos < end_line) {
             try {
                 range_start = std::stoull(req.body.substr(start_val, dash_pos - start_val));
                 std::string end_str = req.body.substr(dash_pos + 1, end_line - (dash_pos + 1));
                 if (!end_str.empty()) range_end = std::stoull(end_str);
             } catch (...) { is_range = false; }
         }
    }

    if (range_start >= file_size || range_end >= file_size || range_start > range_end) {
         std::string cr = "bytes */" + std::to_string(file_size);
         res->status(416);
         res->set_header("Content-Range", cr);
         res->set_header("Content-Length", "0");
         res->send("");
         return;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        res->status(500);
        res->send("Read error");
        return;
    }

    f.seekg(range_start);
    size_t chunk_size = range_end - range_start + 1;
    std::vector<char> buffer(chunk_size);
    f.read(buffer.data(), chunk_size);
    std::string body(buffer.begin(), buffer.end());

    if (is_range) {
         res->status(206);
         res->set_header("Content-Type", "application/octet-stream");
         res->set_header("Content-Range", "bytes " + std::to_string(range_start) + "-" + std::to_string(range_end) + "/" + std::to_string(file_size));
         res->send(body);
    } else {
         res->status(200);
         res->set_header("Content-Type", "application/octet-stream");
         res->send(body);
    }
}

void DashboardManager::register_routes() {
    if (!server_) return;

    // MINIMAL TEST: Zero dependencies, pure stack
    // MINIMAL TEST: Zero dependencies, pure stack
    server_->register_handler("/", [](const auto&, std::shared_ptr<mm_rec::net::Response> res) {
        const char* body = "<html><body><h1>WORKS</h1></body></html>";
        res->set_header("Content-Type", "text/html");
        res->send(body);
    });

    // Static CSS
    server_->register_handler("/static/style.css", [](const mm_rec::net::Request&, std::shared_ptr<mm_rec::net::Response> res) {
        res->set_header("Content-Type", "text/css");
        res->send(std::string(assets::get_style_css()));
    });

    // Static JS
    server_->register_handler("/static/app.js", [](const mm_rec::net::Request&, std::shared_ptr<mm_rec::net::Response> res) {
        res->set_header("Content-Type", "application/javascript");
        res->send(std::string(assets::get_app_js()));
    });
    
    // API Stats
    server_->register_handler("/api/stats", [this](const mm_rec::net::Request&, std::shared_ptr<mm_rec::net::Response> res) {

        std::stringstream ss;
        ss << "{";
        {
             // Protect string access if needed, though usually set once
             ss << "\"run_name\": \"" << stats_.current_run_name << "\",";
        }
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
        
        res->set_header("Content-Type", "application/json");
        res->send(ss.str());
    });
    
    // API Hardware
    server_->register_handler("/api/hardware", [](const mm_rec::net::Request&, std::shared_ptr<mm_rec::net::Response> res) {
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
        res->set_header("Content-Type", "application/json");
        res->send(json.str());
    });
    
    // Stop Signal
    server_->register_handler("/api/stop", [this](const mm_rec::net::Request&, std::shared_ptr<mm_rec::net::Response> res) {
        stats_.should_stop = true;
        LOG_INFO("Stop signal received from Dashboard.");
        res->send("Stopping...");
    });

    // History API
    server_->register_handler("/api/history", [](const mm_rec::net::Request&, std::shared_ptr<mm_rec::net::Response> res) {
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
        res->set_header("Content-Type", "application/json");
        res->send(json.str());
    });
    
    // API Runs List
    server_->register_handler("/api/runs", [](const mm_rec::net::Request&, std::shared_ptr<mm_rec::net::Response> res) {
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
        res->set_header("Content-Type", "application/json");
        res->send(ss.str());
    });

    // API Stop Run
    server_->register_handler("/api/runs/stop", [this](const mm_rec::net::Request&, std::shared_ptr<mm_rec::net::Response> res) {
        if (!RunManager::is_job_running()) {
             res->status(400);
             res->set_header("Content-Type", "application/json");
             res->send("{\"error\": \"No job is running\"}");
             return;
        }
        RunManager::stop_job();
        LOG_INFO("Job stopped via API.");
        res->set_header("Content-Type", "application/json");
        res->send("{\"status\": \"stopped\"}");
    });

    // API Start Run (New)
    server_->register_handler("/api/runs/start", [this](const mm_rec::net::Request& req_obj, std::shared_ptr<mm_rec::net::Response> res) {
        const std::string& req = req_obj.body;
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
            return req.substr(start, end - start);
        };

        std::string name = get_json_val("run_name");
        std::string config_file = get_json_val("config_file");
        std::string data_file = get_json_val("data_file");

        res->set_header("Content-Type", "application/json");

        if (name.empty() || config_file.empty()) {
            res->status(400);
            res->send("{\"error\": \"Missing run_name or config_file\"}");
            return;
        }

        if (RunManager::is_job_running()) {
            res->status(409);
            res->send("{\"error\": \"Job already running\"}");
            return;
        }

        // 1. Setup Run Directory
        std::string run_dir = "runs/" + name;
        if (std::filesystem::exists(run_dir)) {
             res->status(409);
             res->send("{\"error\": \"Run already exists\"}");
             return;
        }
        std::filesystem::create_directories(run_dir);

        // 2. Prepare config object
        TrainingJobConfig job_config;
        job_config.run_name = name;
        
        // Resolve Config Path
        std::string resolved_config = config_file;
        if (std::filesystem::exists("configs/" + config_file)) {
            resolved_config = "configs/" + config_file;
        }
        
        job_config.config_path = resolved_config; 
        job_config.data_path = data_file;     

        // 4. Start
        if (RunManager::start_job(job_config)) {
             res->send("{\"status\": \"started\"}");
        } else {
             res->status(500);
             res->send("{\"error\": \"Failed to start job\"}");
        }
    });

    // API Resume Run
    server_->register_handler("/api/runs/resume", [this](const mm_rec::net::Request& req_obj, std::shared_ptr<mm_rec::net::Response> res) {
         const std::string& req = req_obj.body;
         auto get_param = [&](const std::string& key) -> std::string {
             size_t pos = req.find(key + "=");
             if (pos == std::string::npos) return "";
             size_t start = pos + key.size() + 1;
             size_t end = req.find_first_of("& \r\n", start);
             if (end == std::string::npos) return req.substr(start);
             return req.substr(start, end - start);
         };

         std::string run_name = get_param("name");
         res->set_header("Content-Type", "application/json");

         if (run_name.empty()) {
             res->status(400);
             res->send("{\"error\": \"Missing name parameter\"}");
             return;
         }

         if (RunManager::is_job_running()) {
             res->status(409);
             res->send("{\"error\": \"Job already running\"}");
             return;
         }

         if (!RunManager::run_exists(run_name)) {
             res->status(404);
             res->send("{\"error\": \"Run not found\"}");
             return;
         }
         
         std::string run_dir = RunManager::get_run_dir(run_name);
         
         // 1. Verify Config (Isolated)
         std::string config_path = run_dir + "/config.ini";
         if (!std::filesystem::exists(config_path)) {
             config_path = run_dir + "/config.txt"; // Fallback
         }
         
         if (!std::filesystem::exists(config_path)) {
              res->status(500);
              res->send("{\"error\": \"Missing config in run dir\"}");
              return;
         }
         
         // 2. Verify Checkpoint (Data Integrity)
         std::string checkpoint_path = run_dir + "/checkpoint.bin";
         if (!std::filesystem::exists(checkpoint_path)) {
              res->status(500);
              res->send("{\"error\": \"No checkpoint found (cannot resume)\"}");
              return;
         }

         TrainingJobConfig config;
         config.run_name = run_name;
         config.config_path = config_path;
         config.data_path = "training_data.bin"; 
         
         if (RunManager::start_job(config)) {
             res->send("{\"status\": \"started\"}");
         } else {
             res->status(500);
             res->send("{\"error\": \"Failed to start\"}");
         }
    });

    // API Delete Run
    server_->register_handler("/api/runs/delete", [this](const mm_rec::net::Request& req_obj, std::shared_ptr<mm_rec::net::Response> res) {
         const std::string& req = req_obj.body;
         auto get_param = [&](const std::string& key) -> std::string {
             size_t pos = req.find(key + "=");
             if (pos == std::string::npos) return "";
             size_t start = pos + key.size() + 1;
             size_t end = req.find_first_of("& \r\n", start);
             if (end == std::string::npos) return req.substr(start);
             return req.substr(start, end - start);
         };

         std::string run_name = get_param("name");
         res->set_header("Content-Type", "application/json");

         if (run_name.empty()) {
             res->status(400);
             res->send("{\"error\": \"Missing name parameter\"}");
             return;
         }

         if (RunManager::is_job_running()) {
             res->status(409);
             res->send("{\"error\": \"Cannot delete whle a job is running\"}");
             return;
         }

         if (RunManager::delete_run(run_name)) {
             res->send("{\"status\": \"deleted\"}");
         } else {
             res->status(500);
             res->send("{\"error\": \"Delete failed\"}");
         }
    });

    // API Get Run Config
    server_->register_handler("/api/runs/config", [](const mm_rec::net::Request& req_obj, std::shared_ptr<mm_rec::net::Response> res) {
         const std::string& req = req_obj.body;
         std::string first_line = req.substr(0, req.find("\r\n"));
         size_t q_pos = first_line.find("?name=");
         
         res->set_header("Content-Type", "application/json");

         if (q_pos == std::string::npos) {
             res->status(400);
             res->send("{\"error\": \"Missing name parameter\"}");
             return;
         }
         
         size_t end_pos = first_line.find(" ", q_pos);
         std::string run_name = first_line.substr(q_pos + 6, end_pos - (q_pos + 6));
         
         if (run_name.empty() || run_name.find("..") != std::string::npos || run_name.find("/") != std::string::npos) {
             res->status(400);
             res->send("{\"error\": \"Invalid run name\"}");
             return;
         }

         std::string config_path = "runs/" + run_name + "/config.ini";
         if (!std::filesystem::exists(config_path)) {
             // Fallback for older runs
             config_path = "runs/" + run_name + "/config.txt";
             if (!std::filesystem::exists(config_path)) {
                res->status(404);
                res->send("{\"error\": \"Config not found\"}");
                return;
             }
         }
         
         std::ifstream f(config_path);
         if (!f.good()) {
             res->status(500);
             res->send("{\"error\": \"Read error\"}");
             return;
         }
         
         std::stringstream buffer;
         buffer << f.rdbuf();
         
         // Escape JSON string
         std::string content = buffer.str();
         std::string json_content;
         for (char c : content) {
             if (c == '"') json_content += "\\\"";
             else if (c == '\n') json_content += "\\n";
             else if (c == '\r') {} 
             else if (c == '\\') json_content += "\\\\";
             else json_content += c;
         }
         
         res->send("{\"content\": \"" + json_content + "\"}");
    });

    // API Get Run Logs
    server_->register_handler("/api/runs/logs", [](const mm_rec::net::Request& req_obj, std::shared_ptr<mm_rec::net::Response> res) {
         const std::string& req = req_obj.body;
         std::string first_line = req.substr(0, req.find("\r\n"));
         size_t q_pos = first_line.find("?name=");
         res->set_header("Content-Type", "application/json");
         
         if (q_pos == std::string::npos) {
             res->status(400);
             res->send("{\"error\": \"Missing name parameter\"}");
             return;
         }
         
         size_t end_pos = first_line.find(" ", q_pos);
         std::string run_name = first_line.substr(q_pos + 6, end_pos - (q_pos + 6));
         
         if (run_name.empty() || run_name.find("..") != std::string::npos || run_name.find("/") != std::string::npos) {
             res->status(400);
             res->send("{\"error\": \"Invalid run name\"}");
             return;
         }

         std::string log_path = "runs/" + run_name + "/train.log";
         if (!std::filesystem::exists(log_path)) {
             res->status(404);
             res->send("{\"error\": \"Log not found\"}");
             return;
         }
         
         std::ifstream f(log_path, std::ios::binary);
         if (!f.good()) {
             res->status(500);
             res->send("{\"error\": \"Read error\"}");
             return;
         }
         
         // Read last N bytes
         const size_t MAX_LOG_SIZE = 16384; // 16KB tail
         f.seekg(0, std::ios::end);
         size_t file_size = f.tellg();
         size_t start_pos = (file_size > MAX_LOG_SIZE) ? (file_size - MAX_LOG_SIZE) : 0;
         
         f.seekg(start_pos);
         std::string content(file_size - start_pos, '\0');
         f.read(&content[0], content.size());
         
         // JSON Escape
         std::string json_content;
         for (char c : content) {
             if (c == '"') json_content += "\\\"";
             else if (c == '\n') json_content += "\\n";
             else if (c == '\r') {}
             else if (c == '\\') json_content += "\\\\";
             else if ((unsigned char)c < 32) {} // Skip control chars
             else json_content += c;
         }
         
         res->send("{\"content\": \"" + json_content + "\"}");
    });

    // --- Phase 3 Config & Data ---

    // API List Configs
    server_->register_handler("/api/configs", [](const mm_rec::net::Request&, std::shared_ptr<mm_rec::net::Response> res) {
        std::stringstream ss;
        ss << "[";
        bool first = true;
        try {
            if (std::filesystem::exists("configs")) {
                for (const auto& entry : std::filesystem::directory_iterator("configs")) {
                    std::string params = entry.path().extension().string();
                    if (params == ".ini" || params == ".txt") {
                        if (!first) ss << ",";
                        ss << "\"" << entry.path().filename().string() << "\"";
                        first = false;
                    }
                }
            }
        } catch (...) {}
        ss << "]";
        res->set_header("Content-Type", "application/json");
        res->send(ss.str());
    });
    
    // API Read Config
    server_->register_handler("/api/configs/read", [](const mm_rec::net::Request& req_obj, std::shared_ptr<mm_rec::net::Response> res) {
        const std::string& req = req_obj.body;
        std::string first_line = req.substr(0, req.find("\r\n"));
        size_t q_pos = first_line.find("?name=");
        
        res->set_header("Content-Type", "application/json");
        
        if (q_pos == std::string::npos) {
            res->status(400);
            res->send("{\"error\": \"Missing name parameter\"}");
            return;
        }
         
        size_t end_pos = first_line.find(" ", q_pos);
        std::string filename = first_line.substr(q_pos + 6, end_pos - (q_pos + 6));
         
        if (filename.find("/") != std::string::npos || filename.find("..") != std::string::npos) {
             res->status(403);
             res->send("{\"error\": \"Invalid filename\"}");
             return;
        }

        std::string path = "configs/" + filename;
        if (!std::filesystem::exists(path)) {
            res->status(404);
            res->send("{\"error\": \"Config not found\"}");
            return;
        }
        
        std::ifstream f(path);
        if (!f.good()) {
            res->status(500);
            res->send("{\"error\": \"Read error\"}");
            return;
        }
        
        std::stringstream buffer;
        buffer << f.rdbuf();
        
        std::string content = buffer.str();
        std::string json_content;
         for (char c : content) {
             if (c == '"') json_content += "\\\"";
             else if (c == '\n') json_content += "\\n";
             else if (c == '\r') {}
             else if (c == '\\') json_content += "\\\\";
             else json_content += c;
         }
         
         res->send("{\"content\": \"" + json_content + "\"}");
    });

    // API Create Config
    server_->register_handler("/api/configs/create", [](const mm_rec::net::Request& req_obj, std::shared_ptr<mm_rec::net::Response> res) {
        const std::string& req = req_obj.body;
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
        
        res->set_header("Content-Type", "application/json");

        if (filename.empty() || content.empty()) {
            res->status(400);
            res->send("{\"error\": \"Missing filename or content\"}");
            return;
        }
        
        if (filename.find(".ini") == std::string::npos) filename += ".ini";
        if (filename.find("/") != std::string::npos || filename.find("..") != std::string::npos) {
            res->status(403);
            res->send("{\"error\": \"Invalid filename\"}");
            return;
        }

        std::filesystem::create_directories("configs"); // Ensure exists
        std::string full_path = "configs/" + filename;

        std::ofstream file(full_path);
        if (!file.good()) {
            res->status(500);
            res->send("{\"error\": \"Failed to write file\"}");
            return;
        }
        file << content;
        file.close();
        res->send("{\"status\": \"created\", \"file\": \"" + filename + "\"}");
    });

    // API List Datasets
    server_->register_handler("/api/datasets", [](const mm_rec::net::Request&, std::shared_ptr<mm_rec::net::Response> res) {
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
        res->set_header("Content-Type", "application/json");
        res->send(ss.str());
    });

    // API Upload Dataset
    server_->register_handler("/api/datasets/upload", [](const mm_rec::net::Request& req_obj, std::shared_ptr<mm_rec::net::Response> res) {
        const std::string& req = req_obj.body;
        size_t header_end = req.find("\r\n\r\n");
        if (header_end == std::string::npos) {
            res->status(400);
            res->send("Bad Request");
            return;
        }
        
        std::string body = req.substr(header_end + 4); 
        std::string first_line = req.substr(0, req.find("\r\n"));
        size_t q_pos = first_line.find("?name=");
        
        res->set_header("Content-Type", "application/json");

        if (q_pos == std::string::npos) {
            res->status(400);
            res->send("{\"error\": \"Missing name query param\"}");
            return;
        }
        
        size_t end_pos = first_line.find(" ", q_pos);
        std::string filename = first_line.substr(q_pos + 6, end_pos - (q_pos + 6));
        
        if (filename.find(".bin") == std::string::npos) filename += ".bin";
        if (filename.find("/") != std::string::npos || filename.find("..") != std::string::npos) {
             res->status(403);
             res->send("{\"error\": \"Invalid filename\"}");
             return;
        }
        
        std::ofstream file(filename, std::ios::binary);
        if (!file.good()) {
            res->status(500);
            res->send("{\"error\": \"Failed to write file\"}");
            return;
        }
        file.write(body.data(), body.size());
        file.close();
        res->send("{\"status\": \"uploaded\", \"size\": " + std::to_string(body.size()) + "}");
    });

    // --- Phase 3 Model & Inference ---

    // --- Phase 3 Model & Inference ---

    // API List Models
    server_->register_handler("/api/models", [](const mm_rec::net::Request&, std::shared_ptr<mm_rec::net::Response> res) {
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
        res->set_header("Content-Type", "application/json");
        res->send(ss.str());
    });

    // API Download Model
    server_->register_handler("/api/models/download", [](const mm_rec::net::Request& req, std::shared_ptr<mm_rec::net::Response> res) {
        std::string first_line = req.body.substr(0, req.body.find("\r\n"));
        size_t q_pos = first_line.find("?name=");
        if (q_pos == std::string::npos) {
            res->status(400); 
            res->send("Missing name");
            return;
        }
        size_t end_pos = first_line.find(" ", q_pos);
        std::string name = first_line.substr(q_pos + 6, end_pos - (q_pos + 6));
        
        if (name.find("..") != std::string::npos) {
            res->status(403);
            res->send("Invalid path");
            return;
        }
        
        std::string path = name;
        if (name.find("/") != std::string::npos) {
             path = "runs/" + name; 
        } else {
             path = "./" + name;
        }

        // Use the common helper
        serve_file_with_range(path, req, res);
    });

    // --- SSE Real-time Events ---
    
    server_->register_handler("/api/events", [this](const auto& req, std::shared_ptr<mm_rec::net::Response> res) {
        res->enable_sse();
        server_->register_sse_client(res);
        
        // Send initial connection event
        res->send_event("connected", "{\"status\": \"ready\", \"timestamp\": " + std::to_string(std::time(nullptr)) + "}");
        
        // Keep connection alive with heartbeat
        // This handler blocks but on a ThreadPool thread, so it's OK
        while (req.is_connected()) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            
            // Send ping to keep connection alive and detect client disconnect
            try {
                res->send_event("ping", "{\"timestamp\": " + std::to_string(std::time(nullptr)) + "}");
            } catch (...) {
                // Client disconnected
                break;
            }
        }
        
        // Client disconnected, will be cleaned up by weak_ptr
    });

    // API Inference


} // register_routes

} // namespace mm_rec
