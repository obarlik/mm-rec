#include "mm_rec/application/run_manager.h"
#include "mm_rec/utils/ui.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <memory>
#include <mutex>
#include "mm_rec/jobs/job_training.h"

namespace mm_rec {
namespace fs = std::filesystem;

// Static state
static std::unique_ptr<JobTraining> active_job_ = nullptr;
static std::string active_run_name_ = "";  // Track which run owns the active job
static std::mutex run_manager_mutex_;

bool RunManager::start_job(const TrainingJobConfig& config_in) {
    std::lock_guard<std::mutex> lock(run_manager_mutex_);
    if (active_job_ && active_job_->is_running()) {
        ui::error("Job already running for: " + active_run_name_);
        return false;
    }

    // --- ISOLATION LOGIC ---
    TrainingJobConfig isolated_config = config_in;
    
    // 1. Create Run Directory
    if (isolated_config.run_name.empty()) {
        isolated_config.run_name = "run_" + std::to_string(std::time(nullptr));
    }
    if (!create_run(isolated_config.run_name)) {
        ui::error("Failed to create run directory: " + isolated_config.run_name);
        return false;
    }
    std::string run_dir = get_run_dir(isolated_config.run_name);

    // 2. Copy Config File (if source exists and is different from dest)
    if (fs::exists(config_in.config_path)) {
        std::string dest_config = run_dir + "/config.ini";
        try {
            // Check if source and dest are the same file to avoid self-copy error
            bool same_file = false;
            if (fs::exists(dest_config)) {
               if (fs::equivalent(config_in.config_path, dest_config)) same_file = true;
            }

            if (!same_file) {
                 fs::copy_file(config_in.config_path, dest_config, fs::copy_options::overwrite_existing);
            }
            isolated_config.config_path = dest_config; // Point to isolated copy
        } catch (const std::exception& e) {
            ui::error(std::string("Failed to copy config to run dir: ") + e.what());
            return false;
        }
    }

    // 3. Copy Dataset (STRICT ISOLATION - Physical Copy)
    if (fs::exists(config_in.data_path)) {
        std::string data_filename = fs::path(config_in.data_path).filename().string();
        std::string dest_data = run_dir + "/" + data_filename;
        try {
            // Remove if exists
            if (fs::exists(dest_data)) fs::remove(dest_data);
            
            // Physical Copy
            fs::copy_file(config_in.data_path, dest_data, fs::copy_options::overwrite_existing);
            
            isolated_config.data_path = dest_data; // Point to local copy
        } catch (const std::exception& e) {
             // Fallback: Use absolute original path if copy fails
             ui::warning(std::string("Dataset copy failed: ") + e.what() + ". Using original path.");
             isolated_config.data_path = fs::absolute(config_in.data_path).string();
        }
    }

    // Set active run ownership
    active_run_name_ = isolated_config.run_name;
    
    active_job_ = std::make_unique<JobTraining>();
    try {
        // Start the job (Training happens in a separate thread inside, 
        // but initialization like dataset loading might happen here depending on implementation)
        // If start() is blocking or does init, we catch errors here.
        // Assuming active_job_->start() spawns a thread, but might throw during setup.
        bool success = active_job_->start(isolated_config);
        if (!success) {
            // Failed to start - clear ownership
            active_run_name_ = "";
        }
        return success;
    } catch (const std::exception& e) {
        std::cerr << "Message: Failed to start job: " << e.what() << std::endl;
        active_run_name_ = "";  // Clear ownership on failure
        return false;
    }
}

// In .cpp file, add static mutex
// static std::mutex run_manager_mutex_; // Moved to top

void RunManager::stop_job() {
    std::lock_guard<std::mutex> lock(run_manager_mutex_);
    if (active_job_ && active_job_->is_running()) {
        std::cerr << "[RunManager] Stopping job..." << std::endl;
        active_job_->stop();
        std::cerr << "[RunManager] Joining job..." << std::endl;
        active_job_->join();
        std::cerr << "[RunManager] Job joined. Resetting..." << std::endl;
        active_job_.reset();
        active_run_name_ = "";
        std::cerr << "[RunManager] Job stopped." << std::endl;
    } else {
        std::cerr << "[RunManager] No job to stop." << std::endl;
    }
}

bool RunManager::is_job_running() {
    return active_job_ && active_job_->is_running();
}

std::string RunManager::get_run_dir(const std::string& run_name) {
    return get_runs_dir() + "/" + run_name;
}

bool RunManager::run_exists(const std::string& run_name) {
    return fs::exists(get_run_dir(run_name));
}

bool RunManager::create_run(const std::string& run_name) {
    std::string run_dir = get_run_dir(run_name);
    
    if (fs::exists(run_dir)) {
        return true; // Already exists (Idempotent success)
    }
    
    try {
        // Create runs directory if needed
        if (!fs::exists(get_runs_dir())) {
            fs::create_directory(get_runs_dir());
        }
        
        // Create run directory
        fs::create_directory(run_dir);
        return true;
    } catch (...) {
        return false;
    }
}

bool RunManager::delete_run(const std::string& run_name) {
    std::string run_dir = get_run_dir(run_name);
    
    if (!fs::exists(run_dir)) {
        return false;
    }
    
    try {
        fs::remove_all(run_dir);
        return true;
    } catch (...) {
        return false;
    }
}

RunStatus RunManager::get_run_status(const std::string& run_name) {
    std::string run_dir = get_run_dir(run_name);
    
    if (!fs::exists(run_dir)) {
        return RunStatus::UNKNOWN;
    }
    
    // Check for lock file (running)
    if (fs::exists(run_dir + "/.lock")) {
        return RunStatus::RUNNING;
    }
    
    // Check for log file with errors
    std::string log_path = run_dir + "/train.log";
    if (fs::exists(log_path)) {
        std::ifstream log(log_path);
        std::string line;
        while (std::getline(log, line)) {
            if (line.find("ERROR") != std::string::npos || 
                line.find("FAILED") != std::string::npos ||
                line.find("Explosion") != std::string::npos) {
                return RunStatus::FAILED;
            }
        }
    }
    
    // Check for final checkpoint
    if (fs::exists(run_dir + "/kernel_adaptive_final.bin")) {
        return RunStatus::COMPLETED;
    }
    
    // Has checkpoint but not final
    if (fs::exists(run_dir + "/checkpoint_latest.bin")) {
        return RunStatus::STOPPED;
    }
    
    return RunStatus::UNKNOWN;
}

std::string RunManager::status_to_string(RunStatus status) {
    switch (status) {
        case RunStatus::RUNNING: return "RUNNING";
        case RunStatus::STOPPED: return "STOPPED";
        case RunStatus::COMPLETED: return "COMPLETED";
        case RunStatus::FAILED: return "FAILED";
        default: return "UNKNOWN";
    }
}

size_t RunManager::get_directory_size(const fs::path& dir) {
    size_t size = 0;
    try {
        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            if (fs::is_regular_file(entry)) {
                size += fs::file_size(entry);
            }
        }
    } catch (...) {}
    return size;
}

RunInfo RunManager::get_run_info(const std::string& run_name) {
    RunInfo info;
    info.name = run_name;
    
    std::string run_dir = get_run_dir(run_name);
    
    if (!fs::exists(run_dir)) {
        info.status = RunStatus::UNKNOWN;
        info.status_str = "NOT FOUND";
        return info;
    }
    
    // Get status
    info.status = get_run_status(run_name);
    info.status_str = status_to_string(info.status);
    
    // Get timestamps
    try {
        auto ftime = fs::last_write_time(run_dir);
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
        info.modified = sctp;
        info.created = sctp; // Approximate
    } catch (...) {}
    
    // Check for files
    info.has_checkpoint = fs::exists(run_dir + "/checkpoint_latest.bin");
    info.has_log = fs::exists(run_dir + "/train.log");
    info.has_config = fs::exists(run_dir + "/config.txt");
    
    // Parse checkpoint metadata if available
    std::string latest_ckpt = run_dir + "/checkpoint_latest.bin";
    if (fs::exists(latest_ckpt)) {
        std::ifstream ckpt(latest_ckpt, std::ios::binary);
        if (ckpt.good()) {
            // Read epoch (first 4 bytes after magic/version)
            ckpt.seekg(8); // Skip magic and version
            ckpt.read(reinterpret_cast<char*>(&info.current_epoch), sizeof(int));
            ckpt.read(reinterpret_cast<char*>(&info.current_loss), sizeof(float));
        }
    }
    
    std::string best_ckpt = run_dir + "/checkpoint_best.bin";
    if (fs::exists(best_ckpt)) {
        std::ifstream ckpt(best_ckpt, std::ios::binary);
        if (ckpt.good()) {
            ckpt.seekg(12); // Skip to loss field
            ckpt.read(reinterpret_cast<char*>(&info.best_loss), sizeof(float));
        }
    }
    
    // Get directory size
    info.total_size_mb = get_directory_size(run_dir) / (1024 * 1024);
    
    return info;
}

std::vector<RunInfo> RunManager::list_runs() {
    std::vector<RunInfo> runs;
    
    if (!fs::exists(get_runs_dir())) {
        return runs;
    }
    
    try {
        for (const auto& entry : fs::directory_iterator(get_runs_dir())) {
            if (fs::is_directory(entry)) {
                std::string run_name = entry.path().filename().string();
                runs.push_back(get_run_info(run_name));
            }
        }
    } catch (...) {}
    
    // Sort by modified time (newest first)
    std::sort(runs.begin(), runs.end(), [](const RunInfo& a, const RunInfo& b) {
        return a.modified > b.modified;
    });
    
    return runs;
}

} // namespace mm_rec
