#include "mm_rec/utils/run_manager.h"
#include "mm_rec/utils/ui.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <ctime>

namespace mm_rec {
namespace fs = std::filesystem;

std::string RunManager::get_run_dir(const std::string& run_name) {
    return get_runs_dir() + "/" + run_name;
}

bool RunManager::run_exists(const std::string& run_name) {
    return fs::exists(get_run_dir(run_name));
}

bool RunManager::create_run(const std::string& run_name) {
    std::string run_dir = get_run_dir(run_name);
    
    if (fs::exists(run_dir)) {
        return false; // Already exists
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
