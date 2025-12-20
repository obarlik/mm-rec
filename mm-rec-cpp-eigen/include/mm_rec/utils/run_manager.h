#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

namespace mm_rec {

enum class RunStatus {
    RUNNING,
    STOPPED,
    COMPLETED,
    FAILED,
    UNKNOWN
};

struct RunInfo {
    std::string name;
    RunStatus status;
    std::string status_str;
    
    // Timestamps
    std::chrono::system_clock::time_point created;
    std::chrono::system_clock::time_point modified;
    
    // Metrics
    int total_epochs = 0;
    int current_epoch = 0;
    float best_loss = 0.0f;
    float current_loss = 0.0f;
    
    // Files
    bool has_checkpoint = false;
    bool has_log = false;
    bool has_config = false;
    size_t total_size_mb = 0;
};

class RunManager {
public:
    // List all runs in runs/ directory
    static std::vector<RunInfo> list_runs();
    
    // Get detailed info about a specific run
    static RunInfo get_run_info(const std::string& run_name);
    
    // Get current status of a run
    static RunStatus get_run_status(const std::string& run_name);
    
    // Create new run directory structure
    static bool create_run(const std::string& run_name);
    
    // Delete a run directory
    static bool delete_run(const std::string& run_name);
    
    // Check if run exists
    static bool run_exists(const std::string& run_name);
    
    // Get runs directory path
    static std::string get_runs_dir() { return "runs"; }
    
    // Get specific run directory path
    static std::string get_run_dir(const std::string& run_name);
    
private:
    static std::string status_to_string(RunStatus status);
    static size_t get_directory_size(const std::filesystem::path& dir);
};

} // namespace mm_rec
