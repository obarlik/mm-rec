#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <mutex>
#include "mm_rec/application/i_run_manager.h"
#include "mm_rec/jobs/job_training.h"

namespace mm_rec {

// RunStatus is defined in IRunManager

// RunInfo is defined in IRunManager

/**
 * RunManager Implementation
 * 
 * Manages training runs (directories, configs, logs) and the active training job.
 * Note: This service is thread-safe.
 */
class RunManager : public IRunManager {
public:
    RunManager();
    ~RunManager() override;

    // Run Listing & Info
    std::vector<RunInfo> list_runs() override;
    RunInfo get_run_info(const std::string& run_name) override;
    RunStatus get_run_status(const std::string& run_name) override;
    bool run_exists(const std::string& run_name) override;
    
    // Directory Management
    bool create_run(const std::string& run_name) override;
    bool delete_run(const std::string& run_name) override;
    
    // Active Job Control
    bool start_job(const TrainingJobConfig& config) override;
    void stop_job() override;
    bool is_job_running() override;

    // Paths
    std::string get_runs_dir() const override { return "runs"; }
    std::string get_run_dir(const std::string& run_name) override;
    
private:
    std::string status_to_string(RunStatus status);
    size_t get_directory_size(const std::filesystem::path& dir);
    
    // State
    std::unique_ptr<JobTraining> active_job_;
    std::string active_run_name_;
    std::mutex mutex_;
};

} // namespace mm_rec
