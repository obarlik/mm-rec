#pragma once

#include <string>
#include <vector>
#include "mm_rec/jobs/job_training.h"

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

/**
 * Interface for Run Management Service
 */
class IRunManager {
public:
    virtual ~IRunManager() = default;

    // Run Listing & Info
    virtual std::vector<RunInfo> list_runs() = 0;
    virtual RunInfo get_run_info(const std::string& run_name) = 0;
    virtual RunStatus get_run_status(const std::string& run_name) = 0;
    virtual bool run_exists(const std::string& run_name) = 0;
    
    // Directory Management
    virtual bool create_run(const std::string& run_name) = 0;
    virtual bool delete_run(const std::string& run_name) = 0;
    
    // Active Job Control
    virtual bool start_job(const TrainingJobConfig& config) = 0;
    virtual void stop_job() = 0;
    virtual bool is_job_running() = 0;

    // Paths
    virtual std::string get_runs_dir() const = 0;
    virtual std::string get_run_dir(const std::string& run_name) = 0;
};

} // namespace mm_rec
