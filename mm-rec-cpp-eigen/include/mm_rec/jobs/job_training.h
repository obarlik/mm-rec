#pragma once

#include <string>
#include <atomic>
#include <thread>
#include <functional>

namespace mm_rec {

struct TrainingJobConfig {
    std::string config_path;
    std::string data_path;
    std::string run_name;
    bool enable_metrics = true;
};

class JobTraining {
public:
    JobTraining();
    ~JobTraining();

    // Non-blocking start
    bool start(const TrainingJobConfig& config);
    
    // Stop the job
    void stop();
    
    // Wait for completion (blocking)
    void join();
    
    bool is_running() const { return running_; }
    
private:
    void run_internal(TrainingJobConfig config);

    std::atomic<bool> running_{false};
    std::atomic<bool> stop_signal_{false};
    std::thread worker_;
};

} // namespace mm_rec
