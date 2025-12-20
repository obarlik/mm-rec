/**
 * Interactive Server Mode Command
 * 
 * Provides a REPL to manage jobs and query status.
 */

#include "commands.h"
#include "mm_rec/jobs/job_training.h"
#include "mm_rec/utils/dashboard_manager.h"
#include "mm_rec/utils/ui.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/core/vulkan_backend.h"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <atomic>
#include <thread>

using namespace mm_rec;
using namespace mm_rec::ui;

// Global job instance (simple single-job server for now)
static std::unique_ptr<JobTraining> current_job;

void print_help() {
    std::cout << "\nAvailable Commands:\n";
    std::cout << "  start <run_name> <config> <data>  Start a training job\n";
    std::cout << "  stop                              Stop current job\n";
    std::cout << "  status                            Show current status\n";
    std::cout << "  metrics [on|off]                  Toggle metric collection\n";
    std::cout << "  tune                              Run hardware auto-tuner\n";
    std::cout << "  help                              Show this help\n";
    std::cout << "  exit                              Shutdown server\n";
    std::cout << "\n";
}

int cmd_server(int argc, char* argv[]) {
    ui::print_header("MM-Rec Interactive Server");
    ui::info("Type 'help' for commands.");
    
    // Start Dashboard immediately in idle mode
    DashboardManager::instance().start(8085);
    
    bool server_running = true;
    std::string line;
    
    while (server_running) {
        std::cout << color::BOLD << "> " << color::RESET << std::flush;
        
        if (!std::getline(std::cin, line)) break; // EOF
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string cmd;
        ss >> cmd;
        
        if (cmd == "exit" || cmd == "quit") {
            if (current_job && current_job->is_running()) {
                ui::info("Stopping active job...");
                current_job->stop();
                current_job->join();
            }
            server_running = false;
        }
        else if (cmd == "help") {
            print_help();
        }
        else if (cmd == "tune") {
             // Hardware Tuner
             // TODO: Expose from AutoTuner
             ui::info("Tuning logic not yet exposed to server, running init check...");
             if (VulkanBackend::get().init()) {
                 ui::success("GPU Available");
             }
        }
        else if (cmd == "status") {
            if (current_job && current_job->is_running()) {
                auto& stats = DashboardManager::instance().stats();
                std::cout << "Job: RUNNING\n";
                std::cout << "Step: " << stats.current_step << "/" << stats.total_steps << "\n";
                std::cout << "Loss: " << stats.current_loss << "\n";
                std::cout << "Speed: " << stats.current_speed << " tps\n";
            } else {
                std::cout << "Job: IDLE\n";
            }
        }
        else if (cmd == "stop") {
            if (current_job && current_job->is_running()) {
                current_job->stop();
                current_job->join();
                ui::success("Job stopped.");
            } else {
                ui::warning("No job running.");
            }
        }
        else if (cmd == "start") {
            std::string run_name, config_path, data_path;
            if (ss >> run_name >> config_path >> data_path) {
                if (current_job && current_job->is_running()) {
                    ui::error("A job is already running. Stop it first.");
                    continue;
                }
                
                TrainingJobConfig config;
                config.run_name = run_name;
                config.config_path = config_path;
                config.data_path = data_path;
                
                current_job = std::make_unique<JobTraining>();
                if (current_job->start(config)) {
                    ui::success("Job started: " + run_name);
                }
            } else {
                ui::error("Usage: start <run_name> <config_path> <data_path>");
            }
        }
        else {
            ui::warning("Unknown command: " + cmd);
        }
    }
    
    DashboardManager::instance().stop();
    return 0;
}
