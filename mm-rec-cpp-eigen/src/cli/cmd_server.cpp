/**
 * Interactive Server Mode Command
 * 
 * Provides a REPL to manage jobs and query status.
 */

#include "commands.h"
#include "mm_rec/jobs/job_training.h"
#include "mm_rec/utils/dashboard_manager.h"
#include "mm_rec/utils/run_manager.h"
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
    std::cout << "\n" << ui::color::BOLD << "Training Management:" << ui::color::RESET << "\n";
    std::cout << "  list                              List all training runs\n";
    std::cout << "  info <run_name>                   Show detailed run information\n";
    std::cout << "  start <run_name> <config> <data>  Start a training job\n";
    std::cout << "  resume <run_name>                 Resume existing training\n";
    std::cout << "  stop                              Stop current job\n";
    std::cout << "  status                            Show current job status\n";
    std::cout << "  clean [run_name]                  Delete a run\n";
    std::cout << "\n" << ui::color::BOLD << "System:" << ui::color::RESET << "\n";
    std::cout << "  tune                              Run hardware auto-tuner\n";
    std::cout << "  help                              Show this help\n";
    std::cout << "  exit                              Shutdown server\n";
    std::cout << "\n";
}

int cmd_server(int argc, char* argv[]) {
    // Check for --daemon flag
    bool daemon_mode = false;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--daemon") {
            daemon_mode = true;
            break;
        }
    }
    
    ui::print_header("MM-Rec Interactive Server");
    
    if (daemon_mode) {
        ui::info("Running in daemon mode (no REPL)");
        ui::info("Press Ctrl+C to stop");
    } else {
        ui::info("Type 'help' for commands.");
    }
    
    // Start Dashboard immediately in idle mode
    DashboardManager::instance().start(8085);
    
    // Daemon mode: just wait for signal
    if (daemon_mode) {
        // Keep running until signal received
        while (!DashboardManager::instance().should_stop()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        DashboardManager::instance().stop();
        return 0;
    }
    
    // Interactive mode: REPL
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
        else if (cmd == "list") {
            auto runs = RunManager::list_runs();
            
            if (runs.empty()) {
                ui::info("No training runs found.");
            } else {
                std::cout << "\n" << ui::color::BOLD << "Training Runs:" << ui::color::RESET << "\n\n";
                
                ui::Table table({"Name", "Status", "Epoch", "Loss", "Size"}, 15);
                for (const auto& run : runs) {
                    std::string epoch_str = std::to_string(run.current_epoch);
                    std::string loss_str = (run.current_loss > 0) ? std::to_string(run.current_loss).substr(0, 6) : "-";
                    std::string size_str = std::to_string(run.total_size_mb) + " MB";
                    table.add_row({run.name, run.status_str, epoch_str, loss_str, size_str});
                }
                table.finish();
            }
        }
        else if (cmd == "info") {
            std::string run_name;
            if (ss >> run_name) {
                auto info = RunManager::get_run_info(run_name);
                
                if (info.status == RunStatus::UNKNOWN) {
                    ui::error("Run not found: " + run_name);
                } else {
                    std::cout << "\n" << ui::color::BOLD << "Run: " << run_name << ui::color::RESET << "\n";
                    std::cout << "Status: " << info.status_str << "\n";
                    std::cout << "Epoch: " << info.current_epoch << "\n";
                    std::cout << "Current Loss: " << info.current_loss << "\n";
                    std::cout << "Best Loss: " << info.best_loss << "\n";
                    std::cout << "Size: " << info.total_size_mb << " MB\n\n";
                    std::cout << "Files:\n";
                    std::cout << "  Config: " << (info.has_config ? "✓" : "✗") << "\n";
                    std::cout << "  Log: " << (info.has_log ? "✓" : "✗") << "\n";
                    std::cout << "  Checkpoint: " << (info.has_checkpoint ? "✓" : "✗") << "\n\n";
                }
            } else {
                ui::error("Usage: info <run_name>");
            }
        }
        else if (cmd == "resume") {
            std::string run_name;
            if (ss >> run_name) {
                if (current_job && current_job->is_running()) {
                    ui::error("A job is already running. Stop it first.");
                    continue;
                }
                
                if (!RunManager::run_exists(run_name)) {
                    ui::error("Run not found: " + run_name);
                    continue;
                }
                
                std::string run_dir = RunManager::get_run_dir(run_name);
                std::string config_path = run_dir + "/config.txt";
                std::string data_path = "training_data.bin";
                
                if (!std::filesystem::exists(config_path)) {
                    ui::error("Config file not found in run directory.");
                    continue;
                }
                
                TrainingJobConfig config;
                config.run_name = run_name;
                config.config_path = config_path;
                config.data_path = data_path;
                
                current_job = std::make_unique<JobTraining>();
                if (current_job->start(config)) {
                    ui::success("Resumed: " + run_name);
                }
            } else {
                ui::error("Usage: resume <run_name>");
            }
        }
        else if (cmd == "clean") {
            std::string run_name;
            if (ss >> run_name) {
                if (current_job && current_job->is_running()) {
                    ui::error("Cannot delete a running job. Stop it first.");
                    continue;
                }
                
                if (!RunManager::run_exists(run_name)) {
                    ui::error("Run not found: " + run_name);
                    continue;
                }
                
                std::cout << ui::color::YELLOW << "Delete '" << run_name << "'? (y/N): " << ui::color::RESET;
                std::string confirm;
                std::getline(std::cin, confirm);
                
                if (confirm == "y" || confirm == "Y") {
                    if (RunManager::delete_run(run_name)) {
                        ui::success("Deleted: " + run_name);
                    } else {
                        ui::error("Failed to delete run.");
                    }
                } else {
                    ui::info("Cancelled.");
                }
            } else {
                ui::error("Usage: clean <run_name>");
            }
        }
        else {
            ui::warning("Unknown command: " + cmd);
        }
    }
    
    DashboardManager::instance().stop();
    return 0;
}
