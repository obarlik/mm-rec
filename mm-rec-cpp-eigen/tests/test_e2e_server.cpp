/**
 * End-to-End Integration Test
 * 
 * Tests the complete system flow:
 * - DashboardManager initialization
 * - JobTraining lifecycle (start/stop)
 * - API endpoint responses
 * - Data consistency
 */

#include "mm_rec/jobs/job_training.h"
#include "mm_rec/utils/dashboard_manager.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/ui.h"
#include "mm_rec/core/vulkan_backend.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <cassert>

using namespace mm_rec;
namespace fs = std::filesystem;

// Simple HTTP GET utility (without external dependencies)
std::string http_get(const std::string& url) {
    // Use curl as subprocess for simplicity
    std::string cmd = "curl -s " + url;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    
    char buffer[128];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    return result;
}

bool test_dashboard_api() {
    ui::print_header("Test: Dashboard API", 50);
    
    // Start Dashboard on test port
    int test_port = 9999;
    if (!DashboardManager::instance().start(test_port)) {
        ui::error("Failed to start Dashboard on port " + std::to_string(test_port));
        return false;
    }
    
    // Give server time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Test /api/stats endpoint
    std::string url = "http://localhost:" + std::to_string(test_port) + "/api/stats";
    std::string response = http_get(url);
    
    ui::info("API Response: " + response);
    
    // Basic validation: check if response contains expected fields
    bool has_loss = response.find("\"loss\"") != std::string::npos;
    bool has_step = response.find("\"step\"") != std::string::npos;
    bool has_gpu_ratio = response.find("\"gpu_ratio\"") != std::string::npos;
    
    if (!has_loss || !has_step || !has_gpu_ratio) {
        ui::error("API response missing expected fields!");
        return false;
    }
    
    ui::success("Dashboard API test passed");
    DashboardManager::instance().stop();
    return true;
}

bool test_job_lifecycle() {
    ui::print_header("Test: Job Lifecycle", 50);
    
    // Create minimal test config
    std::string test_run = "test_e2e_run";
    std::string runs_dir = "runs";
    std::string run_dir = runs_dir + "/" + test_run;
    
    // Clean up any previous test run
    if (fs::exists(run_dir)) {
        fs::remove_all(run_dir);
    }
    
    // Create minimal config file
    std::ofstream cfg("test_config_e2e.txt");
    cfg << "vocab_size=256\n";
    cfg << "hidden_dim=64\n";
    cfg << "mem_dim=64\n";
    cfg << "ffn_dim=128\n";
    cfg << "num_layers=2\n";
    cfg << "num_experts=2\n";
    cfg << "top_k=1\n";
    cfg << "batch_size=2\n";
    cfg << "max_seq_len=32\n";
    cfg << "max_iterations=1\n";  // Just 1 epoch for testing
    cfg << "learning_rate=0.001\n";
    cfg.close();
    
    // Create minimal dataset (just a few tokens)
    std::ofstream data("test_data_e2e.bin", std::ios::binary);
    int32_t magic = 0x4D4D5243;
    int32_t version = 2;
    int64_t total = 100;  // 100 tokens
    
    data.write(reinterpret_cast<char*>(&magic), sizeof(int32_t));
    data.write(reinterpret_cast<char*>(&version), sizeof(int32_t));
    data.write(reinterpret_cast<char*>(&total), sizeof(int64_t));
    
    // Write dummy tokens and masks
    for (int i = 0; i < 100; ++i) {
        int32_t token = i % 256;
        int32_t mask = 1;
        data.write(reinterpret_cast<char*>(&token), sizeof(int32_t));
    }
    for (int i = 0; i < 100; ++i) {
        int32_t mask = 1;
        data.write(reinterpret_cast<char*>(&mask), sizeof(int32_t));
    }
    data.close();
    
    // Test 1: Start Job
    ui::info("Starting training job...");
    JobTraining job;
    TrainingJobConfig config;
    config.run_name = test_run;
    config.config_path = "test_config_e2e.txt";
    config.data_path = "test_data_e2e.bin";
    config.enable_metrics = false;  // Disable for faster testing
    
    if (!job.start(config)) {
        ui::error("Failed to start job!");
        return false;
    }
    
    ui::success("Job started successfully");
    
    // Test 2: Wait for training to run a bit
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Test 3: Verify job is running
    if (!job.is_running()) {
        ui::error("Job should be running but isn't!");
        return false;
    }
    
    ui::success("Job is running");
    
    // Test 4: Stop job
    ui::info("Stopping job...");
    job.stop();
    job.join();
    
    if (job.is_running()) {
        ui::error("Job should have stopped!");
        return false;
    }
    
    ui::success("Job stopped successfully");
    
    // Test 5: Verify artifacts were created
    bool has_log = fs::exists(run_dir + "/train.log");
    bool has_checkpoint = fs::exists(run_dir + "/checkpoint_latest.bin");
    
    if (!has_log || !has_checkpoint) {
        ui::error("Expected output files not found!");
        ui::info("Has log: " + std::string(has_log ? "yes" : "no"));
        ui::info("Has checkpoint: " + std::string(has_checkpoint ? "yes" : "no"));
        return false;
    }
    
    ui::success("All output files created correctly");
    
    // Cleanup
    fs::remove("test_config_e2e.txt");
    fs::remove("test_data_e2e.bin");
    fs::remove_all(run_dir);
    
    return true;
}

int main() {
    ui::print_header("End-to-End Integration Test Suite", 60);
    
    // Initialize logger
    Logger::instance().start_writer("test_e2e.log", LogLevel::INFO);
    
    int passed = 0;
    int total = 0;
    
    // Test 1: Dashboard API
    total++;
    if (test_dashboard_api()) {
        passed++;
    }
    
    // Test 2: Job Lifecycle
    total++;
    if (test_job_lifecycle()) {
        passed++;
    }
    
    // Summary
    ui::print_header("Test Results", 60);
    std::cout << "Passed: " << passed << "/" << total << "\n";
    
    Logger::instance().stop_writer();
    
    if (passed == total) {
        ui::success("All tests passed! ✓");
        return 0;
    } else {
        ui::error("Some tests failed!");
        return 1;
    }
}
