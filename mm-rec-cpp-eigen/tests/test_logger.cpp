#include "mm_rec/utils/logger_v2.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

using namespace mm_rec;

int main() {
    std::cout << "=== Logger Functionality Test ===" << std::endl;
    
    // Start logger with INFO level (DEBUG should be filtered)
    Logger::instance().start_writer("test_logger.log", LogLevel::INFO);
    
    // Test different log levels
    LOG_UI("UI message - should appear on console");
    LOG_INFO("INFO message - should go to file only");
    LOG_DEBUG("DEBUG message - should be filtered (not appear)");
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    Logger::instance().stop_writer();
    
    // Verify log file contents
    std::ifstream log_file("test_logger.log");
    if (!log_file) {
        std::cerr << "❌ Failed to open log file!" << std::endl;
        return 1;
    }
    
    std::string line;
    int ui_count = 0, info_count = 0, debug_count = 0;
    
    while (std::getline(log_file, line)) {
        if (line.find("[UI]") != std::string::npos) ui_count++;
        if (line.find("[INFO]") != std::string::npos) info_count++;
        if (line.find("[DEBUG]") != std::string::npos) debug_count++;
    }
    
    std::cout << "\n📊 Results:" << std::endl;
    std::cout << "  UI logs: " << ui_count << " (expected: 1)" << std::endl;
    std::cout << "  INFO logs: " << info_count << " (expected: 1)" << std::endl;
    std::cout << "  DEBUG logs: " << debug_count << " (expected: 0 - filtered)" << std::endl;
    
    bool passed = (ui_count == 1) && (info_count == 1) && (debug_count == 0);
    
    if (passed) {
        std::cout << "\n✅ TEST PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ TEST FAILED!" << std::endl;
        return 1;
    }
}
