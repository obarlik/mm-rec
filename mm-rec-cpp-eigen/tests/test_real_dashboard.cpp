#include "mm_rec/application/dashboard_manager.h"
#include "mm_rec/infrastructure/logger.h"
#include <iostream>
#include <thread>
#include <chrono>

// Test utilizing the REAL DashboardManager class
// This isolates whether the issue is internal to DashboardManager logic

int main(int argc, char** argv) {
    int port = argc > 1 ? std::atoi(argv[1]) : 9996;
    
    std::cout << "=== REAL DashboardManager Isolation Test ===\n";
    std::cout << "Port: " << port << "\n";
    
    // Initialize Logger
    mm_rec::Logger::instance().start_writer("test_real_dashboard.log", mm_rec::LogLevel::INFO);
    
    // Prepare Config
    mm_rec::net::HttpServerConfig config;
    config.port = port;
    config.threads = 4;
    config.timeout_sec = 3;

    std::cout << "Starting dashboard...\n";
    // instance() returns the singleton ref
    if (!mm_rec::DashboardManager::instance().start(config)) {
        std::cerr << "Failed to start dashboard!\n";
        return 1;
    }
    
    std::cout << "✓ Dashboard started successfully.\n";
    
    // Check if running in automated mode (no args or specific flag)
    bool automated = true;
    if (argc > 1 && std::string(argv[1]) == "--interactive") {
        automated = false;
    }

    if (automated) {
        std::cout << "Automated mode: Sleeping 1s then stopping...\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    } else {
        std::cout << "Interactive mode: Press Enter to exit...\n";
        std::cout << "Run: curl http://localhost:" << port << "\n";
        std::cin.get();
    }
    
    mm_rec::DashboardManager::instance().stop();
    return 0;
}
