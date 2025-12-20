#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <csignal>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/http_server.h"
#include "mm_rec/utils/dashboard_manager.h"
#include "mm_rec/utils/config.h"
#include "mm_rec/utils/ui.h"
#include "cli/commands.h"
#include "mm_rec/jobs/job_training.h"
#include "mm_rec/utils/run_manager.h" 

using namespace mm_rec;
using namespace mm_rec::ui;

// Signal Handling
static std::atomic<bool> g_shutdown_requested(false);
static std::condition_variable g_shutdown_cv;
static std::mutex g_shutdown_mutex;

void handle_sigint(int sig) {
    if (g_shutdown_requested.exchange(true)) {
        // Force exit on second signal
        std::cerr << "\nForced shutdown.\n";
        std::exit(1);
    }
    std::cout << "\nSignal received (" << sig << "). Shutting down gracefully...\n";
    // Notify main thread to wake up
    g_shutdown_cv.notify_all();
}

void print_help() {
    std::cout << "\n" << ui::color::BOLD << "MM-Rec Server:" << ui::color::RESET << "\n";
    std::cout << "  --port <N>                        Server port (Default: 8085)\n";
    std::cout << "  --threads <N>                     Worker threads (Default: 4)\n";
    std::cout << "  --timeout <N>                     Socket timeout (Default: 3)\n";
    std::cout << "  --daemon                          Run as daemon (background)\n";
    std::cout << "\n";
}

int cmd_server(int argc, char* argv[]) {
    // 1. Initialize Global Config
    auto& app_config = mm_rec::Config::instance();
    app_config.load_from_file("config.ini");
    app_config.load_from_env();
    app_config.parse_args(argc, argv);
    
    // Register Signal Handlers
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);
    
    // 2. Retrieve Settings
    auto server_conf = app_config.section("server");
    int port = server_conf.get<int>("port", 8085);
    int threads = server_conf.get<int>("threads", 4);
    int timeout = server_conf.get<int>("timeout", 3);
    
    // New Management Configs
    int max_connections = server_conf.get<int>("max_connections", 100);
    int rate_limit = server_conf.get<int>("rate_limit", 1000);
    int throttle_ms = server_conf.get<int>("throttle_ms", 0);
    
    bool daemon_mode = false;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--daemon") {
            daemon_mode = true;
            break;
        }
    }
    
#ifdef _WIN32
    if (daemon_mode) {
         ui::warning("Daemon mode is not supported on Windows yet. Running in console mode.");
         daemon_mode = false;
    }
#else
    if (daemon_mode) {
        pid_t pid = fork();
        if (pid < 0) return 1;
        if (pid > 0) {
            std::cout << "✓ Daemon started (PID: " << pid << ")\n";
            std::cout << "  Dashboard: http://localhost:" << port << "\n";
            exit(0);
        }
        if (setsid() < 0) return 1;
        
        pid = fork();
        if (pid < 0) return 1;
        if (pid > 0) {
            int status;
            waitpid(-1, &status, WNOHANG);
            exit(0);
        }
        
        // Close standard file descriptors
        close(STDIN_FILENO);
        close(STDOUT_FILENO);
        close(STDERR_FILENO);
        open("/dev/null", O_RDONLY);
        open("/dev/null", O_WRONLY);
        open("/dev/null", O_WRONLY);
        umask(0);
        
        std::ofstream pid_file("/tmp/mm_rec_server.pid");
        if (pid_file.is_open()) {
            pid_file << getpid();
            pid_file.close();
        }
    }
#endif

    if (!daemon_mode) {
        ui::print_header("MM-Rec Server");
        ui::info("Server running on port: " + std::to_string(port));
        ui::info("Press Ctrl+C to stop.");
    }
    
    // Initialize Logger
    std::filesystem::create_directories("logs");
    Logger::instance().start_writer("logs/server.log", LogLevel::INFO);
    LOG_INFO("MM-Rec Server starting on port " + std::to_string(port));
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Start Dashboard
    net::HttpServerConfig config;
    config.port = port;
    config.threads = threads;
    config.timeout_sec = timeout;
    config.max_connections = max_connections;
    config.max_req_per_min = rate_limit;
    config.throttle_ms = throttle_ms;
    
    DashboardManager::instance().start(config);
    
    // Main Wait Loop (Non-interactive)
    {
        std::unique_lock<std::mutex> lock(g_shutdown_mutex);
        g_shutdown_cv.wait(lock, []{ return g_shutdown_requested.load(); });
    }
    
    DashboardManager::instance().stop();
    Logger::instance().stop_writer();
    return 0;
}
