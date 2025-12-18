#include "commands.h"
#include "mm_rec/utils/ui.h"
#include "mm_rec/utils/logger.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <thread>
#include <atomic>
#include <cstring>
#include <csignal>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>

using namespace mm_rec;
using namespace mm_rec::ui;

namespace {
    std::atomic<bool> server_running{true};

    void signal_handler(int) {
        server_running = false;
    }

    // MIME type helper
    std::string get_mime_type(const std::string& path) {
        if (path.find(".json") != std::string::npos) return "application/json";
        if (path.find(".html") != std::string::npos) return "text/html";
        return "text/plain";
    }

    // Simple HTTP Response builder
    std::string build_response(int status, const std::string& content_type, const std::string& body, bool cors = false) {
        std::stringstream ss;
        ss << "HTTP/1.1 " << status << " OK\r\n";
        ss << "Content-Type: " << content_type << "\r\n";
        ss << "Content-Length: " << body.size() << "\r\n";
        if (cors) {
            ss << "Access-Control-Allow-Origin: *\r\n";
        }
        ss << "Connection: close\r\n";
        ss << "\r\n";
        ss << body;
        return ss.str();
    }
}

int cmd_view_trace(int argc, char* argv[]) {
    if (argc < 2) {
        ui::error("Usage: mm_rec view-trace <trace.json> [port]");
        return 1;
    }

    std::string trace_path = argv[1];
    int port = (argc >= 3) ? std::stoi(argv[2]) : 8080;

    // Check if file exists
    if (access(trace_path.c_str(), F_OK) == -1) {
        ui::error("Trace file not found: " + trace_path);
        return 1;
    }

    // Determine filename for URL
    std::string filename = trace_path.substr(trace_path.find_last_of("/\\") + 1);

    // Create socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) {
        ui::error("Socket creation failed");
        return 1;
    }

    // Set socket options to reuse address
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    // Bind
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        ui::error("Bind failed. Port " + std::to_string(port) + " might be in use.");
        return 1;
    }

    // Listen
    if (listen(server_fd, 3) < 0) {
        ui::error("Listen failed");
        return 1;
    }

    ui::print_header("Trace Viewer Server");
    ui::success("Serving: " + trace_path);
    
    std::string url = "http://localhost:" + std::to_string(port) + "/" + filename;
    std::string perfetto_link = "https://ui.perfetto.dev/#!/?url=" + url;
    
    std::cout << "\n✅  VIEW TRACE HERE:\n";
    std::cout << "   " << color::CYAN << color::BOLD << perfetto_link << color::RESET << "\n\n";
    
    ui::info("Press Ctrl+C to stop server...");

    // Setup signal handler for clean exit
    signal(SIGINT, signal_handler);

    while (server_running) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        
        // Accept with timeout logic for non-blocking loop check could be better, 
        // but robust blocking accept is fine for this simple tool.
        // We use select to allow checking server_running flag periodically.
        
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(server_fd, &readfds);
        
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        
        int activity = select(server_fd + 1, &readfds, NULL, NULL, &timeout);
        
        if (activity < 0 && errno != EINTR) {
             break; 
        }
        
        if (activity == 0) continue; // Timeout, check loop condition

        int new_socket = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (new_socket < 0) continue;

        // Read request
        char buffer[4096] = {0};
        read(new_socket, buffer, 4096);
        std::string request(buffer);

        // Simple parsing
        std::stringstream ss(request);
        std::string method, path;
        ss >> method >> path;
        
        // Remove leading slash
        if (!path.empty() && path[0] == '/') path = path.substr(1);

        LOG_INFO("Request: " + method + " /" + path);

        std::string response;
        if (path == filename || path == trace_path) {
             // Read file content
            std::ifstream f(trace_path);
            if (f) {
                std::stringstream buffer;
                buffer << f.rdbuf();
                response = build_response(200, "application/json", buffer.str(), true); // Enable CORS
            } else {
                response = build_response(404, "text/plain", "File not found");
            }
        } else {
             response = build_response(404, "text/plain", "Not Found");
        }

        send(new_socket, response.c_str(), response.size(), 0);
        close(new_socket);
    }
    
    close(server_fd);
    ui::info("Server stopped.");
    return 0;
}
