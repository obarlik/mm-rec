#pragma once

#include "mm_rec/utils/logger.h"
#include <string>
#include <functional>
#include <map>
#include <vector>
#include <sstream>
#include <thread>
#include <atomic>
#include <cstring>
#include <memory>
#include <iostream>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>

namespace mm_rec {

namespace net {

class HttpServer {
public:
    using Handler = std::function<std::string(const std::string&)>;

    HttpServer(int port) : port_(port), running_(false) {}

    ~HttpServer() {
        stop();
    }

    void register_handler(const std::string& path, Handler handler) {
        handlers_[path] = handler;
    }

    bool start() {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ == 0) return false;

        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port_);

        if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
            return false;
        }

        if (listen(server_fd_, 10) < 0) {
            return false;
        }

        // Set non-blocking for the main loop handling
        // We actually keep the server socket blocking for the thread, 
        // but we might want to set a timeout on accept.
        // For simplicity, we'll use a standard blocking accept in a thread.
        
        running_ = true;
        worker_thread_ = std::thread(&HttpServer::run_loop, this);
        
        LOG_INFO("HttpServer started on port " + std::to_string(port_));
        return true;
    }

    void stop() {
        if (running_) {
            running_ = false;
            // Connect to self to unblock accept if needed, or close fd
            shutdown(server_fd_, SHUT_RDWR);
            close(server_fd_);
            if (worker_thread_.joinable()) {
                worker_thread_.join();
            }
        }
    }

    // MIME type helper
    static std::string get_mime_type(const std::string& path) {
        if (path.find(".json") != std::string::npos) return "application/json";
        if (path.find(".html") != std::string::npos) return "text/html";
        return "text/plain";
    }

    // Build response helper
    static std::string build_response(int status, const std::string& content_type, const std::string& body, bool cors = true) {
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

private:
    int port_;
    int server_fd_;
    std::atomic<bool> running_;
    std::thread worker_thread_;
    std::map<std::string, Handler> handlers_;

    void run_loop() {
        while (running_) {
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            
            int new_socket = accept(server_fd_, (struct sockaddr*)&client_addr, &addr_len);
            if (new_socket < 0) {
                if (running_) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                continue;
            }

            // Handle request in same thread for simplicity (or spawn new thread if high load needed)
            // For a dashboard, serialized handling is fine.
            handle_client(new_socket);
        }
    }

    void handle_client(int socket) {
        char buffer[4096] = {0};
        read(socket, buffer, 4096);
        std::string request(buffer);

        std::stringstream ss(request);
        std::string method, path;
        ss >> method >> path;
        
        if (path.empty()) {
             close(socket);
             return;
        }

        // Remove parameters
        size_t query_pos = path.find('?');
        if (query_pos != std::string::npos) {
            path = path.substr(0, query_pos);
        }

        std::string response;
        if (handlers_.count(path)) {
            response = handlers_[path](request);
        } else {
            response = build_response(404, "text/plain", "Not Found");
        }

        send(socket, response.c_str(), response.size(), 0);
        close(socket);
    }
};

} // namespace net
} // namespace mm_rec
