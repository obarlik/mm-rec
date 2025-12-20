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
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
#endif

namespace mm_rec {

namespace net {

// Simple Thread Pool
class ThreadPool {
    // ... (keep existing ThreadPool implementation) ...
public:
    ThreadPool(size_t threads) : stop_(false) {
        for(size_t i = 0; i < threads; ++i)
            workers_.emplace_back(
                [this] {
                    for(;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex_);
                            this->condition_.wait(lock,
                                [this]{ return this->stop_ || !this->tasks_.empty(); });
                            if(this->stop_ && this->tasks_.empty())
                                return;
                            task = std::move(this->tasks_.front());
                            this->tasks_.pop();
                        }
                        task();
                    }
                }
            );
    }
    // ... (rest of ThreadPool) ...
    // destrctor etc are fine
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if(stop_)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks_.emplace([task](){ (*task)(); });
        }
        condition_.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for(std::thread &worker: workers_)
            worker.join();
    }
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

struct HttpServerConfig {
    int port = 8085;
    size_t threads = 4;
    int timeout_sec = 3;
};

class HttpServer {
public:
    using Handler = std::function<std::string(const std::string&)>;

    HttpServer(const HttpServerConfig& config) 
        : config_(config), running_(false), pool_(config.threads) {
#ifdef _WIN32
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
    }
    
    // Legacy constructor for backward compatibility
    HttpServer(int port) : HttpServer(HttpServerConfig{port}) {}

    ~HttpServer() {
        stop();
#ifdef _WIN32
        WSACleanup();
#endif
    }
    
    int port() const { return config_.port; }

    void register_handler(const std::string& path, Handler handler) {
        std::lock_guard<std::mutex> lock(handlers_mutex_);
        handlers_[path] = handler;
    }

    bool start() {
        if (running_) return true;

        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ == -1) return false;

        int opt = 1;
#ifndef _WIN32
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(config_.port);

        if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
            return false;
        }

        if (listen(server_fd_, 100) < 0) { // Increased backlog
            return false;
        }
        
        running_ = true;
        listener_thread_ = std::thread(&HttpServer::listener_loop, this);
        
        LOG_INFO("High-Performance HTTP Server started on port " + std::to_string(config_.port) + 
                 " (Threads: " + std::to_string(config_.threads) + ")");
        return true;
    }

    // ... (rest is same) ...

    void stop() {
        if (running_) {
            running_ = false;
            // Unblock accept
#ifndef _WIN32
            shutdown(server_fd_, SHUT_RDWR);
            close(server_fd_);
#else
            closesocket(server_fd_);
#endif
            if (listener_thread_.joinable()) {
                listener_thread_.join();
            }
        }
    }

    static std::string get_mime_type(const std::string& path) {
        if (path.find(".html") != std::string::npos) return "text/html";
        if (path.find(".css") != std::string::npos) return "text/css";
        if (path.find(".js") != std::string::npos) return "application/javascript";
        if (path.find(".json") != std::string::npos) return "application/json";
        if (path.find(".png") != std::string::npos) return "image/png";
        if (path.find(".jpg") != std::string::npos) return "image/jpeg";
        if (path.find(".svg") != std::string::npos) return "image/svg+xml";
        if (path.find(".ico") != std::string::npos) return "image/x-icon";
        return "text/plain";
    }

    static std::string build_response(int status, const std::string& content_type, const std::string& body, bool cors = true) {
        std::stringstream ss;
        ss << "HTTP/1.1 " << status << " OK\r\n";
        ss << "Content-Type: " << content_type << "\r\n";
        ss << "Content-Length: " << body.size() << "\r\n";
        if (cors) {
            ss << "Access-Control-Allow-Origin: *\r\n";
            ss << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
            ss << "Access-Control-Allow-Headers: Content-Type\r\n";
        }
        ss << "Connection: close\r\n"; // Keep-alive parsing is complex, stick to close for reliability
        ss << "\r\n";
        ss << body;
        return ss.str();
    }

private:
    HttpServerConfig config_;
#ifdef _WIN32
    SOCKET server_fd_;
#else
    int server_fd_;
#endif
    std::atomic<bool> running_;
    std::thread listener_thread_;
    std::map<std::string, Handler> handlers_;
    std::mutex handlers_mutex_;
    ThreadPool pool_;

    void listener_loop() {
        while (running_) {
            struct sockaddr_in client_addr;
#ifdef _WIN32
            int addr_len = sizeof(client_addr);
#else
            socklen_t addr_len = sizeof(client_addr);
#endif
            
            auto new_socket = accept(server_fd_, (struct sockaddr*)&client_addr, &addr_len);
            if (!running_) break;
            
#ifdef _WIN32
             if (new_socket == INVALID_SOCKET) continue;
#else
             if (new_socket < 0) continue;
#endif

            // Offload to thread pool
            pool_.enqueue([this, new_socket] {
                this->handle_client(new_socket);
            });
        }
    }

    void handle_client(int socket) {
        // PRODUCTION SAFETY: Set timeouts
        // This prevents "Slowloris" attacks or stuck clients from consuming a thread forever.
#ifdef _WIN32
        DWORD timeout = config_.timeout_sec * 1000;
        setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
        setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
#else
        struct timeval tv;
        tv.tv_sec = config_.timeout_sec;
        tv.tv_usec = 0;
        setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
        setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof tv);
#endif

        std::vector<char> buffer(8192); // Increased buffer
        
#ifndef _WIN32
        ssize_t bytes_read = read(socket, buffer.data(), buffer.size());
#else
        int bytes_read = recv(socket, buffer.data(), static_cast<int>(buffer.size()), 0);
#endif

        if (bytes_read <= 0) {
#ifndef _WIN32
            close(socket);
#else
            closesocket(socket);
#endif
            return;
        }

        std::string request(buffer.data(), bytes_read);
        std::stringstream ss(request);
        std::string method, path;
        ss >> method >> path;
        
        if (path.empty()) {
#ifndef _WIN32
             close(socket);
#else
             closesocket(socket);
#endif
             return;
        }

        // Remove parameters
        size_t query_pos = path.find('?');
        if (query_pos != std::string::npos) {
            path = path.substr(0, query_pos);
        }

        std::string response;
        Handler handler = nullptr;

        {
            std::lock_guard<std::mutex> lock(handlers_mutex_);
            // Basic routing (exact match for now)
            if (handlers_.count(path)) {
                handler = handlers_[path];
            } else {
                 // Try wildcards (simple suffix check)
                 for(auto const& [key, h] : handlers_) {
                     if (key.back() == '*' && path.find(key.substr(0, key.size()-1)) == 0) {
                         handler = h;
                         break;
                     }
                 }
            }
        }

        if (handler) {
            try {
                response = handler(request);
            } catch(const std::exception& e) {
                response = build_response(500, "text/plain", std::string("Internal Error: ") + e.what());
            }
        } else {
            response = build_response(404, "text/plain", "Not Found");
        }

#ifndef _WIN32
        send(socket, response.c_str(), response.size(), 0);
        close(socket);
#else
        send(socket, response.c_str(), static_cast<int>(response.size()), 0);
        closesocket(socket);
#endif
    }
};

} // namespace net
} // namespace mm_rec
