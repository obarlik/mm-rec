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


struct Request {
    std::string method;
    std::string path;
    std::string body;
    std::function<bool()> is_connected;
};

class HttpServer {
public:
    using Handler = std::function<std::string(const Request&)>;
    // NextFn represents the next step in the pipeline (either another middleware or the final handler)
    using NextFn = std::function<std::string(const Request&)>;
    
    // Bidirectional Middleware:
    // - Pre-processing: Code before calling next(req)
    // - Post-processing: Code after calling next(req)
    // - Short-circuit: Return response WITHOUT calling next(req)
    using Middleware = std::function<std::string(const Request&, NextFn)>;

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

    // Register a middleware to the pipeline
    // Order matters: use(A); use(B) -> A intercepts first, calls B, B calls Handler.
    // Response flows back: Handler -> B -> A.
    void use(Middleware middleware) {
        std::lock_guard<std::mutex> lock(middlewares_mutex_);
        middlewares_.push_back(middleware);
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

        if (listen(server_fd_, 100) < 0) {
            return false;
        }
        
        running_ = true;
        listener_started_.store(false, std::memory_order_release);
        
        try {
            listener_thread_ = std::thread(&HttpServer::listener_loop, this);
            
            // Wait for thread to actually start (atomic synchronization)
            while (!listener_started_.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            
        } catch (const std::exception& e) {
            running_ = false;
            return false;
        }
        
        LOG_INFO("High-Performance HTTP Server started on port " + std::to_string(config_.port) + 
                 " (Threads: " + std::to_string(config_.threads) + ")");
        return true;
    }

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
        ss << "X-MMRec-Version: 0.1.0\r\n";
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
    std::vector<Middleware> middlewares_;
    std::mutex middlewares_mutex_;
    ThreadPool pool_;
    std::atomic<bool> listener_started_{false};  // Track listener thread startup

    void listener_loop() {
        listener_started_.store(true, std::memory_order_release);
        
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
        
        // LOG_INFO("handle_client: about to read from socket");
        
#ifndef _WIN32
        ssize_t bytes_read = read(socket, buffer.data(), buffer.size());
#else
        int bytes_read = recv(socket, buffer.data(), static_cast<int>(buffer.size()), 0);
#endif

        // LOG_INFO("handle_client: read " + std::to_string(bytes_read) + " bytes");
        if (bytes_read <= 0) {
#ifndef _WIN32
            close(socket);
#else
            closesocket(socket);
#endif
            return;
        }

        std::string raw_request(buffer.data(), bytes_read);
        std::stringstream ss(raw_request);
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

        // Construct Request Object
        Request req;
        req.method = method;
        req.path = path;
        req.body = raw_request; // Storing full raw request as body for now/compatibility, ideally split headers/body
        
        // Connectivity Check Lambda
        req.is_connected = [socket]() -> bool {
            char buf;
#ifndef _WIN32
            ssize_t r = recv(socket, &buf, 1, MSG_PEEK | MSG_DONTWAIT);
            if (r == 0) return false; // Graceful closed
            if (r < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) return true; // Still connected, just no data
                return false; // Error detected
            }
            return true;
#else
            int r = recv(socket, &buf, 1, MSG_PEEK);
            if (r == 0) return false;
            if (r < 0) {
                int err = WSAGetLastError();
                if (err == WSAEWOULDBLOCK) return true;
                return false;
            }
            return true;
#endif
        };

        std::string response;

        // --- Chain of Responsibility Composition ---
        
        // 1. Initial "Bottom" Handler: The Router
        NextFn dispatch = [this](const Request& r) -> std::string {
             Handler handler = nullptr;
             {
                 std::lock_guard<std::mutex> lock(handlers_mutex_);
                 if (handlers_.count(r.path)) {
                     handler = handlers_[r.path];
                 } else {
                     // Suffix wildcard
                      for(auto const& [key, h] : handlers_) {
                         if (key.back() == '*' && r.path.find(key.substr(0, key.size()-1)) == 0) {
                             handler = h;
                             break;
                         }
                     }
                 }
             }

             if (handler) {
                 try {
                     return handler(r);
                 } catch(const std::exception& e) {
                     return build_response(500, "text/plain", std::string("Internal Error: ") + e.what());
                 }
             }
             return build_response(404, "text/plain", "Not Found");
        };

        // 2. Wrap Middleware Layers (Reverse Order)
        // Last added middleware is the "outer" most layer, called first.
        {
            std::lock_guard<std::mutex> lock(middlewares_mutex_);
            for (auto it = middlewares_.rbegin(); it != middlewares_.rend(); ++it) {
                auto current_mw = *it;
                auto next = dispatch; // Capture current 'next'
                dispatch = [current_mw, next](const Request& r) -> std::string {
                    return current_mw(r, next);
                };
            }
        }

        // 3. Execute Chain
        response = dispatch(req);

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

