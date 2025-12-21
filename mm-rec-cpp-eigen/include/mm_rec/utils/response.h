#pragma once

#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <atomic>
#include <mutex>

#ifndef _WIN32
#include <sys/socket.h>
#include <unistd.h>
#else
#include <winsock2.h>
#endif

namespace mm_rec {
namespace net {

class Response {
public:
    Response(int socket_fd) : socket_fd_(socket_fd), headers_sent_(false) {
        // Default Headers
        set_header("Content-Type", "text/plain");
        set_header("Connection", "close");
        set_header("Access-Control-Allow-Origin", "*");
        set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        set_header("Access-Control-Allow-Headers", "Content-Type");
        set_header("X-MMRec-Version", "0.1.0");
    }

    // Set HTTP Status Code
    void status(int code) {
        status_code_ = code;
    }

    // Set Header
    void set_header(const std::string& key, const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!headers_sent_) {
            headers_[key] = value;
        }
    }

    // Send body (auto-flushes headers if needed)
    void send(const std::string& body) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!headers_sent_) {
            send_headers(body.size());
        }
        write_to_socket(body);
    }


    // Explicitly end response (usually called after send, or empty send)
    void end() {
        // If headers not sent, send them now with 0 content length
        // (Only if we haven't sent anything yet)
        // But send() handles it.
    }

    int get_status() const { return status_code_; }

    // --- SSE (Server-Sent Events) Support ---
    
    // Enable SSE mode (sets appropriate headers)
    void enable_sse() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!headers_sent_) {
            sse_mode_ = true;
            set_header("Content-Type", "text/event-stream");
            set_header("Cache-Control", "no-cache");
            set_header("Connection", "keep-alive");
            set_header("X-Accel-Buffering", "no"); // Disable nginx buffering
        }
    }

    // Send SSE event (auto-formats as "data: ...\n\n")
    void send_event(const std::string& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!headers_sent_) {
            send_headers(0); // SSE has no fixed content-length
        }
        
        std::string sse_data = "data: " + data + "\n\n";
        write_to_socket(sse_data);
        flush_socket();
    }

    // Send named SSE event (e.g., "event: training.step\ndata: {...}\n\n")
    void send_event(const std::string& event, const std::string& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!headers_sent_) {
            send_headers(0);
        }
        
        std::string sse_data = "event: " + event + "\ndata: " + data + "\n\n";
        write_to_socket(sse_data);
        flush_socket();
    }

    // Explicit flush for SSE (ensures immediate delivery)
    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        flush_socket();
    }


private:
    int socket_fd_;
    int status_code_ = 200;
    bool headers_sent_;
    bool sse_mode_ = false;
    std::map<std::string, std::string> headers_;
    std::mutex mutex_;

    void send_headers(size_t content_length) {
        std::stringstream ss;
        ss << "HTTP/1.1 " << status_code_ << " OK\r\n"; // TODO: Status text map
        
        // For SSE, don't set Content-Length
        if (!sse_mode_ && headers_.find("Content-Length") == headers_.end()) {
            headers_["Content-Length"] = std::to_string(content_length);
        }

        for (const auto& [k, v] : headers_) {
            ss << k << ": " << v << "\r\n";
        }
        ss << "\r\n";
        
        std::string header_str = ss.str();
        write_to_socket(header_str);
        headers_sent_ = true;
    }

    void write_to_socket(const std::string& data) {
#ifndef _WIN32
        ::send(socket_fd_, data.c_str(), data.size(), 0);
#else
        ::send(socket_fd_, data.c_str(), static_cast<int>(data.size()), 0);
#endif
    }

    void flush_socket() {
        // Force kernel to send buffered data immediately
        // Using TCP_NODELAY or similar could be added here if needed
#ifndef _WIN32
        fsync(socket_fd_); // Doesn't work on sockets, but leaving for reference
#endif
    }
};

} // namespace net
} // namespace mm_rec
