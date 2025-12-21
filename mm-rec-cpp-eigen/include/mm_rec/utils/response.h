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

private:
    int socket_fd_;
    int status_code_ = 200;
    bool headers_sent_;
    std::map<std::string, std::string> headers_;
    std::mutex mutex_;

    void send_headers(size_t content_length) {
        std::stringstream ss;
        ss << "HTTP/1.1 " << status_code_ << " OK\r\n"; // TODO: Status text map
        
        // Ensure Content-Length is set if not present (simple mode)
        // In streaming/chunked mode, we would omit this or use Transfer-Encoding
        if (headers_.find("Content-Length") == headers_.end()) {
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
};

} // namespace net
} // namespace mm_rec
