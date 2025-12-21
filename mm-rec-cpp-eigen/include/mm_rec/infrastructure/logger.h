#pragma once

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <sstream> // Added
#include <ctime>   // Added
#include <cstdlib> // Added

#include "mm_rec/infrastructure/request_context.h" // Full include for RequestContext methods

#include <cstring>
#include <filesystem>
#include <iomanip>

namespace mm_rec {

namespace fs = std::filesystem;

/**
 * Simple file rotation helper
 */
inline void rotate_file_if_needed(const std::string& path_str, size_t max_bytes, int max_backups) {
    fs::path path(path_str);
    if (!fs::exists(path)) return;
    
    // Check size
    if (fs::file_size(path) < max_bytes) return;
    
    // Rotate: file.log -> file.1.log -> ...
    // Delete oldest
    fs::path oldest = path.parent_path() / (path.stem().string() + "." + std::to_string(max_backups) + path.extension().string());
    if (fs::exists(oldest)) fs::remove(oldest);
    
    // Shift others
    for (int i = max_backups - 1; i >= 1; --i) {
        fs::path src = path.parent_path() / (path.stem().string() + "." + std::to_string(i) + path.extension().string());
        fs::path dst = path.parent_path() / (path.stem().string() + "." + std::to_string(i + 1) + path.extension().string());
        if (fs::exists(src)) fs::rename(src, dst);
    }
    
    // Rename current to .1
    fs::path first_backup = path.parent_path() / (path.stem().string() + ".1" + path.extension().string());
    fs::rename(path, first_backup);
}

/**
 * Zero-Overhead Logging System v2
 * 
 * Design:
 * - Lock-free ring buffer (like metrics)
 * - Async background writer
 * - UI/System log separation
 * - Runtime enable/disable per level
 * - Immortal pattern (static destruction safe)
 */

/**
 * Log Levels
 */
enum class LogLevel : uint8_t {
    UI = 0,      // User-facing (always shown on stdout)
    INFO = 1,    // System status
    WARNING = 2, // Potential issues
    ERROR = 3,   // Critical failures
    DEBUG = 4    // Debug details (disabled by default)
};

/**
 * Log Event (72 bytes due to alignment, still cache-friendly)
 */
struct LogEvent {
    LogLevel level;           // 1 byte
    uint64_t timestamp_us;    // 8 bytes (aligned at offset 8)
    char message[55];         // 55 bytes
    // Total: 72 bytes (8-byte aligned due to uint64_t)
    
    LogEvent() = default;
    
    LogEvent(LogLevel lvl, const char* msg) : level(lvl) {
        auto now = std::chrono::high_resolution_clock::now();
        timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
        
        // Safe string copy (truncate if needed)
        int i = 0;
        while (i < 53 && msg && msg[i] != '\0') {
            message[i] = msg[i];
            ++i;
        }
        message[i] = '\0';
    }
};


/**
 * Lock-Free Ring Buffer for Log Events
 */
class LogBuffer {
public:
    static constexpr size_t BUFFER_SIZE = 8192;  // 8K events = 512KB
    
    LogBuffer() : write_idx_(0), read_idx_(0) {
        buffer_.resize(BUFFER_SIZE);
    }
    
    bool push(const LogEvent& event) {
        size_t current = write_idx_.load(std::memory_order_relaxed);
        size_t next = (current + 1) % BUFFER_SIZE;
        
        if (next == read_idx_.load(std::memory_order_acquire)) {
            return false;  // Buffer full, drop event
        }
        
        buffer_[current] = event;
        write_idx_.store(next, std::memory_order_release);
        return true;
    }
    
    size_t consume(std::vector<LogEvent>& out, size_t max_count) {
        size_t count = 0;
        size_t current = read_idx_.load(std::memory_order_relaxed);
        size_t end = write_idx_.load(std::memory_order_acquire);
        
        while (current != end && count < max_count) {
            out.push_back(buffer_[current]);
            current = (current + 1) % BUFFER_SIZE;
            ++count;
        }
        
        read_idx_.store(current, std::memory_order_release);
        return count;
    }
    
private:
    std::vector<LogEvent> buffer_;
    std::atomic<size_t> write_idx_;
    std::atomic<size_t> read_idx_;
};

/**
 * Logger Interface for Dependency Injection
 * 
 * Abstract interface to enable:
 * - Testability (inject MockLogger)
 * - Flexibility (FileLogger, NetworkLogger, etc.)
 * - SOLID principles (Dependency Inversion)
 */
class ILogger {
public:
    virtual ~ILogger() = default;
    
    // Core logging methods
    virtual void log(LogLevel level, const char* msg) = 0;
    virtual void log(LogLevel level, const std::string& msg) = 0;
    
    // Convenience methods
    virtual void ui(const char* msg) = 0;
    virtual void info(const char* msg) = 0;
    virtual void warning(const char* msg) = 0;
    virtual void error(const char* msg) = 0;
    virtual void debug(const char* msg) = 0;
    
    virtual void ui(const std::string& msg) = 0;
    virtual void info(const std::string& msg) = 0;
    virtual void warning(const std::string& msg) = 0;
    virtual void error(const std::string& msg) = 0;
    virtual void debug(const std::string& msg) = 0;
    
    // Writer control
    virtual void start_writer(const std::string& log_file = "mm_rec.log", 
                             LogLevel max_level = LogLevel::INFO) = 0;
    virtual void stop_writer() = 0;
    
    // RequestContext integration (optional, default no-op)
    virtual void set_request_context(mm_rec::net::RequestContext* ctx) {}
};

/**
 * Console Logger Implementation (Production)
 * 
 * Lock-free, high-performance logger with async file writing.
 * Can be used via:
 * 1. DI Container: auto logger = container.resolve<ILogger>();
 * 2. Singleton (backward compat): Logger::instance()
 */
class Logger : public ILogger {
public:
    // DI-friendly: Public constructor
    Logger() = default;
    
    /**
     * Set current RequestContext for automatic diagnostic tracing.
     * Called by DI middleware to enable auto-trace.
     * 
     * Usage:
     *   logger->set_request_context(ctx.get());
     *   logger->info("Processing user");  // Auto-added to ctx diagnostic trace!
     *   logger->set_request_context(nullptr);  // Clear after request
     */
    void set_request_context(mm_rec::net::RequestContext* ctx) override {
        current_request_context_ = ctx;
    }

    // Singleton access (backward compatibility)
    static Logger& instance() {
        static Logger inst;
        return inst;
    }
    
    // Thread-local buffer accessor
    static LogBuffer& get_local_buffer() {
        static thread_local LogBuffer buffer;
        static thread_local bool registered = false;
        
        if (!registered) {
            instance().register_buffer(&buffer);
            registered = true;
        }
        
        return buffer;
    }
    
    // Fast logging (inline, essentially free)
    void log(LogLevel level, const char* msg) override {
        auto& logger = instance();
        
        // Skip if writer not running (zero overhead when disabled)
        if (!logger.writer_running_.load(std::memory_order_relaxed)) {
            return;
        }
        
        // Filter by level
        if (static_cast<uint8_t>(level) > static_cast<uint8_t>(logger.max_level_)) {
            return;  // Level too verbose, skip
        }
        
        // ========================================
        // AUTOMATIC DIAGNOSTIC TRACE
        // ========================================
        // If RequestContext is available (in HTTP request scope),
        // automatically add to diagnostic trace buffer.
        if (current_request_context_ != nullptr) {
            std::string level_str;
            switch (level) {
                case LogLevel::UI: level_str = "UI"; break;
                case LogLevel::INFO: level_str = "INFO"; break;
                case LogLevel::WARNING: level_str = "WARN"; break;
                case LogLevel::ERROR: level_str = "ERROR"; break;
                case LogLevel::DEBUG: level_str = "DEBUG"; break;
            }
            
            // Extract component from message (first word after brackets, or "App")
            std::string component = "App";
            std::string msg_str(msg);
            
            // Try to extract component from log prefix [COR:xxx|TR:yyy]
            size_t bracket_end = msg_str.find(']');
            if (bracket_end != std::string::npos && bracket_end < msg_str.length() - 1) {
                size_t component_start = bracket_end + 2;  // Skip "] "
                size_t component_end = msg_str.find(':', component_start);
                if (component_end != std::string::npos && component_end < component_start + 20) {
                    component = msg_str.substr(component_start, component_end - component_start);
                } else {
                    // Use first word after bracket
                    component_end = msg_str.find(' ', component_start);
                    if (component_end != std::string::npos && component_end < component_start + 20) {
                        component = msg_str.substr(component_start, component_end - component_start);
                    }
                }
            }
            
            // TODO: Re-enable when RequestContext is fully defined
            // current_request_context_->add_trace(level_str, component, msg);
        }
        
        // UI level: Write to stdout immediately (users expect instant feedback)
        if (level == LogLevel::UI) {
            std::cout << msg << std::endl;
        }
        
        // Record to log file
        LogEvent event(level, msg);
        get_local_buffer().push(event);
    }
    
    // String overload
    void log(LogLevel level, const std::string& msg) override {
        log(level, msg.c_str());
    }
    
    // Convenience methods
    void ui(const char* msg) override    { log(LogLevel::UI, msg); }
    void info(const char* msg) override  { log(LogLevel::INFO, msg); }
    void warning(const char* msg) override { log(LogLevel::WARNING, msg); }
    void error(const char* msg) override { log(LogLevel::ERROR, msg); }
    void debug(const char* msg) override { log(LogLevel::DEBUG, msg); }
    
    void ui(const std::string& msg) override    { log(LogLevel::UI, msg); }
    void info(const std::string& msg) override  { log(LogLevel::INFO, msg); }
    void warning(const std::string& msg) override { log(LogLevel::WARNING, msg); }
    void error(const std::string& msg) override   { log(LogLevel::ERROR, msg); }
    void debug(const std::string& msg) override { log(LogLevel::DEBUG, msg); }
    
    // Start background writer
    void start_writer(const std::string& log_file = "mm_rec.log", 
                     LogLevel max_level = LogLevel::INFO) override {
        if (writer_running_.load(std::memory_order_acquire)) return;
        
        log_file_path_ = log_file;
        max_level_ = max_level;
        
        // Retention Policy: Max 10MB, Keep 3 backups
        rotate_file_if_needed(log_file, 10 * 1024 * 1024, 3);
        
        writer_running_.store(true, std::memory_order_release);
        writer_thread_ = new std::thread(&Logger::writer_loop, this);
    }
    
    // Stop background writer
    void stop_writer() override {
        if (!writer_running_.exchange(false)) return;
        
        if (writer_thread_ && writer_thread_->joinable()) {
            writer_thread_->join();
            delete writer_thread_;
            writer_thread_ = nullptr;
        }
        
        flush_all();
    }
    
    // ========================================
    // Static Helpers (Backward Compatibility)
    // ========================================
    
    // Static wrappers for convenience macros (LOG_INFO, etc.)
    static void s_log(LogLevel level, const char* msg) { instance().log(level, msg); }
    static void s_log(LogLevel level, const std::string& msg) { instance().log(level, msg); }
    static void s_ui(const char* msg) { instance().ui(msg); }
    static void s_info(const char* msg) { instance().info(msg); }
    static void s_warning(const char* msg) { instance().warning(msg); }
    static void s_error(const char* msg) { instance().error(msg); }
    static void s_debug(const char* msg) { instance().debug(msg); }
    static void s_ui(const std::string& msg) { instance().ui(msg); }
    static void s_info(const std::string& msg) { instance().info(msg); }
    static void s_warning(const std::string& msg) { instance().warning(msg); }
    static void s_error(const std::string& msg) { instance().error(msg); }
    static void s_debug(const std::string& msg) { instance().debug(msg); }
    
    // Minimal destructor - immortal pattern
    ~Logger() {
        writer_running_.store(false, std::memory_order_release);
        writer_thread_ = nullptr;  // Leak (singleton lives until exit)
    }
    
private:
    // Thread-local RequestContext for automatic diagnostic tracing
    // Thread-local context for request tracking
    inline static thread_local mm_rec::net::RequestContext* current_request_context_ = nullptr;
    
    void register_buffer(LogBuffer* buf) {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        buffers_.push_back(buf);
    }
    
    void writer_loop() {
        std::ofstream ofs(log_file_path_, std::ios::out | std::ios::app);
        if (!ofs) return;
        
        // Write header
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        ofs << "=== MM-Rec Log Session Started: " << std::ctime(&time_t);
        ofs.flush();
        
        std::vector<LogEvent> batch;
        batch.reserve(1024);
        
        while (writer_running_.load(std::memory_order_acquire)) {
            // Collect from all thread-local buffers
            {
                std::lock_guard<std::mutex> lock(registry_mutex_);
                for (auto* buf : buffers_) {
                    if (buf) {
                        buf->consume(batch, 1024);
                    }
                }
            }
            
            // Write to file
            if (!batch.empty()) {
                for (const auto& event : batch) {
                    write_event(ofs, event);
                }
                batch.clear();
                ofs.flush();
            }
            
            // Sleep to avoid busy-wait
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    void write_event(std::ofstream& ofs, const LogEvent& event) {
        // Format: ISO 8601 timestamp [LEVEL] Message
        const char* level_str = "UI";
        if (event.level == LogLevel::INFO) level_str = "INFO";
        else if (event.level == LogLevel::WARNING) level_str = "WARN";
        else if (event.level == LogLevel::ERROR) level_str = "ERROR";
        else if (event.level == LogLevel::DEBUG) level_str = "DEBUG";
        
        // Convert microseconds to ISO 8601: YYYY-MM-DD HH:MM:SS.mmm
        auto total_sec = event.timestamp_us / 1000000;
        auto millis = (event.timestamp_us % 1000000) / 1000;
        
        time_t time_sec = static_cast<time_t>(total_sec);
        std::tm* tm_info = std::localtime(&time_sec);
        
        char time_buf[32];
        std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);
        
        ofs << time_buf << "." << std::setfill('0') << std::setw(3) << millis
            << " [" << level_str << "] "
            << event.message << "\n";
    }
    
    void flush_all() {
        if (writer_running_.load(std::memory_order_acquire)) {
            return;
        }
        
        std::vector<LogEvent> events;
        
        {
            std::lock_guard<std::mutex> lock(registry_mutex_);
            for (auto* buf : buffers_) {
                if (buf) {
                    buf->consume(events, 100000);
                }
            }
        }
        
        if (!events.empty() && !log_file_path_.empty()) {
            std::ofstream ofs(log_file_path_, std::ios::out | std::ios::app);
            if (ofs) {
                for (const auto& e : events) {
                    write_event(ofs, e);
                }
            }
        }
    }
    
    std::string log_file_path_;
    LogLevel max_level_{LogLevel::INFO};
    
    std::atomic<bool> writer_running_{false};
    std::thread* writer_thread_{nullptr};
    
    std::mutex registry_mutex_;
    std::vector<LogBuffer*> buffers_;
};

// Convenience macros (backward compatibility)
#define LOG_UI(msg)    mm_rec::Logger::s_ui(msg)
#define LOG_INFO(msg)  mm_rec::Logger::s_info(msg)
#define LOG_WARN(msg)  mm_rec::Logger::s_warning(msg)
#define LOG_ERROR(msg) mm_rec::Logger::s_error(msg)
#define LOG_DEBUG(msg) mm_rec::Logger::s_debug(msg)

} // namespace mm_rec
