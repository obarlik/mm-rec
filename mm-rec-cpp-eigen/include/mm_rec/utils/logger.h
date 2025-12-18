#pragma once

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <cstring>

namespace mm_rec {

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
    INFO = 1,    // System status (configurable)
    DEBUG = 2    // Debug details (disabled by default)
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
 * Global Logger
 */
class Logger {
public:
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
    static inline void log(LogLevel level, const char* msg) {
        auto& logger = instance();
        
        // Skip if writer not running (zero overhead when disabled)
        if (!logger.writer_running_.load(std::memory_order_relaxed)) {
            return;
        }
        
        // Filter by level
        if (static_cast<uint8_t>(level) > static_cast<uint8_t>(logger.max_level_)) {
            return;  // Level too verbose, skip
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
    static inline void log(LogLevel level, const std::string& msg) {
        log(level, msg.c_str());
    }
    
    // Convenience methods
    static void ui(const char* msg)    { log(LogLevel::UI, msg); }
    static void info(const char* msg)  { log(LogLevel::INFO, msg); }
    static void debug(const char* msg) { log(LogLevel::DEBUG, msg); }
    
    static void ui(const std::string& msg)    { log(LogLevel::UI, msg); }
    static void info(const std::string& msg)  { log(LogLevel::INFO, msg); }
    static void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    
    // Start background writer
    void start_writer(const std::string& log_file = "mm_rec.log", 
                     LogLevel max_level = LogLevel::INFO) {
        if (writer_running_.load(std::memory_order_acquire)) return;
        
        log_file_path_ = log_file;
        max_level_ = max_level;
        
        writer_running_.store(true, std::memory_order_release);
        writer_thread_ = new std::thread(&Logger::writer_loop, this);
    }
    
    // Stop background writer
    void stop_writer() {
        if (!writer_running_.exchange(false)) return;
        
        if (writer_thread_ && writer_thread_->joinable()) {
            writer_thread_->join();
            delete writer_thread_;
            writer_thread_ = nullptr;
        }
        
        flush_all();
    }
    
    // Minimal destructor - immortal pattern
    ~Logger() {
        writer_running_.store(false, std::memory_order_release);
        writer_thread_ = nullptr;  // Leak (singleton lives until exit)
    }
    
private:
    Logger() = default;
    
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
        // Format: [TIMESTAMP] [LEVEL] Message
        const char* level_str = "UI";
        if (event.level == LogLevel::INFO) level_str = "INFO";
        else if (event.level == LogLevel::DEBUG) level_str = "DEBUG";
        
        // Convert microseconds to readable time
        auto sec = event.timestamp_us / 1000000;
        auto usec = event.timestamp_us % 1000000;
        
        ofs << "[" << sec << "." << usec << "] "
            << "[" << level_str << "] "
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

// Convenience macros
#define LOG_UI(msg)    mm_rec::Logger::ui(msg)
#define LOG_INFO(msg)  mm_rec::Logger::info(msg)
#define LOG_DEBUG(msg) mm_rec::Logger::debug(msg)

} // namespace mm_rec
