#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <map>

namespace mm_rec {
namespace diagnostics {

/**
 * Error Trace Record
 * Stores full diagnostic trace for a failed request.
 */
struct ErrorTrace {
    std::string correlation_id;
    std::string trace_id;
    std::string error_message;
    std::string error_code;
    std::chrono::system_clock::time_point timestamp;
    std::string full_trace;         // Complete diagnostic trace
    std::string request_path;
    std::string request_method;
    int status_code;
    int64_t duration_ms;
    
    // Metadata
    std::map<std::string, std::string> metadata;
};

/**
 * IDiagnosticManager Interface
 * 
 * Interface for diagnostic tracing management.
 * Allows for easy mocking in tests.
 */
class IDiagnosticManager {
public:
    virtual ~IDiagnosticManager() = default;
    
    // Configuration
    virtual void set_tracing_enabled(bool enabled) = 0;
    virtual bool is_tracing_enabled() const = 0;
    virtual void set_max_error_traces(size_t max) = 0;
    
    // Error trace storage
    virtual void record_error_trace(const ErrorTrace& trace) = 0;
    virtual std::vector<ErrorTrace> get_recent_errors(size_t limit = 50) const = 0;
    virtual std::optional<ErrorTrace> get_error_by_correlation_id(const std::string& corr_id) const = 0;
    virtual void clear_error_traces() = 0;
    
    // Statistics
    virtual void increment_request_count() = 0;
    virtual void increment_trace_collected() = 0;
    virtual void increment_trace_dropped(size_t count) = 0;
    virtual Statistics get_statistics() const = 0;
    virtual void reset_statistics() = 0;
    
    // JSON serialization
    virtual std::string error_trace_to_json(const ErrorTrace& trace) const = 0;
    virtual std::string statistics_to_json() const = 0;
};

/**
 * Diagnostic Manager
 * 
 * Centralized management for diagnostic tracing:
 * - Runtime enable/disable tracing
 * - Store error traces
 * - Dashboard API endpoints
 * - Statistics tracking
 */
class DiagnosticManager : public IDiagnosticManager {
private:
    // Configuration
    bool tracing_globally_enabled_ = true;
    size_t max_error_traces_ = 100;  // Keep last 100 errors
    
    // Error trace storage (circular buffer)
    std::vector<ErrorTrace> error_traces_;
    mutable std::mutex traces_mutex_;
    
    // Statistics
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> total_errors_{0};
    std::atomic<uint64_t> total_traces_collected_{0};
    std::atomic<uint64_t> total_traces_dropped_{0};
    
public:
    // DI-friendly: Public constructor
    DiagnosticManager() = default;
    
    // ========================================
    // Configuration
    // ========================================
    
    /**
     * Enable/disable tracing globally.
     * When disabled, all TRACE_FUNC() calls become no-ops.
     */
    void set_tracing_enabled(bool enabled) override {
        tracing_globally_enabled_ = enabled;
    }
    
    bool is_tracing_enabled() const override {
        return tracing_globally_enabled_;
    }
    
    /**
     * Set maximum error traces to keep in memory.
     */
    void set_max_error_traces(size_t max) override {
        max_error_traces_ = max;
    }
    
    // ========================================
    // Error Trace Storage
    // ========================================
    
    /**
     * Record an error trace.
     * Called automatically when request fails.
     */
    void record_error_trace(const ErrorTrace& trace) override {
        std::lock_guard<std::mutex> lock(traces_mutex_);
        
        total_errors_++;
        
        // Circular buffer
        if (error_traces_.size() >= max_error_traces_) {
            error_traces_.erase(error_traces_.begin());
        }
        
        error_traces_.push_back(trace);
    }
    
    /**
     * Get recent error traces.
     */
    std::vector<ErrorTrace> get_recent_errors(size_t limit = 50) const {
        std::lock_guard<std::mutex> lock(traces_mutex_);
        
        size_t start = error_traces_.size() > limit ? error_traces_.size() - limit : 0;
        return std::vector<ErrorTrace>(
            error_traces_.begin() + start,
            error_traces_.end()
        );
    }
    
    /**
     * Get error trace by correlation ID.
     */
    std::optional<ErrorTrace> get_error_by_correlation_id(const std::string& corr_id) const {
        std::lock_guard<std::mutex> lock(traces_mutex_);
        
        for (auto it = error_traces_.rbegin(); it != error_traces_.rend(); ++it) {
            if (it->correlation_id == corr_id) {
                return *it;
            }
        }
        
        return std::nullopt;
    }
    
    /**
     * Clear all error traces.
     */
    void clear_error_traces() override {
        std::lock_guard<std::mutex> lock(traces_mutex_);
        error_traces_.clear();
    }
    
    // ========================================
    // Statistics
    // ========================================
    
    void increment_request_count() override {
        total_requests_++;
    }
    
    void increment_trace_collected() override {
        total_traces_collected_++;
    }
    
    void increment_trace_dropped(size_t count) override {
        total_traces_dropped_ += count;
    }
    
    struct Statistics {
        uint64_t total_requests;
        uint64_t total_errors;
        uint64_t total_traces_collected;
        uint64_t total_traces_dropped;
        size_t error_traces_in_memory;
        double error_rate;
        bool tracing_enabled;
    };
    
    Statistics get_statistics() const {
        std::lock_guard<std::mutex> lock(traces_mutex_);
        
        Statistics stats;
        stats.total_requests = total_requests_.load();
        stats.total_errors = total_errors_.load();
        stats.total_traces_collected = total_traces_collected_.load();
        stats.total_traces_dropped = total_traces_dropped_.load();
        stats.error_traces_in_memory = error_traces_.size();
        stats.error_rate = stats.total_requests > 0 
            ? (static_cast<double>(stats.total_errors) / stats.total_requests * 100.0)
            : 0.0;
        stats.tracing_enabled = tracing_globally_enabled_;
        
        return stats;
    }
    
    /**
     * Reset statistics (useful for testing).
     */
    void reset_statistics() override {
        total_requests_ = 0;
        total_errors_ = 0;
        total_traces_collected_ = 0;
        total_traces_dropped_ = 0;
    }
    
    // ========================================
    // JSON Serialization (for Dashboard API)
    // ========================================
    
    /**
     * Export error trace as JSON.
     */
    std::string error_trace_to_json(const ErrorTrace& trace) const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"correlation_id\": \"" << trace.correlation_id << "\",\n";
        oss << "  \"trace_id\": \"" << trace.trace_id << "\",\n";
        oss << "  \"error_message\": \"" << json_escape(trace.error_message) << "\",\n";
        oss << "  \"error_code\": \"" << trace.error_code << "\",\n";
        oss << "  \"timestamp\": " << std::chrono::system_clock::to_time_t(trace.timestamp) << ",\n";
        oss << "  \"request_path\": \"" << trace.request_path << "\",\n";
        oss << "  \"request_method\": \"" << trace.request_method << "\",\n";
        oss << "  \"status_code\": " << trace.status_code << ",\n";
        oss << "  \"duration_ms\": " << trace.duration_ms << ",\n";
        oss << "  \"full_trace\": \"" << json_escape(trace.full_trace) << "\"\n";
        oss << "}";
        return oss.str();
    }
    
    /**
     * Export statistics as JSON.
     */
    std::string statistics_to_json() const {
        auto stats = get_statistics();
        
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"total_requests\": " << stats.total_requests << ",\n";
        oss << "  \"total_errors\": " << stats.total_errors << ",\n";
        oss << "  \"error_rate\": " << stats.error_rate << ",\n";
        oss << "  \"total_traces_collected\": " << stats.total_traces_collected << ",\n";
        oss << "  \"total_traces_dropped\": " << stats.total_traces_dropped << ",\n";
        oss << "  \"error_traces_in_memory\": " << stats.error_traces_in_memory << ",\n";
        oss << "  \"tracing_enabled\": " << (stats.tracing_enabled ? "true" : "false") << "\n";
        oss << "}";
        return oss.str();
    }

private:
    std::string json_escape(const std::string& str) const {
        std::ostringstream oss;
        for (char c : str) {
            switch (c) {
                case '"': oss << "\\\""; break;
                case '\\': oss << "\\\\"; break;
                case '\n': oss << "\\n"; break;
                case '\r': oss << "\\r"; break;
                case '\t': oss << "\\t"; break;
                default: oss << c; break;
            }
        }
        return oss.str();
    }
};

} // namespace diagnostics
} // namespace mm_rec
