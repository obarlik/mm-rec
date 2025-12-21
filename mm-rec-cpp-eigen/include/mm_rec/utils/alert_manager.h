#pragma once

#include "mm_rec/utils/diagnostic_manager.h"
#include <queue>
#include <chrono>

namespace mm_rec {
namespace diagnostics {

/**
 * Alert Severity Levels
 */
enum class AlertLevel {
    INFO,      // Informational
    WARNING,   // Potential issue
    ERROR,     // Requires attention
    CRITICAL   // Requires immediate action
};

/**
 * Alert Record
 */
struct Alert {
    AlertLevel level;
    std::string title;
    std::string message;
    std::chrono::system_clock::time_point timestamp;
    std::string correlation_id;  // Optional
    std::map<std::string, std::string> metadata;
    
    bool acknowledged = false;
};

/**
 * Performance Metrics
 */
struct PerformanceMetrics {
    // Response time stats (in milliseconds)
    double avg_response_time = 0.0;
    double p50_response_time = 0.0;
    double p95_response_time = 0.0;
    double p99_response_time = 0.0;
    double max_response_time = 0.0;
    
    // Throughput (requests per second)
    double requests_per_second = 0.0;
    
    // Slow requests (> threshold)
    uint64_t slow_requests_count = 0;
    double slow_request_threshold_ms = 1000.0;  // Default: 1 second
    
    // Memory usage
    size_t trace_buffer_usage_percent = 0;
    size_t trace_buffer_max_entries = 1000;
};

/**
 * Log Entry
 */
struct LogEntry {
    std::string level;
    std::string component;
    std::string message;
    std::chrono::system_clock::time_point timestamp;
    std::string correlation_id;
};

/**
 * Alert Manager
 * 
 * Monitors system health and generates alerts:
 * - High error rate
 * - Slow response times
 * - Memory pressure
 * - Custom thresholds
 */
class AlertManager {
private:
    // Alert storage (last N alerts)
    std::deque<Alert> alerts_;
    mutable std::mutex alerts_mutex_;
    size_t max_alerts_ = 1000;
    
    // Alert thresholds
    double error_rate_threshold_ = 5.0;      // 5% error rate
    double slow_request_threshold_ms_ = 1000.0;  // 1 second
    size_t slow_request_count_threshold_ = 10;   // 10 slow requests
    
    // Performance tracking
    std::deque<double> response_times_;  // Last N response times
    mutable std::mutex perf_mutex_;
    size_t max_response_times_ = 1000;
    
    // Log storage (ring buffer)
    std::deque<LogEntry> logs_;
    mutable std::mutex logs_mutex_;
    size_t max_logs_ = 10000;  // Keep last 10K log entries
    
public:
    // DI-friendly: Public constructor
    AlertManager() = default;
    
    // ========================================
    // Alert Management
    // ========================================
    
    /**
     * Generate alert.
     */
    void create_alert(AlertLevel level, const std::string& title, 
                     const std::string& message, 
                     const std::string& corr_id = "") {
        std::lock_guard<std::mutex> lock(alerts_mutex_);
        
        Alert alert;
        alert.level = level;
        alert.title = title;
        alert.message = message;
        alert.timestamp = std::chrono::system_clock::now();
        alert.correlation_id = corr_id;
        
        // Ring buffer
        if (alerts_.size() >= max_alerts_) {
            alerts_.pop_front();
        }
        
        alerts_.push_back(alert);
    }
    
    /**
     * Get unacknowledged alerts.
     */
    std::vector<Alert> get_active_alerts() const {
        std::lock_guard<std::mutex> lock(alerts_mutex_);
        
        std::vector<Alert> active;
        for (const auto& alert : alerts_) {
            if (!alert.acknowledged) {
                active.push_back(alert);
            }
        }
        return active;
    }
    
    /**
     * Get all recent alerts.
     */
    std::vector<Alert> get_recent_alerts(size_t limit = 100) const {
        std::lock_guard<std::mutex> lock(alerts_mutex_);
        
        size_t start = alerts_.size() > limit ? alerts_.size() - limit : 0;
        return std::vector<Alert>(alerts_.begin() + start, alerts_.end());
    }
    
    /**
     * Acknowledge alert.
     */
    void acknowledge_alert(size_t index) {
        std::lock_guard<std::mutex> lock(alerts_mutex_);
        if (index < alerts_.size()) {
            alerts_[index].acknowledged = true;
        }
    }
    
    /**
     * Clear all alerts.
     */
    void clear_alerts() {
        std::lock_guard<std::mutex> lock(alerts_mutex_);
        alerts_.clear();
    }
    
    // ========================================
    // Performance Monitoring
    // ========================================
    
    /**
     * Record response time for performance tracking.
     */
    void record_response_time(double milliseconds) {
        std::lock_guard<std::mutex> lock(perf_mutex_);
        
        // Ring buffer
        if (response_times_.size() >= max_response_times_) {
            response_times_.pop_front();
        }
        
        response_times_.push_back(milliseconds);
        
        // Check for slow request alert
        if (milliseconds > slow_request_threshold_ms_) {
            size_t recent_slow_count = 0;
            for (auto it = response_times_.rbegin(); 
                 it != response_times_.rend() && recent_slow_count < 100; 
                 ++it) {
                if (*it > slow_request_threshold_ms_) {
                    recent_slow_count++;
                }
            }
            
            if (recent_slow_count >= slow_request_count_threshold_) {
                create_alert(
                    AlertLevel::WARNING,
                    "High Slow Request Rate",
                    "Detected " + std::to_string(recent_slow_count) + 
                    " slow requests (>" + std::to_string(slow_request_threshold_ms_) + "ms)"
                );
            }
        }
    }
    
    /**
     * Get performance metrics.
     */
    PerformanceMetrics get_performance_metrics() const {
        std::lock_guard<std::mutex> lock(perf_mutex_);
        
        PerformanceMetrics metrics;
        
        if (response_times_.empty()) {
            return metrics;
        }
        
        // Calculate stats
        std::vector<double> sorted(response_times_.begin(), response_times_.end());
        std::sort(sorted.begin(), sorted.end());
        
        // Average
        double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        metrics.avg_response_time = sum / sorted.size();
        
        // Percentiles
        metrics.p50_response_time = sorted[sorted.size() * 50 / 100];
        metrics.p95_response_time = sorted[sorted.size() * 95 / 100];
        metrics.p99_response_time = sorted[sorted.size() * 99 / 100];
        metrics.max_response_time = sorted.back();
        
        // Slow requests count
        metrics.slow_requests_count = std::count_if(
            sorted.begin(), sorted.end(),
            [this](double t) { return t > slow_request_threshold_ms_; }
        );
        
        return metrics;
    }
    
    // ========================================
    // Log Management
    // ========================================
    
    /**
     * Add log entry.
     */
    void add_log(const std::string& level, const std::string& component,
                const std::string& message, const std::string& corr_id = "") {
        std::lock_guard<std::mutex> lock(logs_mutex_);
        
        LogEntry entry;
        entry.level = level;
        entry.component = component;
        entry.message = message;
        entry.timestamp = std::chrono::system_clock::now();
        entry.correlation_id = corr_id;
        
        // Ring buffer
        if (logs_.size() >= max_logs_) {
            logs_.pop_front();
        }
        
        logs_.push_back(entry);
    }
    
    /**
     * Get recent logs.
     */
    std::vector<LogEntry> get_recent_logs(size_t limit = 100) const {
        std::lock_guard<std::mutex> lock(logs_mutex_);
        
        size_t start = logs_.size() > limit ? logs_.size() - limit : 0;
        return std::vector<LogEntry>(logs_.begin() + start, logs_.end());
    }
    
    /**
     * Search logs by correlation ID.
     */
    std::vector<LogEntry> search_logs_by_correlation(const std::string& corr_id) const {
        std::lock_guard<std::mutex> lock(logs_mutex_);
        
        std::vector<LogEntry> results;
        for (const auto& log : logs_) {
            if (log.correlation_id == corr_id) {
                results.push_back(log);
            }
        }
        return results;
    }
    
    // ========================================
    // Health Checks
    // ========================================
    
    /**
     * Check system health and generate alerts if needed.
     * Requires DiagnosticManager to be injected separately.
     */
    void check_health(DiagnosticManager* diag_mgr) {
        if (!diag_mgr) return;
        
        auto stats = diag_mgr->get_statistics();
        
        // Check error rate
        if (stats.error_rate > error_rate_threshold_) {
            create_alert(
                AlertLevel::ERROR,
                "High Error Rate",
                "Error rate is " + std::to_string(stats.error_rate) + 
                "% (threshold: " + std::to_string(error_rate_threshold_) + "%)"
            );
        }
        
        // Check trace buffer usage
        if (stats.total_traces_dropped > 0) {
            create_alert(
                AlertLevel::WARNING,
                "Trace Buffer Overflow",
                "Dropped " + std::to_string(stats.total_traces_dropped) + " trace entries"
            );
        }
    }
    
    // ========================================
    // Configuration
    // ========================================
    
    void set_error_rate_threshold(double threshold) {
        error_rate_threshold_ = threshold;
    }
    
    void set_slow_request_threshold(double ms) {
        slow_request_threshold_ms_ = ms;
    }
};

} // namespace diagnostics
} // namespace mm_rec
