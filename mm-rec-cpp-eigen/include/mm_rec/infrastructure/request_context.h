#pragma once

#include <string>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <map>
#include <vector>
#include <algorithm>

namespace mm_rec {
namespace net {

/**
 * Request Context - Distributed Tracing & Correlation
 * 
 * Provides comprehensive request tracking for:
 * - End-to-end correlation across services
 * - Distributed tracing (OpenTelemetry-style)
 * - User/tenant context
 * - Performance monitoring
 * 
 * Compatible with W3C Trace Context standard.
 */
class RequestContext {
public:
    // ========================================
    // Tracing & Correlation IDs
    // ========================================
    
    /**
     * Correlation ID - Unique identifier for end-to-end request flow.
     * Same across all services/components in the request chain.
     * Format: COR-{hex-timestamp}-{random}
     */
    std::string correlation_id;
    
    /**
     * Trace ID - Distributed tracing identifier (128-bit, hex).
     * Follows W3C Trace Context specification.
     * Shared across microservices for same logical operation.
     */
    std::string trace_id;
    
    /**
     * Span ID - Current operation identifier (64-bit, hex).
     * Unique per operation/service in the trace.
     */
    std::string span_id;
    
    /**
     * Parent Span ID - Calling operation's span ID.
     * Used to build call hierarchy/tree.
     */
    std::string parent_span_id;
    
    // ========================================
    // Request Metadata
    // ========================================
    
    std::string method;           // HTTP method (GET, POST, etc.)
    std::string path;             // Request path
    std::string ip_address;       // Client IP
    std::string user_agent;       // Client user agent
    
    // ========================================
    // User & Tenant Context
    // ========================================
    
    std::string user_id;          // Authenticated user ID
    std::string username;         // Authenticated username
    std::string tenant_id;        // Multi-tenancy support
    std::string session_id;       // User session ID
    
    // ========================================
    // Baggage - Context Propagation
    // ========================================
    
    /**
     * Baggage - Key-value pairs propagated across service boundaries.
     * Use for custom context (e.g., experiment ID, feature flags).
     */
    std::map<std::string, std::string> baggage;
    
    // ========================================
    // Diagnostic Trace (In-Memory Buffer)
    // ========================================
    
    /**
     * Diagnostic trace - In-memory log buffer for error context.
     * 
     * Logs are kept in memory during request processing.
     * On success: Buffer is discarded (no disk I/O).
     * On error: Full trace is written to disk/log system.
     * 
     * This provides zero-overhead logging for successful requests
     * while capturing full context for debugging failures.
     */
    struct TraceEntry {
        std::chrono::steady_clock::time_point timestamp;
        std::string level;      // INFO, WARN, ERROR, DEBUG
        std::string component;  // Which component logged (e.g., "Auth", "Database")
        std::string message;    // Log message
        
        int64_t elapsed_ms;     // Milliseconds since request start
        uint32_t repeat_count = 1;  // For aggregation (same message repeated)
        int depth = 0;          // Scope depth (for indentation)
    };
    
    std::vector<TraceEntry> diagnostic_trace;
    bool trace_enabled = true;  // Enable/disable tracing
    size_t max_trace_entries = 1000;  // Limit memory usage (circular buffer)
    
    mutable size_t entries_dropped_ = 0;  // Track overflow
    
    // Scope depth tracking (for indentation)
    int current_scope_depth_ = 0;
    
    // ========================================
    // Timing
    // ========================================
    
    std::chrono::steady_clock::time_point start_time;
    
    // ========================================
    // Constructor
    // ========================================
    
    RequestContext() {
        correlation_id = generate_correlation_id();
        trace_id = generate_trace_id();
        span_id = generate_span_id();
        start_time = std::chrono::steady_clock::now();
    }
    
    // ========================================
    // Diagnostic Trace Methods
    // ========================================
    
    /**
     * Add diagnostic trace entry (in-memory only, with protection).
     * 
     * Memory Protection Strategies:
     * 1. Circular Buffer: Oldest entries overwritten when full
     * 2. Sampling: Can sample every N entries in hot paths
     * 3. Aggregation: Repeated messages are counted, not duplicated
     */
    void add_trace(const std::string& level, const std::string& component, const std::string& message) {
        if (!trace_enabled) return;
        
        // Check for message aggregation (prevent spam)
        if (!diagnostic_trace.empty()) {
            auto& last = diagnostic_trace.back();
            
            // If same message repeated within 1ms, just increment counter
            auto now = std::chrono::steady_clock::now();
            auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last.timestamp
            ).count();
            
            if (last.level == level && 
                last.component == component && 
                last.message == message &&
                time_diff < 1) {
                // Aggregate: increment repeat count instead of new entry
                last.repeat_count++;
                return;
            }
        }
        
        TraceEntry entry;
        entry.timestamp = std::chrono::steady_clock::now();
        entry.level = level;
        entry.component = component;
        entry.message = message;
        entry.repeat_count = 1;
        entry.depth = current_scope_depth_;  // Capture current depth
        entry.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            entry.timestamp - start_time
        ).count();
        
        // Circular buffer: Overwrite oldest if full
        if (diagnostic_trace.size() >= max_trace_entries) {
            // Shift entries (move [1..N] to [0..N-1])
            std::rotate(diagnostic_trace.begin(), diagnostic_trace.begin() + 1, diagnostic_trace.end());
            diagnostic_trace.back() = entry;
            entries_dropped_++;
        } else {
            diagnostic_trace.push_back(entry);
        }
    }
    
    /**
     * Add trace with sampling (for hot loops).
     * Only logs every Nth call to prevent memory bloat.
     * 
     * Usage:
     *   for (int i = 0; i < 100000; i++) {
     *       ctx->add_trace_sampled("INFO", "Loop", "Processing item", 1000);
     *       // Only logs every 1000th iteration
     *   }
     */
    void add_trace_sampled(const std::string& level, const std::string& component, 
                          const std::string& message, uint32_t sample_rate = 100) {
        static thread_local uint32_t counter = 0;
        counter++;
        
        if (counter % sample_rate == 0) {
            add_trace(level, component, message + " (sampled: 1/" + std::to_string(sample_rate) + ")");
        }
    }
    
    /**
     * Flush diagnostic trace to string (for error reports).
     * Returns full trace with timing information.
     */
    std::string flush_trace() const {
        if (diagnostic_trace.empty()) return "";
        
        std::ostringstream oss;
        oss << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        oss << "DIAGNOSTIC TRACE (Correlation: " << correlation_id << ")\n";
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        if (entries_dropped_ > 0) {
            oss << "⚠️  WARNING: " << entries_dropped_ << " entries dropped (circular buffer overflow)\n";
            oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        }
        
        for (const auto& entry : diagnostic_trace) {
            oss << "[+" << std::setw(6) << entry.elapsed_ms << "ms] "
                << "[" << std::setw(5) << entry.level << "] ";
            
            // Add indentation based on scope depth
            for (int i = 0; i < entry.depth; i++) {
                oss << "  ";  // 2 spaces per depth level
            }
            
            oss << std::setw(15) << std::left << entry.component << " "
                << entry.message;
            
            // Show repeat count if aggregated
            if (entry.repeat_count > 1) {
                oss << " (×" << entry.repeat_count << ")";
            }
            
            oss << "\n";
        }
        
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        oss << "Total entries: " << diagnostic_trace.size();
        if (entries_dropped_ > 0) {
            oss << " (+" << entries_dropped_ << " dropped)";
        }
        oss << " | Duration: " << elapsed_ms() << "ms\n";
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        return oss.str();
    }
    
    /**
     * Clear diagnostic trace (called on successful completion).
     */
    void clear_trace() {
        diagnostic_trace.clear();
        entries_dropped_ = 0;
    }
    
    /**
     * Get trace entry count.
     */
    size_t trace_count() const {
        return diagnostic_trace.size();
    }
    
    /**
     * Get total entries (including dropped).
     */
    size_t total_trace_attempts() const {
        return diagnostic_trace.size() + entries_dropped_;
    }
    
    /**
     * Enter a new trace scope (increment depth).
     * Used by TraceScope RAII helper.
     */
    void enter_scope(const std::string& scope_name) {
        add_trace("INFO", "Scope", "→ " + scope_name);
        current_scope_depth_++;
    }
    
    /**
     * Exit current trace scope (decrement depth).
     * Used by TraceScope RAII helper.
     */
    void exit_scope(const std::string& scope_name) {
        current_scope_depth_--;
        add_trace("INFO", "Scope", "← " + scope_name);
    }
    
    /**
     * Get current scope depth.
     */
    int scope_depth() const {
        return current_scope_depth_;
    }
    
    // ========================================
    // Timing Helpers
    // ========================================
    
    /**
     * Get elapsed time since request started (milliseconds).
     */
    int64_t elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start_time
        ).count();
    }
    
    /**
     * Get elapsed time since request started (microseconds).
     */
    int64_t elapsed_us() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            now - start_time
        ).count();
    }
    
    // ========================================
    // Logging Helpers
    // ========================================
    
    /**
     * Format as structured log prefix with correlation ID.
     * Example: [COR:abc123|TRACE:xyz|SPAN:456]
     */
    std::string log_prefix() const {
        std::ostringstream oss;
        oss << "[COR:" << correlation_id.substr(4, 6);  // Short form
        if (!trace_id.empty()) {
            oss << "|TR:" << trace_id.substr(0, 8);     // First 8 chars
        }
        if (!span_id.empty()) {
            oss << "|SP:" << span_id.substr(0, 6);      // First 6 chars
        }
        if (!user_id.empty()) {
            oss << "|U:" << user_id;
        }
        oss << "] ";
        return oss.str();
    }
    
    /**
     * Format as compact log prefix (correlation ID only).
     * Example: [COR:abc123]
     */
    std::string short_log_prefix() const {
        return "[COR:" + correlation_id.substr(4, 6) + "] ";
    }
    
    // ========================================
    // W3C Trace Context Headers
    // ========================================
    
    /**
     * Generate W3C traceparent header value.
     * Format: 00-{trace-id}-{span-id}-{flags}
     */
    std::string to_traceparent() const {
        return "00-" + trace_id + "-" + span_id + "-01";
    }
    
    /**
     * Parse W3C traceparent header.
     * Returns true if successfully parsed.
     */
    bool from_traceparent(const std::string& header) {
        // Format: 00-trace_id-parent_span_id-flags
        if (header.length() < 55) return false;
        
        try {
            trace_id = header.substr(3, 32);
            parent_span_id = header.substr(36, 16);
            span_id = generate_span_id();  // New span for this service
            return true;
        } catch (...) {
            return false;
        }
    }
    
    /**
     * Set baggage item (context propagation).
     */
    void set_baggage(const std::string& key, const std::string& value) {
        baggage[key] = value;
    }
    
    /**
     * Get baggage item.
     */
    std::string get_baggage(const std::string& key) const {
        auto it = baggage.find(key);
        return it != baggage.end() ? it->second : "";
    }

private:
    // ========================================
    // ID Generators
    // ========================================
    
    /**
     * Generate correlation ID: COR-{hex-timestamp}-{random}
     */
    static std::string generate_correlation_id() {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()
        ).count();
        
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0x1000, 0x9fff);
        
        std::ostringstream oss;
        oss << "COR-" << std::hex << (ms & 0xFFFFFF) << "-" << dis(gen);
        return oss.str();
    }
    
    /**
     * Generate trace ID (128-bit, 32 hex chars) - W3C compliant.
     */
    static std::string generate_trace_id() {
        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        oss << std::setw(16) << gen();
        oss << std::setw(16) << gen();
        return oss.str();
    }
    
    /**
     * Generate span ID (64-bit, 16 hex chars) - W3C compliant.
     */
    static std::string generate_span_id() {
        static std::random_device rd;
        static std::mt19937_64 gen(rd());
        
        std::ostringstream oss;
        oss << std::hex << std::setfill('0') << std::setw(16) << gen();
        return oss.str();
    }
};

/**
 * RAII Trace Scope Helper
 * 
 * Automatically tracks function entry/exit with indentation.
 * Use TRACE_FUNC() macro for automatic function name capture.
 * 
 * Example:
 *   void UserService::process_user(int id) {
 *       TRACE_FUNC();  // Compiler inserts function name!
 *       // ... work ...
 *   } // Auto exit trace
 */
class TraceScope {
private:
    RequestContext* ctx_;
    const char* scope_name_;  // Pointer, not copy!
    bool is_active_;
    
public:
    /**
     * Enter scope (zero-copy when disabled).
     * Uses const char* to avoid string allocation overhead.
     */
    TraceScope(RequestContext* ctx, const char* name)
        : ctx_(ctx), scope_name_(name), is_active_(false) {
        // Early exit if no context or tracing disabled (ZERO overhead!)
        if (!ctx_ || !ctx_->trace_enabled) {
            ctx_ = nullptr;  // Mark as inactive
            return;
        }
        
        is_active_ = true;
        ctx_->enter_scope(scope_name_);  // Only allocate string if needed!
    }
    
    /**
     * Exit scope (RAII destructor).
     */
    ~TraceScope() {
        if (is_active_ && ctx_) {
            ctx_->exit_scope(scope_name_);
        }
    }
    
    // Non-copyable
    TraceScope(const TraceScope&) = delete;
    TraceScope& operator=(const TraceScope&) = delete;
};

} // namespace net
} // namespace mm_rec

// ========================================
// Convenience Macros
// ========================================

/**
 * Automatic function tracing with compiler-provided name.
 * 
 * Usage:
 *   void MyClass::process_order(int id) {
 *       TRACE_FUNC();  // Automatically traces "MyClass::process_order"
 *       // ... method body ...
 *   }
 * 
 * GCC/Clang: Uses __PRETTY_FUNCTION__ for full signature
 * MSVC: Uses __FUNCSIG__
 * Fallback: Uses __FUNCTION__
 */
#if defined(__GNUC__) || defined(__clang__)
    #define TRACE_FUNC() \
        mm_rec::net::TraceScope __trace_scope_guard__(get_request_context(), __PRETTY_FUNCTION__)
#elif defined(_MSC_VER)
    #define TRACE_FUNC() \
        mm_rec::net::TraceScope __trace_scope_guard__(get_request_context(), __FUNCSIG__)
#else
    #define TRACE_FUNC() \
        mm_rec::net::TraceScope __trace_scope_guard__(get_request_context(), __FUNCTION__)
#endif

/**
 * Manual scope tracing with custom name.
 * 
 * Usage:
 *   TRACE_SCOPE("Database Query");
 */
#define TRACE_SCOPE(name) \
    mm_rec::net::TraceScope __trace_scope_guard__(get_request_context(), name)

/**
 * Get RequestContext from current DI scope.
 * Helper for TRACE_FUNC macro.
 * IMPORTANT: This must be implemented by the application.
 * Default implementation returns nullptr (no tracing).
 */
mm_rec::net::RequestContext* get_request_context();
