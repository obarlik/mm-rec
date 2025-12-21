#pragma once

#include <variant>
#include <stdexcept>
#include <string>
#include <functional>
#include <vector>
#include <chrono>
#include <sstream>
#include <map>

namespace mm_rec {

/**
 * Enhanced Error with Correlation ID and Stack Trace
 * 
 * Provides comprehensive error context for debugging:
 * - Correlation ID for request tracking
 * - Error chain/stack trace
 * - Timestamp
 * - Error code
 */
struct ErrorContext {
    std::string message;              // Error message
    std::string correlation_id;       // Request correlation ID
    std::string trace_id;             // Distributed trace ID
    std::string error_code;           // Machine-readable error code (e.g., "DB_TIMEOUT")
    std::string location;             // Where error occurred (file:line or component)
    
    // Error chain (stack trace)
    std::vector<std::string> stack_trace;
    
    // Timestamp
    std::chrono::system_clock::time_point timestamp;
    
    // Additional context
    std::map<std::string, std::string> metadata;
    
    ErrorContext(const std::string& msg = "")
        : message(msg), timestamp(std::chrono::system_clock::now()) {}
    
    // Add to stack trace
    ErrorContext& with_trace(const std::string& location) {
        stack_trace.push_back(location);
        return *this;
    }
    
    // Set correlation ID
    ErrorContext& with_correlation(const std::string& corr_id) {
        correlation_id = corr_id;
        return *this;
    }
    
    // Set trace ID
    ErrorContext& with_trace_id(const std::string& tid) {
        trace_id = tid;
        return *this;
    }
    
    // Set error code
    ErrorContext& with_code(const std::string& code) {
        error_code = code;
        return *this;
    }
    
    // Set location
    ErrorContext& at(const std::string& loc) {
        location = loc;
        return *this;
    }
    
    // Add metadata
    ErrorContext& with_metadata(const std::string& key, const std::string& value) {
        metadata[key] = value;
        return *this;
    }
    
    /**
     * Format as detailed error report.
     */
    std::string to_string() const {
        std::ostringstream oss;
        
        // Header
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        oss << "ERROR: " << message << "\n";
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        // Correlation tracking
        if (!correlation_id.empty()) {
            oss << "Correlation ID: " << correlation_id << "\n";
        }
        if (!trace_id.empty()) {
            oss << "Trace ID:       " << trace_id << "\n";
        }
        
        // Error details
        if (!error_code.empty()) {
            oss << "Error Code:     " << error_code << "\n";
        }
        if (!location.empty()) {
            oss << "Location:       " << location << "\n";
        }
        
        // Metadata
        if (!metadata.empty()) {
            oss << "\nContext:\n";
            for (const auto& [key, value] : metadata) {
                oss << "  " << key << ": " << value << "\n";
            }
        }
        
        // Stack trace
        if (!stack_trace.empty()) {
            oss << "\nStack Trace:\n";
            for (size_t i = 0; i < stack_trace.size(); ++i) {
                oss << "  " << (i + 1) << ". " << stack_trace[i] << "\n";
            }
        }
        
        oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        return oss.str();
    }
    
    /**
     * Format as compact log line.
     */
    std::string to_log_line() const {
        std::ostringstream oss;
        oss << "[";
        if (!correlation_id.empty()) {
            oss << "COR:" << correlation_id;
        }
        if (!error_code.empty()) {
            oss << "|ERR:" << error_code;
        }
        oss << "] " << message;
        if (!location.empty()) {
            oss << " @ " << location;
        }
        return oss.str();
    }
};

/**
 * Type-safe error handling with enhanced error context.
 * Similar to Rust's Result<T, E> or C++23's std::expected.
 * 
 * Usage:
 *   Result<Model> load_model(const std::string& path, RequestContext* ctx) {
 *       if (!exists(path)) {
 *           return Result<Model>::err(
 *               ErrorContext("File not found")
 *                   .with_correlation(ctx->correlation_id)
 *                   .with_code("FILE_NOT_FOUND")
 *                   .at("ModelLoader::load")
 *                   .with_metadata("path", path)
 *           );
 *       }
 *       return Result<Model>::ok(Model::from_file(path));
 *   }
 * 
 *   auto result = load_model("model.bin", ctx);
 *   if (result.is_err()) {
 *       LOG_ERROR(result.error().to_string());  // Full error report
 *       // Or compact: LOG_ERROR(result.error().to_log_line());
 *   }
 */
template<typename T, typename E = ErrorContext>
class Result {
private:
    std::variant<T, E> data_;
    bool is_ok_;

public:
    // Constructors (private, use static factory methods)
    static Result ok(T value) {
        Result r;
        r.data_ = std::move(value);
        r.is_ok_ = true;
        return r;
    }

    static Result err(E error) {
        Result r;
        r.data_ = std::move(error);
        r.is_ok_ = false;
        return r;
    }
    
    // Convenience: Create error from string (backward compat)
    static Result err(const std::string& message) {
        return err(ErrorContext(message));
    }

    // State checks
    bool is_ok() const { return is_ok_; }
    bool is_err() const { return !is_ok_; }

    // Value access (throws if wrong state)
    T& value() {
        if (!is_ok_) {
            throw std::runtime_error("Called value() on error Result");
        }
        return std::get<T>(data_);
    }

    const T& value() const {
        if (!is_ok_) {
            throw std::runtime_error("Called value() on error Result");
        }
        return std::get<T>(data_);
    }

    E& error() {
        if (is_ok_) {
            throw std::runtime_error("Called error() on ok Result");
        }
        return std::get<E>(data_);
    }

    const E& error() const {
        if (is_ok_) {
            throw std::runtime_error("Called error() on ok Result");
        }
        return std::get<E>(data_);
    }

    // Safe value access with default
    T value_or(T default_val) const {
        return is_ok_ ? std::get<T>(data_) : default_val;
    }

    // Monadic operations
    template<typename F>
    auto map(F&& func) -> Result<decltype(func(std::declval<T>())), E> {
        using U = decltype(func(std::declval<T>()));
        
        if (is_ok_) {
            return Result<U, E>::ok(func(std::get<T>(data_)));
        } else {
            return Result<U, E>::err(std::get<E>(data_));
        }
    }

    template<typename F>
    auto and_then(F&& func) -> decltype(func(std::declval<T>())) {
        using ResultType = decltype(func(std::declval<T>()));
        
        if (is_ok_) {
            return func(std::get<T>(data_));
        } else {
            return ResultType::err(std::get<E>(data_));
        }
    }
    
    /**
     * Map error (add context to error chain).
     */
    Result& map_err(std::function<void(E&)> func) {
        if (is_err()) {
            func(std::get<E>(data_));
        }
        return *this;
    }

    // Convenience operator
    explicit operator bool() const { return is_ok_; }

private:
    Result() = default;
};

// Convenience macro for error creation with location
#define ERR(msg) \
    mm_rec::ErrorContext(msg).at(__FILE__ ":" + std::to_string(__LINE__))

#define ERR_WITH_CTX(ctx, msg, code) \
    mm_rec::ErrorContext(msg) \
        .with_correlation((ctx)->correlation_id) \
        .with_trace_id((ctx)->trace_id) \
        .with_code(code) \
        .at(__FILE__ ":" + std::to_string(__LINE__))

} // namespace mm_rec
