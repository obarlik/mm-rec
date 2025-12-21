#pragma once

#include <variant>
#include <stdexcept>
#include <string>
#include <functional>

namespace mm_rec {

/**
 * Type-safe error handling alternative to exceptions.
 * Similar to Rust's Result<T, E> or C++23's std::expected.
 * 
 * Usage:
 *   Result<Model, std::string> load_model(const std::string& path) {
 *       if (!exists(path)) return Result::err("File not found");
 *       return Result::ok(Model::from_file(path));
 *   }
 * 
 *   auto result = load_model("model.bin");
 *   if (result.is_ok()) {
 *       Model& model = result.value();
 *   } else {
 *       LOG_ERROR(result.error());
 *   }
 */
template<typename T, typename E = std::string>
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

    // Convenience operator
    explicit operator bool() const { return is_ok_; }

private:
    Result() = default;
};

} // namespace mm_rec
