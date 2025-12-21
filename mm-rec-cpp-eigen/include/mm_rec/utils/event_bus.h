#pragma once

#include <functional>
#include <map>
#include <vector>
#include <mutex>
#include <string>
#include <sstream>

namespace mm_rec {

// Simple event data: key-value pairs that can be serialized to JSON
using EventData = std::map<std::string, std::string>;

// Helper to convert EventData to JSON string
inline std::string to_json(const EventData& data) {
    if (data.empty()) return "{}";
    
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (const auto& [key, value] : data) {
        if (!first) ss << ", ";
        ss << "\"" << key << "\": " << value; // Assumes value is already JSON-formatted (e.g., "\"string\"" or "123")
        first = false;
    }
    ss << "}";
    return ss.str();
}

/**
 * Event-driven publish/subscribe system for decoupled communication.
 * 
 * Events:
 *   - training.started: {{"run_name", "\"...\""}}
 *   - training.step: {{"step", "123"}, {"loss", "0.456"}, {"lr", "0.001"}}
 *   - training.stopped: {{"reason", "\"...\""}}
 *   - system.gpu_oom: {{"details", "\"...\""}}
 *   - inference.token: {{"token", "\"hello\""}, {"logprob", "-2.3"}}
 * 
 * Usage:
 *   EventBus::instance().on("training.step", [](const EventData& data) {
 *       std::cout << "Step: " << data.at("step") << std::endl;
 *   });
 * 
 *   EventBus::instance().emit("training.step", {{"step", "123"}, {"loss", "0.456"}});
 */
class EventBus {
public:
    using Handler = std::function<void(const EventData&)>;
    using Subscription = size_t;

    // DI-friendly: Public constructor
    EventBus() = default;
    ~EventBus() = default;
    
    // Non-copyable
    EventBus(const EventBus&) = delete;
    EventBus& operator=(const EventBus&) = delete;

    // Singleton access (for backward compatibility)
    static EventBus& instance() {
        static EventBus inst;
        return inst;
    }

    // Subscribe to event
    Subscription on(const std::string& event, Handler handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        Subscription id = next_id_++;
        listeners_[event].push_back({id, handler});
        
        return id;
    }

    // Unsubscribe from event
    void off(const std::string& event, Subscription id) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto& handlers = listeners_[event];
        handlers.erase(
            std::remove_if(handlers.begin(), handlers.end(),
                [id](const auto& pair) { return pair.first == id; }),
            handlers.end()
        );
    }

    // Emit event to all subscribers
    void emit(const std::string& event, const EventData& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (listeners_.find(event) == listeners_.end()) {
            return;
        }

        // Copy handlers to avoid deadlock if handler modifies listeners
        auto handlers_copy = listeners_[event];
        
        // Release lock before calling handlers
        mutex_.unlock();
        
        for (const auto& [id, handler] : handlers_copy) {
            try {
                handler(data);
            } catch (const std::exception& e) {
                // Log error but don't crash
                // TODO: Use proper logging
            }
        }
        
        mutex_.lock();
    }

    // Clear all subscribers for event
    void clear(const std::string& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        listeners_.erase(event);
    }

    // Clear all subscribers
    void clear_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        listeners_.clear();
    }

private:
    std::map<std::string, std::vector<std::pair<Subscription, Handler>>> listeners_;
    std::mutex mutex_;
    Subscription next_id_ = 0;
};

} // namespace mm_rec
