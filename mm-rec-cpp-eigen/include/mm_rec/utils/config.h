#pragma once

#include <string>
#include <map>
#include <mutex>
#include <vector>

namespace mm_rec {

/**
 * Global Configuration Manager
 * Supports Layered Configuration:
 * 1. CLI Arguments (Highest Priority)
 * 2. Environment Variables
 * 3. Config File (config.ini)
 * 4. Default Values (Lowest Priority)
 */
class Config {
public:
    static Config& instance();

    // Loaders
    void load_from_file(const std::string& filename = "mm_rec.ini");
    void load_from_env(); // Loads known env vars
    void parse_args(int argc, char* argv[]); // Parses known CLI args

    // Generic Getters
    template<typename T>
    T get(const std::string& key, const T& default_val = T()) const;

    // Generic Setters
    template<typename T>
    void set(const std::string& key, const T& value);

    // Section Proxy for scoped access
    class Section {
    public:
        Section(Config& parent, std::string prefix) 
            : parent_(parent), prefix_(std::move(prefix)) {
            if (!prefix_.empty() && prefix_.back() != '.') prefix_ += ".";
        }

        template<typename T>
        T get(const std::string& key, const T& default_val = T()) const {
            return parent_.get<T>(prefix_ + key, default_val);
        }

        template<typename T>
        Section& set(const std::string& key, const T& value) {
            parent_.set<T>(prefix_ + key, value);
            return *this;
        }
        
        // Subsection support
        Section section(const std::string& name) const {
            return Section(parent_, prefix_ + name);
        }

    private:
        Config& parent_;
        std::string prefix_;
    };

    // Create a section view
    Section section(const std::string& prefix) {
        return Section(*this, prefix);
    }

    // DI-friendly: Public constructor
    // DI-friendly: Public constructor
    Config() = default;
    ~Config() = default;
    
    // Typed convenience getters (explicit methods for common types)
    // defined inline below specializations to avoid instantiation errors
    int get_int(const std::string& key, int default_val = 0) const;
    bool get_bool(const std::string& key, bool default_val = false) const;
    double get_double(const std::string& key, double default_val = 0.0) const;
    std::string get_string(const std::string& key, const std::string& default_val = "") const;

private:
    // Internal string-based storage helpers
    std::string get_raw(const std::string& key) const;
    void set_raw(const std::string& key, const std::string& val);

    mutable std::mutex mutex_;
    std::map<std::string, std::string> settings_;
};

// Template Specializations (must be declared BEFORE usage in inline methods)
template<> int Config::get<int>(const std::string& key, const int& default_val) const;
template<> float Config::get<float>(const std::string& key, const float& default_val) const;
template<> double Config::get<double>(const std::string& key, const double& default_val) const;
template<> bool Config::get<bool>(const std::string& key, const bool& default_val) const;
template<> std::string Config::get<std::string>(const std::string& key, const std::string& default_val) const;

// Inline Implementations (now safe to call specializations)
inline int Config::get_int(const std::string& key, int default_val) const {
    return get<int>(key, default_val);
}

inline bool Config::get_bool(const std::string& key, bool default_val) const {
    return get<bool>(key, default_val);
}

inline double Config::get_double(const std::string& key, double default_val) const {
    return get<double>(key, default_val);
}

inline std::string Config::get_string(const std::string& key, const std::string& default_val) const {
    return get<std::string>(key, default_val);
}

template<typename T>
void Config::set(const std::string& key, const T& value) {
    set_raw(key, std::to_string(value));
}

// Partial specialization for string to avoid to_string(string)
template<>
inline void Config::set<std::string>(const std::string& key, const std::string& value) {
    set_raw(key, value);
}


} // namespace mm_rec
