#include "mm_rec/infrastructure/config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>

namespace mm_rec {

Config& Config::instance() {
    static Config instance;
    return instance;
}

void Config::set_raw(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    settings_[key] = value;
}

std::string Config::get_raw(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = settings_.find(key);
    if (it != settings_.end()) return it->second;
    return "";
}

// Template Implementations
template<> int Config::get<int>(const std::string& key, const int& default_val) const {
    std::string val = get_raw(key);
    if (val.empty()) return default_val;
    try { return std::stoi(val); } catch (...) { return default_val; }
}

template<> float Config::get<float>(const std::string& key, const float& default_val) const {
    std::string val = get_raw(key);
    if (val.empty()) return default_val;
    try { return std::stof(val); } catch (...) { return default_val; }
}

template<> double Config::get<double>(const std::string& key, const double& default_val) const {
    std::string val = get_raw(key);
    if (val.empty()) return default_val;
    try { return std::stod(val); } catch (...) { return default_val; }
}

template<> bool Config::get<bool>(const std::string& key, const bool& default_val) const {
    std::string val = get_raw(key);
    if (val.empty()) return default_val;
    std::transform(val.begin(), val.end(), val.begin(), ::tolower);
    return (val == "true" || val == "1" || val == "yes" || val == "on");
}

template<> std::string Config::get<std::string>(const std::string& key, const std::string& default_val) const {
    std::string val = get_raw(key);
    return val.empty() ? default_val : val;
}

void Config::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return;

    std::string line;
    std::string current_section = "";

    while (std::getline(file, line)) {
        // Trim leading whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        
        // Trim trailing whitespace
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Check for section [Header]
        if (line.front() == '[' && line.back() == ']') {
            current_section = line.substr(1, line.size() - 2);
            // Trim section name just in case
            current_section.erase(0, current_section.find_first_not_of(" \t"));
            current_section.erase(current_section.find_last_not_of(" \t") + 1);
            if (!current_section.empty()) current_section += ".";
            continue;
        }

        std::stringstream ss(line);
        std::string key, val;
        if (std::getline(ss, key, '=') && std::getline(ss, val)) {
            // Trim key
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            
            // Trim val
            val.erase(0, val.find_first_not_of(" \t"));
            val.erase(val.find_last_not_of(" \t") + 1);

            // Strip quotes if present
            if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
                val = val.substr(1, val.size() - 2);
            }

            set_raw(current_section + key, val);
        }
    }
}

void Config::load_from_env() {
    const std::vector<std::pair<std::string, std::string>> env_map = {
        {"MM_REC_PORT", "server.port"},
        {"MM_REC_THREADS", "server.threads"},
        {"MM_REC_TIMEOUT", "server.timeout"},
        {"MM_REC_LOG_LEVEL", "log.level"},
        {"MM_REC_DATA_PATH", "data.path"}
    };

    for (const auto& [env_var, config_key] : env_map) {
        if (const char* env_val = std::getenv(env_var.c_str())) {
            set_raw(config_key, env_val);
        }
    }
}

void Config::parse_args(int argc, char* argv[]) {
    auto server_conf = section("server");
    
    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (i + 1 < argc) {
            if (arg == "--port") server_conf.set("port", std::stoi(argv[++i]));
            else if (arg == "--threads") server_conf.set("threads", std::stoi(argv[++i]));
            else if (arg == "--timeout") server_conf.set("timeout", std::stoi(argv[++i]));
            else if (arg == "--data") set("data.path", std::string(argv[++i]));
        }
        if (arg == "--debug") set("log.level", std::string("debug"));
    }
}

} // namespace mm_rec
