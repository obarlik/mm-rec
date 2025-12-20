#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace mm_rec {
namespace ui {

/**
 * Terminal Colors (ANSI escape codes)
 */
namespace color {
    constexpr const char* RESET   = "\033[0m";
    constexpr const char* BOLD    = "\033[1m";
    constexpr const char* DIM     = "\033[2m";
    
    // Foreground colors
    constexpr const char* BLACK   = "\033[30m";
    constexpr const char* RED     = "\033[31m";
    constexpr const char* GREEN   = "\033[32m";
    constexpr const char* YELLOW  = "\033[33m";
    constexpr const char* BLUE    = "\033[34m";
    constexpr const char* MAGENTA = "\033[35m";
    constexpr const char* CYAN    = "\033[36m";
    constexpr const char* WHITE   = "\033[37m";
    
    // Bright colors
    constexpr const char* BRIGHT_BLACK   = "\033[90m";
    constexpr const char* BRIGHT_RED     = "\033[91m";
    constexpr const char* BRIGHT_GREEN   = "\033[92m";
    constexpr const char* BRIGHT_YELLOW  = "\033[93m";
    constexpr const char* BRIGHT_BLUE    = "\033[94m";
    constexpr const char* BRIGHT_MAGENTA = "\033[95m";
    constexpr const char* BRIGHT_CYAN    = "\033[96m";
    constexpr const char* BRIGHT_WHITE   = "\033[97m";
} // namespace color

/**
 * Box drawing characters
 */
namespace box {
    constexpr const char* HORIZONTAL = "─";
    constexpr const char* VERTICAL   = "│";
    constexpr const char* TOP_LEFT   = "┌";
    constexpr const char* TOP_RIGHT  = "┐";
    constexpr const char* BOTTOM_LEFT = "└";
    constexpr const char* BOTTOM_RIGHT = "┘";
    constexpr const char* T_DOWN    = "┬";
    constexpr const char* T_UP      = "┴";
    constexpr const char* T_RIGHT   = "├";
    constexpr const char* T_LEFT    = "┤";
    constexpr const char* CROSS     = "┼";
} // namespace box

/**
 * Unicode symbols
 */
namespace symbol {
    constexpr const char* CHECKMARK = "✓";
    constexpr const char* CROSS     = "✗";
    constexpr const char* ARROW_RIGHT = "→";
    constexpr const char* ARROW_LEFT  = "←";
    constexpr const char* BULLET    = "•";
    constexpr const char* STAR      = "★";
    constexpr const char* HEART     = "♥";
    constexpr const char* SPINNER[4] = {"⠋", "⠙", "⠹", "⠸"};
} // namespace symbol

/**
 * Progress Bar
 * 
 * Usage:
 *   ProgressBar bar(100);
 *   for (int i = 0; i <= 100; ++i) {
 *       bar.update(i);
 *       std::this_thread::sleep_for(std::chrono::milliseconds(50));
 *   }
 *   bar.finish();
 */
class ProgressBar {
public:
    ProgressBar(int total, int width = 50, const std::string& prefix = "")
        : total_(total), width_(width), prefix_(prefix), current_(0) {}
    
    void update(int current, const std::string& suffix = "") {
        current_ = current;
        
        int progress = (current_ * width_) / total_;
        int percentage = (current_ * 100) / total_;
        
        std::cout << "\r" << prefix_;
        std::cout << color::CYAN << "[";
        
        for (int i = 0; i < width_; ++i) {
            if (i < progress) {
                std::cout << color::GREEN << "█";
            } else {
                std::cout << color::DIM << "░";
            }
        }
        
        std::cout << color::CYAN << "] ";
        std::cout << color::BOLD << std::setw(3) << percentage << "%";
        std::cout << color::RESET;
        
        if (!suffix.empty()) {
            std::cout << " " << suffix;
        }
        
        std::cout << std::flush;
    }
    
    void finish() {
        update(total_, color::GREEN + std::string(symbol::CHECKMARK) + " Done" + std::string(color::RESET));
        std::cout << std::endl;
    }
    
private:
    int total_;
    int width_;
    std::string prefix_;
    int current_;
};

/**
 * Spinner (for indefinite progress)
 * 
 * Usage:
 *   Spinner spinner("Loading");
 *   for (int i = 0; i < 100; ++i) {
 *       spinner.tick();
 *       std::this_thread::sleep_for(std::chrono::milliseconds(100));
 *   }
 *   spinner.finish("Done!");
 */
class Spinner {
public:
    explicit Spinner(const std::string& message = "Working")
        : message_(message), frame_(0) {}
    
    void tick() {
        std::cout << "\r" << color::CYAN << symbol::SPINNER[frame_ % 4] 
                  << color::RESET << " " << message_ << std::flush;
        ++frame_;
    }
    
    void finish(const std::string& final_msg = "") {
        std::string msg = final_msg.empty() ? message_ : final_msg;
        std::cout << "\r" << color::GREEN << symbol::CHECKMARK 
                  << color::RESET << " " << msg << std::endl;
    }
    
private:
    std::string message_;
    int frame_;
};

/**
 * Formatted Headers
 */
inline void print_header(const std::string& title, int width = 60) {
    std::cout << color::BOLD << color::CYAN;
    std::cout << "\n" << box::TOP_LEFT;
    for (int i = 0; i < width - 2; ++i) std::cout << box::HORIZONTAL;
    std::cout << box::TOP_RIGHT << "\n";
    
    int padding = (width - title.length() - 2) / 2;
    std::cout << box::VERTICAL;
    for (int i = 0; i < padding; ++i) std::cout << " ";
    std::cout << color::BOLD << color::WHITE << title << color::CYAN;
    for (int i = 0; i < width - static_cast<int>(title.length()) - padding - 2; ++i) std::cout << " ";
    std::cout << box::VERTICAL << "\n";
    
    std::cout << box::BOTTOM_LEFT;
    for (int i = 0; i < width - 2; ++i) std::cout << box::HORIZONTAL;
    std::cout << box::BOTTOM_RIGHT;
    std::cout << color::RESET << "\n" << std::endl;
}

/**
 * Status Messages
 */
inline void success(const std::string& msg) {
    std::cout << color::GREEN << symbol::CHECKMARK << " " 
              << color::RESET << msg << std::endl;
}

inline void error(const std::string& msg) {
    std::cout << color::RED << symbol::CROSS << " " 
              << color::RESET << msg << std::endl;
}

inline void info(const std::string& msg) {
    std::cout << color::BLUE << "ℹ " 
              << color::RESET << msg << std::endl;
}

inline void warning(const std::string& msg) {
    std::cout << color::YELLOW << "⚠ " 
              << color::RESET << msg << std::endl;
}

/**
 * Table Helper
 */
class Table {
public:
    Table(const std::vector<std::string>& headers, int col_width = 15)
        : headers_(headers), col_width_(col_width) {
        print_border(true);
        print_row(headers_, true);
        print_border(false);
    }
    
    void add_row(const std::vector<std::string>& row) {
        print_row(row, false);
    }
    
    void finish() {
        print_border(true);
    }
    
private:
    void print_border(bool thick) {
        std::cout << color::CYAN << box::VERTICAL;
        for (size_t i = 0; i < headers_.size(); ++i) {
            for (int j = 0; j < col_width_; ++j) {
                std::cout << (thick ? box::HORIZONTAL : box::HORIZONTAL);
            }
            if (i < headers_.size() - 1) {
                std::cout << box::T_DOWN;
            }
        }
        std::cout << box::VERTICAL << color::RESET << "\n";
    }
    
    void print_row(const std::vector<std::string>& row, bool header) {
        std::cout << color::CYAN << box::VERTICAL;
        for (size_t i = 0; i < row.size(); ++i) {
            if (header) std::cout << color::BOLD << color::WHITE;
            std::cout << std::setw(col_width_) << std::left << row[i];
            std::cout << color::RESET << color::CYAN << box::VERTICAL;
        }
        std::cout << color::RESET << "\n";
    }
    
    std::vector<std::string> headers_;
    int col_width_;
};

/**
 * Simple Timer for displaying elapsed time
 */
class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}
    
    std::string elapsed() const {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_).count();
        
        int hours = duration / 3600;
        int minutes = (duration % 3600) / 60;
        int seconds = duration % 60;
        
        std::ostringstream oss;
        if (hours > 0) {
            oss << hours << "h " << minutes << "m " << seconds << "s";
        } else if (minutes > 0) {
            oss << minutes << "m " << seconds << "s";
        } else {
            oss << seconds << "s";
        }
        return oss.str();
    }
    
private:
    std::chrono::steady_clock::time_point start_;
};

} // namespace ui
} // namespace mm_rec
