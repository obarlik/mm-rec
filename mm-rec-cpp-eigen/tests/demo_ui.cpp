#include "mm_rec/utils/ui.h"
#include "mm_rec/utils/logger.h"
#include <thread>
#include <chrono>

using namespace mm_rec;
using namespace mm_rec::ui;

int main() {
    // Start logger
    Logger::instance().start_writer("ui_demo.log", LogLevel::INFO);
    
    // Header
    print_header("MM-Rec UI Components Demo", 60);
    
    // Status messages
    success("Logger initialized successfully");
    info("Demonstrating UI components");
    warning("This is just a demo");
    
    std::cout << "\n";
    
    // Progress bar demo
    LOG_UI("1. Progress Bar Demo:");
    ProgressBar bar(100, 40, "Processing: ");
    for (int i = 0; i <= 100; i += 5) {
        bar.update(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    bar.finish();
    
    std::cout << "\n";
    
    // Spinner demo
    LOG_UI("2. Spinner Demo:");
    Spinner spinner("Loading model");
    for (int i = 0; i < 20; ++i) {
        spinner.tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    spinner.finish("Model loaded!");
    
    std::cout << "\n";
    
    // Table demo
    LOG_UI("3. Table Demo:");
    std::vector<std::string> headers = {"Metric", "Value", "Unit"};
    Table table(headers, 15);
    table.add_row(std::vector<std::string>{"Loss", "0.234", "CE"});
    table.add_row(std::vector<std::string>{"Accuracy", "94.2", "%"});
    table.add_row(std::vector<std::string>{"Speed", "1250", "tok/s"});
    table.finish();
    
    std::cout << "\n";
    
    // Colors demo
    LOG_UI("4. Colors Demo:");
    std::cout << color::RED << "  ● Red" << color::RESET << std::endl;
    std::cout << color::GREEN << "  ● Green" << color::RESET << std::endl;
    std::cout << color::BLUE << "  ● Blue" << color::RESET << std::endl;
    std::cout << color::YELLOW << "  ● Yellow" << color::RESET << std::endl;
    std::cout << color::CYAN << "  ● Cyan" << color::RESET << std::endl;
    std::cout << color::MAGENTA << "  ● Magenta" << color::RESET << std::endl;
    
    std::cout << "\n";
    
    // Timer demo
    LOG_UI("5. Timer Demo:");
    Timer timer;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << "  Elapsed: " << timer.elapsed() << std::endl;
    
    std::cout << "\n";
    success("Demo complete!");
    
    Logger::instance().stop_writer();
    return 0;
}
