/**
 * Parse Metrics Command - Convert binary metrics to human-readable format
 */

#include "mm_rec/utils/metrics.h"
#include "commands.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace mm_rec;

int cmd_parse_metrics(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: mm_rec parse-metrics <metrics_file.bin>" << std::endl;
        return 1;
    }

    std::string metrics_path = argv[1];
    
    // Open binary file
    std::ifstream ifs(metrics_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "❌ Failed to open: " << metrics_path << std::endl;
        return 1;
    }
    
    // Read header: [MAGIC:4][VERSION:4][RESERVED:4]
    char magic[4];
    uint32_t version, reserved;
    
    ifs.read(magic, 4);
    ifs.read(reinterpret_cast<char*>(&version), 4);
    ifs.read(reinterpret_cast<char*>(&reserved), 4);
    
    // Validate magic
    if (magic[0] != 'M' || magic[1] != 'M' || magic[2] != 'R' || magic[3] != 'C') {
        std::cerr << "❌ Invalid magic number. Not a valid metrics file." << std::endl;
        return 1;
    }
    
    std::cout << "📊 Metrics File: " << metrics_path << std::endl;
    std::cout << "   Version: " << version << std::endl;
    std::cout << std::endl;
    
    // Read events
    std::vector<MetricEvent> events;
    MetricEvent event;
    
    while (ifs.read(reinterpret_cast<char*>(&event), sizeof(MetricEvent))) {
        events.push_back(event);
    }
    
    std::cout << "Total Events: " << events.size() << std::endl;
    std::cout << std::endl;
    
    // Print table
    std::cout << std::setw(10) << "Type" 
              << std::setw(18) << "Timestamp (μs)" 
              << std::setw(12) << "Value1" 
              << std::setw(12) << "Value2" 
              << std::setw(10) << "Extra" 
              << std::setw(10) << "Label" 
              << std::endl;
    std::cout << std::string(72, '-') << std::endl;
    
    const char* type_names[] = {
        "TRAIN", "INFER", "FORWARD", "BACKWARD", 
        "OPTIM", "CKPT", "MEMORY", "BRAKE", "CUSTOM"
    };
    
    for (const auto& e : events) {
        int type_idx = static_cast<int>(e.type);
        const char* type_str = (type_idx >= 0 && type_idx < 9) ? type_names[type_idx] : "UNKNOWN";
        
        std::cout << std::setw(10) << type_str
                  << std::setw(18) << e.timestamp_us
                  << std::setw(12) << std::fixed << std::setprecision(4) << e.value1
                  << std::setw(12) << std::fixed << std::setprecision(4) << e.value2
                  << std::setw(10) << e.extra
                  << std::setw(10) << e.label
                  << std::endl;
    }
    
    return 0;
}
