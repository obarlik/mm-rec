/**
 * Parse Metrics Command - Convert binary metrics to human-readable format
 */

#include "mm_rec/utils/metrics.h"
#include "commands.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <cmath>
#include <numeric> // For std::accumulate
#include <algorithm> // For std::min/max
#include <sstream> // For std::stringstream

#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/ui.h"

using namespace mm_rec;
using namespace mm_rec::ui;

// Define MMRC_MAGIC for validation
const char MMRC_MAGIC[4] = {'M', 'M', 'R', 'C'};

// Helper struct to accumulate statistics for each metric type
struct MetricStats {
    long count = 0;
    double sum = 0.0;
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::lowest();

    void add_value(double val) {
        count++;
        sum += val;
        min = std::min(min, val);
        max = std::max(max, val);
    }

    double avg() const {
        return count > 0 ? sum / count : 0.0;
    }
};

// Chrome Tracing JSON Writer
void export_trace_json(const std::vector<MetricEvent>& events, const std::string& out_path) {
    std::ofstream ofs(out_path);
    if (!ofs) {
        ui::error("Failed to create trace file: " + out_path);
        return;
    }
    
    ofs << "[\n";
    
    bool first = true;
    const char* type_names[] = {
        "TRAIN", "INFER", "FORWARD", "BACKWARD", 
        "OPTIM", "CKPT", "MEMORY", "BRAKE", "CUSTOM"
    };

    for (const auto& e : events) {
        if (!first) ofs << ",\n";
        first = false;
        
        int type_idx = static_cast<int>(e.type);
        std::string name = (type_idx >= 0 && type_idx < 9) ? type_names[type_idx] : "UNKNOWN";
        
        // Manual JSON construction for speed (no dependency)
        // ts: timestamp in microseconds
        // dur: duration in microseconds (value1 is ms, so * 1000)
        // ph: "X" (Complete Event) for duration, "i" (Instant) for point events
        
        ofs << "  {";
        ofs << "\"name\": \"" << name << "\", ";
        ofs << "\"cat\": \"mm_rec\", ";
        ofs << "\"ph\": \"X\", "; // Assume all are timed events for now
        ofs << "\"ts\": " << e.timestamp_us << ", ";
        ofs << "\"dur\": " << (e.value1 * 1000.0) << ", "; // value1 is ms
        ofs << "\"pid\": 1, ";
        ofs << "\"tid\": 1, ";
        ofs << "\"args\": { \"label\": \"" << e.label << "\", \"val2\": " << e.value2 << " }";
        ofs << "}";
    }
    
    ofs << "\n]\n";
    ui::success("Trace exported to: " + out_path);
    ui::info("Open in browser: https://ui.perfetto.dev/ or chrome://tracing");
}

int cmd_parse_metrics(int argc, char* argv[]) {
    if (argc < 2) {
        ui::error("Usage: mm_rec parse-metrics <metrics_file.bin> [--export-trace <out.json>]");
        return 1;
    }
    
    std::string metrics_path = argv[1];
    std::string trace_out_path = "";
    
    if (argc >= 4 && std::string(argv[2]) == "--export-trace") {
        trace_out_path = argv[3];
    }
    
    // Open binary file
    std::ifstream ifs(metrics_path, std::ios::binary);
    if (!ifs) {
        ui::error("Failed to open: " + metrics_path);
        return 1;
    }
    
    ui::info("Parsing: " + metrics_path);
    // Read header: [MAGIC:4][VERSION:4][RESERVED:4]
    char magic[4];
    uint32_t version, reserved;
    
    ifs.read(magic, 4);
    ifs.read(reinterpret_cast<char*>(&version), 4);
    ifs.read(reinterpret_cast<char*>(&reserved), 4);
    
    // Validate magic
    if (std::memcmp(magic, MMRC_MAGIC, 4) != 0) {
        ui::error("Invalid magic number. Not a valid metrics file.");
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
    
    // Aggregate statistics
    std::map<std::string, MetricStats> stats;
    const char* type_names[] = {
        "TRAIN", "INFER", "FORWARD", "BACKWARD", 
        "OPTIM", "CKPT", "MEMORY", "BRAKE", "CUSTOM"
    };

    for (const auto& e : events) {
        int type_idx = static_cast<int>(e.type);
        const char* type_str = (type_idx >= 0 && type_idx < 9) ? type_names[type_idx] : "UNKNOWN";
        stats[type_str].add_value(e.value1); // Assuming value1 is the primary metric
    }

    ui::print_header("Performance Summary", 50);
    
    Table table({"Metric Type", "Count", "Avg", "Min", "Max"}, 12);
    
    for (const auto& pair : stats) {
        const std::string& name = pair.first;
        const MetricStats& s = pair.second;
        
        std::stringstream ss_avg; ss_avg << std::fixed << std::setprecision(4) << s.avg();
        std::stringstream ss_min; ss_min << std::fixed << std::setprecision(4) << s.min;
        std::stringstream ss_max; ss_max << std::fixed << std::setprecision(4) << s.max;
        
        table.add_row({
            name,
            std::to_string(s.count),
            ss_avg.str(),
            ss_min.str(),
            ss_max.str()
        });
    }
    table.finish();
    
    table.finish();
    
    if (!trace_out_path.empty()) {
        ui::print_header("Trace Export");
        export_trace_json(events, trace_out_path);
    }
    
    Logger::instance().stop_writer();
    return 0;
}

