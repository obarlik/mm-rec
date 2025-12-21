#pragma once

#include "mm_rec/business/metric_types.h"
#include <string>

namespace mm_rec {
namespace infrastructure {

/**
 * Interface for Metrics Exporting Strategy
 * 
 * Allows decoupling the Application/Jobs from the specific
 * export implementation (File, Network, Console, etc.)
 */
class IMetricsExporter {
public:
    virtual ~IMetricsExporter() = default;

    /**
     * Start the export process.
     * @param output_path File path or resource identifier (connection string)
     * @param sampling Sampling configuration
     */
    virtual void start(const std::string& output_path, 
                       const MetricsSamplingConfig& sampling = MetricsSamplingConfig()) = 0;

    /**
     * Stop the export process and flush remaining data.
     */
    virtual void stop() = 0;
};

} // namespace infrastructure
} // namespace mm_rec
