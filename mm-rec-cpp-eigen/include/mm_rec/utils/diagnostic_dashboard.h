// Dashboard API Endpoints for Diagnostics

#pragma once

#include "mm_rec/utils/diagnostic_manager.h"
#include "mm_rec/utils/alert_manager.h"
#include "mm_rec/utils/http_server.h"
#include <sstream>

using namespace mm_rec;
using namespace mm_rec::diagnostics;

// External symbols from embedded ops.html
extern "C" {
    extern const char _binary_ops_html_start[];
    extern const char _binary_ops_html_end[];
}

/**
 * Get embedded ops.html content.
 */
inline std::string get_ops_dashboard_html() {
    size_t size = _binary_ops_html_end - _binary_ops_html_start;
    return std::string(_binary_ops_html_start, size);
}

/**
 * Register diagnostic API endpoints with HTTP server.
 * 
 * Endpoints:
 *   GET  /ops                            - Operations dashboard UI
 *   GET  /api/diagnostics/stats          - Get statistics
 *   GET  /api/diagnostics/errors         - List recent errors  
 *   GET  /api/diagnostics/errors/:id     - Get specific error trace
 *   POST /api/diagnostics/config         - Update config (enable/disable)
 *   POST /api/diagnostics/clear          - Clear error history
 *   GET  /api/alerts/active              - Get active alerts
 *   GET  /api/performance/metrics        - Get performance metrics
 *   GET  /api/logs/recent                - Get recent logs
 */
inline void register_diagnostic_endpoints(HttpServer* server, 
                                          DiagnosticManager* diag_mgr,
                                          AlertManager* alert_mgr) {
    // ========================================
    // GET /ops - Operations Dashboard UI
    // ========================================
    server->get("/ops", [](const auto& req, auto res) {
        std::string html = get_ops_dashboard_html();
        res->status(200)
           ->header("Content-Type", "text/html; charset=utf-8")
           ->send(html);
    });
    
    // ========================================
    // GET /api/diagnostics/stats
    // ========================================
    server->get("/api/diagnostics/stats", [&diag_mgr](const auto& req, auto res) {
        std::string json = diag_mgr.statistics_to_json();
        res->status(200)
           ->header("Content-Type", "application/json")
           ->send(json);
    });
    
    // ========================================
    // GET /api/diagnostics/errors?limit=50
    // ========================================
    server->get("/api/diagnostics/errors", [&diag_mgr](const auto& req, auto res) {
        // Parse limit from query params
        size_t limit = 50;
        // TODO: Extract from req.query("limit")
        
        auto errors = diag_mgr.get_recent_errors(limit);
        
        std::ostringstream oss;
        oss << "{\n  \"errors\": [\n";
        
        for (size_t i = 0; i < errors.size(); i++) {
            oss << "    " << diag_mgr.error_trace_to_json(errors[i]);
            if (i < errors.size() - 1) oss << ",";
            oss << "\n";
        }
        
        oss << "  ],\n";
        oss << "  \"count\": " << errors.size() << "\n";
        oss << "}";
        
        res->status(200)
           ->header("Content-Type", "application/json")
           ->send(oss.str());
    });
    
    // ========================================
    // GET /api/diagnostics/errors/:correlation_id
    // ========================================
    server->get("/api/diagnostics/errors/:id", [&diag_mgr](const auto& req, auto res) {
        std::string corr_id = req.params.at("id");
        
        auto trace = diag_mgr.get_error_by_correlation_id(corr_id);
        
        if (trace) {
            std::string json = diag_mgr.error_trace_to_json(*trace);
            res->status(200)
               ->header("Content-Type", "application/json")
               ->send(json);
        } else {
            res->status(404)
               ->send(R"({"error": "Trace not found"})");
        }
    });
    
    // ========================================
    // POST /api/diagnostics/config
    // Body: {"tracing_enabled": true/false}
    // ========================================
    server->post("/api/diagnostics/config", [&diag_mgr](const auto& req, auto res) {
        // TODO: Parse JSON body
        // For now, simple toggle
        bool current = diag_mgr.is_tracing_enabled();
        diag_mgr.set_tracing_enabled(!current);
        
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"tracing_enabled\": " << (diag_mgr.is_tracing_enabled() ? "true" : "false") << ",\n";
        oss << "  \"message\": \"Tracing " << (diag_mgr.is_tracing_enabled() ? "enabled" : "disabled") << "\"\n";
        oss << "}";
        
        res->status(200)
           ->header("Content-Type", "application/json")
           ->send(oss.str());
    });
    
    // ========================================
    // POST /api/diagnostics/clear
    // ========================================
    server->post("/api/diagnostics/clear", [&diag_mgr](const auto& req, auto res) {
        diag_mgr.clear_error_traces();
        
        res->status(200)
           ->send(R"({"message": "Error traces cleared"})");
    });
    
    // ========================================
    // GET /api/alerts/active
    // ========================================
    server->get("/api/alerts/active", [&alert_mgr](const auto& req, auto res) {
        auto alerts = alert_mgr.get_active_alerts();
        
        std::ostringstream oss;
        oss << "{\n  \"alerts\": [\n";
        
        for (size_t i = 0; i < alerts.size(); i++) {
            const auto& alert = alerts[i];
            
            std::string level_str;
            switch (alert.level) {
                case AlertLevel::INFO: level_str = "INFO"; break;
                case AlertLevel::WARNING: level_str = "WARNING"; break;
                case AlertLevel::ERROR: level_str = "ERROR"; break;
                case AlertLevel::CRITICAL: level_str = "CRITICAL"; break;
            }
            
            oss << "    {\n";
            oss << "      \"level\": \"" << level_str << "\",\n";
            oss << "      \"title\": \"" << alert.title << "\",\n";
            oss << "      \"message\": \"" << alert.message << "\",\n";
            oss << "      \"timestamp\": " << std::chrono::system_clock::to_time_t(alert.timestamp) << ",\n";
            oss << "      \"correlation_id\": \"" << alert.correlation_id << "\",\n";
            oss << "      \"acknowledged\": " << (alert.acknowledged ? "true" : "false") << "\n";
            oss << "    }";
            
            if (i < alerts.size() - 1) oss << ",";
            oss << "\n";
        }
        
        oss << "  ],\n";
        oss << "  \"count\": " << alerts.size() << "\n";
        oss << "}";
        
        res->status(200)
           ->header("Content-Type", "application/json")
           ->send(oss.str());
    });
    
    // ========================================
    // GET /api/performance/metrics
    // ========================================
    server->get("/api/performance/metrics", [&alert_mgr](const auto& req, auto res) {
        auto metrics = alert_mgr.get_performance_metrics();
        
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"avg_response_time\": " << metrics.avg_response_time << ",\n";
        oss << "  \"p50_response_time\": " << metrics.p50_response_time << ",\n";
        oss << "  \"p95_response_time\": " << metrics.p95_response_time << ",\n";
        oss << "  \"p99_response_time\": " << metrics.p99_response_time << ",\n";
        oss << "  \"max_response_time\": " << metrics.max_response_time << ",\n";
        oss << "  \"slow_requests_count\": " << metrics.slow_requests_count << ",\n";
        oss << "  \"requests_per_second\": " << metrics.requests_per_second << "\n";
        oss << "}";
        
        res->status(200)
           ->header("Content-Type", "application/json")
           ->send(oss.str());
    });
    
    // ========================================
    // GET /api/logs/recent?limit=100
    // ========================================
    server->get("/api/logs/recent", [&alert_mgr](const auto& req, auto res) {
        // TODO: Parse limit from query
        size_t limit = 100;
        
        auto logs = alert_mgr.get_recent_logs(limit);
        
        std::ostringstream oss;
        oss << "{\n  \"logs\": [\n";
        
        for (size_t i = 0; i < logs.size(); i++) {
            const auto& log = logs[i];
            
            oss << "    {\n";
            oss << "      \"level\": \"" << log.level << "\",\n";
            oss << "      \"component\": \"" << log.component << "\",\n";
            oss << "      \"message\": \"" << log.message << "\",\n";
            oss << "      \"timestamp\": " << std::chrono::system_clock::to_time_t(log.timestamp) << ",\n";
            oss << "      \"correlation_id\": \"" << log.correlation_id << "\"\n";
            oss << "    }";
            
            if (i < logs.size() - 1) oss << ",";
            oss << "\n";
        }
        
        oss << "  ],\n";
        oss << "  \"count\": " << logs.size() << "\n";
        oss << "}";
        
        res->status(200)
           ->header("Content-Type", "application/json")
           ->send(oss.str());
    });
}
