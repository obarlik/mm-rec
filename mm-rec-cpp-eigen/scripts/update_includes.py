import os

# Define the replacements map
replacements = {
    # Infrastructure
    "mm_rec/utils/logger.h": "mm_rec/infrastructure/logger.h",
    "mm_rec/utils/config.h": "mm_rec/infrastructure/config.h",
    "mm_rec/utils/http_server.h": "mm_rec/infrastructure/http_server.h",
    "mm_rec/utils/di_container.h": "mm_rec/infrastructure/di_container.h",
    "mm_rec/utils/thread_pool.h": "mm_rec/infrastructure/thread_pool.h",
    "mm_rec/utils/event_bus.h": "mm_rec/infrastructure/event_bus.h",
    "mm_rec/utils/connection_manager.h": "mm_rec/infrastructure/connection_manager.h",
    "mm_rec/utils/traffic_manager.h": "mm_rec/infrastructure/traffic_manager.h",
    "mm_rec/utils/request_context.h": "mm_rec/infrastructure/request_context.h",
    "mm_rec/utils/response.h": "mm_rec/infrastructure/response.h",
    "mm_rec/utils/system_optimizer.h": "mm_rec/infrastructure/system_optimizer.h",
    "mm_rec/utils/middlewares.h": "mm_rec/infrastructure/middlewares.h",
    "mm_rec/utils/result.h": "mm_rec/infrastructure/result.h",
    "mm_rec/utils/service_layers.h": "mm_rec/infrastructure/service_layers.h",
    "mm_rec/utils/error_handling.h": "mm_rec/infrastructure/error_handling.h",

    # Business
    "mm_rec/utils/diagnostic_manager.h": "mm_rec/business/diagnostic_manager.h",
    "mm_rec/utils/alert_manager.h": "mm_rec/business/alert_manager.h",
    "mm_rec/utils/metrics.h": "mm_rec/business/metrics.h",
    "mm_rec/utils/checkpoint.h": "mm_rec/business/checkpoint.h",

    # Application
    "mm_rec/utils/service_configurator.h": "mm_rec/application/service_configurator.h",
    "mm_rec/utils/dashboard_manager.h": "mm_rec/application/dashboard_manager.h",
    "mm_rec/utils/diagnostic_dashboard.h": "mm_rec/application/diagnostic_dashboard.h",
    "mm_rec/utils/run_manager.h": "mm_rec/application/run_manager.h",
    "mm_rec/utils/session.h": "mm_rec/application/session.h",
    "mm_rec/utils/dashboard_html.h": "mm_rec/application/dashboard_html.h",
}

# Directories to search
target_dirs = ["include", "src", "tests"]

def process_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        original_content = content
        
        for old, new in replacements.items():
            content = content.replace(old, new)
            
        if content != original_content:
            print(f"Updating {filepath}")
            with open(filepath, 'w') as f:
                f.write(content)
                
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    root_dir = os.getcwd()
    for target_dir in target_dirs:
        dir_path = os.path.join(root_dir, target_dir)
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith((".h", ".cpp", ".hpp", ".cc")):
                    process_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
