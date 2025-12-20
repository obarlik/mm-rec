#!/bin/bash
# Silent wrapper for CMake integration
# Suppresses all output and always returns 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/cleanup_processes.sh" >/dev/null 2>&1
exit 0
