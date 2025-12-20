#!/bin/bash

# Quick Restart - Cleanup and start in one command
# Perfect for rapid development iterations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COLOR_CYAN='\033[0;36m'
COLOR_RESET='\033[0m'

echo -e "${COLOR_CYAN}Quick Restart - Cleanup & Launch${COLOR_RESET}"
echo ""

# Run cleanup silently
bash "$SCRIPT_DIR/cleanup_processes.sh" > /dev/null 2>&1 || true

# Small delay to ensure ports are freed
sleep 0.5

# Start server
exec bash "$SCRIPT_DIR/start_server.sh"
