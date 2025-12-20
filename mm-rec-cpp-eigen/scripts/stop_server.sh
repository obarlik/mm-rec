#!/bin/bash

# Stop MM-Rec Server Daemon

COLOR_CYAN='\033[0;36m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_RESET='\033[0m'

PID_FILE="/tmp/mm_rec_server.pid"

echo -e "${COLOR_CYAN}Stopping MM-Rec Server...${COLOR_RESET}"

if [ ! -f "$PID_FILE" ]; then
    echo -e "${COLOR_YELLOW}⚠${COLOR_RESET}  No PID file found"
    echo -e "${COLOR_CYAN}→${COLOR_RESET} Running cleanup anyway..."
    bash "$(dirname "$0")/cleanup_processes.sh"
    exit 0
fi

SERVER_PID=$(cat "$PID_FILE")

if ! ps -p $SERVER_PID > /dev/null 2>&1; then
    echo -e "${COLOR_YELLOW}⚠${COLOR_RESET}  Server not running (PID: $SERVER_PID)"
    rm -f "$PID_FILE"
    exit 0
fi

echo -e "${COLOR_CYAN}→${COLOR_RESET} Sending SIGTERM to PID $SERVER_PID..."
kill -15 $SERVER_PID 2>/dev/null || true

# Wait for graceful shutdown
for i in {1..5}; do
    if ! ps -p $SERVER_PID > /dev/null 2>&1; then
        echo -e "${COLOR_GREEN}✓${COLOR_RESET} Server stopped gracefully"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still alive
echo -e "${COLOR_YELLOW}⚠${COLOR_RESET}  Force killing..."
kill -9 $SERVER_PID 2>/dev/null || true
rm -f "$PID_FILE"

echo -e "${COLOR_GREEN}✓${COLOR_RESET} Server stopped"
