#!/bin/bash

# Daemonize MM-Rec Server
# Properly starts server in background without creating zombie processes

set -e

COLOR_CYAN='\033[0;36m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_ROOT/mm_rec"
PID_FILE="/tmp/mm_rec_server.pid"
LOG_FILE="/tmp/mm_rec_server.log"

echo -e "${COLOR_CYAN}========================================${COLOR_RESET}"
echo -e "${COLOR_CYAN}  MM-REC Server Daemonizer${COLOR_RESET}"
echo -e "${COLOR_CYAN}========================================${COLOR_RESET}"
echo ""

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo -e "${COLOR_YELLOW}⚠${COLOR_RESET}  Server already running (PID: $OLD_PID)"
        echo -e "${COLOR_CYAN}→${COLOR_RESET} Use:scripts/stop_server.sh to stop it first"
        exit 1
    else
        echo -e "${COLOR_YELLOW}⚠${COLOR_RESET}  Stale PID file found, cleaning up..."
        rm -f "$PID_FILE"
    fi
fi

# Check binary exists
if [ ! -f "$BINARY" ]; then
    echo -e "${COLOR_RED}✗${COLOR_RESET} Binary not found: $BINARY"
    echo -e "${COLOR_YELLOW}→${COLOR_RESET} Build first: cd mm-rec-cpp-eigen && make mm_rec"
    exit 1
fi

# Cleanup old processes
echo -e "${COLOR_CYAN}[1/3]${COLOR_RESET} Cleaning up old processes..."
bash "$SCRIPT_DIR/cleanup_silent.sh" 2>/dev/null || true

# Start server as proper daemon
echo -e "${COLOR_CYAN}[2/3]${COLOR_RESET} Starting server in daemon mode..."

#  Use setsid to completely detach from terminal
# This prevents zombie processes by making init (PID 1) the parent
setsid "$BINARY" server --daemon > "$LOG_FILE" 2>&1 &

# Get the PID
SERVER_PID=$!

# Save PID
echo $SERVER_PID > "$PID_FILE"

# Wait a bit to check if it started successfully
sleep 2

if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo -e "${COLOR_GREEN}✓${COLOR_RESET} Server started successfully"
    echo -e "  ${COLOR_CYAN}→${COLOR_RESET} PID: $SERVER_PID"
    echo -e "  ${COLOR_CYAN}→${COLOR_RESET} Dashboard: http://localhost:8085"
    echo -e "  ${COLOR_CYAN}→${COLOR_RESET} Logs: $LOG_FILE"
    echo -e "  ${COLOR_CYAN}→${COLOR_RESET} Stop:scripts/stop_server.sh"
else
    echo -e "${COLOR_RED}✗${COLOR_RESET} Server failed to start"
    echo -e "${COLOR_YELLOW}→${COLOR_RESET} Check logs: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

echo ""
echo -e "${COLOR_CYAN}[3/3]${COLOR_RESET} Verification..."
sleep 1

# Test dashboard
if curl -s http://localhost:8085/api/stats > /dev/null 2>&1; then
    echo -e "${COLOR_GREEN}✓${COLOR_RESET} Dashboard is responding"
else
    echo -e "${COLOR_YELLOW}⚠${COLOR_RESET}  Dashboard may still be initializing..."
fi

echo ""
echo -e "${COLOR_GREEN}========================================${COLOR_RESET}"
echo -e "${COLOR_GREEN}  Server Running!${COLOR_RESET}"
echo -e "${COLOR_GREEN}========================================${COLOR_RESET}"
