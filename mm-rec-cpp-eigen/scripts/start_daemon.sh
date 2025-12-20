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

# Parse port from args for verification
PORT=8085
for i in "$@"; do
    if [[ "$i" == "--port"* ]]; then
        # Handle --port=1234 format if used, or assume next arg (simplified)
        continue
    fi
     # Simple check: if prev arg was --port, this is the port
    if [[ "$PREV" == "--port" ]]; then
        PORT="$i"
    fi
    PREV="$i"
done

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

# Remove old PID file if exists (cleanup handled above, extra safety)
rm -f "$PID_FILE"

# Start the binary
# We don't need 'nohup' or '&' because the binary self-daemonizes (double-fork)
# But we suppress output to avoid clutter
# Pass any extra arguments to the binary (like --port, --threads)
"$BINARY" server --daemon "$@" > /dev/null 2>&1

# Wait for PID file to be created by the daemon
echo -n "Waiting for daemon..."
for i in {1..50}; do
    if [ -f "$PID_FILE" ]; then
        break
    fi
    echo -n "."
    sleep 0.1
done
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo -e "${COLOR_RED}✗${COLOR_RESET} Timeout waiting for PID file"
    exit 1
fi

SERVER_PID=$(cat "$PID_FILE")

if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo -e "${COLOR_GREEN}✓${COLOR_RESET} Server started successfully"
    echo -e "  ${COLOR_CYAN}→${COLOR_RESET} PID: $SERVER_PID"
    echo -e "  ${COLOR_CYAN}→${COLOR_RESET} Dashboard: http://localhost:$PORT"
    echo -e "  ${COLOR_CYAN}→${COLOR_RESET} Logs: $LOG_FILE"
    echo -e "  ${COLOR_CYAN}→${COLOR_RESET} Stop: scripts/stop_server.sh"
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
if curl -s http://localhost:$PORT/api/stats > /dev/null 2>&1; then
    echo -e "${COLOR_GREEN}✓${COLOR_RESET} Dashboard is responding"
else
    echo -e "${COLOR_YELLOW}⚠${COLOR_RESET}  Dashboard may still be initializing..."
fi

echo ""
echo -e "${COLOR_GREEN}========================================${COLOR_RESET}"
echo -e "${COLOR_GREEN}  Server Running!${COLOR_RESET}"
echo -e "${COLOR_GREEN}========================================${COLOR_RESET}"
