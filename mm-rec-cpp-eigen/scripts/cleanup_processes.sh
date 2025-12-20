#!/bin/bash

# mm_rec Process Cleanup Script - ENHANCED VERSION
# Aggressively kills all running mm_rec processes and frees up ports
# Uses multiple strategies to ensure complete cleanup

COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_MAGENTA='\033[0;35m'
COLOR_RESET='\033[0m'

echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
echo -e "${COLOR_BLUE}  MM-REC Process Cleanup Utility${COLOR_RESET}"
echo -e "${COLOR_BLUE}  [ENHANCED - Multi-Strategy Kill]${COLOR_RESET}"
echo -e "${COLOR_BLUE}========================================${COLOR_RESET}"
echo ""

KILLED_ANY=false

# Function to kill processes with escalating force
kill_with_escalation() {
    local pattern=$1
    local name=$2
    
    # Find PIDs
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    
    if [ -z "$pids" ]; then
        echo -e "${COLOR_GREEN}✓${COLOR_RESET} No $name processes found"
        return 0
    fi
    
    echo -e "${COLOR_YELLOW}⚠${COLOR_RESET}  Found $name processes (PIDs: $pids)"
    KILLED_ANY=true
    
    # Strategy 1: Polite SIGTERM first
    echo -e "  ${COLOR_MAGENTA}→${COLOR_RESET} Attempting graceful shutdown (SIGTERM)..."
    for pid in $pids; do
        kill -15 $pid 2>/dev/null || true
    done
    sleep 0.5
    
    # Check if still alive
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        echo -e "  ${COLOR_GREEN}✓${COLOR_RESET} Gracefully terminated"
        return 0
    fi
    
    # Strategy 2: SIGKILL (kill -9)
    echo -e "  ${COLOR_YELLOW}→${COLOR_RESET} Force killing (SIGKILL)..."
    for pid in $pids; do
        kill -9 $pid 2>/dev/null || true
    done
    sleep 0.3
    
    # Check again
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        echo -e "  ${COLOR_GREEN}✓${COLOR_RESET} Force killed successfully"
        return 0
    fi
    
    # Strategy 3: Kill entire process tree
    echo -e "  ${COLOR_YELLOW}→${COLOR_RESET} Killing process tree..."
    for pid in $pids; do
        pkill -9 -P $pid 2>/dev/null || true  # Kill children first
        kill -9 $pid 2>/dev/null || true      # Then parent
    done
    sleep 0.3
    
    # Final check
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        echo -e "  ${COLOR_GREEN}✓${COLOR_RESET} Process tree killed"
        return 0
    fi
    
    echo -e "  ${COLOR_RED}✗${COLOR_RESET} WARNING: Some processes may still be alive: $pids"
}

# Function to kill by exact binary name (more aggressive)
kill_by_binary() {
    local binary_name=$1
    
    # Using killall for exact binary name matching
    if killall -9 "$binary_name" 2>/dev/null; then
        echo -e "${COLOR_GREEN}✓${COLOR_RESET} Killed all $binary_name instances"
        KILLED_ANY=true
    fi
}

# Function to kill process using a specific port (enhanced)
kill_by_port() {
    local port=$1
    
    # Find all PIDs using the port
    local pids=$(lsof -ti:$port 2>/dev/null || true)
    
    if [ -z "$pids" ]; then
        echo -e "${COLOR_GREEN}✓${COLOR_RESET} Port $port is free"
        return 0
    fi
    
    echo -e "${COLOR_YELLOW}⚠${COLOR_RESET}  Port $port in use (PIDs: $pids)"
    KILLED_ANY=true
    
    # Kill all processes using this port
    for pid in $pids; do
        echo -e "  ${COLOR_MAGENTA}→${COLOR_RESET} Killing process $pid using port $port..."
        kill -15 $pid 2>/dev/null || true
    done
    sleep 0.3
    
    # Force kill if still alive
    pids=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            kill -9 $pid 2>/dev/null || true
        done
        sleep 0.2
    fi
    
    # Verify port is free
    if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "  ${COLOR_GREEN}✓${COLOR_RESET} Port $port freed successfully"
    else
        echo -e "  ${COLOR_RED}✗${COLOR_RESET} Port $port still in use!"
    fi
}

echo "Phase 1: Killing processes by pattern..."
echo ""

# Kill by pattern (most comprehensive)
kill_with_escalation "demo_training_cpp" "demo_training_cpp"
kill_with_escalation "demo_inference_cpp" "demo_inference_cpp"
kill_with_escalation "mm_rec_cli" "mm_rec_cli"
kill_with_escalation "mm-rec-cpp-eigen" "mm-rec-cpp-eigen"

echo ""
echo "Phase 2: Killing by binary name..."
echo ""

# Kill by exact binary name
kill_by_binary "demo_training_cpp"
kill_by_binary "demo_inference_cpp"
kill_by_binary "mm_rec_cli"

echo ""
echo "Phase 3: Freeing ports..."
echo ""

# Clean up ports
kill_by_port 8085
kill_by_port 8080

echo ""
echo "Phase 4: Final verification..."
echo ""

# Check for any remaining processes (excluding language servers and this script)
remaining_pattern=$(pgrep -f "mm_rec|demo_training|demo_inference" 2>/dev/null | while read pid; do
    # Exclude language servers and this cleanup script
    ps -p $pid -o cmd= | grep -v "language_server" | grep -v "cleanup_processes" | grep -q . && echo $pid
done || true)

remaining_binary=$(pgrep "demo_training_cpp|demo_inference_cpp|mm_rec_cli" 2>/dev/null || true)
remaining="$remaining_pattern $remaining_binary"

if [ -n "$(echo $remaining | xargs)" ]; then
    echo -e "${COLOR_RED}========================================${COLOR_RESET}"
    echo -e "${COLOR_RED}  ⚠ WARNING: Zombie Processes Detected${COLOR_RESET}"
    echo -e "${COLOR_RED}========================================${COLOR_RESET}"
    echo ""
    echo "The following processes could not be killed:"
    ps aux | grep -E "mm_rec|demo_training|demo_inference" | grep -v grep | grep -v cleanup_processes | grep -v language_server || true
    echo ""
    echo -e "${COLOR_YELLOW}Possible solutions:${COLOR_RESET}"
    echo "1. Try running this script with sudo: sudo $0"
    echo "2. Reboot the system if processes are in uninterruptible sleep (D state)"
    echo "3. Check for I/O blocking: iotop -o"
else
    echo -e "${COLOR_GREEN}========================================${COLOR_RESET}"
    echo -e "${COLOR_GREEN}  ✓ All Clean!${COLOR_RESET}"
    echo -e "${COLOR_GREEN}========================================${COLOR_RESET}"
    
    if [ "$KILLED_ANY" = true ]; then
        echo -e "\n${COLOR_GREEN}✓${COLOR_RESET} All mm_rec processes terminated successfully"
    else
        echo -e "\n${COLOR_GREEN}✓${COLOR_RESET} No processes were running"
    fi
fi

# Always exit with success (cleanup is best-effort)
exit 0

