#!/bin/bash

# mm_rec Server Starter with Auto-Cleanup
# Ensures clean start by removing residual processes first

set -e

COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_CYAN='\033[0;36m'
COLOR_RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

echo -e "${COLOR_CYAN}========================================${COLOR_RESET}"
echo -e "${COLOR_CYAN}  MM-REC Server Launcher${COLOR_RESET}"
echo -e "${COLOR_CYAN}========================================${COLOR_RESET}"
echo ""

# Step 1: Cleanup existing processes
echo -e "${COLOR_BLUE}[1/4]${COLOR_RESET} Cleaning up existing processes..."
if [ -f "$SCRIPT_DIR/cleanup_processes.sh" ]; then
    bash "$SCRIPT_DIR/cleanup_processes.sh"
else
    echo -e "${COLOR_YELLOW}⚠${COLOR_RESET}  Cleanup script not found, skipping..."
fi

echo ""
sleep 1

# Step 2: Check build directory
echo -e "${COLOR_BLUE}[2/4]${COLOR_RESET} Checking build directory..."
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${COLOR_RED}✗${COLOR_RESET} Build directory not found: $BUILD_DIR"
    echo -e "${COLOR_YELLOW}→${COLOR_RESET} Please build the project first with: cd $PROJECT_ROOT && mkdir -p build && cd build && cmake .. && make -j$(nproc)"
    exit 1
fi

# Find the executable
DEMO_TRAINING="$BUILD_DIR/demo_training_cpp"
MM_REC_CLI="$BUILD_DIR/mm_rec_cli"

if [ -f "$MM_REC_CLI" ]; then
    EXECUTABLE="$MM_REC_CLI"
    EXEC_NAME="mm_rec_cli"
elif [ -f "$DEMO_TRAINING" ]; then
    EXECUTABLE="$DEMO_TRAINING"
    EXEC_NAME="demo_training_cpp"
else
    echo -e "${COLOR_RED}✗${COLOR_RESET} No executable found in $BUILD_DIR"
    echo -e "${COLOR_YELLOW}→${COLOR_RESET} Looking for: mm_rec_cli or demo_training_cpp"
    exit 1
fi

echo -e "${COLOR_GREEN}✓${COLOR_RESET} Found executable: $EXEC_NAME"
echo ""

# Step 3: Check port availability
echo -e "${COLOR_BLUE}[3/4]${COLOR_RESET} Checking port 8085..."
if lsof -Pi :8085 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${COLOR_RED}✗${COLOR_RESET} Port 8085 is still in use!"
    echo -e "${COLOR_YELLOW}→${COLOR_RESET} Run cleanup again or wait a moment..."
    exit 1
fi
echo -e "${COLOR_GREEN}✓${COLOR_RESET} Port 8085 is available"
echo ""

# Step 4: Start the server
echo -e "${COLOR_BLUE}[4/4]${COLOR_RESET} Starting server..."
echo -e "${COLOR_CYAN}========================================${COLOR_RESET}"
echo -e "${COLOR_GREEN}✓${COLOR_RESET} Launching $EXEC_NAME"
echo -e "${COLOR_GREEN}✓${COLOR_RESET} Dashboard will be available at: ${COLOR_CYAN}http://localhost:8085${COLOR_RESET}"
echo -e "${COLOR_CYAN}========================================${COLOR_RESET}"
echo ""
echo -e "${COLOR_YELLOW}Press Ctrl+C to stop the server${COLOR_RESET}"
echo ""

# Trap to ensure cleanup on exit
cleanup_on_exit() {
    echo ""
    echo -e "${COLOR_YELLOW}⚠${COLOR_RESET}  Shutting down server..."
    bash "$SCRIPT_DIR/cleanup_processes.sh" 2>/dev/null || true
    exit 0
}

trap cleanup_on_exit INT TERM

# Run the server
cd "$PROJECT_ROOT"
"$EXECUTABLE"
