#!/bin/bash

# Build & Run Script with Auto-Cleanup
# Usage: ./build_and_run.sh [target_name]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/mm-rec-cpp-eigen/build"

COLOR_CYAN='\033[0;36m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RESET='\033[0m'

TARGET=${1:-mm_rec}

echo -e "${COLOR_CYAN}========================================${COLOR_RESET}"
echo -e "${COLOR_CYAN}  MM-REC Build & Run${COLOR_RESET}"
echo -e "${COLOR_CYAN}========================================${COLOR_RESET}"
echo ""

# Step 1: Cleanup
echo -e "${COLOR_YELLOW}[1/3]${COLOR_RESET} Cleaning up old processes..."
bash "$SCRIPT_DIR/cleanup_processes.sh" > /dev/null 2>&1 || true
echo -e "${COLOR_GREEN}✓${COLOR_RESET} Cleanup done"
echo ""

# Step 2: Build
echo -e "${COLOR_YELLOW}[2/3]${COLOR_RESET} Building project..."
cd "$PROJECT_ROOT/mm-rec-cpp-eigen"

if [ ! -d "build" ]; then
    mkdir -p build
    cd build
    cmake ..
else
    cd build
fi

make -j$(nproc) $TARGET

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${COLOR_GREEN}✓${COLOR_RESET} Build successful!"
else
    echo ""
    echo -e "${COLOR_RED}✗${COLOR_RESET} Build failed!"
    exit 1
fi
echo ""

# Step 3: Run
echo -e "${COLOR_YELLOW}[3/3]${COLOR_RESET} Running $TARGET..."
echo -e "${COLOR_CYAN}========================================${COLOR_RESET}"
echo ""

# Trap Ctrl+C to cleanup on exit
cleanup_on_exit() {
    echo ""
    echo -e "${COLOR_YELLOW}Cleaning up...${COLOR_RESET}"
    bash "$SCRIPT_DIR/cleanup_processes.sh" > /dev/null 2>&1 || true
    exit 0
}

trap cleanup_on_exit INT TERM

# Run the target
./$TARGET

# Cleanup after normal exit
cleanup_on_exit
