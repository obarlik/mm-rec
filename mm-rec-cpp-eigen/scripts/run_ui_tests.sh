#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
UI_TEST_DIR="$ROOT_DIR/tests/ui"
VENV_DIR="$UI_TEST_DIR/venv"

echo "========================================"
echo "   Running Selenium UI Tests"
echo "========================================"

# Check for Node
if ! command -v npm &> /dev/null; then
    echo "Error: npm is required but not installed."
    exit 1
fi

cd "$UI_TEST_DIR"

# Install Dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing Node dependencies..."
    npm install
fi

# Run Tests
echo "Starting Mocha..."
npm test

echo "========================================"
echo "   Tests Completed"
echo "========================================"
