#!/bin/bash
# Server Restart Script
# Kills old server, pulls code, and starts new server

echo "🔄 Restarting server..."

# Get server directory
SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SERVER_DIR"

# Kill old server
echo "⏹️  Stopping old server..."
pkill -f "train_server.py" || true
sleep 2

# Pull latest code
echo "📥 Pulling latest code..."
git pull

# Build C++ extensions if needed
echo "🔨 Building C++ extensions..."
if [ -f "mm_rec/cpp/setup.py" ]; then
    cd mm_rec/cpp
    if python setup.py build_ext --inplace 2>/dev/null; then
        echo "  ✅ C++ extensions built"
    else
        echo "  ⚠️  C++ build failed, using Python fallback"
    fi
    cd ../..
fi

# Activate environment and restart
echo "🚀 Starting new server..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif command -v conda &> /dev/null; then
    conda activate mm-rec
fi

# Start server in background
nohup python server/train_server.py > server.log 2>&1 &

echo "✅ Server restarted!"
echo "📝 Logs: tail -f server.log"
