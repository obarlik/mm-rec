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
