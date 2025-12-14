#!/bin/bash
# Manual Start Script for Phoenix (Miniconda)
# Usage: ./phoenix_manual_start.sh

# Configuration
CONDA_ENV="mm-rec"
# Get the directory where the script is located
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GATEWAY_PORT=8090

echo "================================================"
echo "🚀 Phoenix Manual Start (Miniconda)"
echo "================================================"

# 1. Navigate to directory
cd "$PROJECT_DIR" || { echo "❌ Directory not found: $PROJECT_DIR"; exit 1; }

# 2. Activate Conda
# Ensure conda is initialized
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV || { echo "❌ Conda env '$CONDA_ENV' not found"; exit 1; }

echo "✅ Environment activated: $(which python)"

# 3. Install/Update Dependencies
echo "📦 Checking dependencies..."
pip install httpx>=0.25.0
pip install -r requirements.txt
pip install -r server/requirements.txt

# 4. Stop existing processes
echo "🛑 Stopping old processes..."
pkill -9 -f gateway.py || true
pkill -9 -f train_server.py || true
pkill -9 -f inference_server.py || true

# 5. Start Servers
echo "▶️  Starting Gateway (Manager) on port $GATEWAY_PORT..."
# Gateway manages Training (:8001) and Inference (:8002) subprocesses
nohup python server/gateway.py --port $GATEWAY_PORT > gateway.log 2>&1 &
PID_GATEWAY=$!
echo "   ✅ Gateway PID: $PID_GATEWAY"

echo ""
echo "📝 Logs:"
echo "   tail -f gateway.log"
echo "   tail -f server_internal.log"
echo "   tail -f inference_internal.log"
echo ""
echo "✅ Stack is UP."
echo "   Gateway Health: curl http://localhost:$GATEWAY_PORT/gateway/health"
