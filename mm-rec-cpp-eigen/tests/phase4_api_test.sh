#!/bin/bash

# Configuration
PORT=8085
BASE_URL="http://localhost:$PORT"
RUN_NAME="test_run_bash_v1"
CONFIG_FILE="test_bash.ini"
DATA_FILE="test_bash.bin"

COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_RESET='\033[0m'

fail() {
    echo -e "${COLOR_RED}[FAIL] $1${COLOR_RESET}"
    exit 1
}

pass() {
    echo -e "${COLOR_GREEN}[PASS] $1${COLOR_RESET}"
}

echo "============================================"
echo "   Run-Centric API Test Suite (Bash)        "
echo "============================================"

# 1. Check Server
echo "[1] Checking Server..."
curl -s "$BASE_URL/api/stats" > /dev/null || fail "Server is not running on port $PORT"
pass "Server is UP"

# 2. Cleanup Old
echo "[2] Cleaning up..."
curl -s -X DELETE "$BASE_URL/api/runs/delete?name=$RUN_NAME" > /dev/null
rm -f "$CONFIG_FILE" "$DATA_FILE"

# 3. Create Files
echo "[3] creating dummy resources..."
echo "[model]" > $CONFIG_FILE
echo "hidden_dim=999" >> $CONFIG_FILE
echo "dummy_data" > $DATA_FILE

# 4. Upload/Create via API (Config)
echo "[4] Creating Global Config via API..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/configs/create" \
    -H "Content-Type: application/json" \
    -d "{\"filename\": \"$CONFIG_FILE\", \"content\": \"[model]\nhidden_dim=999\"}")

echo "$RESPONSE" | grep -q "created" || fail "Failed to create config"
pass "Config Created"

# 5. Start Run (The Critical Test)
echo "[5] STARTING RUN..."
# JSON payload matching the C++ handler implementation
# keys: run_name, config_file, data_file
RESPONSE=$(curl -s -X POST "$BASE_URL/api/runs/start" \
    -H "Content-Type: application/json" \
    -d "{\"run_name\": \"$RUN_NAME\", \"config_file\": \"$CONFIG_FILE\", \"data_file\": \"$DATA_FILE\"}")

echo "Response: $RESPONSE"
echo "$RESPONSE" | grep -q "started" || fail "Failed to start run. Likely 404/500."
pass "Run Started successfully"

sleep 2

# 6. Verify Run in List
echo "[6] Verifying Run List..."
RESPONSE=$(curl -s "$BASE_URL/api/runs")
echo "$RESPONSE" | grep -q "$RUN_NAME" || fail "Run not found in list"
pass "Run visible in list"

# 7. Verify ISOLATION (Get Run Config)
echo "[7] Verifying ISOLATED Config..."
RESPONSE=$(curl -s "$BASE_URL/api/runs/config?name=$RUN_NAME")
# Check if it contains the specific value 'hidden_dim=999'
echo "$RESPONSE" | grep -q "hidden_dim=999" || fail "Isolated config content mismatch"
pass "Isolated Config Verified"

# 8. Verify Logs
echo "[8] Verifying Logs..."
RESPONSE=$(curl -s "$BASE_URL/api/runs/logs?name=$RUN_NAME")
# Should return some JSON with content
echo "$RESPONSE" | grep -q "content" || fail "Log endpoint failed"
pass "Logs endpoint working"

# 9. Stop Run
echo "[9] Stopping Run..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/runs/stop")
echo "$RESPONSE" | grep -q "stopped" || fail "Failed to stop run"
sleep 1
pass "Run Stopped"

# 10. Delete Run
echo "[10] Deleting Run..."
RESPONSE=$(curl -s -X DELETE "$BASE_URL/api/runs/delete?name=$RUN_NAME")
echo "$RESPONSE" | grep -q "deleted" || fail "Failed to delete run"
pass "Run Deleted"

echo "============================================"
echo -e "${COLOR_GREEN}ALL TESTS PASSED${COLOR_RESET}"
echo "============================================"
exit 0
