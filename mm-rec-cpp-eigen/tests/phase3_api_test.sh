#!/bin/bash
# Test Script for Phase 3 APIs

PORT=8085
BASE_URL="http://localhost:$PORT"

echo "Testing Phase 3 APIs on Port $PORT..."

# 1. List Configs (Existing)
echo "--- GET /api/configs ---"
curl -s "$BASE_URL/api/configs" | grep "\[" && echo " [OK]" || echo " [FAIL]"

# 2. Create Config
echo "--- POST /api/configs/create ---"
curl -s -X POST -H "Content-Type: application/json" \
     -d '{"filename": "test_api_config.ini", "content": "[model]\ntest=1"}' \
     "$BASE_URL/api/configs/create"
echo ""

# Verify creation
if [ -f "test_api_config.ini" ]; then
    echo "test_api_config.ini created. [OK]"
    rm test_api_config.ini
else
    echo "test_api_config.ini NOT created. [FAIL]"
fi

# 3. List Datasets
echo "--- GET /api/datasets ---"
curl -s "$BASE_URL/api/datasets" | grep "\[" && echo " [OK]" || echo " [FAIL]"

# 4. Upload Dataset
echo "Creating dummy dataset..."
echo "dummy data" > dummy.bin
echo "--- PUT /api/datasets/upload ---"
curl -s -X PUT --data-binary @dummy.bin "$BASE_URL/api/datasets/upload?name=uploaded_dummy.bin"
echo ""

# Verify upload
if [ -f "uploaded_dummy.bin" ]; then
    echo "uploaded_dummy.bin created. [OK]"
    rm uploaded_dummy.bin
else
    echo "uploaded_dummy.bin NOT created. [FAIL]"
fi
rm dummy.bin

# 5. List Models
echo "--- GET /api/models ---"
curl -s "$BASE_URL/api/models" | grep "\[" && echo " [OK]" || echo " [FAIL]"

echo "Test Complete."
