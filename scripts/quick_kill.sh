#!/bin/bash

# Super Quick Cleanup
# Use this while developing - kills everything instantly

killall -9 demo_training_cpp demo_inference_cpp mm_rec_cli 2>/dev/null || true
pkill -9 -f "mm_rec|demo_training|demo_inference" 2>/dev/null || true

# Free ports
for port in 8085 8080; do
    pid=$(lsof -ti:$port 2>/dev/null || true)
    [ -n "$pid" ] && kill -9 $pid 2>/dev/null || true
done

echo "✓ Quick cleanup done!"
