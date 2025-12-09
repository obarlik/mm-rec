#!/bin/bash
# Real-time training watcher

LOG_FILE="${1:-training_real.log}"
INTERVAL="${2:-3}"

echo "📊 MM-Rec Training Watcher"
echo "=========================="
echo "📁 Log: $LOG_FILE"
echo "🔄 Update: every ${INTERVAL}s"
echo "=========================="
echo ""

while true; do
    clear
    echo "📊 MM-Rec Training - Live Status"
    echo "================================"
    echo "⏰ $(date '+%H:%M:%S')"
    echo ""
    
    if [ -f "$LOG_FILE" ]; then
        # Show last steps
        echo "📈 Recent Steps:"
        tail -50 "$LOG_FILE" 2>/dev/null | grep -E "(Step [0-9]+ completed|📈 Step)" | tail -5
        echo ""
        
        # Show current progress
        echo "🔄 Current Status:"
        tail -10 "$LOG_FILE" 2>/dev/null | grep -E "(Training|Processing|completed)" | tail -3
        echo ""
        
        # Show loss trend
        echo "📉 Loss Trend:"
        tail -50 "$LOG_FILE" 2>/dev/null | grep "Loss:" | tail -5
        echo ""
        
        # Show checkpoints
        echo "💾 Checkpoints:"
        tail -20 "$LOG_FILE" 2>/dev/null | grep "Checkpoint" | tail -2
        echo ""
        
        # Process status
        if pgrep -f train_real.py > /dev/null; then
            echo "✅ Training: RUNNING"
        else
            echo "⚠️  Training: STOPPED"
        fi
    else
        echo "⏳ Waiting for log file..."
    fi
    
    sleep "$INTERVAL"
done

