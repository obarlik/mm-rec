#!/usr/bin/env python3
"""
Real-time Training Monitor
Monitors training progress and shows live updates
"""

import time
import sys
import os
from pathlib import Path

def monitor_training(log_file="training_real.log", interval=2):
    """Monitor training log file and show live updates."""
    print("="*80)
    print("📊 MM-Rec Training Monitor - Live Updates")
    print("="*80)
    print(f"📁 Log file: {log_file}")
    print(f"🔄 Update interval: {interval}s")
    print("="*80 + "\n")
    
    log_path = Path(log_file)
    last_size = 0
    
    try:
        while True:
            if log_path.exists():
                current_size = log_path.stat().st_size
                
                if current_size > last_size:
                    # Read new content
                    with open(log_path, 'r', encoding='utf-8') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        
                        if new_content.strip():
                            # Clear line and print new content
                            lines = new_content.strip().split('\n')
                            for line in lines[-10:]:  # Show last 10 lines
                                if line.strip():
                                    print(f"  {line}")
                            
                            last_size = current_size
                else:
                    # No new content, show status
                    print(f"⏳ Waiting for updates... ({time.strftime('%H:%M:%S')})", end='\r')
            else:
                print(f"⏳ Waiting for log file... ({time.strftime('%H:%M:%S')})", end='\r')
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")


if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else "training_real.log"
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    
    monitor_training(log_file, interval)

