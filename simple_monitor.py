import requests
import time
import argparse
import datetime

# Simple Monitor - No libraries, just prints.
# Usage: python simple_monitor.py --job-id <JOB_ID> --server http://phoenix:8090

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='http://phoenix:8090')
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()

    print(f"👀 Watching Job: {args.job_id}")
    print("-" * 50)

    last_step = -1
    
    while True:
        try:
            resp = requests.get(f"{args.server.rstrip('/')}/api/train/status/{args.job_id}", timeout=5)
            if resp.status_code != 200:
                print(f"⚠️ API Error: {resp.status_code}")
                time.sleep(5)
                continue
                
            data = resp.json()
            prog = data.get('progress', {})
            status = data.get('status', 'unknown')
            
            # Print only if step changed or status changed
            step = prog.get('step', 0)
            
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            
            if status != 'training':
                print(f"[{ts}] Status: {status.upper()}")
                if status in ['completed', 'failed', 'stopped']:
                    break
            
            if step != last_step or status == 'compiling':
                loss = prog.get('loss', 'N/A')
                speed = prog.get('speed', 'N/A')
                eta = prog.get('eta', 'N/A')
                msg = prog.get('message', '')
                
                # Clear line if compiling to show animation effect or just print new line
                # Prioritize metrics if available
                if 'epoch' in prog:
                    vram = prog.get('vram', 'N/A')
                    gnorm = prog.get('gnorm', 'N/A')
                    state = prog.get('max_state', 'N/A')
                    print(f"[{ts}] Epoch {prog.get('epoch')} | Step {step} | Loss: {loss} | {speed} | ETA: {eta} | VRAM: {vram} | GNorm: {gnorm} | State: {state}")
                elif msg:
                     print(f"[{ts}] {msg}")
                
                last_step = step
                
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Connection error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
