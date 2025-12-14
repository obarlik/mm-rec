#!/usr/bin/env python3
"""
Remote Training Client
Submit jobs, sync code, monitor progress, download models
"""

import requests
import time
import zipfile
from pathlib import Path
from typing import Optional
import json

class RemoteTrainer:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        
    def sync_code(self, project_dir: str = "."):
        """Sync local code to remote server."""
        print("📦 Packaging code...")
        
        # Create zip archive
        project_path = Path(project_dir)
        archive_path = Path("/tmp/mm-rec-code.zip")
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in project_path.rglob('*'):
                if file_path.is_file():
                    # Skip unnecessary files
                    if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.venv', 'checkpoints']):
                        continue
                    
                    arcname = file_path.relative_to(project_path.parent)
                    zipf.write(file_path, arcname)
        
        print(f"📤 Uploading code ({archive_path.stat().st_size / 1024 / 1024:.1f} MB)...")
        
        # Upload to server
        with open(archive_path, 'rb') as f:
            response = requests.post(
                f"{self.server_url}/api/code/sync",
                files={'file': f}
            )
        
        if response.status_code == 200:
            print("✅ Code synced successfully!")
            return True
        else:
            print(f"❌ Sync failed: {response.text}")
            return False
    
    def submit_job(self, config: dict) -> str:
        """Submit training job."""
        print(f"🚀 Submitting job: {config.get('job_name', 'unnamed')}...")
        
        response = requests.post(
            f"{self.server_url}/api/train/submit",
            json=config
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            print(f"✅ Job submitted: {job_id}")
            return job_id
        else:
            print(f"❌ Submit failed: {response.text}")
            return None
    
    def get_status(self, job_id: str) -> dict:
        """Get job status."""
        response = requests.get(f"{self.server_url}/api/train/status/{job_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    
    def monitor(self, job_id: str, update_interval: int = 1):
        """Monitor training progress (Curses TUI)."""
        import curses
        import time
        import requests
        from collections import deque
        import datetime
        import sys

        def tui(stdscr):
            # Setup
            curses.curs_set(0) # Hide cursor
            stdscr.nodelay(True) # Non-blocking input
            # Timeout handles the loop sleep (1000ms = 1s)
            stdscr.timeout(1000) 
            
            history = deque(maxlen=30) # Store enough for large screens
            last_step = -1
            
            # Colors if supported
            if curses.has_colors():
                curses.start_color()
                curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK) # Header
                curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK) # Good
                curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Warning
                curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK) # Err
            
            while True:
                # 1. Fetch Data
                status_txt = "CONNECTING..."
                try:
                    resp = requests.get(
                        f"{self.server_url}/api/train/status/{job_id}",
                        timeout=1
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        status = data.get('status', 'unknown')
                        status_txt = status.upper()
                        
                        progress = data.get('progress', {})
                        
                        if status == 'training' and isinstance(progress, dict) and 'loss' in progress:
                            # Extract
                            ep = str(progress.get('epoch', '?'))
                            st = int(progress.get('step', 0))
                            total_st = int(progress.get('total_steps', 1)) # Prevent div/0
                            
                            # Percentage
                            percent = (st / total_st) * 100 if total_st > 0 else 0
                            prog_str = f"{percent:.1f}%"
                            
                            loss = float(progress.get('loss', 0.0))
                            spd = str(progress.get('speed', '?'))
                            eta = str(progress.get('eta', '?'))
                            
                            # VRAM (Assuming RTX 4090 - 24GB)
                            vram_val = progress.get('vram', 'N/A')
                            vram_str = "N/A"
                            if vram_val != 'N/A':
                                try:
                                    vram_num = float(str(vram_val).split()[0])
                                    vram_gb = vram_num / 1024
                                    vram_pct = (vram_gb / 24.0) * 100
                                    vram_str = f"{vram_gb:.1f}G ({int(vram_pct)}%)"
                                except:
                                    vram_str = str(vram_val)
                            
                            gnorm = str(progress.get('gnorm', progress.get('grad_norm', '?')))
                            mstate = str(progress.get('max_state', progress.get('MaxState', '?')))

                            # Update history
                            current_step_num = st
                            if current_step_num != last_step or last_step == -1:
                                loss_str = f"{loss:.4f}"
                                # COLUMNS: EP | STEP | PROG | LOSS | GNORM | MST | VRAM | SPEED | ETA
                                row = f"{ep:<4} {st:<6} {prog_str:<6} {loss_str:<8} {gnorm:<8} {mstate:<8} {vram_str:<12} {spd:<10} {eta:<10}"
                                history.appendleft(row)
                                last_step = current_step_num
                        
                        elif status in ['completed', 'failed', 'stopped']:
                             history.appendleft(f"JOB FINISHED: {status.upper()}")
                             status_txt = status.upper()

                except requests.exceptions.RequestException:
                    status_txt = "NETWORK ERROR"

                # 2. Draw UI
                stdscr.erase()
                height, width = stdscr.getmaxyx()
                
                # Title Bar
                title = f" JAX MONITOR: {job_id} | STATUS: {status_txt} | {datetime.datetime.now().strftime('%H:%M:%S')}"
                safe_addstr(stdscr, 0, 0, title[:width], curses.color_pair(1) | curses.A_BOLD)
                safe_addstr(stdscr, 1, 0, "=" * (width), curses.color_pair(1))
                
                # Column Headers
                headers = f"{'EP':<4} {'STEP':<6} {'PROG':<6} {'LOSS':<8} {'GNORM':<8} {'MST':<8} {'VRAM':<12} {'SPEED':<10} {'ETA':<10}"
                safe_addstr(stdscr, 2, 0, headers[:width], curses.A_UNDERLINE)
                
                # Data Rows
                # Max rows specific to screen
                max_rows = height - 8 # Reserve 8 lines for header/footer/legend
                
                for i, row in enumerate(history):
                    if i >= max_rows: break
                    y = 3 + i
                    safe_addstr(stdscr, y, 0, row[:width])

                # Legend / Info Panel (Bottom)
                # Ensure we have space
                if height > 12:
                    info_y = height - 5
                    safe_addstr(stdscr, info_y, 0, "-" * width, curses.color_pair(1))
                    
                    # Row 1
                    s1 = " GUIDANCE: "
                    safe_addstr(stdscr, info_y + 1, 0, s1, curses.A_BOLD)
                    safe_addstr(stdscr, info_y + 1, len(s1), "LOSS: Decoding Error (Lower is better, Target < 2.0)")
                    
                    # Row 2
                    s2 = "           "
                    safe_addstr(stdscr, info_y + 2, 0, s2)
                    safe_addstr(stdscr, info_y + 2, len(s2), "GNORM: Stability (Should be ~1.0, >10.0 is unstable)")
                    
                    # Row 3
                    safe_addstr(stdscr, info_y + 3, 0, s2)
                    # Use color warning for MST explanation
                    safe_addstr(stdscr, info_y + 3, len(s2), "MST (MaxState): Activation Magnitude (Risk if > 100)")

                # Footer
                footer = " [q] Quit | [r] Refresh"
                safe_addstr(stdscr, height-1, 0, footer[:width], curses.color_pair(1) | curses.A_REVERSE)

                stdscr.refresh()
                
                # Input Handling
                c = stdscr.getch()
                if c == ord('q'):
                    break
        
        def safe_addstr(win, y, x, s, attr=0):
            """Robust addstr that avoids errors at screen corners."""
            h, w = win.getmaxyx()
            if y >= h or x >= w: return
            try:
                # Clip string if too long
                if len(s) > w - x: s = s[:w-x]
                win.addstr(y, x, s, attr)
            except curses.error:
                pass

        try:
            curses.wrapper(tui)
        except Exception as e:
            print(f"Curses Error: {e}")
    
    def download_model(self, job_id: str, output_path: str):
        """Download trained model."""
        print(f"📥 Downloading model from job {job_id}...")
        
        response = requests.get(
            f"{self.server_url}/api/train/download/{job_id}",
            stream=True
        )
        
        if response.status_code == 200:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Model downloaded to {output_path}")
            return True
        else:
            print(f"❌ Download failed: {response.text}")
            return False
    
    def list_jobs(self):
        """List all jobs on server."""
        response = requests.get(f"{self.server_url}/api/jobs")
        if response.status_code == 200:
            jobs = response.json()['jobs']
            print("\n📋 Jobs on server:")
            print("=" * 80)
            for job in jobs:
                print(f"  {job['job_id']}: {job['status']} - {job['config']['job_name']}")
            return jobs
        return []
    
    def health_check(self):
        """Check server health."""
        try:
            response = requests.get(f"{self.server_url}/api/health")
            if response.status_code == 200:
                health = response.json()
                print("🏥 Server Health:")
                print(f"  Status: {health['status']}")
                print(f"  GPU: {health['gpu_name'] if health['gpu_available'] else 'Not available'}")
                print(f"  Active jobs: {health['active_jobs']}")
                return True
        except Exception as e:
            print(f"❌ Server unreachable: {e}")
            return False
    
    def update_server(self):
        """Trigger server to pull latest code from git."""
        print("🔄 Updating server code...")
        try:
            response = requests.post(f"{self.server_url}/api/update")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {result['message']}")
                print(f"📝 Git output: {result.get('git_output', 'No output')}")
                if result.get('note'):
                    print(f"⚠️  {result['note']}")
                return True
            else:
                print(f"❌ Update failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Update failed: {e}")
            return False

    def get_logs(self, log_type: str, job_id: str = None):
        """Get logs from server."""
        print(f"📜 Fetching logs (Type: {log_type})...")
        
        try:
            if log_type == 'job':
                if not job_id:
                    print("❌ Error: --job-id required for job logs")
                    return
                url = f"{self.server_url}/api/logs/file/{job_id}"
            elif log_type in ['gateway', 'server', 'inference']:
                # Gateway logs might only work if connected to gateway port
                url = f"{self.server_url}/gateway/logs/{log_type}"
            else:
                print(f"❌ Unknown log type: {log_type}")
                return

            response = requests.get(url)
            
            if response.status_code == 200:
                print("\n" + "="*80)
                print(response.text)
                print("="*80 + "\n")
            else:
                print(f"❌ Failed to fetch logs: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error fetching logs: {e}")

def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Remote GPU Training Client")
    parser.add_argument('--server', type=str, default='http://phoenix:8090', help='Server URL')
    parser.add_argument('--action', required=True, 
                        choices=['sync', 'submit', 'monitor', 'download', 'list', 'health', 'update', 'stop', 'upload', 'logs'],
                        help='Action to perform')
    parser.add_argument("--job-id", help="Job ID (for monitor/download/logs)")
    parser.add_argument("--config", help="Config file (for submit)")
    parser.add_argument("--output", help="Output path (for download)")
    parser.add_argument("--project-dir", default=".", help="Project directory (for sync)")
    parser.add_argument('--file', type=str, help='File path for upload')
    parser.add_argument('--type', type=str, choices=['job', 'gateway', 'server', 'inference'], help='Log type')
    parser.add_argument('--force', action='store_true', help='Force operation (check active jobs)')
    
    args = parser.parse_args()
    
    trainer = RemoteTrainer(args.server)
    
    if args.action == 'health':
        trainer.health_check()
    
    elif args.action == 'logs':
        if not args.type:
            print("❌ --type required for logs (job, gateway, server)")
            return
        trainer.get_logs(args.type, args.job_id)
        
    elif args.action == 'sync':
        trainer.sync_code(args.project_dir)
    
    elif args.action == 'submit':
        if not args.config:
            print("❌ --config required for submit")
            return
        
        with open(args.config) as f:
            config = json.load(f)
        
        job_id = trainer.submit_job(config)
        if job_id:
            print(f"\n💡 Monitor with: python client/train_client.py --action monitor --job-id {job_id}")
    
    elif args.action == 'stop':
        if not args.job_id:
            print("❌ Error: --job-id required for stop")
            return
        
        try:
            response = requests.post(f"{args.server}/api/train/stop/{args.job_id}")
            if response.status_code == 200:
                print(f"🛑 Job stopped: {response.json()}")
            else:
                print(f"❌ Stop failed: {response.text}")
        except Exception as e:
            print(f"❌ Connection error: {e}")

    elif args.action == 'upload':
        if not args.file:
            print("❌ Error: --file required for upload")
            return
            
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return
            
        print(f"📤 Uploading {file_path.name}...")
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{args.server}/api/data/upload", files=files)
            
            if response.status_code == 200:
                print(f"✅ Upload successful: {response.json()}")
            else:
                print(f"❌ Upload failed: {response.text}")
        except Exception as e:
            print(f"❌ Connection error: {e}")

    elif args.action == 'monitor':
        if not args.job_id:
            print("❌ --job-id required for monitor")
            return
        
        trainer.monitor(args.job_id)
    
    elif args.action == 'download':
        if not args.job_id or not args.output:
            print("❌ --job-id and --output required for download")
            return
        
        trainer.download_model(args.job_id, args.output)
    
    elif args.action == 'update':
        trainer.update_server()
    
    elif args.action == 'list':
        trainer.list_jobs()


if __name__ == "__main__":
    main()
