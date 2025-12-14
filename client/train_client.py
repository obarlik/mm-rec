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
    
    def monitor(self, job_id: str, update_interval: int = 10):
        """Monitor training progress."""
        print(f"👀 Monitoring job: {job_id}")
        print("=" * 80)
        
        while True:
            status = self.get_status(job_id)
            if not status:
                print("❌ Failed to get status")
                break
            
            if status['status'] == 'completed':
                print("\n✅ Training completed!")
                break
            
            if status['status'] == 'failed':
                print(f"\n❌ Training failed: {status['progress'].get('error', 'Unknown error')}")
                break
            
                # Rich TUI update
                from rich.table import Table
                from rich.panel import Panel
                from rich.layout import Layout
                from rich.console import Console
                from rich.live import Live
                from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
                
                console = Console(force_terminal=True)
                layout = Layout()
                
                # Split Layout
                layout.split(
                    Layout(name="header", size=3),
                    Layout(name="main", ratio=1),
                    Layout(name="footer", size=3)
                )
                
                layout["main"].split_row(
                    Layout(name="metrics"),
                    Layout(name="progress")
                )
                
                prog = status['progress']
                
                # Header
                status_msg = prog.get('message', status['status'].upper())
                header = Panel(f"[bold cyan]🚀 Job: {job_id} | Status: {status_msg}[/]", style="white on blue")
                layout["header"].update(header)
                
                # Metrics Table
                table = Table(title="Training Metrics", expand=True)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                
                table.add_row("Message", f"[yellow]{status_msg}[/]")
                if 'epoch' in prog:
                    table.add_row("Epoch", f"{prog.get('epoch', 'N/A')}")
                    table.add_row("Step", f"{prog.get('step', 'N/A')}/{prog.get('total_steps', 'N/A')}")
                    table.add_row("Loss", f"{prog.get('loss', 0.0):.4f}")
                    table.add_row("Speed", f"{prog.get('speed', 'N/A')}")
                    table.add_row("ETA", f"{prog.get('eta', 'N/A')}")
                    table.add_row("VRAM", f"{prog.get('vram', 'N/A')}")
                    table.add_row("GNorm", f"{prog.get('gnorm', 'N/A')}")
                    table.add_row("State", f"{prog.get('max_state', 'N/A')}")
                else:
                    table.add_row("Info", "Waiting for metrics...")
                
                layout["metrics"].update(Panel(table))
                
                # Progress Bar
                total = prog.get('total_steps', 0)
                current = prog.get('step', 0)
                pct = (current / total * 100) if total > 0 else 0
                
                # ASCII Bar
                bar_len = 30
                filled = int(pct / 100 * bar_len)
                bar_str = "█" * filled + "░" * (bar_len - filled)
                
                prog_panel = Panel(
                    f"\n[bold green]{pct:.1f}% Complete[/]\n\n"
                    f"[{bar_str}]\n\n"
                    f"[dim]Total Steps: {total}[/]",
                    title="Progress",
                    border_style="green"
                )
                layout["progress"].update(prog_panel)

                print("✨ Starting Dashboard...", flush=True)
                
                # Live Update Loop
                # Use screen=True for proper full-screen dashboard
                with Live(layout, console=console, refresh_per_second=4, screen=True) as live:
                    while True:
                        try:
                            status = self.get_status(job_id)
                            
                            # Handle Connection Error
                            if not status:
                                import datetime
                                now_str = datetime.datetime.now().strftime("%H:%M:%S")
                                layout["header"].update(Panel(f"[bold red]🚀 Job: {job_id} | Status: CONNECTION LOST | Last Upd: {now_str}[/]", style="white on red"))
                                layout["footer"].update(Panel("[bold red]Connection lost. Retrying...[/]", style="white on red"))
                                live.refresh()
                                time.sleep(2) # Wait before retry
                                continue

                            prog = status['progress']
                            
                            # Check completion
                            if status['status'] != 'training' and status['status'] != 'queued':
                                # Final update
                                live.update(layout)
                                print(f"\nJob finished with status: {status['status']}")
                                break
                            
                            import datetime
                            # Update Header
                            now_str = datetime.datetime.now().strftime("%H:%M:%S")
                            status_msg = prog.get('message', status['status'].upper())
                            layout["header"].update(Panel(f"[bold cyan]🚀 Job: {job_id} | Status: {status_msg} | Last Upd: {now_str}[/]", style="white on blue"))
                            
                            # Explicitly refresh screen for screen=False mode
                            live.refresh()
                            
                            # Update Table
                            table = Table(title="Training Metrics", expand=True)
                            table.add_column("Metric", style="cyan")
                            table.add_column("Value", style="magenta")
                            table.add_row("Message", f"[yellow]{status_msg}[/]")
                            
                            if 'epoch' in prog:
                                table.add_row("Epoch", f"{prog.get('epoch', 'N/A')}")
                                table.add_row("Step", f"{prog.get('step', 'N/A')}/{prog.get('total_steps', 'N/A')}")
                                table.add_row("Loss", f"{prog.get('loss', 0.0):.4f}")
                                table.add_row("Speed", f"{prog.get('speed', 'N/A')}")
                                table.add_row("ETA", f"{prog.get('eta', 'N/A')}")
                                table.add_row("VRAM", f"{prog.get('vram', 'N/A')}")
                                table.add_row("GNorm", f"{prog.get('gnorm', 'N/A')}")
                                table.add_row("State", f"{prog.get('max_state', 'N/A')}")
                            else:
                                table.add_row("Info", "Waiting for metrics...")
                                
                            layout["metrics"].update(Panel(table))
                            
                            # Update Progress
                            total = prog.get('total_steps', 0)
                            current = prog.get('step', 0)
                            pct = (current / total * 100) if total > 0 else 0
                            filled = int(pct / 100 * bar_len)
                            bar_str = "█" * filled + "░" * (bar_len - filled)
                            
                            layout["progress"].update(Panel(
                                f"\n[bold green]{pct:.1f}% Complete[/]\n\n"
                                f"[{bar_str}]\n\n"
                                f"[dim]Total Steps: {total}[/]",
                                title="Progress",
                                border_style="green"
                            ))
                            
                            time.sleep(update_interval)
                            
                        except KeyboardInterrupt:
                            break
                        except Exception as e:
                            # If connection fails, print error but try to continue
                            # If critical error, break
                            layout["footer"].update(Panel(f"[bold red]Error: {str(e)}[/]", style="white on red"))
                            time.sleep(1)
                # Only return if loop broke
                return
    
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
            elif log_type in ['gateway', 'server']:
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
    parser.add_argument('--type', type=str, choices=['job', 'gateway', 'server'], help='Log type')
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
