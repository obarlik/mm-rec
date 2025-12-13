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
        response = requests.get(f"{self.server_url}/api/train/status/{job_id}")
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
            
            if status['status'] == 'training':
                prog = status['progress']
                print(f"\r[Epoch {prog['epoch']}, Step {prog['step']}/{prog['total_steps']}] "
                      f"Loss: {prog['loss']:.4f}, ETA: {prog['eta_minutes']}m    ", 
                      end='', flush=True)
            
            time.sleep(update_interval)
    
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

def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Remote GPU Training Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument('--action', required=True, 
                        choices=['sync', 'submit', 'monitor', 'download', 'list', 'health', 'update', 'stop', 'upload'],
                        help='Action to perform')
    parser.add_argument("--job-id", help="Job ID (for monitor/download)")
    parser.add_argument("--config", help="Config file (for submit)")
    parser.add_argument("--output", help="Output path (for download)")
    parser.add_argument("--project-dir", default=".", help="Project directory (for sync)")
    parser.add_argument('--file', type=str, help='File path for upload')
    
    args = parser.parse_args()
    
    trainer = RemoteTrainer(args.server)
    
    if args.action == 'health':
        trainer.health_check()
    
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
