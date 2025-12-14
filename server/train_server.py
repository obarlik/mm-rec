#!/usr/bin/env python3
"""
Remote GPU Training Server
Handles training jobs, code sync, and model serving
"""

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import torch
import asyncio
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional
import time

app = FastAPI(title="MM-Rec Training Server")

# Configuration
WORKSPACE_DIR = Path("./workspace")
CHECKPOINTS_DIR = Path("./checkpoints")
WORKSPACE_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

# Job storage
jobs: Dict[str, 'TrainingJob'] = {}

class TrainingConfig(BaseModel):
    job_name: str
    model_dim: int = 64
    num_layers: int = 2
    num_heads: int = 2
    ffn_dim: int = 128
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    max_length: int = 256
    data_path: Optional[str] = None
    vocab_size: Optional[int] = None
    warmup_fraction: float = 0.05
    early_stop_patience: int = 3
    min_delta: float = 0.01
    use_uboo: bool = False
    use_moe: bool = False

class TrainingJob:
    def __init__(self, job_id: str, config: TrainingConfig):
        self.job_id = job_id
        self.config = config
        self.status = "queued"
        self.progress = {
            'epoch': 0,
            'step': 0,
            'total_steps': 0,
            'loss': 0.0,
            'eta_minutes': 0
        }
        self.model_path = None
        self.log_file = WORKSPACE_DIR / f"{job_id}.log"
        self.start_time = None
        
    def run(self):
        """Run training via External JAX Process."""
        try:
            self.status = "training"
            self.start_time = time.time()
            
            # 1. Prepare Config File
            config_path = WORKSPACE_DIR / f"{self.job_id}_config.json"
            
            config_dict = self.config.dict()
            
            # Enforce data_path presence - NO FALLBACKS
            if 'data_path' not in config_dict or not config_dict['data_path']:
                raise ValueError("CRITICAL: 'data_path' is missing in config! Default dataset fallback is disabled.")
            
            # Verify file exists
            data_file = Path(config_dict['data_path'])
            if not data_file.exists():
                raise FileNotFoundError(f"CRITICAL: Dataset file not found at: {data_file}")
            
            # Explicitly set vocab_size for JAX (tiktoken cl100k_base + margin)
            config_dict['vocab_size'] = 100300
 
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            print(f"ℹ️  Job {self.job_id}: Config saved to {config_path}")

            # 2. Build JAX Command
            # Assumes running from project root
            cmd = [
                "python", 
                "mm_rec_jax/training/train_server_jax.py",
                "--config", str(config_path.absolute())
            ]
            
            print(f"🚀 Launching JAX Training: {' '.join(cmd)}")
            
            # 3. Launch Process
            with open(self.log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1, # Line buffered
                    cwd=str(Path.cwd())
                )
                
                # 4. Monitor & Stream Logs
                self.process = process # Store ref for stopping
                
                for line in iter(process.stdout.readline, ''):
                    if not line: break
                    
                    # Write to log file
                    log_f.write(line)
                    log_f.flush()
                    
                    # Optional: Parse line to update self.progress for API
                    # Optional: Parse line to update self.progress for API
                    # New JAX Format: "... | Speed: 4.18 it/s | ETA: 02:45 | VRAM: ..."
                    import re
                    # Updated Regex to include ETA
                    match = re.search(r'Epoch (\d+) \| Step (\d+) \(E:\d+/(\d+)\): Loss ([\d\.]+) \| Speed: ([\d\.]+) it/s \| ETA: ([\d\:]+) \| VRAM: ([\w/]+) MiB \| GNorm: ([\d\.]+) \| MaxState: ([\d\.]+)', line)
                    if match:
                        try:
                            self.progress['epoch'] = int(match.group(1))
                            self.progress['step'] = int(match.group(2))
                            batches_per_epoch = int(match.group(3))
                            total_epochs = self.config.num_epochs
                            self.progress['total_steps'] = batches_per_epoch * total_epochs
                            
                            self.progress['loss'] = float(match.group(4))
                            self.progress['speed'] = match.group(5) + " it/s"
                            self.progress['eta'] = match.group(6) # Captured ETA
                            self.progress['vram'] = match.group(7) + " MiB"
                            self.progress['gnorm'] = match.group(8)
                            self.progress['max_state'] = match.group(9)
                            self.progress['message'] = "Training..."
                        except:
                            pass
                    else:
                        # Capture other important lines as status messages
                        if "Compiling" in line:
                            self.progress['message'] = "Compiling XLA Kernels... (This may take 1-2 mins)"
                        elif "Initializing" in line:
                             self.progress['message'] = "Initializing JAX..."
                        elif "Loading dataset" in line:
                             self.progress['message'] = "Loading Dataset..."
                    
                    # Fallback or other formats could go here
                    
                    # Check for external stop signal (Gateway API set status='stopped')
                    if self.status == "stopped":
                        print(f"🛑 Kill signal received for Job {self.job_id}")
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except:
                            process.kill()
                        return

            # 5. Completion
            return_code = process.wait()
            if return_code == 0:
                self.status = "completed"
                # Locate Saved Model (Expected: {job_id}_model.msgpack)
                expected_model_path = WORKSPACE_DIR / f"{self.job_id}_model.msgpack"
                if expected_model_path.exists():
                    self.model_path = expected_model_path
                    print(f"✅ Job {self.job_id} Completed. Model available at {self.model_path}")
                else:
                    print(f"⚠️ Job {self.job_id} Completed but model file not found at {expected_model_path}")
            else:
                self.status = "failed"
                print(f"❌ Job {self.job_id} Failed with code {return_code}.")

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.status = "failed"
            self.progress['error'] = str(e)
            with open(self.log_file, 'a') as f:
                f.write(f"\nCRITICAL SERVER ERROR: {error_msg}\n")

@app.get("/api/debug/benchmark")
async def run_benchmark():
    """Run speed benchmark and return results."""
    import sys
    import io
    from contextlib import redirect_stdout
    import debug_benchmark
    
    # Capture output
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            debug_benchmark.benchmark()
        except Exception as e:
            import traceback
            print(f"Error: {e}\n{traceback.format_exc()}")
            
    return {"output": f.getvalue()}

@app.post("/api/code/sync")
async def sync_code(file: UploadFile = File(...)):
    """Receive and extract code archive."""
    try:
        # Save uploaded file
        archive_path = WORKSPACE_DIR / "code.zip"
        with open(archive_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract
        shutil.unpack_archive(archive_path, WORKSPACE_DIR)
        
        # Install dependencies
        subprocess.run([
            "pip", "install", "-r", 
            str(WORKSPACE_DIR / "mm-rec" / "requirements.txt")
        ], check=True)
        
        return {"status": "success", "message": "Code synced and dependencies installed"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train/submit")
async def submit_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Submit a training job."""
    job_id = str(uuid.uuid4())[:8]
    job = TrainingJob(job_id, config)
    jobs[job_id] = job
    
    # Start training in background (in a thread to avoid blocking API)
    from fastapi.concurrency import run_in_threadpool
    background_tasks.add_task(run_in_threadpool, job.run)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Job {config.job_name} submitted"
    }

@app.post("/api/train/stop/{job_id}")
async def stop_job(job_id: str):
    """Stop a training job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status in ["training", "queued"]:
        job.status = "stopped"
        return {"status": "stopped", "message": f"Job {job_id} stopping..."}
    else:
        return {"status": job.status, "message": f"Job {job_id} is already {job.status}"}

@app.post("/api/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload data file to workspace/data."""
    try:
        data_dir = WORKSPACE_DIR / "data"
        data_dir.mkdir(exist_ok=True)
        
        file_path = data_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        return {"status": "success", "filename": file.filename, "path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/train/status/{job_id}")
async def get_status(job_id: str):
    """Get training job status."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "config": job.config.dict()
    }

@app.get("/api/train/logs/{job_id}")
async def stream_logs(job_id: str):
    """Stream training logs via Server-Sent Events."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator():
        while job.status in ["queued", "training"]:
            data = {
                "status": job.status,
                "progress": job.progress
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)
        
        # Final update
        yield f"data: {json.dumps({'status': job.status, 'progress': job.progress})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.get("/api/train/download/{job_id}")
async def download_model(job_id: str):
    """Download trained model."""
    # First check memory
    job = jobs.get(job_id)
    
    # Check disk for model file (Persistence fallback)
    model_path = WORKSPACE_DIR / f"{job_id}_model.msgpack"
    
    # Legacy fallback for old .pt files if needed
    if not model_path.exists():
        model_path = WORKSPACE_DIR / f"{job_id}_model.pt"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        model_path,
        filename=model_path.name,
        media_type="application/octet-stream"
    )

@app.get("/api/logs/file/{job_id}")
async def get_job_log(job_id: str):
    """Get log file for a specific job."""
    job = jobs.get(job_id)
    if not job:
        # Check if file exists purely on disk (for past jobs)
        log_path = WORKSPACE_DIR / f"{job_id}.log"
    else:
        log_path = job.log_file
        
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
        
    return FileResponse(
        log_path,
        media_type="text/plain",
        filename=f"{job_id}.log"
    )

@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job.status,
                "config": job.config.model_dump()
            }
            for job_id, job in jobs.items()
        ]
    }

@app.post("/api/update")
async def update_server():
    """
    Legacy update endpoint. 
    In Gateway mode, this should be handled by the Gateway.
    If reached here, it means we are running standalone or Gateway proxied it incorrectly.
    """
    return {
        "status": "ignored",
        "message": "Update should be handled by Gateway (Layer 1). This is Layer 2."
    }


SERVER_VERSION = "v0.3.0 (Gateway Supported)"

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    gpu_info = {
        "status": "healthy",
        "version": SERVER_VERSION,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "active_jobs": len([j for j in jobs.values() if j.status == "training"]),
        "vocab_size": tokenizer.vocab_size if 'tokenizer' in globals() else "Not Initialized",
        "mode": "worker_process"
    }
    return gpu_info

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    args = parser.parse_args()
    
    print("\n🚀 Starting MM-Rec Training Server (Worker)...")
    print(f"ℹ️  Version: {SERVER_VERSION}")
    print(f"📁 Workspace: {WORKSPACE_DIR.absolute()}")
    print(f"🔧 Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"📡 Listening on: {args.host}:{args.port}")
    
    uvicorn.run(app, host=args.host, port=args.port)
