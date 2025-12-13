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
        """Run training on GPU."""
        try:
            self.status = "training"
            self.start_time = time.time()
            
            # Import training modules
            import sys
            # Code is extracted to workspace root
            sys.path.insert(0, str(WORKSPACE_DIR))
            
            from mm_rec.model import MMRecModel
            from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
            from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig
            from mm_rec.data.chat_format import ChatMessage
            
            # Setup
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Initialize tokenizer without hardcoded size (tiktoken will set it correct, e.g. 100277)
            tokenizer = get_tokenizer()
            vocab_size = tokenizer.vocab_size + 1000  # Safety margin for special tokens/mismatches
            print(f"ℹ️  Tokenizer initialized with vocab_size={vocab_size} (inc. +1000 margin)")
            
            # Load data
            # Load data - check multiple locations
            possible_paths = [
                WORKSPACE_DIR / "data" / "train.json",  # Uploaded
                WORKSPACE_DIR / "data" / "phase1" / "train.json"  # Git synced
            ]
            
            data_path = None
            for p in possible_paths:
                if p.exists():
                    data_path = p
                    break
            
            if not data_path:
                raise FileNotFoundError(f"train.json not found in: {[str(p) for p in possible_paths]}")
                
            print(f"ℹ️  Loading data from: {data_path}")
            with open(data_path) as f:
                data = json.load(f)  # Full dataset (no limit)
            
            conversations = []
            for item in data:
                messages = [ChatMessage(role=msg['role'], content=msg['content']) 
                           for msg in item['conversations']]
                conversations.append(messages)
            
            # Create model
            model = MMRecModel(
                vocab_size=vocab_size,
                model_dim=self.config.model_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                ffn_dim=self.config.ffn_dim
            ).to(device)
            
            # Training setup
            sft_config = SFTConfig(
                max_length=self.config.max_length,
                label_smoothing=0.1
            )
            trainer = SFTTrainer(model, tokenizer, sft_config)
            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
            
            # Training loop
            total_steps = (len(conversations) // self.config.batch_size) * self.config.num_epochs
            self.progress['total_steps'] = total_steps
            
            # OneCycleLR: Warmup + Cosine Decay
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1, # 10% warmup
                anneal_strategy='cos'
            )
            
            for epoch in range(self.config.num_epochs):
                epoch_losses = []
                
                # Batch Loop
                batch_size = self.config.batch_size
                for i in range(0, len(conversations), batch_size):
                    step_start_time = time.time()
                    
                    # Create Batch
                    batch_conversations = conversations[i : i + batch_size]
                    if not batch_conversations:
                        continue
                        
                    # Check for stop signal
                    if self.status == "stopped":
                        print(f"🛑 Job {self.job_id} stopped by user.")
                        return
                    
                    # Get current LR (before step)
                    current_lr = scheduler.get_last_lr()[0]

                    # Training step (Batched)
                    try:
                        result = trainer.train_batch(batch_conversations, optimizer, device, verbose=False)
                        epoch_losses.append(result['loss'])
                        # Step Scheduler
                        scheduler.step()
                    except Exception as e:
                        print(f"⚠️ Batch failed: {e}")
                        continue
                    
                    # Update progress
                    current_step = (epoch * total_steps) + (i // batch_size) + 1
                    elapsed = time.time() - self.start_time
                    step_duration = time.time() - step_start_time
                    steps_per_sec = 1.0 / step_duration if step_duration > 0 else 0.0
                    
                    # Calculate ETA based on remaining batches
                    remaining_steps = (total_steps * self.config.num_epochs) - current_step
                    # Moving average speed could be better, but instantaneous is fine for now
                    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    
                    self.progress = {
                        'epoch': epoch + 1,
                        'step': current_step,
                        'total_steps': total_steps,
                        'loss': result['loss'],
                        'eta_minutes': int(eta_seconds / 60),
                        'speed': f"{steps_per_sec:.2f} it/s",
                        'lr': f"{current_lr:.2e}"
                    }
                    
                    # Save checkpoint every 500 steps
                    if (i + 1) % 500 == 0:
                        checkpoint_path = CHECKPOINTS_DIR / f"{self.job_id}_epoch{epoch+1}_step{i+1}.pt"
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'config': self.config.dict(),
                            'progress': self.progress
                        }, checkpoint_path)
            
            # Save final model
            self.model_path = CHECKPOINTS_DIR / f"{self.job_id}_final.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': self.config.dict(),
                'final_loss': sum(epoch_losses) / len(epoch_losses)
            }, self.model_path)
            
            self.status = "completed"
            
        except Exception as e:
            self.status = "failed"
            self.progress['error'] = str(e)
            with open(self.log_file, 'a') as f:
                f.write(f"\nERROR: {str(e)}\n")

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
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    if not job.model_path or not job.model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        job.model_path,
        filename=f"{job_id}_model.pt",
        media_type="application/octet-stream"
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
async def update_server(restart: bool = True, force: bool = False):
    """Pull latest code from git and optionally restart server."""
    try:
        # Check for active jobs if restarting
        if restart and not force:
            active_jobs = [j.job_id for j in jobs.values() if j.status in ["training", "queued"]]
            if active_jobs:
                raise HTTPException(
                    status_code=409, 
                    detail=f"Active jobs running: {active_jobs}. Use force=True to restart anyway."
                )

        import subprocess
        import os
        import sys
        
        # Get current directory
        server_dir = Path(__file__).parent.parent
        
        # Pull latest code
        result = subprocess.run(
            ["git", "pull"],
            cwd=server_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Git pull failed: {result.stderr}")
        
        response_data = {
            "status": "updated",
            "message": "Code updated successfully.",
            "git_output": result.stdout
        }
        
        # Auto-restart if requested
        if restart:
            response_data["message"] += " Server restarting..."
            response_data["restart"] = True
            
            # Schedule restart after response is sent
            async def restart_server():
                await asyncio.sleep(1)  # Give time for response to be sent
                
                # Determine which restart script to use
                server_dir = Path(__file__).parent.parent
                
                # Try PowerShell script first (Windows)
                ps_script = server_dir / "server" / "restart_server.ps1"
                sh_script = server_dir / "server" / "restart_server.sh"
                
                if ps_script.exists() and sys.platform == 'win32':
                    # Windows PowerShell (Only on actual Windows)
                    subprocess.Popen(
                        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps_script)],
                        cwd=server_dir,
                        # DETACHED_PROCESS needed for Windows
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == 'win32' else 0,
                        start_new_session=True # For WSL/Linux
                    )
                elif sh_script.exists():
                    # Linux/WSL/Mac - Use os.execv for robust process replacement
                    # Re-build extensions first if needed
                    cpp_dir = server_dir / "mm_rec" / "cpp"
                    if (cpp_dir / "setup.py").exists():
                        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], 
                                     cwd=cpp_dir, check=False)
                    
                    # Restart process directly
                    print("🔄 Executing os.execv restart...")
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                else:
                    # Fallback to os.execv
                    os.execv(sys.executable, ['python'] + sys.argv)
            
            asyncio.create_task(restart_server())
        else:
            response_data["note"] = "Server restart required manually"
        
        return response_data
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Git pull timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


SERVER_VERSION = "v0.2.16 (Batching + LR Scheduler)"

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
        "features": ["Triton-Free", "Threaded-Training", "Auto-Restart", "Vocab-Safe-Margin"]
    }
    return gpu_info

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting MM-Rec Training Server...")
    print(f"ℹ️  Version: {SERVER_VERSION}")
    print("✅ Features: [Triton-Free GPU] [Threaded Execution] [Auto-Restart]")
    print(f"📁 Workspace: {WORKSPACE_DIR.absolute()}")
    print(f"💾 Checkpoints: {CHECKPOINTS_DIR.absolute()}")
    print(f"🔧 Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    uvicorn.run(app, host="0.0.0.0", port=8001)
