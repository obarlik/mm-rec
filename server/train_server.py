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
        
    async def run(self):
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
            tokenizer = get_tokenizer(vocab_size=100256)
            
            # Load data
            data_path = WORKSPACE_DIR / "mm-rec" / "data" / "phase1" / "train.json"
            with open(data_path) as f:
                data = json.load(f)[:5000]  # Limit for testing
            
            conversations = []
            for item in data:
                messages = [ChatMessage(role=msg['role'], content=msg['content']) 
                           for msg in item['conversations']]
                conversations.append(messages)
            
            # Create model
            model = MMRecModel(
                vocab_size=100256,
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
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
            
            # Training loop
            total_steps = len(conversations) * self.config.num_epochs
            self.progress['total_steps'] = total_steps
            
            for epoch in range(self.config.num_epochs):
                epoch_losses = []
                
                for i, conv in enumerate(conversations):
                    # Training step
                    result = trainer.train_step(conv, optimizer, device, verbose=False)
                    epoch_losses.append(result['loss'])
                    
                    # Update progress
                    step = epoch * len(conversations) + i + 1
                    elapsed = time.time() - self.start_time
                    steps_per_sec = step / elapsed if elapsed > 0 else 0
                    remaining_steps = total_steps - step
                    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    
                    self.progress = {
                        'epoch': epoch + 1,
                        'step': i + 1,
                        'total_steps': len(conversations),
                        'loss': result['loss'],
                        'eta_minutes': int(eta_seconds / 60)
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
    
    # Start training in background
    background_tasks.add_task(job.run)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Job {config.job_name} submitted"
    }

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
                "config": job.config.dict()
            }
            for job_id, job in jobs.items()
        ]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "active_jobs": len([j for j in jobs.values() if j.status == "training"])
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting MM-Rec Training Server...")
    print(f"📁 Workspace: {WORKSPACE_DIR.absolute()}")
    print(f"💾 Checkpoints: {CHECKPOINTS_DIR.absolute()}")
    uvicorn.run(app, host="0.0.0.0", port=8001)
