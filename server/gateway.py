import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI, Request, HTTPException, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
import os

# Configuration
GATEWAY_PORT = 8090
TRAIN_SERVER_PORT = 8001
INFERENCE_SERVER_PORT = 8002
TRAIN_SERVER_HOST = "127.0.0.1"
TRAIN_SERVER_URL = f"http://{TRAIN_SERVER_HOST}:{TRAIN_SERVER_PORT}"
INFERENCE_SERVER_URL = f"http://{TRAIN_SERVER_HOST}:{INFERENCE_SERVER_PORT}"
SERVER_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="MM-Rec Training & Inference Gateway")
train_client = httpx.AsyncClient(base_url=TRAIN_SERVER_URL, timeout=None)
inference_client = httpx.AsyncClient(base_url=INFERENCE_SERVER_URL, timeout=None)

# Process Management
train_process: Optional[subprocess.Popen] = None
inference_process: Optional[subprocess.Popen] = None

def start_train_server():
    """Start the training server as a subprocess."""
    global train_process
    
    cmd = [sys.executable, "server/train_server.py", "--port", str(TRAIN_SERVER_PORT)]
    print(f"🚀 Gateway: Starting Training Server on port {TRAIN_SERVER_PORT}...")
    log_file = open(SERVER_DIR / "server_internal.log", "a")
    
    train_process = subprocess.Popen(
        cmd,
        cwd=SERVER_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return train_process

def start_inference_server():
    """Start the inference server (CPU) as a subprocess."""
    global inference_process
    
    # We use CPU for inference to avoid VRAM clash
    env = os.environ.copy()
    env["JAX_PLATFORM_NAME"] = "cpu"
    
    # Find the most recent checkpoint in workspace/
    workspace_dir = SERVER_DIR / "workspace"
    if workspace_dir.exists():
        checkpoints = list(workspace_dir.glob("*_ckpt_epoch_*.msgpack"))
        if checkpoints:
            # Sort by modification time, get latest
            model_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
            print(f"🧠 Gateway: Found checkpoint: {model_path}")
        else:
            print("⚠️ Gateway: No checkpoints found in workspace/, skipping Inference Server.")
            return None
    else:
        print("⚠️ Gateway: workspace/ directory not found, skipping Inference Server.")
        return None
    
    config_path = "configs/baseline.json"
    
    cmd = [
        sys.executable, "server/inference_server.py",
        "--port", str(INFERENCE_SERVER_PORT),
        "--model", model_path,
        "--config", config_path
    ]
    
    print(f"🧠 Gateway: Starting Inference Server (CPU) on port {INFERENCE_SERVER_PORT}...")
    log_file = open(SERVER_DIR / "inference_internal.log", "a")
    
    inference_process = subprocess.Popen(
        cmd,
        cwd=SERVER_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return inference_process

def stop_processes():
    """Stop all managed servers."""
    global train_process, inference_process
    
    for name, proc in [("Training", train_process), ("Inference", inference_process)]:
        if proc and proc.poll() is None:
            print(f"🛑 Gateway: Stopping {name} Server...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    
    train_process = None
    inference_process = None

@app.on_event("startup")
async def startup_event():
    start_train_server()
    # Check if necessary files exist before starting inference
    if (SERVER_DIR / "configs/moe_activation.json").exists():
        start_inference_server()

@app.on_event("shutdown")
async def shutdown_event():
    stop_processes()
    await train_client.aclose()
    await inference_client.aclose()

# Management Endpoints

@app.get("/gateway/health")
async def gateway_health():
    """Check health of gateway and downstream servers."""
    train_status = "down"
    inference_status = "down"
    
    try:
        resp = await train_client.get("/api/health")
        if resp.status_code == 200:
            train_status = "up"
    except Exception:
        pass
        
    try:
        resp = await inference_client.get("/health")
        if resp.status_code == 200:
            inference_status = "up"
    except Exception:
        pass
        
    # Get Git info
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            cwd=SERVER_DIR, 
            text=True
        ).strip()
    except:
        pass

    return {
        "gateway": "online",
        "git_commit": git_commit,
        "train_server": {
            "status": train_status,
            "pid": train_process.pid if train_process else None
        },
        "inference_server": {
            "status": inference_status,
            "pid": inference_process.pid if inference_process else None
        }
    }

@app.get("/gateway/logs/{log_type}")
async def get_system_logs(log_type: str):
    """Get system logs."""
    log_map = {
        "gateway": "gateway.log",
        "server": "server_internal.log",
        "inference": "inference_internal.log"
    }
    
    if log_type not in log_map:
        raise HTTPException(status_code=400, detail="Invalid log type.")
    
    log_file = SERVER_DIR / log_map[log_type]
    if not log_file.exists():
         raise HTTPException(status_code=404, detail="Log file not found.")
         
    return FileResponse(log_file, media_type="text/plain")

@app.post("/api/update")
async def update_server():
    """Update code and restart servers."""
    try:
        print("🔄 Gateway: Pulling code...")
        result = subprocess.run(
            ["git", "pull"], cwd=SERVER_DIR, capture_output=True, text=True, timeout=30
        )
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
            
        print("🔄 Gateway: Restarting servers...")
        stop_processes()
        time.sleep(1)
        start_train_server()
        if (SERVER_DIR / "configs/moe_activation.json").exists():
            start_inference_server()
        
        return {"status": "updated", "git_output": result.stdout}
        
    except Exception as e:
        print(f"❌ Update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Routing Logic

@app.post("/api/chat")
async def proxy_chat(request: Request):
    """Proxy chat requests to Inference Server."""
    if not inference_process:
        # Try to autostart if checkpoints exist now
        start_inference_server()
        if not inference_process:
             raise HTTPException(status_code=503, detail="Inference Server not running and no checkpoint found")
        await asyncio.sleep(5) # Wait for startup
        
    try:
        # Forward the raw JSON body
        body = await request.body()
        r = await inference_client.post(
            "/chat", 
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=30.0
        )
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
    except Exception as e:
        print(f"Inference Proxy Error: {e}")
        raise HTTPException(status_code=502, detail="Inference Server unavailable")

# Catch-all Proxy to Training Server
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path_name: str, request: Request):
    # Specialized routes check (just in case)
    if path_name == "api/update": return await update_server()
    if path_name == "gateway/health": return await gateway_health()
    if path_name == "api/chat": return await proxy_chat(request)

    # Proxy to Train Server
    url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
    
    try:
        req = train_client.build_request(
            request.method, url, headers=request.headers, content=request.stream()
        )
        r = await train_client.send(req, stream=True)
        return StreamingResponse(
            r.aiter_raw(), status_code=r.status_code, headers=r.headers, background=None
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Training Server unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=GATEWAY_PORT)
    args = parser.parse_args()
    
    print(f"🚀 Gateway Service Starting on {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
