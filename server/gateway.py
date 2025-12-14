import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI, Request, HTTPException, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse

# Configuration
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
    
    # Hardcoded/Default paths for now - ideally configurable
    # We use CPU for inference to avoid VRAM clash
    env = os.environ.copy()
    env["JAX_PLATFORM_NAME"] = "cpu"
    
    # Check for model/config availability
    # Fallback to defaults if specific files don't exist?
    # For Phase 9, we assume standard paths
    model_path = "checkpoints/last_ckpt.msgpack" # or best_model.msgpack
    config_path = "configs/moe_activation.json"
    
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
    else:
        print("⚠️ Gateway: Config not found, skipping Inference Server start.")

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
        raise HTTPException(status_code=503, detail="Inference Server not running")
        
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
    # Specialized routes
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
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=GATEWAY_PORT)
    args = parser.parse_args()
    
    print(f"🚀 Gateway Service Starting on {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

def start_train_server():
    """Start the training server as a subprocess."""
    global server_process
    
    # We pass the port using an environment variable or flag if the server supports it
    # Currently train_server.py has hardcoded 8001, but we will modify it to accept --port
    cmd = [sys.executable, "server/train_server.py", "--port", str(TRAIN_SERVER_PORT)]
    
    print(f"🚀 Gateway: Starting training server on port {TRAIN_SERVER_PORT}...")
    # Open logs
    log_file = open(SERVER_DIR / "server_internal.log", "a")
    
    server_process = subprocess.Popen(
        cmd,
        cwd=SERVER_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return server_process

def stop_train_server():
    """Stop the training server."""
    global server_process
    if server_process and server_process.poll() is None:
        print("🛑 Gateway: Stopping training server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
    server_process = None

@app.on_event("startup")
async def startup_event():
    start_train_server()

@app.on_event("shutdown")
async def shutdown_event():
    stop_train_server()
    await client.aclose()

# Management Endpoints

@app.get("/gateway/health")
async def gateway_health():
    """Check health of gateway and downstream server."""
    server_status = "down"
    server_details = {}
    
    try:
        resp = await client.get("/api/health")
        if resp.status_code == 200:
            server_status = "up"
            server_details = resp.json()
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
        "server_status": server_status,
        "server_pid": server_process.pid if server_process else None,
        "server_details": server_details
    }

@app.get("/gateway/logs/{log_type}")
async def get_system_logs(log_type: str):
    """Get system logs (gateway or server)."""
    if log_type == "gateway":
        log_file = SERVER_DIR / "gateway.log"
    elif log_type == "server":
        # Log is created in CWD=SERVER_DIR
        log_file = SERVER_DIR / "server_internal.log"
    else:
        raise HTTPException(status_code=400, detail="Invalid log type. Use 'gateway' or 'server'.")
        
    if not log_file.exists():
         raise HTTPException(status_code=404, detail=f"Log file not found: {log_file}")
         
    return FileResponse(log_file, media_type="text/plain")

@app.post("/api/update")
async def update_server():
    """Update code via git and restart the training server."""
    try:
        # 1. Pull Code
        print("🔄 Gateway: Pulling latest code...")
        result = subprocess.run(
            ["git", "pull"],
            cwd=SERVER_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Git pull failed: {result.stderr}")
            
        git_output = result.stdout
        
        # 2. Restart Server
        print("🔄 Gateway: Restarting training server...")
        stop_train_server()
        time.sleep(1) # Give it a moment to release ports
        start_train_server()
        
        return {
            "status": "updated", 
            "message": "Code updated and server restarted.",
            "git_output": git_output
        }
        
    except Exception as e:
        print(f"❌ Update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Proxy Logic
async def proxy_request(request: Request):
    """Proxy request to the training server."""
    if not server_process or server_process.poll() is not None:
        # Auto-restart if dead?
        if not server_process or server_process.poll() is not None:
             print("⚠️ Gateway: Server found dead, restarting...")
             start_train_server()
             await asyncio.sleep(2) # Wait for startup

    url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
    
    # Exclude gateway-specific headers if any
    headers = dict(request.headers)
    headers.pop("host", None) # Let httpx set the correct host
    
    try:
        req = client.build_request(
            request.method,
            url,
            headers=headers,
            content=request.stream(),
        )
        r = await client.send(req, stream=True)
        
        return StreamingResponse(
            r.aiter_raw(),
            status_code=r.status_code,
            headers=r.headers,
            background=None
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Training server unavailable (starting up?)")
    except Exception as e:
        print(f"Proxy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Catch-all proxy
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path_name: str, request: Request):
    # Specialized handling for update, otherwise proxy
    if path_name == "api/update":
        return await update_server()
    if path_name == "gateway/health": # Should be caught by specific route but just in case
        return await gateway_health()
        
    return await proxy_request(request)

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=GATEWAY_PORT)
    args = parser.parse_args()
    
    print(f"🚀 Starting Gateway on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
