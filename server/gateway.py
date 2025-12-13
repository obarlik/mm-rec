import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI, Request, HTTPException, Response, BackgroundTasks
from fastapi.responses import StreamingResponse

# Configuration
GATEWAY_PORT = 8000
TRAIN_SERVER_PORT = 8001
TRAIN_SERVER_HOST = "127.0.0.1"
TRAIN_SERVER_URL = f"http://{TRAIN_SERVER_HOST}:{TRAIN_SERVER_PORT}"
SERVER_DIR = Path(__file__).parent.parent

app = FastAPI(title="MM-Rec Training Gateway")
client = httpx.AsyncClient(base_url=TRAIN_SERVER_URL, timeout=None)

# Process Management
server_process: Optional[subprocess.Popen] = None

def start_train_server():
    """Start the training server as a subprocess."""
    global server_process
    
    # We pass the port using an environment variable or flag if the server supports it
    # Currently train_server.py has hardcoded 8001, but we will modify it to accept --port
    cmd = [sys.executable, "server/train_server.py", "--port", str(TRAIN_SERVER_PORT)]
    
    print(f"🚀 Gateway: Starting training server on port {TRAIN_SERVER_PORT}...")
    # Open logs
    log_file = open("server_internal.log", "a")
    
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
        
    return {
        "gateway": "online",
        "server_status": server_status,
        "server_pid": server_process.pid if server_process else None,
        "server_details": server_details
    }

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
