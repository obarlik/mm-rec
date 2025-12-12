# Remote GPU Training System

## Quick Start

### 1. GPU Machine (Server)

```bash
# Install dependencies
cd server
pip install -r requirements.txt

# Start server
python train_server.py
# Server running on http://0.0.0.0:8000
```

### 2. Local Machine (Client)

```bash
# Check server health
python client/train_client.py --server http://gpu-machine:8000 --action health

# Sync code to server
python client/train_client.py --server http://gpu-machine:8000 --action sync

# Submit training job
python client/train_client.py \
  --server http://gpu-machine:8000 \
  --action submit \
  --config configs/stage1_gpu.json

# Monitor progress (real-time)
python client/train_client.py \
  --server http://gpu-machine:8000 \
  --action monitor \
  --job-id abc12345

# Download trained model
python client/train_client.py \
  --server http://gpu-machine:8000 \
  --action download \
  --job-id abc12345 \
  --output models/stage1.pt
```

## Complete Workflow

```bash
# 1. Health check
python client/train_client.py --server http://gpu-machine:8000 --action health

# 2. Sync code
python client/train_client.py --server http://gpu-machine:8000 --action sync

# 3. Submit job
JOB_ID=$(python client/train_client.py \
  --server http://gpu-machine:8000 \
  --action submit \
  --config configs/stage1_gpu.json | grep "Job submitted" | awk '{print $3}')

# 4. Monitor
python client/train_client.py \
  --server http://gpu-machine:8000 \
  --action monitor \
  --job-id $JOB_ID

# 5. Download
python client/train_client.py \
  --server http://gpu-machine:8000 \
  --action download \
  --job-id $JOB_ID \
  --output models/stage1.pt
```

## API Endpoints

- `GET /api/health` - Server health check
- `POST /api/code/sync` - Upload code archive
- `POST /api/train/submit` - Submit training job
- `GET /api/train/status/{job_id}` - Get job status
- `GET /api/train/logs/{job_id}` - Stream logs (SSE)
- `GET /api/train/download/{job_id}` - Download model
- `GET /api/jobs` - List all jobs

## Security (SSH Tunnel)

```bash
# On local machine
ssh -L 8000:localhost:8000 user@gpu-machine

# Now use localhost
python client/train_client.py --server http://localhost:8000 --action health
```

## Configuration

Edit `configs/stage1_gpu.json`:
```json
{
  "job_name": "my_training",
  "model_dim": 64,
  "num_epochs": 10,
  "batch_size": 16
}
```

## Troubleshooting

### Server not starting
```bash
# Check port
lsof -i :8000

# Try different port
uvicorn server.train_server:app --port 8001
```

### Code sync fails
```bash
# Check file size
du -sh .

# Exclude large files
# Edit client/train_client.py, add to skip list
```

### GPU not detected
```bash
# On server
python -c "import torch; print(torch.cuda.is_available())"
```
