# Deployment Workflow

We utilize a **Git-Based Deployment** strategy. This ensures version control and consistency between the local development environment and the Phoenix training server.

## Standard Workflow

### 1. Local Development
Make changes locally, commit, and push to GitHub.

```bash
git add .
git commit -m "Your feature description"
git push origin main
```

### 2. Deploy to Phoenix
Connect to the server and pull changes.

```bash
ssh onurbarlik@hotmail.com@phoenix
cd ~/mm-rec-training
git pull origin main
```

### 3. Run Training/Benchmarks
Execute the training script remotely.

**JAX Training (High Performance):**
```bash
python mm_rec_jax/training/train_server_jax.py
```

**PyTorch Training (Legacy):**
```bash
python server/train_server.py
```

## Legacy Scripts (Deprecated)
The following scripts in the root directory are considered deprecated for active development but kept for reference:
- `deploy_to_phoenix.sh` (Rsync based)
- `run_jax_benchmark_phoenix.sh` (SSH remote exec based)

**Please prefer the Git workflow.**
