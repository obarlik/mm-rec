#!/bin/bash
#SBATCH --job-name=mmrec-100m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/mmrec-100m-%j.out
#SBATCH --error=logs/mmrec-100m-%j.err

# Load modules (adjust for your cluster)
module load cuda/11.8
module load python/3.10

# Activate virtual environment
source /path/to/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=mmrec-100m

# Create logs directory
mkdir -p logs

# Run training
python3 -m mm_rec.scripts.train_modular \
    --stage all \
    --batch_size 4 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_interval 100

echo "Training completed!"

