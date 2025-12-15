#!/usr/bin/env python3
"""
Emergency disk cleanup script
Removes old checkpoints keeping only the most recent ones
"""
import os
from pathlib import Path
import sys

WORKSPACE = Path("workspace")

def cleanup_checkpoints(keep_count=2):
    """Keep only the most recent N checkpoints per job."""
    if not WORKSPACE.exists():
        print(f"❌ Workspace directory not found: {WORKSPACE}")
        return
    
    # Find all checkpoint files
    checkpoints = list(WORKSPACE.glob("*_ckpt_epoch_*.msgpack"))
    
    # Also include .saved checkpoints
    saved_checkpoints = list(WORKSPACE.glob("*_ckpt_epoch_*.saved.msgpack"))
    
    print(f"🔍 Found {len(checkpoints)} regular checkpoints")
    print(f"🔍 Found {len(saved_checkpoints)} saved checkpoints")
    
    # Group by job_id
    job_checkpoints = {}
    for ckpt in checkpoints:
        # Extract job_id from filename (format: {job_id}_ckpt_epoch_{N}.msgpack)
        job_id = ckpt.name.split('_ckpt_')[0]
        if job_id not in job_checkpoints:
            job_checkpoints[job_id] = []
        job_checkpoints[job_id].append(ckpt)
    
    total_freed = 0
    total_removed = 0
    
    # Clean up each job
    for job_id, ckpts in job_checkpoints.items():
        # Sort by modification time (newest first)
        ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the most recent N
        to_remove = ckpts[keep_count:]
        
        if to_remove:
            print(f"\n📦 Job {job_id}: Keeping {min(keep_count, len(ckpts))} newest, removing {len(to_remove)} old checkpoints")
            
            for ckpt in to_remove:
                size = ckpt.stat().st_size
                print(f"   🗑️  Removing {ckpt.name} ({size / 1024 / 1024:.1f} MB)")
                try:
                    ckpt.unlink()
                    total_freed += size
                    total_removed += 1
                except Exception as e:
                    print(f"   ❌ Failed to remove {ckpt.name}: {e}")
    
    # Remove ALL .saved checkpoints (they're backups, not needed for training)
    if saved_checkpoints:
        print(f"\n🧹 Removing {len(saved_checkpoints)} .saved backup checkpoints...")
        for ckpt in saved_checkpoints:
            size = ckpt.stat().st_size
            print(f"   🗑️  Removing {ckpt.name} ({size / 1024 / 1024:.1f} MB)")
            try:
                ckpt.unlink()
                total_freed += size
                total_removed += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {ckpt.name}: {e}")
    
    print(f"\n✅ Cleanup complete!")
    print(f"   Files removed: {total_removed}")
    print(f"   Space freed: {total_freed / 1024 / 1024 / 1024:.2f} GB")

if __name__ == "__main__":
    keep = 2  # Keep only 2 most recent per job
    if len(sys.argv) > 1:
        keep = int(sys.argv[1])
    
    print(f"🚨 Emergency Cleanup - Keeping {keep} most recent checkpoints per job")
    cleanup_checkpoints(keep)
