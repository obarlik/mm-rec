#!/usr/bin/env python3
"""
Automatic Training Script
Downloads data and trains on CPU automatically
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Main function to download data and train automatically."""
    
    print("="*80)
    print("MM-Rec 100M Automatic Training (CPU)")
    print("="*80)
    print("\nBu script:")
    print("  1. İnternetten veri indirir (WikiText, Code datasets)")
    print("  2. Veriyi işler ve hazırlar")
    print("  3. CPU'da eğitime başlar")
    print("\n⚠️  Not: CPU eğitimi GPU'dan çok daha yavaştır.")
    print("   Bu script test ve geliştirme için tasarlanmıştır.\n")
    
    # Check if data exists
    data_dir = Path("./data")
    text_file = data_dir / "text" / "wikitext.jsonl"
    code_file = data_dir / "code" / "code.jsonl"
    
    if not text_file.exists() or not code_file.exists():
        print("📥 Veri bulunamadı. İndiriliyor...")
        print("   Bu işlem birkaç dakika sürebilir.\n")
        
        # Download data
        download_cmd = [
            sys.executable,
            "-m", "mm_rec.data.download_data",
            "--output_dir", "./data",
            "--text_samples", "500",
            "--code_samples", "500"
        ]
        
        result = subprocess.run(download_cmd, capture_output=False)
        if result.returncode != 0:
            print("❌ Veri indirme başarısız!")
            return 1
        
        print("\n✅ Veri indirme tamamlandı!\n")
    else:
        print("✅ Veri zaten mevcut, indirme atlanıyor.\n")
    
    # Start training
    print("🚀 Eğitime başlanıyor...")
    print("   Stage 1: Local Consistency (CPU-optimized)\n")
    
    train_cmd = [
        sys.executable,
        "-m", "mm_rec.scripts.train_cpu",
        "--stage", "stage1",
        "--batch_size", "2",
        "--checkpoint_dir", "./checkpoints_cpu",
        "--checkpoint_interval", "50",
        "--data_dir", "./data",
        "--max_samples", "500"
    ]
    
    result = subprocess.run(train_cmd)
    
    if result.returncode == 0:
        print("\n✅ Eğitim tamamlandı!")
        print(f"   Checkpoint'ler: ./checkpoints_cpu/")
        return 0
    else:
        print("\n❌ Eğitim sırasında hata oluştu!")
        return 1


if __name__ == '__main__':
    sys.exit(main())

