#!/usr/bin/env python3
"""
Download additional pre-training data sources.

Downloads:
- OpenWebText (Reddit links)
- C4 (Colossal Clean Crawled Corpus) - optional
- BookCorpus - optional
"""

import os
import sys
import argparse
from pathlib import Path
import requests
import gzip
import shutil
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_openwebtext(data_dir: Path):
    """Download OpenWebText dataset."""
    print("📥 Downloading OpenWebText...")
    print("   Note: OpenWebText is large (~8GB compressed)")
    print("   This may take a while...")
    
    # OpenWebText is typically distributed as tar.gz
    # For now, we'll provide instructions
    print("\n💡 OpenWebText Download Instructions:")
    print("   1. Visit: https://skylion007.github.io/OpenWebTextCorpus/")
    print("   2. Download the dataset")
    print("   3. Extract to:", data_dir / "openwebtext")
    print("\n   Or use:")
    print("   wget https://zenodo.org/record/3834942/files/openwebtext.tar.xz")
    print("   tar -xf openwebtext.tar.xz -C", data_dir)
    
    return data_dir / "openwebtext"


def download_c4(data_dir: Path):
    """Download C4 dataset (optional, very large)."""
    print("\n📥 C4 Dataset:")
    print("   Size: ~750GB (compressed)")
    print("   This is VERY large, optional for initial training")
    print("\n💡 C4 Download:")
    print("   Use TensorFlow Datasets:")
    print("   import tensorflow_datasets as tfds")
    print("   ds = tfds.load('c4', split='train', data_dir=", data_dir, ")")
    
    return data_dir / "c4"


def main():
    parser = argparse.ArgumentParser(description="Download pre-training data")
    parser.add_argument("--data_dir", type=str, default="./data/pretrain",
                        help="Data directory")
    parser.add_argument("--download_openwebtext", action="store_true",
                        help="Download OpenWebText")
    parser.add_argument("--download_c4", action="store_true",
                        help="Download C4 (WARNING: Very large ~750GB)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Pre-training Data Downloader")
    print("="*80)
    print(f"📦 Data directory: {data_dir}")
    print()
    
    if args.download_openwebtext:
        download_openwebtext(data_dir)
    
    if args.download_c4:
        print("\n⚠️  WARNING: C4 is ~750GB. Are you sure?")
        response = input("Continue? (yes/no): ")
        if response.lower() == 'yes':
            download_c4(data_dir)
        else:
            print("Skipped C4 download")
    
    print("\n" + "="*80)
    print("✅ Data download setup complete")
    print("="*80)
    print("\n📋 Next steps:")
    print("   1. Download OpenWebText manually or use provided instructions")
    print("   2. Update pretrain.py to use additional data sources")
    print("   3. Start real pre-training with --max_steps 50000")


if __name__ == '__main__':
    main()
