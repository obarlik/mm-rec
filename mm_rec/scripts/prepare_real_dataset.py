#!/usr/bin/env python3
"""
Gerçek Dataset Hazırlama Scripti
Küçük bir gerçek dataset indirir ve train/val split yapar
"""

import os
import sys
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.data.text_data_loader import (
    load_texts_from_directory,
    load_text_from_file,
    create_data_loaders
)


def download_wikipedia_subset(output_dir: Path, num_articles: int = 1000):
    """
    Wikipedia'dan küçük bir subset indir (test için).
    Gerçek eğitim için daha büyük dataset gerekir.
    """
    print(f"📥 Downloading Wikipedia subset ({num_articles} articles)...")
    print("   Note: This is a small subset for testing. For real training,")
    print("   you should download full OpenWebText or C4 dataset.")
    
    # Wikipedia API kullanarak küçük bir subset al
    # Gerçek kullanım için: https://dumps.wikimedia.org/
    output_file = output_dir / "wikipedia_subset.txt"
    
    print("\n💡 Wikipedia Download Options:")
    print("   1. Manual download: https://dumps.wikimedia.org/enwiki/latest/")
    print("   2. Use Wikipedia API (limited)")
    print("   3. Use pre-processed datasets (OpenWebText, C4)")
    
    return output_file


def download_tiny_shakespeare(output_dir: Path):
    """
    Tiny Shakespeare dataset (gerçek text data, ilk eğitim için).
    """
    print("📥 Downloading Tiny Shakespeare dataset (real text data)...")
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_file = output_dir / "tiny_shakespeare.txt"
    
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {output_file}")
        print(f"   Size: {output_file.stat().st_size / (1024**2):.2f} MB")
        return output_file
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return None


def prepare_dataset_from_file(
    input_file: Path,
    output_dir: Path,
    val_split: float = 0.1,
    min_chars: int = 100
):
    """
    Tek bir text dosyasından train/val split yap.
    """
    print(f"📂 Processing: {input_file}")
    
    # Dosyayı oku
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"   Total characters: {len(text):,}")
    
    # Split
    split_idx = int(len(text) * (1 - val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Kaydet
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(train_text)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write(val_text)
    
    print(f"✅ Created train.txt: {len(train_text):,} characters")
    print(f"✅ Created val.txt: {len(val_text):,} characters")
    
    return train_file, val_file


def prepare_dataset_from_directory(
    input_dir: Path,
    output_dir: Path,
    val_split: float = 0.1
):
    """
    Bir dizindeki tüm text dosyalarından train/val split yap.
    """
    print(f"📂 Processing directory: {input_dir}")
    
    # Tüm text dosyalarını yükle
    all_texts = load_texts_from_directory(str(input_dir))
    print(f"   Found {len(all_texts)} text files")
    
    # Split
    split_idx = int(len(all_texts) * (1 - val_split))
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:] if len(all_texts) > split_idx else []
    
    # Birleştir ve kaydet
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"
    
    train_text = "\n\n".join(train_texts)
    val_text = "\n\n".join(val_texts) if val_texts else ""
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(train_text)
    
    if val_text:
        with open(val_file, 'w', encoding='utf-8') as f:
            f.write(val_text)
    
    print(f"✅ Created train.txt: {len(train_text):,} characters")
    if val_text:
        print(f"✅ Created val.txt: {len(val_text):,} characters")
    
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(description="Prepare real dataset for training")
    parser.add_argument("--output_dir", type=str, default="./data/real",
                        help="Output directory for processed dataset")
    parser.add_argument("--download_tiny_shakespeare", action="store_true",
                        help="Download Tiny Shakespeare dataset (real text data for first training)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input text file to process")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory with text files")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Real Dataset Preparation")
    print("="*80)
    print(f"📦 Output directory: {output_dir}")
    print()
    
    if args.download_tiny_shakespeare:
        # Tiny Shakespeare indir (gerçek dataset eğitimi için)
        shakespeare_file = download_tiny_shakespeare(output_dir)
        if shakespeare_file:
            prepare_dataset_from_file(shakespeare_file, output_dir, args.val_split)
    
    elif args.input_file:
        # Tek dosyadan hazırla
        input_file = Path(args.input_file)
        if not input_file.exists():
            print(f"❌ File not found: {input_file}")
            return
        prepare_dataset_from_file(input_file, output_dir, args.val_split)
    
    elif args.input_dir:
        # Dizinden hazırla
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"❌ Directory not found: {input_dir}")
            return
        prepare_dataset_from_directory(input_dir, output_dir, args.val_split)
    
    else:
        print("❌ No input specified!")
        print("\nOptions:")
        print("  1. --download_tiny_shakespeare  (real text data for first training)")
        print("  2. --input_file <file>          (process single file)")
        print("  3. --input_dir <dir>            (process directory)")
        return
    
    print()
    print("="*80)
    print("✅ Dataset preparation complete!")
    print("="*80)
    print(f"\n📋 Next steps:")
    print(f"   1. Train with: python mm_rec/scripts/train_base_model.py \\")
    print(f"      --config tiny \\")
    print(f"      --data_dir {output_dir} \\")
    print(f"      --epochs 10")
    print()


if __name__ == '__main__':
    main()
