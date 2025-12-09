#!/usr/bin/env python3
"""
Download Pre-training Data for MM-Rec

Downloads WikiText-103 and other pre-training datasets.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  datasets library not installed. Install with: pip install datasets")


def download_wikitext_103(output_dir: Path, max_samples: Optional[int] = None):
    """Download WikiText-103 dataset."""
    print(f"\n📥 Downloading WikiText-103...")
    output_file = output_dir / "wikitext_103.txt"
    
    try:
        # Load dataset
        print("   Loading dataset from Hugging Face...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        
        # Extract text
        texts = []
        for i, example in enumerate(dataset):
            text = example.get("text", "").strip()
            if text and len(text) > 50:  # Filter very short texts
                texts.append(text)
            
            if max_samples and len(texts) >= max_samples:
                break
            
            if (i + 1) % 1000 == 0:
                print(f"   Processed {i+1} examples, collected {len(texts)} texts...")
        
        # Save to file
        print(f"   Saving {len(texts)} texts to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + "\n\n")
        
        print(f"✅ WikiText-103 downloaded: {output_file}")
        print(f"   Total texts: {len(texts)}")
        print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error downloading WikiText-103: {e}")
        return None


def download_openwebtext(output_dir: Path, max_samples: Optional[int] = None):
    """Download OpenWebText dataset (if available)."""
    print(f"\n📥 Downloading OpenWebText...")
    output_file = output_dir / "openwebtext.txt"
    
    try:
        print("   Loading dataset from Hugging Face...")
        # OpenWebText is large, use streaming
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        
        texts = []
        for i, example in enumerate(dataset):
            text = example.get("text", "").strip()
            if text and len(text) > 100:
                texts.append(text)
            
            if max_samples and len(texts) >= max_samples:
                break
            
            if (i + 1) % 1000 == 0:
                print(f"   Processed {i+1} examples, collected {len(texts)} texts...")
        
        # Save to file
        print(f"   Saving {len(texts)} texts to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + "\n\n")
        
        print(f"✅ OpenWebText downloaded: {output_file}")
        print(f"   Total texts: {len(texts)}")
        
        return output_file
        
    except Exception as e:
        print(f"⚠️  OpenWebText not available: {e}")
        print("   Skipping...")
        return None


def create_synthetic_data(output_dir: Path, num_samples: int = 1000):
    """Create synthetic pre-training data if real data unavailable."""
    print(f"\n📝 Creating synthetic pre-training data...")
    output_file = output_dir / "synthetic_pretrain.txt"
    
    # Simple synthetic text generation
    import random
    
    templates = [
        "The {subject} is a {adjective} {noun} that {verb} in the {location}.",
        "{Subject} {verb} {adverb} through the {location}, {verb_ing} {noun}.",
        "In the {location}, {subject} {verb} {adjective} {noun} with {noun2}.",
    ]
    
    subjects = ["cat", "dog", "bird", "person", "car", "tree", "book", "computer"]
    adjectives = ["beautiful", "fast", "slow", "big", "small", "bright", "dark"]
    nouns = ["house", "garden", "park", "city", "mountain", "river", "ocean"]
    verbs = ["runs", "walks", "flies", "swims", "reads", "writes", "thinks"]
    locations = ["forest", "city", "beach", "mountain", "valley", "desert"]
    adverbs = ["quickly", "slowly", "carefully", "quietly", "loudly"]
    
    texts = []
    for i in range(num_samples):
        template = random.choice(templates)
        text = template.format(
            subject=random.choice(subjects),
            Subject=random.choice(subjects).capitalize(),
            adjective=random.choice(adjectives),
            noun=random.choice(nouns),
            noun2=random.choice(nouns),
            verb=random.choice(verbs),
            verb_ing=random.choice(verbs).replace("s", "ing"),
            adverb=random.choice(adverbs),
            location=random.choice(locations)
        )
        texts.append(text)
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + "\n\n")
    
    print(f"✅ Synthetic data created: {output_file}")
    print(f"   Total texts: {len(texts)}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Download Pre-training Data")
    parser.add_argument("--output_dir", type=str, default="./data/pretrain",
                        help="Output directory for pre-training data")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["wikitext", "openwebtext", "all", "synthetic"],
                        help="Dataset to download")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to download")
    parser.add_argument("--synthetic_fallback", action="store_true",
                        help="Create synthetic data if real data unavailable")
    
    args = parser.parse_args()
    
    if not DATASETS_AVAILABLE:
        print("❌ datasets library required. Install with: pip install datasets")
        if args.synthetic_fallback:
            print("   Using synthetic data fallback...")
        else:
            return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Pre-training Data Download")
    print("="*80)
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Dataset: {args.dataset}")
    
    downloaded_files = []
    
    # Download based on selection
    if args.dataset == "wikitext" or args.dataset == "all":
        file = download_wikitext_103(output_dir, args.max_samples)
        if file:
            downloaded_files.append(file)
    
    if args.dataset == "openwebtext" or args.dataset == "all":
        file = download_openwebtext(output_dir, args.max_samples)
        if file:
            downloaded_files.append(file)
    
    if args.dataset == "synthetic" or (args.synthetic_fallback and not downloaded_files):
        file = create_synthetic_data(output_dir, args.max_samples or 1000)
        if file:
            downloaded_files.append(file)
    
    # Summary
    print("\n" + "="*80)
    print("✅ Download Complete!")
    print("="*80)
    print(f"📁 Files downloaded: {len(downloaded_files)}")
    for file in downloaded_files:
        print(f"   ✅ {file}")
    
    if downloaded_files:
        print(f"\n💡 Next step: Run pre-training")
        print(f"   python mm_rec/scripts/pretrain.py --data_dir {output_dir}")
    else:
        print("\n⚠️  No files downloaded. Check errors above.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

