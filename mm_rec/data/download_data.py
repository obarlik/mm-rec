"""
Automatic Data Downloader for MM-Rec Training
Downloads text and code data from various sources
"""

import os
import sys
import requests
import json
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DataDownloader:
    """
    Downloads text and code data from various sources.
    """
    
    def __init__(self, output_dir: str = "./data"):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_dir = self.output_dir / "text"
        self.code_dir = self.output_dir / "code"
        self.text_dir.mkdir(exist_ok=True)
        self.code_dir.mkdir(exist_ok=True)
    
    def download_huggingface_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        max_samples: Optional[int] = None,
        data_type: str = "text"  # "text" or "code"
    ) -> List[str]:
        """
        Download dataset from Hugging Face.
        
        Args:
            dataset_name: Hugging Face dataset name (e.g., "wikitext", "the_pile")
            split: Dataset split ("train", "validation", etc.)
            text_column: Column name containing text
            max_samples: Maximum number of samples to download
            data_type: "text" or "code"
        
        Returns:
            List of text samples
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("⚠️ datasets library not installed. Installing...")
            os.system("pip install datasets")
            from datasets import load_dataset
        
        print(f"\n📥 Downloading {dataset_name} from Hugging Face...")
        print(f"   Split: {split}, Column: {text_column}, Type: {data_type}")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, split=split, streaming=False)
            
            # Limit samples if specified
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Extract text
            texts = []
            for item in tqdm(dataset, desc="Processing"):
                if text_column in item:
                    text = item[text_column]
                    if isinstance(text, str) and len(text) > 100:  # Filter short texts
                        texts.append(text)
            
            print(f"✅ Downloaded {len(texts)} samples")
            return texts
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            return []
    
    def download_wikitext(self, max_samples: int = 1000) -> List[str]:
        """Download WikiText dataset."""
        return self.download_huggingface_dataset(
            "wikitext",
            split="train",
            text_column="text",
            max_samples=max_samples,
            data_type="text"
        )
    
    def download_code_dataset(self, max_samples: int = 1000) -> List[str]:
        """Download code dataset (e.g., The Stack)."""
        # Try multiple code datasets
        datasets_to_try = [
            ("bigcode/the-stack", "train", "content"),
            ("bigcode/python_code", "train", "content"),
            ("code_search_net", "train", "code"),
        ]
        
        all_code = []
        for dataset_name, split, column in datasets_to_try:
            try:
                code = self.download_huggingface_dataset(
                    dataset_name,
                    split=split,
                    text_column=column,
                    max_samples=max_samples // len(datasets_to_try),
                    data_type="code"
                )
                all_code.extend(code)
                if len(all_code) >= max_samples:
                    break
            except Exception as e:
                print(f"⚠️ Could not download {dataset_name}: {e}")
                continue
        
        return all_code[:max_samples]
    
    def download_from_url(self, url: str, output_file: Path) -> bool:
        """Download file from URL."""
        try:
            print(f"📥 Downloading from {url}...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f, tqdm(
                desc=output_file.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            print(f"✅ Downloaded: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return False
    
    def save_texts(self, texts: List[str], filename: str, data_type: str = "text"):
        """
        Save texts to file.
        
        Args:
            texts: List of text strings
            filename: Output filename
            data_type: "text" or "code"
        """
        output_dir = self.text_dir if data_type == "text" else self.code_dir
        output_file = output_dir / filename
        
        # Save as JSON (one text per line for easy loading)
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                # Escape newlines for JSON
                text_escaped = json.dumps(text)
                f.write(text_escaped + '\n')
        
        print(f"💾 Saved {len(texts)} samples to {output_file}")
    
    def download_all(
        self,
        text_samples: int = 1000,
        code_samples: int = 1000,
        use_wikitext: bool = True,
        use_code_datasets: bool = True
    ):
        """
        Download all data sources.
        
        Args:
            text_samples: Number of text samples to download
            code_samples: Number of code samples to download
            use_wikitext: Download WikiText dataset
            use_code_datasets: Download code datasets
        """
        print("="*80)
        print("MM-Rec Data Downloader")
        print("="*80)
        
        # Download text data
        if use_wikitext:
            print("\n📚 Downloading Text Data...")
            text_data = self.download_wikitext(max_samples=text_samples)
            if text_data:
                self.save_texts(text_data, "wikitext.jsonl", data_type="text")
        
        # Download code data
        if use_code_datasets:
            print("\n💻 Downloading Code Data...")
            code_data = self.download_code_dataset(max_samples=code_samples)
            if code_data:
                self.save_texts(code_data, "code.jsonl", data_type="code")
        
        print("\n✅ Data download completed!")
        print(f"   Text data: {self.text_dir}")
        print(f"   Code data: {self.code_dir}")


def main():
    parser = argparse.ArgumentParser(description='Download training data for MM-Rec')
    
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for downloaded data')
    parser.add_argument('--text_samples', type=int, default=1000,
                       help='Number of text samples to download')
    parser.add_argument('--code_samples', type=int, default=1000,
                       help='Number of code samples to download')
    parser.add_argument('--no_wikitext', action='store_true',
                       help='Skip WikiText download')
    parser.add_argument('--no_code', action='store_true',
                       help='Skip code datasets download')
    
    args = parser.parse_args()
    
    downloader = DataDownloader(output_dir=args.output_dir)
    downloader.download_all(
        text_samples=args.text_samples,
        code_samples=args.code_samples,
        use_wikitext=not args.no_wikitext,
        use_code_datasets=not args.no_code
    )


if __name__ == '__main__':
    main()

