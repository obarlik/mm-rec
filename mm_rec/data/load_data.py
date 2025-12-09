"""
Data Loader for MM-Rec Training
Loads downloaded text and code data
"""

import json
import torch
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset


class TextCodeDataset(Dataset):
    """
    Dataset for Text and Code data.
    Loads from JSONL files.
    """
    
    def __init__(
        self,
        text_file: Optional[Path] = None,
        code_file: Optional[Path] = None,
        seq_len: int = 512,
        vocab_size: int = 32000,
        synthetic: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            text_file: Path to text JSONL file
            code_file: Path to code JSONL file
            seq_len: Sequence length
            vocab_size: Vocabulary size (for synthetic data)
            synthetic: If True, generate synthetic data
            max_samples: Maximum number of samples
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.synthetic = synthetic
        
        if synthetic:
            self.text_data = None
            self.code_data = None
            self.length = 10000
        else:
            # Load real data
            self.text_data = self._load_jsonl(text_file) if text_file and text_file.exists() else []
            self.code_data = self._load_jsonl(code_file) if code_file and code_file.exists() else []
            
            # Limit samples
            if max_samples:
                self.text_data = self.text_data[:max_samples]
                self.code_data = self.code_data[:max_samples]
            
            # Use the longer dataset length
            self.length = max(len(self.text_data), len(self.code_data)) if (self.text_data or self.code_data) else 10000
    
    def _load_jsonl(self, file_path: Path) -> List[str]:
        """Load JSONL file."""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            text = json.loads(line)
                            if isinstance(text, str) and len(text) > 50:
                                texts.append(text)
                        except json.JSONDecodeError:
                            # Try as plain text
                            if len(line) > 50:
                                texts.append(line)
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
        
        return texts
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.synthetic:
            return {
                'text': torch.randint(0, self.vocab_size, (self.seq_len,)),
                'code': torch.randint(0, self.vocab_size, (self.seq_len,))
            }
        
        # Get real data
        text_seq = self.text_data[idx % len(self.text_data)] if self.text_data else None
        code_seq = self.code_data[idx % len(self.code_data)] if self.code_data else None
        
        # Simple tokenization (character-level or word-level)
        # In production, use proper tokenizer
        def simple_tokenize(text: str, vocab_size: int) -> torch.Tensor:
            """Simple tokenization (character-level hash)."""
            if text is None:
                return torch.randint(0, vocab_size, (self.seq_len,))
            
            # Convert to token IDs (simple hash-based)
            tokens = []
            for char in text[:self.seq_len * 10]:  # Take more chars for hashing
                token_id = hash(char) % vocab_size
                tokens.append(token_id)
                if len(tokens) >= self.seq_len:
                    break
            
            # Pad or truncate
            if len(tokens) < self.seq_len:
                tokens.extend([0] * (self.seq_len - len(tokens)))
            else:
                tokens = tokens[:self.seq_len]
            
            return torch.tensor(tokens, dtype=torch.long)
        
        return {
            'text': simple_tokenize(text_seq, self.vocab_size),
            'code': simple_tokenize(code_seq, self.vocab_size)
        }


def create_dataloader(
    data_dir: str = "./data",
    seq_len: int = 512,
    batch_size: int = 4,
    vocab_size: int = 32000,
    num_workers: int = 0,  # CPU için 0 önerilir
    synthetic: bool = False,
    max_samples: Optional[int] = None
):
    """
    Create DataLoader for training.
    
    Args:
        data_dir: Directory containing data files
        seq_len: Sequence length
        batch_size: Batch size
        vocab_size: Vocabulary size
        num_workers: DataLoader workers (0 for CPU)
        synthetic: Use synthetic data
        max_samples: Maximum samples per dataset
    
    Returns:
        DataLoader instance
    """
    data_path = Path(data_dir)
    
    text_file = data_path / "text" / "wikitext.jsonl" if not synthetic else None
    code_file = data_path / "code" / "code.jsonl" if not synthetic else None
    
    dataset = TextCodeDataset(
        text_file=text_file,
        code_file=code_file,
        seq_len=seq_len,
        vocab_size=vocab_size,
        synthetic=synthetic,
        max_samples=max_samples
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # CPU için 0
        pin_memory=False  # CPU için False
    )
    
    return dataloader

