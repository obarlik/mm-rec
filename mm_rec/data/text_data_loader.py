"""
Kaliteli Text Data Loader
Gerçek text data ile eğitim için
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict
import os
from pathlib import Path
import random


class TextDataset(Dataset):
    """
    Text dataset for language modeling.
    Gerçek text dosyalarından sequence'ler oluşturur.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        seq_len: int,
        stride: Optional[int] = None
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance (must have encode method)
            seq_len: Sequence length
            stride: Stride for sliding window (None = no overlap)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        
        # Tokenize all texts
        self.tokenized_sequences = []
        for text in texts:
            tokens = tokenizer.encode(text)
            # Create sliding windows
            for i in range(0, len(tokens) - seq_len + 1, self.stride):
                window = tokens[i:i + seq_len]
                if len(window) == seq_len:
                    self.tokenized_sequences.append(window)
    
    def __len__(self):
        return len(self.tokenized_sequences)
    
    def __getitem__(self, idx):
        sequence = self.tokenized_sequences[idx]
        
        # Input: all tokens
        # Labels: next token prediction (shifted forward by 1)
        # Example: Input [t0, t1, t2, t3] -> Labels [t1, t2, t3, -100]
        input_ids = torch.tensor(sequence, dtype=torch.long)
        
        # Doğru label shifting: forward shift (circular değil!)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift forward: t[i] predicts t[i+1]
        labels[-1] = -100  # Ignore last token (no next token to predict)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class SimpleCharacterTokenizer:
    """
    Basit character-level tokenizer.
    Her karakter bir token olur.
    """
    
    def __init__(self, vocab_size: int = 5000):
        """
        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.unk_token_id = 0
        self.pad_token_id = 1
        
        # Initialize with special tokens
        self.char_to_id['<UNK>'] = self.unk_token_id
        self.char_to_id['<PAD>'] = self.pad_token_id
        self.id_to_char[self.unk_token_id] = '<UNK>'
        self.id_to_char[self.pad_token_id] = '<PAD>'
        
        self.next_id = 2
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        char_counts = {}
        for text in texts:
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
        
        # Sort by frequency
        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add most frequent chars to vocab
        for char, count in sorted_chars:
            if self.next_id < self.vocab_size:
                if char not in self.char_to_id:
                    self.char_to_id[char] = self.next_id
                    self.id_to_char[self.next_id] = char
                    self.next_id += 1
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = []
        for char in text:
            if char in self.char_to_id:
                tokens.append(self.char_to_id[char])
            else:
                tokens.append(self.unk_token_id)
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        chars = []
        for token_id in token_ids:
            if token_id in self.id_to_char:
                chars.append(self.id_to_char[token_id])
        return ''.join(chars)


def load_text_from_file(file_path: str) -> str:
    """Load text from file"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_texts_from_directory(directory: str, max_files: Optional[int] = None) -> List[str]:
    """Load texts from directory"""
    texts = []
    text_files = list(Path(directory).glob('*.txt'))
    
    if max_files:
        text_files = text_files[:max_files]
    
    for file_path in text_files:
        try:
            text = load_text_from_file(str(file_path))
            if len(text.strip()) > 0:
                texts.append(text)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return texts


def create_sample_text_corpus(output_file: str = "sample_corpus.txt", num_samples: int = 100):
    """
    Örnek text corpus oluştur (test için).
    Gerçek eğitimde bu yerine gerçek dataset kullanılmalı.
    """
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models use neural networks with multiple layers.",
        "Transformers revolutionized the field of natural language processing.",
        "Recurrent neural networks process sequences one element at a time.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Language models predict the next token in a sequence.",
        "Training large language models requires significant computational resources.",
        "Fine-tuning adapts pretrained models to specific tasks.",
    ]
    
    # Repeat and combine
    corpus = []
    for i in range(num_samples):
        corpus.append(random.choice(sample_texts))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(corpus))
    
    print(f"✅ Sample corpus created: {output_file}")
    return output_file


def create_data_loaders(
    train_texts: List[str],
    val_texts: Optional[List[str]] = None,
    tokenizer=None,
    vocab_size: int = 5000,
    seq_len: int = 512,
    batch_size: int = 4,
    train_stride: Optional[int] = None,
    val_stride: Optional[int] = None,
    num_workers: int = 0
) -> tuple[DataLoader, Optional[DataLoader], SimpleCharacterTokenizer]:
    """
    Create train and validation data loaders.
    
    Args:
        train_texts: Training texts
        val_texts: Validation texts (optional)
        tokenizer: Tokenizer instance (if None, creates SimpleCharacterTokenizer)
        vocab_size: Vocabulary size
        seq_len: Sequence length
        batch_size: Batch size
        train_stride: Stride for training (None = no overlap)
        val_stride: Stride for validation (None = no overlap)
        num_workers: Number of worker processes
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = SimpleCharacterTokenizer(vocab_size=vocab_size)
        # Build vocabulary from training texts
        tokenizer.build_vocab(train_texts)
        print(f"✅ Vocabulary built: {len(tokenizer.char_to_id)} tokens")
    
    # Create datasets
    train_dataset = TextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=train_stride
    )
    
    val_dataset = None
    if val_texts:
        val_dataset = TextDataset(
            texts=val_texts,
            tokenizer=tokenizer,
            seq_len=seq_len,
            stride=val_stride
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    print(f"✅ Train dataset: {len(train_dataset)} sequences")
    if val_dataset:
        print(f"✅ Validation dataset: {len(val_dataset)} sequences")
    
    return train_loader, val_loader, tokenizer
