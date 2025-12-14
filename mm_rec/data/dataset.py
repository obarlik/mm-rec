import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .chat_format import ChatFormatter, ChatMessage

class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.
    Handles tokenization of chat conversations.
    """
    def __init__(self, conversations: List[List[ChatMessage]], tokenizer, config: Any):
        """
        Args:
            conversations: List of conversation histories (List[ChatMessage])
            tokenizer: Tokenizer (tiktoken/OpenAI-compatible)
            config: SFTConfig object (must have max_length, only_predict_assistant, ignore_index)
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.config = config
        self.chat_formatter = ChatFormatter()
    
    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        messages = self.conversations[idx]
        
        # --- Logic extracted from SFTTrainer.prepare_chat_input ---
        
        # Format messages
        input_text, target_text = self.chat_formatter.create_training_pairs(
            messages,
            only_assistant=self.config.only_predict_assistant
        )
        
        # Tokenize (returns list of ints)
        # tiktoken encode() only accepts 'text' and 'allowed_special'/'disallowed_special'
        # It does not support max_length, truncation, padding natively.
        # We allow all special tokens (like <|endoftext|>) to pass through.
        input_ids = self.tokenizer.encode(input_text, allowed_special="all")
        
        # Manual Truncation
        if len(input_ids) > self.config.max_length:
            input_ids = input_ids[:self.config.max_length]
        
        # Create labels
        if self.config.only_predict_assistant:
            # Tokenize full input to find assistant token position
            full_input_ids = self.tokenizer.encode(input_text, allowed_special="all")
            
            # Assistant token
            assistant_token = self.chat_formatter.assistant_token
            # Assistant token usually doesn't contain special tokens, but safe to allow
            assistant_token_ids = self.tokenizer.encode(assistant_token, allowed_special="all")
            
            # Find assistant token position
            assistant_pos = -1
            if len(assistant_token_ids) > 0:
                # Naive search
                full_ids_list = list(full_input_ids)
                asst_ids_list = list(assistant_token_ids)
                
                for i in range(len(full_ids_list) - len(asst_ids_list) + 1):
                    if full_ids_list[i:i+len(asst_ids_list)] == asst_ids_list:
                        assistant_pos = i + len(asst_ids_list)
                        break
            
            # Create labels: -100 (ignore_index)
            labels = [self.config.ignore_index] * len(input_ids)
            
            if assistant_pos >= 0:
                # Tokenize target
                target_ids = self.tokenizer.encode(target_text, allowed_special="all")
                
                # Manual Truncation for Target (if needed within context)
                # Usually we just fit it into the buffer
                
                # Match truncated input
                truncated_ids = list(input_ids)
                asst_ids_list = list(assistant_token_ids)
                
                assistant_in_truncated = False
                for i in range(len(truncated_ids) - len(asst_ids_list) + 1):
                    if truncated_ids[i:i+len(asst_ids_list)] == asst_ids_list:
                        assistant_start_idx = i + len(asst_ids_list)
                        
                        remaining_len = len(truncated_ids) - assistant_start_idx
                        if remaining_len > 0 and len(target_ids) > 0:
                            label_len = min(remaining_len, len(target_ids))
                            labels[assistant_start_idx:assistant_start_idx+label_len] = target_ids[:label_len]
                            assistant_in_truncated = True
                        break
                
                if not assistant_in_truncated and len(target_ids) > 0:
                     # This fallback is tricky with tiktoken mismatch, simpler to just skip or naive logic
                     # For now, if assistant token was truncated out, we probably shouldn't learn from this sample
                     # But falling back to "learn last part" is existing logic
                     label_len = min(len(input_ids), len(target_ids))
                     labels[-label_len:] = target_ids[:label_len]
            else:
                # Fallback
                labels = list(input_ids)
        else:
            labels = list(input_ids)
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


@dataclass
class SFTDataCollator:
    """
    Data collator for SFT.
    Pads input_ids and labels.
    """
    tokenizer: Any
    ignore_index: int = -100
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Extract
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad
        if hasattr(self.tokenizer, 'pad_token_id'):
            pad_id = self.tokenizer.pad_token_id
        elif hasattr(self.tokenizer, 'eot_token'):
            pad_id = self.tokenizer.eot_token
        else:
            pad_id = 0 # Fallback for simple tokenizers
        
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index
        )
        
        # Mask
        attention_mask = (input_ids_padded != pad_id).long()
        
        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask
        }
