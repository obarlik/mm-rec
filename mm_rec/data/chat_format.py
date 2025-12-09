"""
OpenAI-Compatible Chat Format Support
Handles system/user/assistant message formatting for SFT training
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """Single chat message."""
    role: str  # "system", "user", "assistant"
    content: str


class ChatFormatter:
    """
    Formats chat conversations for OpenAI-compatible training.
    Supports OpenAI Chat API format.
    """
    
    def __init__(
        self,
        system_token: str = "<|system|>",
        user_token: str = "<|user|>",
        assistant_token: str = "<|assistant|>",
        eos_token: str = "<|endoftext|>"
    ):
        """
        Initialize chat formatter.
        
        Args:
            system_token: System message token
            user_token: User message token
            assistant_token: Assistant message token
            eos_token: End of text token
        """
        self.system_token = system_token
        self.user_token = user_token
        self.assistant_token = assistant_token
        self.eos_token = eos_token
    
    def format_messages(self, messages: List[ChatMessage]) -> str:
        """
        Format messages into a single string for training.
        
        Args:
            messages: List of ChatMessage objects
        
        Returns:
            Formatted string
        """
        formatted_parts = []
        
        for msg in messages:
            if msg.role == "system":
                formatted_parts.append(f"{self.system_token}\n{msg.content}\n{self.eos_token}")
            elif msg.role == "user":
                formatted_parts.append(f"{self.user_token}\n{msg.content}\n{self.eos_token}")
            elif msg.role == "assistant":
                formatted_parts.append(f"{self.assistant_token}\n{msg.content}\n{self.eos_token}")
        
        return "\n".join(formatted_parts)
    
    def parse_openai_format(self, data: Dict) -> List[ChatMessage]:
        """
        Parse OpenAI Chat API format.
        
        Args:
            data: OpenAI format dict with "messages" key
        
        Returns:
            List of ChatMessage objects
        """
        messages = []
        
        if "messages" in data:
            for msg in data["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append(ChatMessage(role=role, content=content))
        
        return messages
    
    def create_training_pairs(
        self,
        messages: List[ChatMessage],
        only_assistant: bool = True
    ) -> Tuple[str, str]:
        """
        Create (input, target) pairs for training.
        
        Args:
            messages: List of ChatMessage objects
            only_assistant: If True, only predict assistant responses
        
        Returns:
            (input_text, target_text) tuple
        """
        if only_assistant:
            # Only predict assistant responses
            input_parts = []
            target_parts = []
            
            for i, msg in enumerate(messages):
                if msg.role == "assistant":
                    # Input: everything before this assistant message
                    input_text = self.format_messages(messages[:i])
                    # Target: this assistant message
                    target_text = f"{self.assistant_token}\n{msg.content}\n{self.eos_token}"
                    return input_text, target_text
            
            # Fallback: predict from all messages
            full_text = self.format_messages(messages)
            return full_text, ""
        else:
            # Predict all tokens
            full_text = self.format_messages(messages)
            return full_text, full_text


class ChatDataset:
    """
    Dataset for OpenAI-compatible chat format training.
    """
    
    def __init__(
        self,
        data_file: str,
        formatter: Optional[ChatFormatter] = None,
        max_length: int = 2048
    ):
        """
        Initialize chat dataset.
        
        Args:
            data_file: Path to JSONL file with OpenAI format
            formatter: ChatFormatter instance
            max_length: Maximum sequence length
        """
        self.data_file = data_file
        self.formatter = formatter or ChatFormatter()
        self.max_length = max_length
        self.conversations = self._load_conversations()
    
    def _load_conversations(self) -> List[List[ChatMessage]]:
        """Load conversations from JSONL file."""
        conversations = []
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    messages = self.formatter.parse_openai_format(data)
                    if messages:
                        conversations.append(messages)
                except json.JSONDecodeError:
                    continue
        
        return conversations
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        messages = self.conversations[idx]
        input_text, target_text = self.formatter.create_training_pairs(messages)
        
        return {
            'input': input_text,
            'target': target_text,
            'messages': messages
        }


def create_chat_example() -> Dict:
    """
    Create example OpenAI-compatible chat format.
    
    Returns:
        Example dict in OpenAI format
    """
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."}
        ]
    }

