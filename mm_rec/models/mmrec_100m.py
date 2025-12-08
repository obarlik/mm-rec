"""
MM-Rec 100M Parameter Modular Model
256 channels, 2 experts (Text and Code)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from ..core.memory_state import MemoryState, MemoryBankConfig
from ..blocks.mm_rec_block import MMRecBlock
from ..blocks.mm_rec_block import RMSNorm


class ExpertModule(nn.Module):
    """
    Expert module for specialized domains (Text or Code).
    Each expert has its own FFN and memory channels.
    """
    
    def __init__(
        self,
        model_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        num_memories: int = 1,
        mem_dim: int = 256,
        max_seq_len: int = 32768,
        dropout: float = 0.1,
        expert_name: str = "expert"
    ):
        super().__init__()
        self.expert_name = expert_name
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.mem_dim = mem_dim
        
        # MM-Rec blocks for this expert
        self.blocks = nn.ModuleList([
            MMRecBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                num_memories=num_memories,
                mem_dim=mem_dim,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Expert-specific normalization
        self.norm = RMSNorm(model_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        memory_states: Optional[List[MemoryState]] = None,
        chunk_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[MemoryState]]:
        """
        Forward pass through expert blocks.
        
        Args:
            x: Input [batch, seq_len, model_dim]
            memory_states: List of MemoryState (one per layer)
            chunk_size: Optional chunking for long sequences
        
        Returns:
            output: [batch, seq_len, model_dim]
            updated_states: List of updated MemoryState
        """
        if memory_states is None:
            batch_size = x.shape[0]
            device = x.device
            memory_states = [
                self._create_memory_state(batch_size, device)
                for _ in range(self.num_layers)
            ]
        
        new_memory_states = []
        for i, block in enumerate(self.blocks):
            x, updated_state = block(x, memory_states[i], use_checkpointing=False)
            new_memory_states.append(updated_state)
        
        x = self.norm(x)
        return x, new_memory_states
    
    def _create_memory_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """Create memory state for this expert."""
        M = 1024  # Long-term memory size
        
        short_term_config = {
            'k_dim': self.model_dim,
            'v_dim': self.model_dim,
            'num_slots': 32768,
            'dtype': torch.float32
        }
        
        long_term_config = {
            'k_dim': self.mem_dim,
            'v_dim': self.mem_dim,
            'num_slots': M,
            'dtype': torch.float32
        }
        
        return MemoryState(
            short_term_config=short_term_config,
            long_term_config=long_term_config,
            device=device
        )


class FusionLayer(nn.Module):
    """
    Fusion layer that combines two 256-channel experts into 512 channels.
    Used during inference to merge Text and Code expert memories.
    """
    
    def __init__(
        self,
        expert_dim: int = 256,
        fused_dim: int = 512,
        fusion_method: str = "concatenate"
    ):
        super().__init__()
        self.expert_dim = expert_dim
        self.fused_dim = fused_dim
        self.fusion_method = fusion_method
        
        if fusion_method == "concatenate":
            # Simple concatenation: 256 + 256 = 512
            self.fusion_proj = nn.Linear(expert_dim * 2, fused_dim)
        elif fusion_method == "weighted":
            # Weighted combination
            self.fusion_proj = nn.Linear(expert_dim * 2, fused_dim)
            self.gate = nn.Sequential(
                nn.Linear(expert_dim * 2, expert_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(
        self,
        text_output: torch.Tensor,
        code_output: torch.Tensor,
        text_memory: Optional[torch.Tensor] = None,
        code_memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse two expert outputs into combined representation.
        
        Args:
            text_output: Text expert output [batch, seq_len, 256]
            code_output: Code expert output [batch, seq_len, 256]
            text_memory: Optional text expert memory [batch, M, 256]
            code_memory: Optional code expert memory [batch, M, 256]
        
        Returns:
            fused: [batch, seq_len, 512]
        """
        # Concatenate expert outputs
        if text_memory is not None and code_memory is not None:
            # Include memory in fusion
            text_combined = torch.cat([text_output, text_memory.mean(dim=1, keepdim=True).expand_as(text_output)], dim=-1)
            code_combined = torch.cat([code_output, code_memory.mean(dim=1, keepdim=True).expand_as(code_output)], dim=-1)
            combined = torch.cat([text_combined, code_combined], dim=-1)
        else:
            combined = torch.cat([text_output, code_output], dim=-1)
        
        # Apply fusion projection
        if self.fusion_method == "weighted":
            gate_weights = self.gate(combined)
            combined = combined * gate_weights
        
        fused = self.fusion_proj(combined)
        return fused


class MMRec100M(nn.Module):
    """
    MM-Rec 100M Parameter Modular Model
    
    Architecture:
    - 2 Experts: Text Expert (256 channels) + Code Expert (256 channels)
    - Fusion Layer: Combines experts to 512 channels
    - Total: ~100M parameters
    
    Configuration:
    - model_dim: 256 per expert (512 after fusion)
    - num_layers: 8 per expert
    - num_heads: 8 per expert
    - ffn_dim: 1024 per expert
    - vocab_size: 32000
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        expert_dim: int = 256,
        num_layers: int = 16,  # Increased for ~100M params
        num_heads: int = 8,
        ffn_dim: int = 3072,  # Increased for ~100M params (256 * 12)
        num_memories: int = 1,
        mem_dim: int = 256,
        max_seq_len: int = 32768,
        dropout: float = 0.1,
        fusion_method: str = "concatenate"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.expert_dim = expert_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.fused_dim = expert_dim * 2  # 512 after fusion
        
        # Shared embedding (both experts use same vocabulary)
        self.embedding = nn.Embedding(vocab_size, expert_dim)
        
        # Two expert modules
        self.text_expert = ExpertModule(
            model_dim=expert_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_memories=num_memories,
            mem_dim=mem_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            expert_name="text"
        )
        
        self.code_expert = ExpertModule(
            model_dim=expert_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_memories=num_memories,
            mem_dim=mem_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            expert_name="code"
        )
        
        # Fusion layer (combines 256+256 → 512)
        self.fusion = FusionLayer(
            expert_dim=expert_dim,
            fused_dim=self.fused_dim,
            fusion_method=fusion_method
        )
        
        # Final layers (after fusion)
        self.final_norm = RMSNorm(self.fused_dim)
        self.lm_head = nn.Linear(self.fused_dim, vocab_size, bias=False)
        
        # Note: Weight tying between embedding and lm_head is complex for modular model
        # Each expert uses 256 dims, fusion creates 512 dims
        # For simplicity, we keep them separate (can be tied later if needed)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        expert_type: Optional[str] = None,  # "text", "code", or None (both)
        memory_states: Optional[Dict[str, List[MemoryState]]] = None,
        chunk_size: Optional[int] = None,
        return_memory: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through modular MM-Rec model.
        
        Args:
            input_ids: [batch, seq_len]
            expert_type: "text", "code", or None (use both and fuse)
            memory_states: Dict with "text" and "code" keys
            chunk_size: Optional chunking for long sequences
            return_memory: If True, return memory states
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            (optional) memory_states: Dict of memory states
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embedding
        x = self.embedding(input_ids)  # [batch, seq_len, 256]
        
        # Initialize memory states if not provided
        if memory_states is None:
            memory_states = {
                "text": None,
                "code": None
            }
        
        # Route to experts
        if expert_type == "text":
            # Text expert only
            text_output, text_states = self.text_expert(x, memory_states.get("text"), chunk_size)
            code_output = torch.zeros_like(text_output)
            code_states = []
        elif expert_type == "code":
            # Code expert only
            code_output, code_states = self.code_expert(x, memory_states.get("code"), chunk_size)
            text_output = torch.zeros_like(code_output)
            text_states = []
        else:
            # Both experts (training or inference with fusion)
            text_output, text_states = self.text_expert(x, memory_states.get("text"), chunk_size)
            code_output, code_states = self.code_expert(x, memory_states.get("code"), chunk_size)
        
        # Fusion layer (combines 256+256 → 512)
        fused = self.fusion(text_output, code_output)
        
        # Final layers
        fused = self.final_norm(fused)  # [batch, seq_len, 512]
        logits = self.lm_head(fused)  # [batch, seq_len, vocab_size]
        
        if return_memory:
            return logits, {
                "text": text_states,
                "code": code_states
            }
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_expert_params(self) -> Dict[str, int]:
        """Get parameter count per expert."""
        return {
            "text": sum(p.numel() for p in self.text_expert.parameters() if p.requires_grad),
            "code": sum(p.numel() for p in self.code_expert.parameters() if p.requires_grad),
            "fusion": sum(p.numel() for p in self.fusion.parameters() if p.requires_grad),
            "embedding": self.embedding.weight.numel(),
            "lm_head": self.lm_head.weight.numel(),
            "total": self.get_num_params()
        }

