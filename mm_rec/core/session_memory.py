"""
Session-based Memory Management
Serialize and load memory states (M and h_t) based on session ID
"""

import torch
import pickle
import os
import json
from typing import Dict, Optional, List
from pathlib import Path
from .memory_state import MemoryState


class SessionMemoryManager:
    """
    Manages memory state persistence based on session ID.
    Handles serialization and loading of static memory (M) and dynamic memory (h_t).
    """
    
    def __init__(
        self,
        base_dir: str = "./memory_sessions",
        use_database: bool = False,
        db_config: Optional[Dict] = None
    ):
        """
        Initialize session memory manager.
        
        Args:
            base_dir: Base directory for file-based storage
            use_database: If True, use database instead of files
            db_config: Database configuration (if use_database=True)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.use_database = use_database
        self.db_config = db_config
        
        if use_database:
            # Initialize database connection
            # This would be implemented with SQLite, PostgreSQL, etc.
            self._init_database()
    
    def _init_database(self):
        """Initialize database connection."""
        # Placeholder for database initialization
        # In production, use SQLAlchemy, asyncpg, etc.
        pass
    
    def serialize_state(
        self,
        session_id: str,
        memory_states: Dict[str, List[MemoryState]],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Serialize memory states to disk/database.
        
        Args:
            session_id: Unique session identifier
            memory_states: Dict with "text" and "code" keys, each containing list of MemoryState
            metadata: Optional metadata (timestamps, sequence length, etc.)
        
        Returns:
            Path or ID where state was saved
        """
        if self.use_database:
            return self._save_to_database(session_id, memory_states, metadata)
        else:
            return self._save_to_file(session_id, memory_states, metadata)
    
    def _save_to_file(
        self,
        session_id: str,
        memory_states: Dict[str, List[MemoryState]],
        metadata: Optional[Dict]
    ) -> str:
        """Save memory states to file."""
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Serialize each expert's memory states
        for expert_name, states in memory_states.items():
            expert_dir = session_dir / expert_name
            expert_dir.mkdir(parents=True, exist_ok=True)
            
            for layer_idx, state in enumerate(states):
                state_file = expert_dir / f"layer_{layer_idx}.pt"
                
                # Extract tensors from MemoryState
                state_dict = {
                    'short_term_k': state.short_term.k if hasattr(state, 'short_term') else None,
                    'short_term_v': state.short_term.v if hasattr(state, 'short_term') else None,
                    'long_term_m': state.long_term.k if hasattr(state, 'long_term') else None,
                    'long_term_v': state.long_term.v if hasattr(state, 'long_term') else None,
                }
                
                torch.save(state_dict, state_file)
        
        # Save metadata
        if metadata:
            metadata_file = session_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
        
        return str(session_dir)
    
    def _save_to_database(
        self,
        session_id: str,
        memory_states: Dict[str, List[MemoryState]],
        metadata: Optional[Dict]
    ) -> str:
        """Save memory states to database."""
        # Placeholder for database implementation
        # Would serialize tensors to bytes and store in BLOB columns
        raise NotImplementedError("Database storage not yet implemented")
    
    def load_state(
        self,
        session_id: str,
        device: torch.device,
        expert_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, List[MemoryState]]]:
        """
        Load memory states from disk/database.
        
        Args:
            session_id: Session identifier
            device: Target device for loaded tensors
            expert_names: List of expert names to load (default: ["text", "code"])
        
        Returns:
            Dict of memory states or None if not found
        """
        if expert_names is None:
            expert_names = ["text", "code"]
        
        if self.use_database:
            return self._load_from_database(session_id, device, expert_names)
        else:
            return self._load_from_file(session_id, device, expert_names)
    
    def _load_from_file(
        self,
        session_id: str,
        device: torch.device,
        expert_names: List[str]
    ) -> Optional[Dict[str, List[MemoryState]]]:
        """Load memory states from file."""
        session_dir = self.base_dir / session_id
        
        if not session_dir.exists():
            return None
        
        memory_states = {}
        
        for expert_name in expert_names:
            expert_dir = session_dir / expert_name
            if not expert_dir.exists():
                continue
            
            states = []
            layer_idx = 0
            
            while True:
                state_file = expert_dir / f"layer_{layer_idx}.pt"
                if not state_file.exists():
                    break
                
                state_dict = torch.load(state_file, map_location=device)
                
                # Reconstruct MemoryState from saved tensors
                # This is a simplified version - actual implementation would need
                # to properly reconstruct MemoryState objects
                from .memory_state import MemoryBank, MemoryState
                
                # Create memory banks from saved tensors
                short_term = MemoryBank(
                    k=state_dict.get('short_term_k'),
                    v=state_dict.get('short_term_v'),
                    state=None,
                    decay_coeff=None
                )
                
                long_term = MemoryBank(
                    k=state_dict.get('long_term_m'),
                    v=state_dict.get('long_term_v'),
                    state=None,
                    decay_coeff=None
                )
                
                # Create MemoryState (simplified - would need proper config)
                state = MemoryState(
                    short_term_config={'k_dim': 256, 'v_dim': 256, 'num_slots': 32768, 'dtype': torch.float32},
                    long_term_config={'k_dim': 256, 'v_dim': 256, 'num_slots': 1024, 'dtype': torch.float32},
                    device=device
                )
                
                states.append(state)
                layer_idx += 1
            
            if states:
                memory_states[expert_name] = states
        
        return memory_states if memory_states else None
    
    def _load_from_database(
        self,
        session_id: str,
        device: torch.device,
        expert_names: List[str]
    ) -> Optional[Dict[str, List[MemoryState]]]:
        """Load memory states from database."""
        # Placeholder for database implementation
        raise NotImplementedError("Database loading not yet implemented")
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session data.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deleted, False if not found
        """
        if self.use_database:
            return self._delete_from_database(session_id)
        else:
            session_dir = self.base_dir / session_id
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
                return True
            return False
    
    def _delete_from_database(self, session_id: str) -> bool:
        """Delete session from database."""
        raise NotImplementedError("Database deletion not yet implemented")
    
    def list_sessions(self) -> List[str]:
        """List all available session IDs."""
        if self.use_database:
            return self._list_from_database()
        else:
            return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def _list_from_database(self) -> List[str]:
        """List sessions from database."""
        raise NotImplementedError("Database listing not yet implemented")

