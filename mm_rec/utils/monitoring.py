"""
Monitoring Hooks for MM-Rec Training
NaN/Inf detection, memory norm tracking, automatic recovery
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Callable
import warnings
from pathlib import Path
import json
from datetime import datetime


class NumericalStabilityMonitor:
    """
    Monitors numerical stability during training.
    Detects NaN/Inf in model outputs and memory states.
    """
    
    def __init__(
        self,
        check_interval: int = 10,
        alert_threshold: int = 3
    ):
        """
        Initialize monitor.
        
        Args:
            check_interval: Check every N steps
            alert_threshold: Alert after N consecutive issues
        """
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        self.issue_count = 0
        self.last_checkpoint = None
    
    def check_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        step: int
    ) -> bool:
        """
        Check outputs for NaN/Inf.
        
        Args:
            outputs: Dictionary of model outputs
            step: Current training step
        
        Returns:
            True if stable, False if issues detected
        """
        if step % self.check_interval != 0:
            return True
        
        issues = []
        
        for name, tensor in outputs.items():
            if torch.isnan(tensor).any():
                issues.append(f"NaN detected in {name}")
            if torch.isinf(tensor).any():
                issues.append(f"Inf detected in {name}")
        
        if issues:
            self.issue_count += 1
            warning_msg = f"⚠️ Step {step}: " + "; ".join(issues)
            warnings.warn(warning_msg, RuntimeWarning)
            
            if self.issue_count >= self.alert_threshold:
                raise RuntimeError(
                    f"CRITICAL: {self.issue_count} consecutive numerical stability issues detected!\n"
                    f"Last issues: {issues}\n"
                    f"Recommendation: Restore from last checkpoint and reduce learning rate."
                )
            
            return False
        
        # Reset counter on successful check
        self.issue_count = 0
        return True
    
    def check_memory_states(
        self,
        memory_states: Dict[str, List],
        step: int
    ) -> bool:
        """
        Check memory states for NaN/Inf.
        
        Args:
            memory_states: Dictionary of memory states (per expert)
            step: Current training step
        
        Returns:
            True if stable, False if issues detected
        """
        if step % self.check_interval != 0:
            return True
        
        issues = []
        
        for expert_name, states in memory_states.items():
            for layer_idx, state in enumerate(states):
                # Check short-term memory
                if hasattr(state, 'short_term') and state.short_term is not None:
                    if hasattr(state.short_term, 'k') and state.short_term.k is not None:
                        if torch.isnan(state.short_term.k).any():
                            issues.append(f"NaN in {expert_name} layer {layer_idx} short_term.k")
                        if torch.isinf(state.short_term.k).any():
                            issues.append(f"Inf in {expert_name} layer {layer_idx} short_term.k")
                
                # Check long-term memory
                if hasattr(state, 'long_term') and state.long_term is not None:
                    if hasattr(state.long_term, 'k') and state.long_term.k is not None:
                        if torch.isnan(state.long_term.k).any():
                            issues.append(f"NaN in {expert_name} layer {layer_idx} long_term.k")
                        if torch.isinf(state.long_term.k).any():
                            issues.append(f"Inf in {expert_name} layer {layer_idx} long_term.k")
        
        if issues:
            self.issue_count += 1
            warning_msg = f"⚠️ Step {step}: Memory state issues - " + "; ".join(issues)
            warnings.warn(warning_msg, RuntimeWarning)
            return False
        
        self.issue_count = 0
        return True


class MemoryNormTracker:
    """
    Tracks memory norms (M and h_t) for anomaly detection.
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        alert_threshold: float = 10.0  # Alert if norm changes by >10x
    ):
        """
        Initialize tracker.
        
        Args:
            log_interval: Log every N steps
            alert_threshold: Alert if norm changes exceed this ratio
        """
        self.log_interval = log_interval
        self.alert_threshold = alert_threshold
        self.norm_history: Dict[str, List[float]] = {}
        self.last_norms: Dict[str, float] = {}
    
    def track_norms(
        self,
        memory_states: Dict[str, List],
        step: int
    ) -> Dict[str, float]:
        """
        Track memory norms and detect anomalies.
        
        Args:
            memory_states: Dictionary of memory states
            step: Current training step
        
        Returns:
            Dictionary of current norms
        """
        current_norms = {}
        
        for expert_name, states in memory_states.items():
            for layer_idx, state in enumerate(states):
                # Track short-term memory norm (h_t)
                if hasattr(state, 'short_term') and state.short_term is not None:
                    if hasattr(state.short_term, 'k') and state.short_term.k is not None:
                        key = f"{expert_name}_layer{layer_idx}_short_term"
                        norm = state.short_term.k.norm().item()
                        current_norms[key] = norm
                        
                        # Check for sudden changes
                        if key in self.last_norms:
                            ratio = norm / (self.last_norms[key] + 1e-8)
                            if ratio > self.alert_threshold or ratio < 1.0 / self.alert_threshold:
                                warnings.warn(
                                    f"⚠️ Step {step}: Sudden norm change in {key}: "
                                    f"{self.last_norms[key]:.4f} → {norm:.4f} (ratio: {ratio:.2f}x)",
                                    RuntimeWarning
                                )
                        
                        self.last_norms[key] = norm
                
                # Track long-term memory norm (M)
                if hasattr(state, 'long_term') and state.long_term is not None:
                    if hasattr(state.long_term, 'k') and state.long_term.k is not None:
                        key = f"{expert_name}_layer{layer_idx}_long_term"
                        norm = state.long_term.k.norm().item()
                        current_norms[key] = norm
                        
                        # Check for sudden changes
                        if key in self.last_norms:
                            ratio = norm / (self.last_norms[key] + 1e-8)
                            if ratio > self.alert_threshold or ratio < 1.0 / self.alert_threshold:
                                warnings.warn(
                                    f"⚠️ Step {step}: Sudden norm change in {key}: "
                                    f"{self.last_norms[key]:.4f} → {norm:.4f} (ratio: {ratio:.2f}x)",
                                    RuntimeWarning
                                )
                        
                        self.last_norms[key] = norm
        
        # Store history
        if step % self.log_interval == 0:
            for key, norm in current_norms.items():
                if key not in self.norm_history:
                    self.norm_history[key] = []
                self.norm_history[key].append(norm)
        
        return current_norms
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get norm history."""
        return self.norm_history.copy()


class RecoveryProtocol:
    """
    Automatic recovery protocol for numerical errors.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        lr_reduction_factor: float = 0.5,
        max_recovery_attempts: int = 3
    ):
        """
        Initialize recovery protocol.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            lr_reduction_factor: Factor to reduce learning rate by
            max_recovery_attempts: Maximum recovery attempts
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.lr_reduction_factor = lr_reduction_factor
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_count = 0
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find latest checkpoint."""
        if not self.checkpoint_dir.exists():
            return None
        
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    def recover(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        error_type: str = "numerical_error"
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Attempt recovery from numerical error.
        
        Args:
            model: Model to recover
            optimizer: Optimizer to recover
            error_type: Type of error ("numerical_error", "memory_error", etc.)
        
        Returns:
            Tuple of (success, checkpoint_data)
        """
        if self.recovery_count >= self.max_recovery_attempts:
            raise RuntimeError(
                f"Maximum recovery attempts ({self.max_recovery_attempts}) exceeded. "
                f"Manual intervention required."
            )
        
        self.recovery_count += 1
        
        # Find latest checkpoint
        checkpoint_path = self.find_latest_checkpoint()
        if checkpoint_path is None:
            raise RuntimeError("No checkpoint found for recovery!")
        
        print(f"\n🔄 Recovery attempt {self.recovery_count}/{self.max_recovery_attempts}")
        print(f"   Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=next(model.parameters()).device)
        
        # Restore model
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Restore optimizer
        if 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Reduce learning rate
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * self.lr_reduction_factor
            param_group['lr'] = new_lr
            print(f"   Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        print(f"✅ Recovery completed. Resuming from step {checkpoint_data.get('step', 0)}")
        
        return True, checkpoint_data


def create_monitoring_hooks(
    model: nn.Module,
    checkpoint_dir: str = "./checkpoints"
) -> Dict[str, Callable]:
    """
    Create monitoring hooks for training.
    
    Returns:
        Dictionary of hook functions
    """
    stability_monitor = NumericalStabilityMonitor()
    norm_tracker = MemoryNormTracker()
    recovery = RecoveryProtocol(checkpoint_dir=checkpoint_dir)
    
    def forward_hook(module, input, output):
        """Hook to check forward pass outputs."""
        if isinstance(output, torch.Tensor):
            stability_monitor.check_outputs({"output": output}, step=0)  # step will be set externally
    
    def memory_hook(memory_states, step):
        """Hook to track memory norms."""
        norm_tracker.track_norms(memory_states, step)
        stability_monitor.check_memory_states(memory_states, step)
    
    return {
        "forward_hook": forward_hook,
        "memory_hook": memory_hook,
        "stability_monitor": stability_monitor,
        "norm_tracker": norm_tracker,
        "recovery": recovery
    }

