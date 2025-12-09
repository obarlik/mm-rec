"""
Adaptive Learning Rate Scheduler
Dynamic learning rate adjustment based on loss/metrics
"""

import torch
import torch.optim as optim
from typing import Optional, Union
import warnings


class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduler with dynamic adjustments.
    
    Automatically adjusts learning rate based on:
    - Loss plateau detection
    - Gradient norm monitoring
    - Validation metrics (if provided)
    
    Features:
    - Plateau detection: Reduces LR when loss stops improving
    - Gradient-based adjustment: Adjusts LR based on gradient norm
    - Minimum LR protection: Prevents LR from going too low
    - Patience mechanism: Waits before reducing LR
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        mode: str = 'min',  # 'min' for loss, 'max' for accuracy
        factor: float = 0.5,  # LR reduction factor
        patience: int = 10,  # Steps to wait before reducing LR
        threshold: float = 0.0001,  # Minimum change to qualify as improvement
        min_lr: float = 1e-6,  # Minimum learning rate
        verbose: bool = True,
        cooldown: int = 0,  # Steps to wait after LR reduction
        eps: float = 1e-8
    ):
        """
        Initialize adaptive learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            mode: 'min' (minimize loss) or 'max' (maximize metric)
            factor: Factor to multiply LR by when reducing
            patience: Number of steps with no improvement before reducing LR
            threshold: Minimum change to qualify as improvement
            min_lr: Minimum learning rate
            verbose: Print LR changes
            cooldown: Steps to wait after LR reduction before resuming patience
            eps: Small epsilon for numerical stability
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        self.cooldown = cooldown
        self.eps = eps
        
        # State
        self.best_metric: Optional[float] = None
        self.patience_counter = 0
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.last_lr_reduction_step = -1
        
        # Track LR history
        self.lr_history = []
        
    def step(self, metric: float, step: Optional[int] = None) -> bool:
        """
        Update learning rate based on metric.
        
        Args:
            metric: Current metric value (loss or accuracy)
            step: Current training step (optional, for logging)
        
        Returns:
            True if LR was reduced, False otherwise
        """
        # Handle cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        # Initialize best metric
        if self.best_metric is None:
            self.best_metric = metric
            self._save_lr_history()
            return False
        
        # Check if metric improved
        if self.mode == 'min':
            improved = metric < (self.best_metric - self.threshold)
        else:  # mode == 'max'
            improved = metric > (self.best_metric + self.threshold)
        
        if improved:
            # Metric improved - reset counters
            self.best_metric = metric
            self.patience_counter = 0
            self.num_bad_epochs = 0
            self._save_lr_history()
            return False
        else:
            # Metric did not improve
            self.patience_counter += 1
            self.num_bad_epochs += 1
            
            # Reduce LR if patience exceeded
            if self.patience_counter >= self.patience:
                lr_reduced = self._reduce_lr(step)
                if lr_reduced:
                    self.patience_counter = 0
                    self.cooldown_counter = self.cooldown
                    self.last_lr_reduction_step = step if step is not None else -1
                return lr_reduced
        
        self._save_lr_history()
        return False
    
    def _reduce_lr(self, step: Optional[int] = None) -> bool:
        """Reduce learning rate for all parameter groups."""
        reduced = False
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            
            # Check if LR is already at minimum
            if old_lr <= self.min_lr + self.eps:
                continue
            
            # Calculate new LR
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose:
                step_str = f" (step {step})" if step is not None else ""
                print(f"📉 LR reduced for param_group {i}{step_str}: {old_lr:.2e} → {new_lr:.2e}")
            
            reduced = True
        
        if reduced:
            self._save_lr_history()
        
        return reduced
    
    def _save_lr_history(self):
        """Save current LR to history."""
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.lr_history.append(current_lrs[0] if len(current_lrs) == 1 else current_lrs)
    
    def get_last_lr(self) -> list:
        """Get last learning rates for all parameter groups."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            'best_metric': self.best_metric,
            'patience_counter': self.patience_counter,
            'cooldown_counter': self.cooldown_counter,
            'num_bad_epochs': self.num_bad_epochs,
            'last_lr_reduction_step': self.last_lr_reduction_step,
            'lr_history': self.lr_history
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state dict from checkpoint."""
        self.best_metric = state_dict.get('best_metric')
        self.patience_counter = state_dict.get('patience_counter', 0)
        self.cooldown_counter = state_dict.get('cooldown_counter', 0)
        self.num_bad_epochs = state_dict.get('num_bad_epochs', 0)
        self.last_lr_reduction_step = state_dict.get('last_lr_reduction_step', -1)
        self.lr_history = state_dict.get('lr_history', [])


class GradientBasedLRAdjuster:
    """
    Gradient-based learning rate adjustment.
    
    Adjusts LR based on gradient norm to prevent:
    - Exploding gradients (reduce LR)
    - Vanishing gradients (increase LR)
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        target_grad_norm: float = 1.0,  # Target gradient norm
        factor: float = 0.1,  # Adjustment factor
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        verbose: bool = True
    ):
        self.optimizer = optimizer
        self.target_grad_norm = target_grad_norm
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose
    
    def adjust(self, grad_norm: float) -> bool:
        """
        Adjust LR based on gradient norm.
        
        Args:
            grad_norm: Current gradient norm
        
        Returns:
            True if LR was adjusted, False otherwise
        """
        if grad_norm <= 0:
            return False
        
        # Calculate adjustment factor
        # If grad_norm > target: reduce LR (gradients too large)
        # If grad_norm < target: increase LR (gradients too small)
        ratio = self.target_grad_norm / grad_norm
        
        # Clamp ratio to reasonable range
        ratio = max(0.5, min(2.0, ratio))
        
        adjusted = False
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * (1.0 + self.factor * (ratio - 1.0))
            new_lr = max(self.min_lr, min(self.max_lr, new_lr))
            
            if abs(new_lr - old_lr) > 1e-8:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"📊 Gradient-based LR adjustment: {old_lr:.2e} → {new_lr:.2e} (grad_norm: {grad_norm:.4f})")
                adjusted = True
        
        return adjusted

