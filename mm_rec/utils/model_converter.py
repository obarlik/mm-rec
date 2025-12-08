"""
Model Weight Converter: Transform existing LLM weights to MM-Rec architecture

This utility analyzes existing model weights (e.g., LLaMA, GPT) and converts
them to MM-Rec architecture by:
1. Mapping compatible weights (embedding, FFN, some attention components)
2. Initializing new MM-Rec-specific components (MDI, HDS, Associative Scan)
3. Providing partial loading with missing key warnings
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import warnings


class ModelWeightAnalyzer:
    """
    Analyzes existing model weights to determine compatibility with MM-Rec.
    """
    
    def __init__(self, source_state_dict: Dict[str, torch.Tensor]):
        """
        Initialize analyzer with source model's state_dict.
        
        Args:
            source_state_dict: State dict from source model (e.g., LLaMA, GPT)
        """
        self.source_state_dict = source_state_dict
        self.source_keys = set(source_state_dict.keys())
        
    def analyze_architecture(self) -> Dict[str, any]:
        """
        Analyze source model architecture from weight keys.
        
        Returns:
            Dictionary with architecture information:
            - model_type: "llama", "gpt", "unknown"
            - vocab_size: Inferred from embedding size
            - model_dim: Inferred from layer dimensions
            - num_layers: Number of transformer blocks
            - num_heads: Number of attention heads
            - ffn_dim: Feed-forward dimension
        """
        analysis = {
            'model_type': 'unknown',
            'vocab_size': None,
            'model_dim': None,
            'num_layers': 0,
            'num_heads': None,
            'ffn_dim': None,
            'head_dim': None
        }
        
        # Detect model type from key patterns
        if any('llama' in k.lower() or 'layers' in k.lower() for k in self.source_keys):
            analysis['model_type'] = 'llama'
        elif any('gpt' in k.lower() or 'transformer' in k.lower() for k in self.source_keys):
            analysis['model_type'] = 'gpt'
        
        # Find embedding layer
        embedding_keys = [k for k in self.source_keys if 'embed' in k.lower() and 'weight' in k]
        if embedding_keys:
            emb_key = embedding_keys[0]
            emb_shape = self.source_state_dict[emb_key].shape
            analysis['vocab_size'] = emb_shape[0]
            analysis['model_dim'] = emb_shape[1]
        
        # Find transformer layers
        layer_keys = [k for k in self.source_keys if 'layers' in k.lower() or 'blocks' in k.lower()]
        if layer_keys:
            # Extract layer indices
            layer_indices = set()
            for k in layer_keys:
                parts = k.split('.')
                for i, part in enumerate(parts):
                    if part.isdigit():
                        layer_indices.add(int(part))
            analysis['num_layers'] = max(layer_indices) + 1 if layer_indices else 0
        
        # Find attention heads
        q_proj_keys = [k for k in self.source_keys if 'q_proj' in k.lower() or 'qkv' in k.lower()]
        if q_proj_keys:
            q_key = q_proj_keys[0]
            q_shape = self.source_state_dict[q_key].shape
            # Typical: [model_dim, model_dim] or [model_dim, num_heads * head_dim]
            if len(q_shape) == 2:
                analysis['model_dim'] = q_shape[0] if analysis['model_dim'] is None else analysis['model_dim']
                if q_shape[1] % q_shape[0] == 0:
                    analysis['num_heads'] = q_shape[1] // q_shape[0]
                    analysis['head_dim'] = q_shape[0] // analysis['num_heads']
        
        # Find FFN dimension
        ffn_keys = [k for k in self.source_keys if 'ffn' in k.lower() or 'mlp' in k.lower() or 'feed_forward' in k.lower()]
        if ffn_keys:
            # Look for up_proj or gate_proj (LLaMA style) or intermediate (GPT style)
            up_proj_keys = [k for k in ffn_keys if 'up' in k.lower() or 'gate' in k.lower() or 'intermediate' in k.lower()]
            if up_proj_keys:
                up_key = up_proj_keys[0]
                up_shape = self.source_state_dict[up_key].shape
                if len(up_shape) == 2:
                    analysis['ffn_dim'] = up_shape[1] if up_shape[0] == analysis['model_dim'] else up_shape[0]
        
        return analysis
    
    def get_compatible_keys(self, target_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple[str, float]]:
        """
        Find compatible keys between source and target models.
        
        Args:
            target_state_dict: MM-Rec model's state_dict
        
        Returns:
            Dictionary mapping target_key -> (source_key, match_score)
            match_score: 0.0-1.0 indicating compatibility
        """
        compatible = {}
        target_keys = set(target_state_dict.keys())
        
        # Key mapping patterns
        key_patterns = {
            # Embedding
            'embedding.weight': ['embed', 'tok_emb', 'wte'],
            'lm_head.weight': ['lm_head', 'output', 'head'],
            
            # Layer normalization
            'norm.weight': ['norm', 'ln_f'],
            'blocks.{}.norm1.weight': ['layers.{}.norm', 'layers.{}.ln_1'],
            'blocks.{}.norm2.weight': ['layers.{}.norm', 'layers.{}.ln_2'],
            
            # Attention (partial - MM-Rec has different attention)
            'blocks.{}.multi_mem_attention.W_q.weight': ['layers.{}.attention.q_proj', 'layers.{}.attn.q_proj'],
            'blocks.{}.multi_mem_attention.W_o.weight': ['layers.{}.attention.o_proj', 'layers.{}.attn.o_proj'],
            
            # FFN
            'blocks.{}.ffn.0.weight': ['layers.{}.mlp.up_proj', 'layers.{}.feed_forward.intermediate'],
            'blocks.{}.ffn.3.weight': ['layers.{}.mlp.down_proj', 'layers.{}.feed_forward.output'],
        }
        
        # Match keys
        for target_key in target_keys:
            best_match = None
            best_score = 0.0
            
            for pattern, source_patterns in key_patterns.items():
                # Check if pattern matches target key
                if self._pattern_matches(pattern, target_key):
                    # Try to find matching source key
                    for source_pattern in source_patterns:
                        for source_key in self.source_keys:
                            if self._keys_match(source_pattern, source_key, target_key):
                                # Calculate match score
                                score = self._calculate_match_score(
                                    self.source_state_dict[source_key],
                                    target_state_dict[target_key]
                                )
                                if score > best_score:
                                    best_match = source_key
                                    best_score = score
            
            if best_match and best_score > 0.5:  # Threshold for compatibility
                compatible[target_key] = (best_match, best_score)
        
        return compatible
    
    def _pattern_matches(self, pattern: str, key: str) -> bool:
        """Check if a pattern matches a key (supports {} placeholders)."""
        import re
        # Convert pattern to regex
        regex_pattern = pattern.replace('{}', r'\d+')
        regex_pattern = regex_pattern.replace('.', r'\.')
        return bool(re.match(regex_pattern, key))
    
    def _keys_match(self, source_pattern: str, source_key: str, target_key: str) -> bool:
        """Check if source and target keys match with pattern."""
        # Extract layer index from target key
        import re
        target_layer_match = re.search(r'blocks\.(\d+)', target_key)
        if not target_layer_match:
            return False
        
        target_layer_idx = target_layer_match.group(1)
        source_pattern_with_idx = source_pattern.replace('{}', target_layer_idx)
        
        # Check if source key matches pattern
        return source_pattern_with_idx.lower() in source_key.lower()
    
    def _calculate_match_score(self, source_tensor: torch.Tensor, target_tensor: torch.Tensor) -> float:
        """
        Calculate compatibility score between source and target tensors.
        
        Returns:
            Score between 0.0 and 1.0
        """
        if source_tensor.shape != target_tensor.shape:
            # Check if shapes are compatible (e.g., transpose)
            if source_tensor.shape == target_tensor.shape[::-1]:
                return 0.8  # Transpose needed
            elif len(source_tensor.shape) == 2 and len(target_tensor.shape) == 2:
                # Check if one dimension matches
                if source_tensor.shape[0] == target_tensor.shape[0] or \
                   source_tensor.shape[1] == target_tensor.shape[1]:
                    return 0.5  # Partial match
            return 0.0
        
        return 1.0  # Perfect match


class ModelWeightConverter:
    """
    Converts existing model weights to MM-Rec architecture.
    """
    
    def __init__(
        self,
        source_state_dict: Dict[str, torch.Tensor],
        target_model: nn.Module
    ):
        """
        Initialize converter.
        
        Args:
            source_state_dict: Source model's state_dict
            target_model: MM-Rec model instance
        """
        self.source_state_dict = source_state_dict
        self.target_model = target_model
        self.analyzer = ModelWeightAnalyzer(source_state_dict)
        
    def convert(
        self,
        strict: bool = False,
        initialize_new: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, any]]:
        """
        Convert source weights to MM-Rec format.
        
        Args:
            strict: If True, raise error on missing keys. If False, warn and continue.
            initialize_new: If True, initialize new MM-Rec-specific components randomly.
        
        Returns:
            Tuple of (converted_state_dict, conversion_report)
        """
        target_state_dict = self.target_model.state_dict()
        conversion_report = {
            'total_keys': len(target_state_dict),
            'converted_keys': 0,
            'missing_keys': [],
            'new_keys': [],
            'shape_mismatches': [],
            'compatibility_scores': {}
        }
        
        # Analyze source architecture
        source_analysis = self.analyzer.analyze_architecture()
        conversion_report['source_analysis'] = source_analysis
        
        # Find compatible keys
        compatible = self.analyzer.get_compatible_keys(target_state_dict)
        conversion_report['compatible_keys'] = len(compatible)
        
        # Start with target model's current weights (random initialization)
        converted_state_dict = target_state_dict.copy()
        
        # Convert compatible weights
        for target_key, (source_key, score) in compatible.items():
            source_tensor = self.source_state_dict[source_key]
            target_tensor = target_state_dict[target_key]
            
            if source_tensor.shape == target_tensor.shape:
                # Direct copy
                converted_state_dict[target_key] = source_tensor.clone()
                conversion_report['converted_keys'] += 1
                conversion_report['compatibility_scores'][target_key] = score
            elif source_tensor.shape == target_tensor.shape[::-1]:
                # Transpose needed
                converted_state_dict[target_key] = source_tensor.t().clone()
                conversion_report['converted_keys'] += 1
                conversion_report['compatibility_scores'][target_key] = score
            else:
                # Shape mismatch
                conversion_report['shape_mismatches'].append({
                    'target_key': target_key,
                    'source_key': source_key,
                    'target_shape': list(target_tensor.shape),
                    'source_shape': list(source_tensor.shape)
                })
        
        # Handle missing keys
        converted_keys_set = set(compatible.keys())
        for target_key in target_state_dict.keys():
            if target_key not in converted_keys_set:
                conversion_report['missing_keys'].append(target_key)
                
                # Initialize new MM-Rec-specific components
                if initialize_new:
                    if any(x in target_key for x in ['mdi', 'hds', 'associative_scan']):
                        # These are MM-Rec-specific, keep random initialization
                        conversion_report['new_keys'].append(target_key)
                    elif 'W_g' in target_key or 'W_z' in target_key:
                        # Gating and z projections - initialize from similar weights if available
                        # Try to find similar weight from source
                        similar_key = self._find_similar_weight(target_key, self.source_state_dict)
                        if similar_key:
                            source_tensor = self.source_state_dict[similar_key]
                            if source_tensor.shape == target_state_dict[target_key].shape:
                                converted_state_dict[target_key] = source_tensor.clone()
                            else:
                                # Initialize with Xavier/Glorot
                                nn.init.xavier_uniform_(converted_state_dict[target_key])
                        else:
                            nn.init.xavier_uniform_(converted_state_dict[target_key])
                    else:
                        # Keep random initialization
                        pass
        
        # Warn about missing keys
        if conversion_report['missing_keys']:
            missing_count = len(conversion_report['missing_keys'])
            if strict:
                raise RuntimeError(
                    f"Missing {missing_count} keys. Set strict=False to continue with partial loading."
                )
            else:
                warnings.warn(
                    f"⚠️ {missing_count} keys could not be converted from source model.\n"
                    f"   These will use random initialization or default values.\n"
                    f"   Missing keys: {conversion_report['missing_keys'][:10]}..."
                    if missing_count > 10 else f"   Missing keys: {conversion_report['missing_keys']}",
                    UserWarning
                )
        
        return converted_state_dict, conversion_report
    
    def _find_similar_weight(self, target_key: str, source_state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
        """Find similar weight in source model for initialization."""
        # Look for attention or FFN weights that might be similar
        if 'W_g' in target_key or 'W_z' in target_key:
            # Try to find attention or FFN weights
            for source_key in source_state_dict.keys():
                if any(x in source_key.lower() for x in ['attn', 'mlp', 'ffn', 'proj']):
                    return source_key
        return None
    
    def save_conversion_report(self, report: Dict[str, any], filepath: str):
        """Save conversion report to file."""
        import json
        
        # Convert tensors to shapes for JSON serialization
        serializable_report = {}
        for key, value in report.items():
            if key == 'compatibility_scores':
                serializable_report[key] = value
            elif key == 'shape_mismatches':
                serializable_report[key] = value
            elif key == 'source_analysis':
                serializable_report[key] = value
            else:
                serializable_report[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)


def convert_model_weights(
    source_checkpoint_path: str,
    target_model: nn.Module,
    output_path: Optional[str] = None,
    strict: bool = False,
    initialize_new: bool = True
) -> Tuple[Dict[str, torch.Tensor], Dict[str, any]]:
    """
    High-level function to convert model weights.
    
    Args:
        source_checkpoint_path: Path to source model checkpoint (.pt, .pth, .safetensors)
        target_model: MM-Rec model instance
        output_path: Optional path to save converted weights
        strict: If True, raise error on missing keys
        initialize_new: If True, initialize new components randomly
    
    Returns:
        Tuple of (converted_state_dict, conversion_report)
    
    Example:
        >>> from mm_rec.model import MMRecModel
        >>> model = MMRecModel(vocab_size=32000, model_dim=4096, num_layers=24)
        >>> converted_weights, report = convert_model_weights(
        ...     'llama-7b.pt',
        ...     model,
        ...     output_path='mmrec-7b-converted.pt'
        ... )
        >>> model.load_state_dict(converted_weights, strict=False)
    """
    # Load source checkpoint
    if source_checkpoint_path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            source_state_dict = load_file(source_checkpoint_path)
        except ImportError:
            raise ImportError("safetensors library required for .safetensors files. Install with: pip install safetensors")
    else:
        checkpoint = torch.load(source_checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            source_state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            source_state_dict = checkpoint['model']
        else:
            source_state_dict = checkpoint
    
    # Convert
    converter = ModelWeightConverter(source_state_dict, target_model)
    converted_state_dict, report = converter.convert(
        strict=strict,
        initialize_new=initialize_new
    )
    
    # Save if output path provided
    if output_path:
        torch.save(converted_state_dict, output_path)
        print(f"✅ Converted weights saved to: {output_path}")
        
        # Save report
        report_path = output_path.replace('.pt', '_report.json').replace('.pth', '_report.json')
        converter.save_conversion_report(report, report_path)
        print(f"✅ Conversion report saved to: {report_path}")
    
    return converted_state_dict, report

