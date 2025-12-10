"""
MM-Rec Base Model Configurations
Progressive training için model konfigürasyonları
En küçükten en büyüğe: Tiny -> Mini -> Small -> Base -> Medium -> Large -> 7B
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model konfigürasyonu"""
    name: str
    vocab_size: int
    model_dim: int
    num_layers: int
    num_heads: int
    num_memories: int = 1
    mem_dim: Optional[int] = None
    ffn_dim: Optional[int] = None
    max_seq_len: int = 2048
    dropout: float = 0.1
    # HEM, DPG, UBÖO
    use_hem: bool = True
    use_dpg: bool = False
    use_uboo: bool = False
    dpg_rank: int = 128
    lambda_P: float = 0.1


# En küçük temel model (başlangıç noktası)
TINY_BASE_CONFIG = ModelConfig(
    name="tiny_base",
    vocab_size=5000,
    model_dim=128,
    num_layers=4,
    num_heads=4,
    max_seq_len=1024,
    use_hem=True,
    use_dpg=False,
    use_uboo=False
)

# Mini model (Tiny'dan sonra)
MINI_BASE_CONFIG = ModelConfig(
    name="mini_base",
    vocab_size=10000,
    model_dim=256,
    num_layers=6,
    num_heads=4,
    max_seq_len=2048,
    use_hem=True,
    use_dpg=False,
    use_uboo=False
)

# Small model (Mini'den sonra)
SMALL_BASE_CONFIG = ModelConfig(
    name="small_base",
    vocab_size=20000,
    model_dim=512,
    num_layers=8,
    num_heads=8,
    max_seq_len=4096,
    use_hem=True,
    use_dpg=False,
    use_uboo=False
)

# Base model (Small'dan sonra)
BASE_BASE_CONFIG = ModelConfig(
    name="base_base",
    vocab_size=32000,
    model_dim=1024,
    num_layers=12,
    num_heads=8,
    max_seq_len=8192,
    use_hem=True,
    use_dpg=False,
    use_uboo=False
)

# Medium model (Base'den sonra)
MEDIUM_BASE_CONFIG = ModelConfig(
    name="medium_base",
    vocab_size=32000,
    model_dim=2048,
    num_layers=16,
    num_heads=16,
    max_seq_len=16384,
    use_hem=True,
    use_dpg=True,
    use_uboo=False
)

# Large model (Medium'dan sonra)
LARGE_BASE_CONFIG = ModelConfig(
    name="large_base",
    vocab_size=32000,
    model_dim=3072,
    num_layers=20,
    num_heads=24,
    max_seq_len=32768,
    use_hem=True,
    use_dpg=True,
    use_uboo=True
)

# 7B Model (Large'dan sonra - hedef)
MMREC_7B_CONFIG = ModelConfig(
    name="mmrec_7b",
    vocab_size=32000,
    model_dim=4096,
    num_layers=24,
    num_heads=32,
    num_memories=8,
    mem_dim=512,
    max_seq_len=32768,
    use_hem=True,
    use_dpg=True,
    use_uboo=True,
    dpg_rank=128,
    lambda_P=0.1
)

# Progressive training sırası
PROGRESSIVE_TRAINING_SEQUENCE = [
    TINY_BASE_CONFIG,
    MINI_BASE_CONFIG,
    SMALL_BASE_CONFIG,
    BASE_BASE_CONFIG,
    MEDIUM_BASE_CONFIG,
    LARGE_BASE_CONFIG,
    MMREC_7B_CONFIG
]


def get_config_by_name(name: str) -> Optional[ModelConfig]:
    """İsimle konfigürasyon al"""
    configs = {
        "tiny": TINY_BASE_CONFIG,
        "mini": MINI_BASE_CONFIG,
        "small": SMALL_BASE_CONFIG,
        "base": BASE_BASE_CONFIG,
        "medium": MEDIUM_BASE_CONFIG,
        "large": LARGE_BASE_CONFIG,
        "7b": MMREC_7B_CONFIG,
    }
    return configs.get(name.lower())


def get_next_config(current_config: ModelConfig) -> Optional[ModelConfig]:
    """Progressive training için sonraki konfigürasyonu al"""
    try:
        current_idx = PROGRESSIVE_TRAINING_SEQUENCE.index(current_config)
        if current_idx < len(PROGRESSIVE_TRAINING_SEQUENCE) - 1:
            return PROGRESSIVE_TRAINING_SEQUENCE[current_idx + 1]
    except ValueError:
        pass
    return None
