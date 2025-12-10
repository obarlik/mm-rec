#!/usr/bin/env python3
"""
MM-Rec Sistem Test Scripti
Tüm mekanizmaları (HEM, DPG, UBÖO) test eder ve sistem durumunu raporlar
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mm_rec.model import MMRecModel


def test_system():
    """Sistem testi - tüm mekanizmaları test eder."""
    print("=" * 60)
    print("MM-Rec Sistem Testi")
    print("=" * 60)
    print()
    
    # Sistem bilgileri
    print("🔧 Sistem Bilgileri:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print()
    
    # Test konfigürasyonu
    config = {
        'vocab_size': 1000,
        'model_dim': 128,
        'num_layers': 2,
        'num_heads': 2,
        'batch_size': 2,
        'seq_len': 16
    }
    
    print("📋 Test Konfigürasyonu:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Test 1: Baseline (hiçbir mekanizma yok)
    print("🧪 Test 1: Baseline Model (HEM=False, DPG=False, UBÖO=False)")
    try:
        model_baseline = MMRecModel(
            vocab_size=config['vocab_size'],
            model_dim=config['model_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            use_hem=False,
            use_dpg=False,
            use_uboo=False
        )
        input_ids = torch.randint(0, config['vocab_size'], 
                                 (config['batch_size'], config['seq_len']))
        logits = model_baseline(input_ids)
        print(f"  ✅ Başarılı: {logits.shape}, {model_baseline.get_num_params()} params")
    except Exception as e:
        print(f"  ❌ Hata: {e}")
        return False
    print()
    
    # Test 2: HEM
    print("🧪 Test 2: HEM Aktif")
    try:
        model_hem = MMRecModel(
            vocab_size=config['vocab_size'],
            model_dim=config['model_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            use_hem=True,
            use_dpg=False,
            use_uboo=False
        )
        logits = model_hem(input_ids)
        print(f"  ✅ Başarılı: {logits.shape}, {model_hem.get_num_params()} params")
    except Exception as e:
        print(f"  ❌ Hata: {e}")
        return False
    print()
    
    # Test 3: DPG
    print("🧪 Test 3: DPG Aktif")
    try:
        model_dpg = MMRecModel(
            vocab_size=config['vocab_size'],
            model_dim=config['model_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            use_hem=False,
            use_dpg=True,
            dpg_rank=64,
            use_uboo=False
        )
        logits = model_dpg(input_ids)
        print(f"  ✅ Başarılı: {logits.shape}, {model_dpg.get_num_params()} params")
    except Exception as e:
        print(f"  ❌ Hata: {e}")
        return False
    print()
    
    # Test 4: UBÖO
    print("🧪 Test 4: UBÖO Aktif")
    try:
        model_uboo = MMRecModel(
            vocab_size=config['vocab_size'],
            model_dim=config['model_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            use_hem=False,
            use_dpg=False,
            use_uboo=True,
            lambda_P=0.1
        )
        logits, L_Aux = model_uboo(input_ids, return_auxiliary_loss=True)
        print(f"  ✅ Başarılı: {logits.shape}, L_Aux={L_Aux.item():.6f}, {model_uboo.get_num_params()} params")
    except Exception as e:
        print(f"  ❌ Hata: {e}")
        return False
    print()
    
    # Test 5: Tüm mekanizmalar
    print("🧪 Test 5: Tüm Mekanizmalar Aktif (HEM + DPG + UBÖO)")
    try:
        model_all = MMRecModel(
            vocab_size=config['vocab_size'],
            model_dim=config['model_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            use_hem=True,
            use_dpg=True,
            dpg_rank=64,
            use_uboo=True,
            lambda_P=0.1
        )
        logits, L_Aux = model_all(input_ids, return_auxiliary_loss=True)
        print(f"  ✅ Başarılı: {logits.shape}, L_Aux={L_Aux.item():.6f}, {model_all.get_num_params()} params")
    except Exception as e:
        print(f"  ❌ Hata: {e}")
        return False
    print()
    
    # Özet
    print("=" * 60)
    print("📊 Özet:")
    print(f"  Baseline: {model_baseline.get_num_params()} params")
    print(f"  HEM: {model_hem.get_num_params()} params")
    print(f"  DPG: {model_dpg.get_num_params()} params")
    print(f"  UBÖO: {model_uboo.get_num_params()} params")
    print(f"  All: {model_all.get_num_params()} params")
    print()
    
    print("=" * 60)
    print("✅ Tüm testler başarılı! Sistem çalışıyor.")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
