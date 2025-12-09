#!/usr/bin/env python3
"""
MM-Rec C++/Triton extension doğrulama aracı.

Amaç:
- yüklü mü?
- hangi dosyadan geliyor?
- mtime (build zamanı tahmini)
- SHA256 (build parmak izi)
- opsiyonel __version__ alanı
"""

import importlib
import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _preload_pytorch_libs():
    """Preload PyTorch libraries to fix libc10.so issues."""
    import ctypes
    import os
    import torch
    
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        # Preload libc10.so with RTLD_GLOBAL
        libc10_path = os.path.join(torch_lib, 'libc10.so')
        if os.path.exists(libc10_path):
            try:
                ctypes.CDLL(libc10_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
        
        # Set LD_LIBRARY_PATH
        os.environ['LD_LIBRARY_PATH'] = torch_lib


def _probe_module(name: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"name": name, "available": False}
    
    # Preload PyTorch libraries before importing C++ extensions
    if name in ("mm_rec_cpp_cpu", "mm_rec_scan_cpu"):
        try:
            _preload_pytorch_libs()
        except Exception:
            pass
    
    try:
        mod = importlib.import_module(name)
        info["available"] = True
        if hasattr(mod, "__file__") and mod.__file__:
            p = Path(mod.__file__)
            info["path"] = str(p)
            if p.exists():
                stat = p.stat()
                info["size_bytes"] = stat.st_size
                info["mtime"] = stat.st_mtime
                info["sha256"] = _hash_file(p)
        if hasattr(mod, "__version__"):
            info["version"] = getattr(mod, "__version__")
    except Exception as e:
        info["error"] = str(e)
    return info


def main():
    targets = [
        "mm_rec_cpp_cpu",
        "mm_rec_scan_cpu",
        # Add more if/when available
        "mm_rec_cpp_cuda",
    ]

    print("=" * 80)
    print("MM-Rec Extension Durum Kontrolü")
    print("=" * 80)
    for name in targets:
        info = _probe_module(name)
        print(f"\n{name}:")
        if not info["available"]:
            print(f"  ❌ Yüklenemedi: {info.get('error', 'bilinmiyor')}")
            continue
        print("  ✅ Yüklendi")
        if "version" in info:
            print(f"  Version    : {info['version']}")
        if "path" in info:
            print(f"  Path       : {info['path']}")
        if "size_bytes" in info:
            print(f"  Size       : {info['size_bytes']:,} bytes")
        if "mtime" in info:
            print(f"  mtime      : {info['mtime']}")
        if "sha256" in info:
            print(f"  SHA256     : {info['sha256']}")

    print("\nNot: SHA256 ve mtime, build parmak izi olarak kullanılabilir.")
    print("     __version__ alanı varsa ayrıca listelenir.")


if __name__ == "__main__":
    main()

