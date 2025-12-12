# Phoenix GPU Test Script
# Run this on Phoenix to check GPU

import torch
print("="*50)
print("PyTorch GPU Test")
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("\n⚠️ CUDA NOT AVAILABLE!")
    print("Possible reasons:")
    print("1. PyTorch installed without CUDA support")
    print("2. CUDA drivers not installed")
    print("3. WSL GPU passthrough not configured")
print("="*50)
