
import torch
import sys
import os

# Ensure the workspace is in the path
sys.path.append('/home/onur/workspace/mm-rec')

from mm_rec.model import MMRecModel

def verify_mm_rec_model():
    print("🚀 Starting MM-Rec Model Verification...")
    
    # Configuration for a small test model
    config = {
        "vocab_size": 1000,
        "model_dim": 128,      # Small dimension for testing
        "num_layers": 2,       # Minimum layers
        "num_heads": 4,
        "num_memories": 1,
        "mem_dim": 32,
        "max_seq_len": 64,     # Multiple of 16
        "dropout": 0.0,
        "use_hem": False,      # Keep it simple first
        "use_dpg": False,
        # Sparse Config
        "use_sparse": True,
        "sparse_chunk_size": 16, # Small chunk for testing
        "num_experts": 4
    }
    
    print(f"📋 Model Config: {config}")
    
    try:
        # 1. Instantiate Model
        model = MMRecModel(**config)
        print("✅ Model instantiated successfully.")
        
        # 2. Create Dummy Input
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        print(f"🔹 Input Shape: {input_ids.shape}")
        
        # 3. Forward Pass
        print("🔄 Running Forward Pass...")
        logits = model(input_ids)
        
        # 4. Check Output
        expected_shape = (batch_size, seq_len, config["vocab_size"])
        print(f"🔹 Output Shape: {logits.shape}")
        
        if logits.shape == expected_shape:
            print("✅ Output shape matches expected shape.")
        else:
            print(f"❌ Output shape mismatch! Expected {expected_shape}, got {logits.shape}")
            return
            
        # 5. Check Backward Pass (Basic Gradients)
        print("🔄 Checking Backward Pass...")
        loss = logits.sum()
        loss.backward()
        print("✅ Backward pass completed successfully.")
        
        # Check if gradients exist for a parameter
        if model.lm_head.weight.grad is not None:
             print("✅ Gradients computed for lm_head.")
        else:
             print("❌ No gradients found for lm_head.")

    except Exception as e:
        print(f"❌ Verification Failed with Error:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_mm_rec_model()
