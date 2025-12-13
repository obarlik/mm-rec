#!/usr/bin/env python3
"""
Interactive Inference Demo for MM-Rec
Connects to Phoenix server, downloads the latest model, and runs chat.
"""
import sys
import torch
import torch.nn as nn
import argparse
import requests
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from mm_rec.model import MMRecModel
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer

def download_model(server_url, job_id, output_path):
    print(f"📥 Downloading model for job {job_id} from {server_url}...")
    try:
        # Try to get final model first, then check for checkpoints if needed (future feature)
        # For now, we assume the server exposes the 'latest' available model via download endpoint
        url = f"{server_url}/api/train/download/{job_id}"
        
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end="")
        print("\n✅ Download complete.")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def chat_loop(model_path):
    print("\n🧠 Loading MM-Rec Base Model...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size + 1000 # Safety margin matching training
    
    # Initialize model
    model = MMRecModel(
        vocab_size=vocab_size,
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ffn_dim=config['ffn_dim'],
        # Enable full features for inference
        use_hem=config.get('use_hem', False),
        use_dpg=config.get('use_dpg', False),
        use_uboo=config.get('use_uboo', False)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded! (Loss: {checkpoint.get('final_loss', 'N/A')})")
    print("💬 Start chatting (type 'exit' to quit)\n")
    
    # Simple chat loop
    history = []
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            # Tokenize
            input_ids = torch.tensor([tokenizer.encode(user_input)])
            
            # Generate (Basic greedy search for demo)
            with torch.no_grad():
                # In a real scenario, use ChatCompletionAPI or proper generation loop
                # This is a raw model test
                logits = model(input_ids)
                next_token = logits[0, -1].argmax()
                decoded = tokenizer.decode([next_token.item()])
                
            print(f"MM-Rec: {decoded} (Raw debug output)")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True, help="Job ID to download")
    parser.add_argument("--server", default="http://192.168.1.86:8001", help="Phoenix URL")
    args = parser.parse_args()
    
    model_file = f"{args.job_id}_model.pt"
    if download_model(args.server, args.job_id, model_file):
        chat_loop(model_file)
