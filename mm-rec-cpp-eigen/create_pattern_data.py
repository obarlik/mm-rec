import struct
import os
import json

def create_nano_data():
    output_bin = "nano_data.bin"
    output_vocab = "vocab.json"
    output_merges = "merges.txt"
    
    # 1. Define Pattern (0 1 2 -> 3) repeating
    # Simple pattern: 0 1 2 3, 0 1 2 3...
    # We want model to predict next number.
    
    # 2. Define Vocab (BPE Style)
    # Special Tokens
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
    }
    
    # Add numbers 0-3 to vocab
    # In BPE, these are usually strings.
    for i in range(4):
        vocab[str(i)] = 4 + i
    
    # 3. Create Merges (Empty for this simple digit-level vocab)
    with open(output_merges, "w") as f:
        f.write("# version: 0.2\n")
        # No merges needed for single digit tokens
    
    # 4. Generate Data
    # 5000 tokens of repeating 0 1 2 3
    pattern = [vocab["0"], vocab["1"], vocab["2"], vocab["3"]] # IDs: 4, 5, 6, 7
    
    total_tokens = 5000
    data = []
    for i in range(total_tokens):
        data.append(pattern[i % 4])
        
    # 5. Save Binary
    print(f"Generating {output_bin} with simple repeating pattern [0, 1, 2, 3] (IDs {vocab['0']}..{vocab['3']})...")
    
    with open(output_bin, "wb") as f:
        # Header
        magic = 20240501
        version = 2
        count = len(data)
        
        f.write(struct.pack("i", magic))
        f.write(struct.pack("i", version))
        f.write(struct.pack("q", count))
        
        # Tokens
        for token in data:
            f.write(struct.pack("i", token))
            
        # Masks (All ones for training)
        for _ in data:
            f.write(struct.pack("i", 1))

    # 6. Save Vocab JSON
    with open(output_vocab, "w") as f:
        json.dump(vocab, f, indent=2)
        
    print(f"Generated {output_vocab} and {output_merges}")

if __name__ == "__main__":
    create_nano_data()
