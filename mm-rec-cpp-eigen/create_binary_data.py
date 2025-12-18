import struct
import random

def generate_binary_data(filename, num_tokens=10000):
    magic = 0x4D4D5245
    version = 2
    count = num_tokens
    
    # Tokens: Random integers 0-99
    tokens = [random.randint(0, 99) for _ in range(num_tokens)]
    # Masks: All 1s for simplicity
    masks = [1] * num_tokens
    
    with open(filename, 'wb') as f:
        f.write(struct.pack('<i', magic))
        f.write(struct.pack('<i', version))
        f.write(struct.pack('<q', count))
        
        # Write Tokens (int32)
        for t in tokens:
            f.write(struct.pack('<i', t))
            
        # Write Masks (int32)
        for m in masks:
            f.write(struct.pack('<i', m))
            
    print(f"Generated {filename} with {num_tokens} tokens.")

if __name__ == "__main__":
    generate_binary_data("train_data.bin")
