import struct

INPUT_FILE = "tinystories.txt"
OUTPUT_FILE = "training_data_nano.bin"

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Tokenizing {len(text)} characters (Byte-Level)...")
    # Byte-level tokenization (0-255)
    tokens = list(text.encode('utf-8'))
    count = len(tokens)
    
    print(f"Generated {count} tokens.")

    # Header matching src/demo_training_cpp.cpp
    # Magic: 0x4D4D5245 (MMRE)
    # Version: 1
    magic = 0x4D4D5245
    version = 1
    
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        f.write(struct.pack('<i', magic))
        f.write(struct.pack('<i', version))
        f.write(struct.pack('<q', count))
        
        # Write integers (int32)
        # Note: demo_training_cpp reads int32 tokens. 
        # But our bytes are 0-255. We store them as 32-bit ints.
        data = struct.pack(f'<{count}i', *tokens)
        f.write(data)
        
    print("✅ Done!")

if __name__ == "__main__":
    main()
