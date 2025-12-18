import struct
import os

filename = "nano_data.bin"

if not os.path.exists(filename):
    print(f"File {filename} not found!")
    exit(1)

file_size = os.path.getsize(filename)
print(f"File size: {file_size} bytes")
# Header is typically version (4 bytes) + count (8 bytes)? 
# Let's try to read header.

with open(filename, "rb") as f:
    # Try reading first few integers.
    # Version (int32)
    # Count (int64)
    # Data ...
    
    header = f.read(4)
    if len(header) < 4:
        print("Empty file")
        exit(1)
        
    version = struct.unpack("i", header)[0]
    print(f"Version: {version}")
    
    count_bytes = f.read(8)
    count = struct.unpack("q", count_bytes)[0]
    print(f"Count: {count} tokens")
    
    print("First 50 tokens:")
    tokens = []
    for _ in range(50):
        b = f.read(4)
        if not b: break
        val = struct.unpack("i", b)[0] # Assuming int32 tokens? Or float? mm_rec uses floats usually?
        # Data loader usually reads floats. Let's check data_loader.cpp or try both.
        # But previous inspection said int32.
        tokens.append(val)
        
    print(tokens)
    
    # Check pattern
    pattern = [4, 5, 6, 7]
    valid = True
    for i in range(len(tokens)):
        if tokens[i] != pattern[i % 4]:
            valid = False
            # print(f"Mismatch at {i}: expected {pattern[i%4]}, got {tokens[i]}")
            
    if valid:
        print("\n✅ Pattern [4, 5, 6, 7] verified for first 50 tokens.")
    else:
        print("\n❌ Pattern mismatch!")
