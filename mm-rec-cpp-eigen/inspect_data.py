import struct
import sys

def inspect_data(path):
    print(f"Inspecting {path}...")
    try:
        with open(path, 'rb') as f:
            # Header
            magic = struct.unpack('i', f.read(4))[0]
            version = struct.unpack('i', f.read(4))[0]
            count = struct.unpack('q', f.read(8))[0]
            
            print(f"Magic: {magic}")
            print(f"Version: {version}")
            print(f"Count: {count}")
            
            # Read all tokens
            tokens_bytes = f.read(count * 4)
            tokens = struct.unpack(f'{count}i', tokens_bytes)
            
            min_val = min(tokens)
            max_val = max(tokens)
            
            print(f"Min Token: {min_val}")
            print(f"Max Token: {max_val}")
            
            if max_val >= 16:
                print(f"⚠️  WARNING: Max Token ({max_val}) >= 16 (Nano Vocab Size)")
            else:
                print("✅ Tokens fit within Nano Vocab Size (16)")

            if version >= 2:
                # Check masks if they exist
                masks_bytes = f.read(count * 4)
                if len(masks_bytes) == count * 4:
                     masks = struct.unpack(f'{count}i', masks_bytes)
                     print(f"Masks loaded. Count: {len(masks)}")
                else:
                    print("No masks found (or partial)")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_data.py <bin_file>")
    else:
        inspect_data(sys.argv[1])
