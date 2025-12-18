import json

INPUT_FILE = "tinystories.txt"
OUTPUT_FILE = "tinystories.jsonl"

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by delimiter
    stories = content.split("<|endoftext|>")
    
    print(f"Found {len(stories)} stories. Converting...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        count = 0
        for story in stories:
            story = story.strip()
            if not story:
                continue
                
            # Create Alpaca format
            entry = {
                "instruction": "Write a short story.",
                "input": "",
                "output": story
            }
            
            f.write(json.dumps(entry) + "\n")
            count += 1
            
    print(f"✅ Converted {count} stories to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
