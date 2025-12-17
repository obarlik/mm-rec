import json
import random

def generate_sample(id):
    # Mix of easy (repeat) and hard (random) patterns
    r = random.random()
    if r < 0.3:
        # EASY: repetition
        content = f"Easy sample {id}. " * 50
        diff = "easy"
    elif r < 0.6:
        # MEDIUM: simple structure
        content = f"This is a medium difficulty sample number {id}. It has some structure but varies. " * 20
        diff = "medium"
    else:
        # HARD: random noise/numbers
        content = " ".join([str(random.randint(0, 10000)) for _ in range(200)])
        diff = "hard"
        
    # Alpaca Format
    return {
        "instruction": f"Type: {diff}",
        "input": "",
        "output": content
    }

with open("synthetic_data.jsonl", "w") as f:
    for i in range(500): # 500 samples
        f.write(json.dumps(generate_sample(i)) + "\n")

print("Generated 500 synthetic samples.")
