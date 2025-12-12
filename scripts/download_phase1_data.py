#!/usr/bin/env python3
"""
Download and prepare Phase 1 training datasets
Foundation training: Simple Q&A and short conversations
"""

import os
import json
from datasets import load_dataset
from typing import List, Dict
import argparse


def download_dolly(output_dir: str, max_samples: int = 15000):
    """Download Databricks Dolly-15K dataset."""
    print("📥 Downloading Dolly-15K...")
    
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    # Convert to chat format
    conversations = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        
        # Dolly format: instruction, context, response
        user_content = item['instruction']
        if item.get('context'):
            user_content = f"{item['context']}\n\n{user_content}"
        
        conversations.append({
            "id": f"dolly_{i}",
            "conversations": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item['response']}
            ]
        })
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dolly_15k.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(conversations)} examples to {output_file}")
    return len(conversations)


def download_openassistant(output_dir: str, max_samples: int = 10000):
    """Download OpenAssistant conversations."""
    print("📥 Downloading OpenAssistant...")
    
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    
    # Filter for English, high-quality conversations
    conversations = []
    seen_ids = set()
    
    for item in dataset:
        if len(conversations) >= max_samples:
            break
        
        # Filter criteria
        if item['lang'] != 'en':
            continue
        rank = item.get('rank')
        if rank is not None and rank < 0:  # Low quality
            continue
        
        # Get conversation thread
        conv_id = item.get('message_tree_id', item.get('message_id'))
        if conv_id in seen_ids:
            continue
        
        seen_ids.add(conv_id)
        
        # Simple format: just the message
        if item.get('role') == 'prompter':
            role = 'user'
        else:
            role = 'assistant'
        
        conversations.append({
            "id": f"oasst_{len(conversations)}",
            "conversations": [
                {"role": "user", "content": item.get('text', '')[:512]},  # Truncate for Phase 1
                {"role": "assistant", "content": "I understand. How can I help you with that?"}
            ]
        })
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "openassistant_10k.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(conversations)} examples to {output_file}")
    return len(conversations)


def create_simple_qa(output_dir: str, num_samples: int = 5000):
    """Create simple Q&A dataset for foundation training."""
    print("📝 Creating simple Q&A dataset...")
    
    # Categories
    qa_pairs = []
    
    # Math (1000 examples)
    for i in range(1000):
        a = (i % 10) + 1
        b = ((i // 10) % 10) + 1
        qa_pairs.append({
            "id": f"math_{i}",
            "conversations": [
                {"role": "user", "content": f"What is {a} + {b}?"},
                {"role": "assistant", "content": str(a + b)}
            ]
        })
    
    # Greetings (1000 examples)
    greetings = [
        ("Hello", "Hello! How can I help you today?"),
        ("Hi", "Hi there! What can I do for you?"),
        ("Good morning", "Good morning! How are you?"),
        ("How are you?", "I'm doing well, thank you! How about you?"),
        ("What's your name?", "I'm an AI assistant. How can I help you?")
    ]
    
    for i in range(1000):
        q, a = greetings[i % len(greetings)]
        qa_pairs.append({
            "id": f"greeting_{i}",
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })
    
    # Simple facts (1000 examples)
    facts = [
        ("What is the capital of France?", "Paris"),
        ("What color is the sky?", "Blue"),
        ("How many days in a week?", "7"),
        ("What is 2+2?", "4"),
        ("What is water made of?", "Hydrogen and oxygen (H2O)")
    ]
    
    for i in range(1000):
        q, a = facts[i % len(facts)]
        qa_pairs.append({
            "id": f"fact_{i}",
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })
    
    # Yes/No questions (1000 examples)
    yn_questions = [
        ("Is the Earth round?", "Yes, the Earth is round."),
        ("Can birds fly?", "Yes, most birds can fly."),
        ("Is water wet?", "Yes, water is wet."),
        ("Do cats bark?", "No, cats meow. Dogs bark."),
        ("Is the sun hot?", "Yes, the sun is very hot.")
    ]
    
    for i in range(1000):
        q, a = yn_questions[i % len(yn_questions)]
        qa_pairs.append({
            "id": f"yesno_{i}",
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })
    
    # Short definitions (1000 examples)
    definitions = [
        ("What is a computer?", "A computer is an electronic device that processes data."),
        ("What is rain?", "Rain is water falling from clouds."),
        ("What is a book?", "A book is a written or printed work consisting of pages."),
        ("What is music?", "Music is organized sound that is pleasing to hear."),
        ("What is a tree?", "A tree is a tall plant with a wooden trunk and branches.")
    ]
    
    for i in range(1000):
        q, a = definitions[i % len(definitions)]
        qa_pairs.append({
            "id": f"definition_{i}",
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "simple_qa_5k.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs[:num_samples], f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {min(num_samples, len(qa_pairs))} Q&A examples in {output_file}")
    return min(num_samples, len(qa_pairs))


def merge_datasets(output_dir: str):
    """Merge all Phase 1 datasets into training and validation splits."""
    print("\n📊 Merging datasets...")
    
    all_data = []
    
    # Load all datasets
    for filename in os.listdir(output_dir):
        if filename.endswith('.json') and filename != 'train.json' and filename != 'val.json':
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"  Loaded {len(data)} from {filename}")
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_data)
    
    # Split 90/10
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Save
    with open(os.path.join(output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(output_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Created splits:")
    print(f"   Training: {len(train_data)} examples")
    print(f"   Validation: {len(val_data)} examples")
    
    return len(train_data), len(val_data)


def main():
    parser = argparse.ArgumentParser(description="Download Phase 1 training data")
    parser.add_argument("--output-dir", default="data/phase1", help="Output directory")
    parser.add_argument("--skip-dolly", action="store_true", help="Skip Dolly download")
    parser.add_argument("--skip-oasst", action="store_true", help="Skip OpenAssistant download")
    parser.add_argument("--skip-qa", action="store_true", help="Skip simple Q&A creation")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Phase 1 Dataset Preparation")
    print("="*80)
    
    total_samples = 0
    
    # Download datasets
    if not args.skip_dolly:
        total_samples += download_dolly(args.output_dir, max_samples=15000)
    
    if not args.skip_oasst:
        total_samples += download_openassistant(args.output_dir, max_samples=10000)
    
    if not args.skip_qa:
        total_samples += create_simple_qa(args.output_dir, num_samples=5000)
    
    # Merge into train/val splits
    train_size, val_size = merge_datasets(args.output_dir)
    
    print("\n" + "="*80)
    print(f"✅ Phase 1 data preparation complete!")
    print(f"   Total samples: {total_samples}")
    print(f"   Training: {train_size}")
    print(f"   Validation: {val_size}")
    print(f"   Location: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
