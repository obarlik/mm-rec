"""
Convert Real Text/Code Data to OpenAI Chat Format
Properly formats real data into conversational format
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def load_jsonl(file_path: Path) -> List[str]:
    """Load JSONL file and return list of texts."""
    texts = []
    if not file_path.exists():
        return texts
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                # Try to parse as JSON
                data = json.loads(line)
                if isinstance(data, str):
                    texts.append(data)
                elif isinstance(data, dict):
                    # Extract text from common fields
                    for key in ['text', 'content', 'body', 'message']:
                        if key in data and isinstance(data[key], str):
                            texts.append(data[key])
                            break
            except json.JSONDecodeError:
                # If not JSON, treat as plain text
                if len(line) > 50:
                    texts.append(line)
    
    return texts


def create_qa_from_text(text: str, min_length: int = 100) -> List[Dict]:
    """Create Q&A pairs from text."""
    conversations = []
    
    # Split by sentences
    sentences = re.split(r'[.!?]\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if len(sentences) < 2:
        return conversations
    
    # Create Q&A pairs
    for i in range(0, len(sentences) - 1, 2):
        question_sentences = sentences[i:i+2]
        answer_sentences = sentences[i+1:i+3] if i+1 < len(sentences) else sentences[i:i+2]
        
        question = ' '.join(question_sentences)
        answer = ' '.join(answer_sentences)
        
        if len(question) > min_length and len(answer) > min_length:
            conversations.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question[:500]},
                    {"role": "assistant", "content": answer[:500]}
                ]
            })
    
    return conversations


def create_code_explanation(code: str, min_length: int = 50) -> List[Dict]:
    """Create code explanation conversations."""
    conversations = []
    
    if len(code) < min_length:
        return conversations
    
    # Extract function/class names
    functions = re.findall(r'def\s+(\w+)', code)
    classes = re.findall(r'class\s+(\w+)', code)
    
    if functions or classes:
        # Create explanation request
        entity = functions[0] if functions else classes[0]
        conversations.append({
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": f"Explain this code:\n\n{code[:400]}"},
                {"role": "assistant", "content": f"This code defines {entity}. It implements functionality for processing data."}
            ]
        })
    else:
        # Generic code explanation
        conversations.append({
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": f"What does this code do?\n\n{code[:400]}"},
                {"role": "assistant", "content": "This code implements a function that processes input data and returns results."}
            ]
        })
    
    return conversations


def convert_text_to_chat(
    text_file: Path,
    output_file: Path,
    max_samples: int = 10000,
    min_text_length: int = 100
) -> int:
    """Convert text file to chat format."""
    print(f"📝 Converting text data: {text_file}")
    
    texts = load_jsonl(text_file)
    print(f"   Loaded {len(texts)} text samples")
    
    conversations = []
    
    for text in tqdm(texts[:max_samples], desc="Processing texts"):
        if len(text) < min_text_length:
            continue
        
        # Create Q&A pairs
        qa_pairs = create_qa_from_text(text, min_length=min_text_length)
        conversations.extend(qa_pairs)
        
        if len(conversations) >= max_samples:
            break
    
    # Save to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in conversations[:max_samples]:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {len(conversations[:max_samples])} text conversations")
    return len(conversations[:max_samples])


def convert_code_to_chat(
    code_file: Path,
    output_file: Path,
    max_samples: int = 10000,
    min_code_length: int = 50
) -> int:
    """Convert code file to chat format."""
    print(f"💻 Converting code data: {code_file}")
    
    code_samples = load_jsonl(code_file)
    print(f"   Loaded {len(code_samples)} code samples")
    
    conversations = []
    
    for code in tqdm(code_samples[:max_samples], desc="Processing code"):
        if len(code) < min_code_length:
            continue
        
        # Create code explanations
        explanations = create_code_explanation(code, min_length=min_code_length)
        conversations.extend(explanations)
        
        if len(conversations) >= max_samples:
            break
    
    # Save to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in conversations[:max_samples]:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {len(conversations[:max_samples])} code conversations")
    return len(conversations[:max_samples])


def convert_all_to_chat(
    data_dir: str = "./data",
    output_file: str = "./data/chat_data_real.jsonl",
    max_samples: int = 10000,
    text_ratio: float = 0.6
) -> int:
    """
    Convert all text and code data to chat format.
    
    Args:
        data_dir: Data directory
        output_file: Output chat data file
        max_samples: Maximum total conversations
        text_ratio: Ratio of text vs code (0.6 = 60% text, 40% code)
    
    Returns:
        Total number of conversations created
    """
    data_path = Path(data_dir)
    output_path = Path(output_file)
    
    text_file = data_path / "text" / "wikitext.jsonl"
    code_file = data_path / "code" / "code.jsonl"
    
    print("="*80)
    print("Converting Real Data to Chat Format")
    print("="*80)
    
    all_conversations = []
    
    # Convert text data
    if text_file.exists():
        text_samples = int(max_samples * text_ratio)
        text_convs = convert_text_to_chat(text_file, output_path, max_samples=text_samples)
        all_conversations.extend(text_convs)
    else:
        print(f"⚠️ Text file not found: {text_file}")
    
    # Convert code data
    if code_file.exists():
        code_samples = max_samples - len(all_conversations)
        code_convs = convert_code_to_chat(code_file, output_path, max_samples=code_samples)
        all_conversations.extend(code_convs)
    else:
        print(f"⚠️ Code file not found: {code_file}")
    
    # Combine and save
    if all_conversations:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        
        print(f"\n✅ Total: {len(all_conversations)} conversations saved to {output_path}")
    else:
        print("\n❌ No conversations created!")
    
    return len(all_conversations)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert real data to chat format')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_file', type=str, default='./data/chat_data_real.jsonl')
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--text_ratio', type=float, default=0.6)
    
    args = parser.parse_args()
    
    convert_all_to_chat(
        data_dir=args.data_dir,
        output_file=args.output_file,
        max_samples=args.max_samples,
        text_ratio=args.text_ratio
    )

