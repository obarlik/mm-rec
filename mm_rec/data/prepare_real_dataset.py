"""
Gerçek Dataset Hazırlama
Küçük başlangıç için pratik dataset oluşturma
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import random

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.data.text_data_loader import (
    load_text_from_file,
    load_texts_from_directory,
    create_data_loaders
)


def prepare_wikipedia_sample(
    output_dir: str = "data/real",
    num_articles: int = 50,
    min_length: int = 500
) -> List[str]:
    """
    Wikipedia'dan örnek makaleler hazırla.
    Küçük başlangıç için pratik bir çözüm.
    
    Args:
        output_dir: Çıktı dizini
        num_articles: Kaç makale
        min_length: Minimum makale uzunluğu
    
    Returns:
        List of text strings
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Wikipedia'dan küçük bir örnek dataset oluştur
    # Gerçek kullanımda: Wikipedia dump indirip parse edilmeli
    
    sample_texts = [
        """
        Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
        
        The scope of AI is disputed: as machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
        """,
        """
        Machine learning (ML) is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.
        
        Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
        """,
        """
        Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them.
        
        The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.
        """,
        """
        Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional neural networks and Transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.
        """,
        """
        A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) and computer vision (CV).
        
        Like recurrent neural networks (RNNs), transformers are designed to process sequential input data, such as natural language, with applications towards tasks such as translation and text summarization. However, unlike RNNs, transformers process the entire input all at once. The attention mechanism provides context for any position in the input sequence. For example, if the input data is a natural language sentence, the transformer does not have to process one word at a time. This allows for more parallelization than RNNs and therefore reduces training times.
        """,
        """
        Computer science is the study of computation, automation, and information. Computer science spans theoretical disciplines (such as algorithms, theory of computation, information theory, and automation) to practical disciplines (including the design and implementation of hardware and software). Computer science is generally considered an area of academic research and distinct from computer programming.
        
        Algorithms and data structures are central to computer science. The theory of computation concerns abstract models of computation and general classes of problems that can be solved using them. The fields of cryptography and computer security involve studying the means for secure communication and for preventing security vulnerabilities.
        """,
        """
        Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.
        
        Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to the ABC programming language, which was inspired by SETL, capable of exception handling and interfacing with the Amoeba operating system. Its implementation began in December 1989.
        """,
        """
        The Internet is a global system of interconnected computer networks that use the Internet protocol suite (TCP/IP) to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks of local to global scope, linked by a broad array of electronic, wireless, and optical networking technologies.
        
        The Internet carries a vast range of information resources and services, such as the inter-linked hypertext documents and applications of the World Wide Web (WWW), electronic mail, telephony, and file sharing.
        """,
        """
        Quantum computing is a type of computation whose operations can harness the phenomena of quantum mechanics, such as superposition, interference, and entanglement. Devices that perform quantum computations are known as quantum computers. Though current quantum computers are too small to outperform usual (classical) computers for practical applications, they are believed to be capable of solving certain computational problems, such as integer factorization, substantially faster than classical computers.
        
        The study of quantum computing is a subfield of quantum information science. Quantum computing began in the early 1980s when physicist Paul Benioff proposed a quantum mechanical model of the Turing machine.
        """,
        """
        Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is a natural phenomenon, scientific evidence shows that human activities have been the main driver of climate change since the mid-20th century, primarily due to the burning of fossil fuels, which increases heat-trapping greenhouse gas levels in Earth's atmosphere.
        
        The effects of climate change include rising sea levels, more frequent extreme weather events, changes in precipitation patterns, and shifts in ecosystems. Addressing climate change requires reducing greenhouse gas emissions and adapting to the changes that are already occurring.
        """
    ]
    
    # Expand sample texts to reach desired number
    texts = []
    while len(texts) < num_articles:
        for sample in sample_texts:
            if len(texts) >= num_articles:
                break
            # Clean and add text
            cleaned = sample.strip().replace('\n', ' ').replace('  ', ' ')
            if len(cleaned) >= min_length:
                texts.append(cleaned)
    
    # Save to files
    output_file = output_path / "wikipedia_sample.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(texts))
    
    print(f"✅ Created Wikipedia sample dataset: {output_file}")
    print(f"   Articles: {len(texts)}")
    print(f"   Total characters: {sum(len(t) for t in texts):,}")
    
    return texts


def prepare_from_directory(
    input_dir: str,
    output_dir: str = "data/real",
    max_files: Optional[int] = None
) -> List[str]:
    """
    Bir dizindeki text dosyalarından dataset hazırla.
    
    Args:
        input_dir: Input dizini (text dosyaları içeren)
        output_dir: Output dizini
        max_files: Maksimum dosya sayısı
    
    Returns:
        List of text strings
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Load texts
    texts = load_texts_from_directory(str(input_path), max_files=max_files)
    
    if not texts:
        raise ValueError(f"No text files found in {input_dir}")
    
    # Save combined
    output_file = output_path / "combined_dataset.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(texts))
    
    print(f"✅ Created dataset from directory: {output_file}")
    print(f"   Files: {len(texts)}")
    print(f"   Total characters: {sum(len(t) for t in texts):,}")
    
    return texts


def split_train_val(
    texts: List[str],
    val_split: float = 0.1,
    output_dir: str = "data/real"
) -> tuple[List[str], List[str]]:
    """
    Texts'i train ve validation set'lerine ayır.
    
    Args:
        texts: Text listesi
        val_split: Validation split ratio (0.0-1.0)
        output_dir: Output dizini
    
    Returns:
        train_texts, val_texts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle
    random.seed(42)
    shuffled = texts.copy()
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * (1 - val_split))
    train_texts = shuffled[:split_idx]
    val_texts = shuffled[split_idx:] if split_idx < len(shuffled) else []
    
    # Save
    train_file = output_path / "train.txt"
    val_file = output_path / "val.txt"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_texts))
    
    if val_texts:
        with open(val_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(val_texts))
    
    print(f"✅ Split dataset:")
    print(f"   Train: {len(train_texts)} texts -> {train_file}")
    print(f"   Validation: {len(val_texts)} texts -> {val_file}")
    
    return train_texts, val_texts


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare real dataset for training")
    parser.add_argument(
        "--source",
        type=str,
        choices=["wikipedia", "directory"],
        default="wikipedia",
        help="Data source"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input directory (for 'directory' source)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/real",
        help="Output directory"
    )
    parser.add_argument(
        "--num_articles",
        type=int,
        default=50,
        help="Number of articles (for 'wikipedia' source)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum files to process (for 'directory' source)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📚 Real Dataset Preparation")
    print("=" * 70)
    print()
    
    # Prepare dataset
    if args.source == "wikipedia":
        texts = prepare_wikipedia_sample(
            output_dir=args.output_dir,
            num_articles=args.num_articles
        )
    elif args.source == "directory":
        if not args.input_dir:
            raise ValueError("--input_dir required for 'directory' source")
        texts = prepare_from_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_files=args.max_files
        )
    else:
        raise ValueError(f"Unknown source: {args.source}")
    
    # Split train/val
    train_texts, val_texts = split_train_val(
        texts,
        val_split=args.val_split,
        output_dir=args.output_dir
    )
    
    print()
    print("=" * 70)
    print("✅ Dataset preparation completed!")
    print("=" * 70)
    print()
    print("📝 Usage:")
    print(f"   python mm_rec/scripts/train_base_model.py \\")
    print(f"       --config tiny \\")
    print(f"       --data_dir {args.output_dir} \\")
    print(f"       --epochs 5")


if __name__ == "__main__":
    main()
