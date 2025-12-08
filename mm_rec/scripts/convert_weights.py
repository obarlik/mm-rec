"""
Script to convert existing LLM weights to MM-Rec architecture.

Usage:
    python -m mm_rec.scripts.convert_weights \
        --source llama-7b.pt \
        --output mmrec-7b-converted.pt \
        --vocab_size 32000 \
        --model_dim 4096 \
        --num_layers 24
"""

import argparse
import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.model import MMRecModel
from mm_rec.utils.model_converter import convert_model_weights


def main():
    parser = argparse.ArgumentParser(
        description='Convert existing LLM weights to MM-Rec architecture'
    )
    
    # Model configuration
    parser.add_argument('--source', type=str, required=True,
                       help='Path to source model checkpoint (.pt, .pth, .safetensors)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for converted weights')
    parser.add_argument('--vocab_size', type=int, default=32000,
                       help='Vocabulary size (default: 32000)')
    parser.add_argument('--model_dim', type=int, default=4096,
                       help='Model dimension (default: 4096)')
    parser.add_argument('--num_layers', type=int, default=24,
                       help='Number of layers (default: 24)')
    parser.add_argument('--num_heads', type=int, default=32,
                       help='Number of attention heads (default: 32)')
    parser.add_argument('--ffn_dim', type=int, default=None,
                       help='FFN dimension (default: model_dim * 4)')
    parser.add_argument('--strict', action='store_true',
                       help='Strict mode: raise error on missing keys')
    parser.add_argument('--no_initialize_new', action='store_true',
                       help='Do not initialize new MM-Rec components')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MM-Rec Model Weight Converter")
    print("="*80)
    print(f"\nSource checkpoint: {args.source}")
    print(f"Output path: {args.output}")
    print(f"\nTarget model configuration:")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  model_dim: {args.model_dim}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  ffn_dim: {args.ffn_dim or args.model_dim * 4}")
    
    # Create target model
    print("\n📦 Creating MM-Rec model...")
    target_model = MMRecModel(
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=32768
    )
    
    print(f"✅ Model created: {target_model.get_num_params() / 1e9:.2f}B parameters")
    
    # Convert weights
    print("\n🔄 Converting weights...")
    try:
        converted_weights, report = convert_model_weights(
            source_checkpoint_path=args.source,
            target_model=target_model,
            output_path=args.output,
            strict=args.strict,
            initialize_new=not args.no_initialize_new
        )
        
        # Print report
        print("\n" + "="*80)
        print("📊 Conversion Report")
        print("="*80)
        print(f"\nTotal target keys: {report['total_keys']}")
        print(f"Converted keys: {report['converted_keys']} ({report['converted_keys']/report['total_keys']*100:.1f}%)")
        print(f"Missing keys: {len(report['missing_keys'])} ({len(report['missing_keys'])/report['total_keys']*100:.1f}%)")
        print(f"New MM-Rec keys: {len(report['new_keys'])}")
        
        if report.get('source_analysis'):
            print(f"\n📋 Source Model Analysis:")
            analysis = report['source_analysis']
            for key, value in analysis.items():
                if value is not None:
                    print(f"  {key}: {value}")
        
        if report['shape_mismatches']:
            print(f"\n⚠️ Shape Mismatches ({len(report['shape_mismatches'])}):")
            for mismatch in report['shape_mismatches'][:5]:
                print(f"  {mismatch['target_key']}:")
                print(f"    Target: {mismatch['target_shape']}")
                print(f"    Source: {mismatch['source_shape']}")
        
        if report['missing_keys']:
            print(f"\n⚠️ Missing Keys (first 10):")
            for key in report['missing_keys'][:10]:
                print(f"  - {key}")
            if len(report['missing_keys']) > 10:
                print(f"  ... and {len(report['missing_keys']) - 10} more")
        
        print("\n✅ Conversion completed!")
        print(f"   Converted weights: {args.output}")
        print(f"   Report: {args.output.replace('.pt', '_report.json').replace('.pth', '_report.json')}")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

