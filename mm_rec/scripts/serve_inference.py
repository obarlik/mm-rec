#!/usr/bin/env python3
"""
MM-Rec Standalone Inference Server
OpenAI-compatible API server for MM-Rec's custom architecture

Usage:
    python mm_rec/scripts/serve_inference.py --model_path ./checkpoints/final.pt --port 8000

Features:
    - OpenAI-compatible /v1/chat/completions endpoint
    - Streaming support
    - Explainability (confidence scores)
    - Custom MM-Rec features (uncertainty threshold, sparse routing stats)
"""

import torch
import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
from datetime import datetime

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import ChatCompletionAPI

# Try to import FastAPI (optional)
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  FastAPI not installed. Install with: pip install fastapi uvicorn")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible request format."""
    model: str = "mmrec"
    messages: list
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    # MM-Rec specific
    router_threshold: Optional[float] = None


class InferenceServer:
    """Standalone inference server for MM-Rec."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        vocab_size: int = 100256
    ):
        """
        Initialize inference server.
        
        Args:
            model_path: Path to checkpoint (optional, creates new model if None)
            device: Device (auto, cpu, cuda)
            vocab_size: Vocabulary size
        """
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Device: {self.device}")
        
        # Tokenizer
        print("🔤 Loading tokenizer...")
        self.tokenizer = get_tokenizer(model_name="gpt-4", vocab_size=vocab_size)
        print(f"✅ Tokenizer loaded (vocab_size={self.tokenizer.vocab_size})")
        
        # Model
        print("🤖 Loading model...")
        if model_path and os.path.exists(model_path):
            # Load from checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract config from checkpoint
            if 'args' in checkpoint:
                args = checkpoint['args']
                self.model = MMRec100M(
                    vocab_size=self.tokenizer.vocab_size,
                    expert_dim=args.get('expert_dim', 256),
                    num_layers=args.get('num_layers', 6),
                    num_heads=args.get('num_heads', 8),
                    ffn_dim=args.get('ffn_dim', 2048)
                ).to(self.device)
            else:
                # Default config
                self.model = MMRec100M(
                    vocab_size=self.tokenizer.vocab_size,
                    expert_dim=256,
                    num_layers=6,
                    num_heads=8,
                    ffn_dim=2048
                ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from {model_path}")
        else:
            # Create new model (for demo)
            print("⚠️  No checkpoint provided, creating new model (untrained)")
            self.model = MMRec100M(
                vocab_size=self.tokenizer.vocab_size,
                expert_dim=128,
                num_layers=2,
                num_heads=4,
                ffn_dim=512
            ).to(self.device)
            print("✅ Model created")
        
        self.model.eval()
        
        # API wrapper
        self.api = ChatCompletionAPI(self.model, self.tokenizer)
        
        print(f"✅ Inference server ready!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_completion(self, request: Dict[str, Any]) -> Dict:
        """
        Create chat completion (non-streaming).
        
        Args:
            request: Request dict with messages, max_tokens, etc.
        
        Returns:
            OpenAI-compatible response
        """
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 1.0)
        router_threshold = request.get("router_threshold")
        
        # Call API
        response = self.api.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            device=self.device,
            router_threshold=router_threshold
        )
        
        # Add MM-Rec specific metadata
        response["model"] = "mmrec"
        response["created"] = int(datetime.now().timestamp())
        
        return response
    
    def create_completion_stream(self, request: Dict[str, Any]):
        """
        Create streaming chat completion.
        
        Args:
            request: Request dict
        
        Yields:
            SSE-formatted chunks
        """
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 1.0)
        
        # Get stream generator
        stream = self.api.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            device=self.device,
            stream=True
        )
        
        # Yield chunks in SSE format
        for chunk in stream:
            chunk["model"] = "mmrec"
            yield f"data: {json.dumps(chunk)}\n\n"
        
        yield "data: [DONE]\n\n"


def create_app(server: InferenceServer) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(title="MM-Rec Inference Server", version="1.0.0")
    
    @app.get("/")
    async def root():
        return {
            "message": "MM-Rec Inference Server",
            "version": "1.0.0",
            "endpoints": ["/v1/chat/completions", "/health"]
        }
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "device": str(server.device)}
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint."""
        try:
            request_dict = request.dict()
            
            if request.stream:
                # Streaming response
                return StreamingResponse(
                    server.create_completion_stream(request_dict),
                    media_type="text/event-stream"
                )
            else:
                # Standard response
                response = server.create_completion(request_dict)
                return JSONResponse(content=response)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def main():
    parser = argparse.ArgumentParser(description="MM-Rec Inference Server")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda)")
    parser.add_argument("--vocab_size", type=int, default=100256,
                        help="Vocabulary size")
    
    args = parser.parse_args()
    
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI is required for the server.")
        print("   Install with: pip install fastapi uvicorn")
        return 1
    
    print("="*80)
    print("MM-Rec Standalone Inference Server")
    print("="*80)
    
    # Initialize server
    server = InferenceServer(
        model_path=args.model_path,
        device=args.device,
        vocab_size=args.vocab_size
    )
    
    # Create app
    app = create_app(server)
    
    print("\n" + "="*80)
    print(f"🚀 Starting server on http://{args.host}:{args.port}")
    print("="*80)
    print("\n📖 API Endpoints:")
    print(f"   POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"   GET  http://{args.host}:{args.port}/health")
    print("\n💡 Example usage:")
    print(f"""
    curl -X POST http://{args.host}:{args.port}/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{{
        "messages": [{{"role": "user", "content": "Hello!"}}],
        "max_tokens": 50,
        "stream": false
      }}'
    """)
    print("="*80 + "\n")
    
    # Run server
    uvicorn.run(app, host=args.host, port=args.port)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
