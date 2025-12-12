#!/usr/bin/env python3
"""
Test client for MM-Rec Inference Server
Demonstrates how to use the deployed model
"""

import requests
import json
import sys

def test_standard_request(base_url: str = "http://localhost:8000"):
    """Test standard (non-streaming) request."""
    print("🧪 Testing Standard Request...")
    
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in simple terms."}
            ],
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": False
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Request successful!")
        print(f"   Response: {result['choices'][0]['message']['content']}")
        print(f"   Confidence: {result['choices'][0]['logprobs']['average_token_confidence']}")
        print(f"   Tokens: {result['usage']['total_tokens']}")
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(f"   {response.text}")


def test_streaming_request(base_url: str = "http://localhost:8000"):
    """Test streaming request."""
    print("\n🌊 Testing Streaming Request...")
    
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Count from 1 to 10"}
            ],
            "max_tokens": 30,
            "stream": True
        },
        stream=True
    )
    
    if response.status_code == 200:
        print("✅ Stream started!")
        print("   Output: ", end="", flush=True)
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk['choices'][0]['delta']
                        if 'content' in delta:
                            print(delta['content'], end="", flush=True)
                    except:
                        pass
        print("\n   ✅ Stream completed!")
    else:
        print(f"❌ Stream failed: {response.status_code}")


def test_health(base_url: str = "http://localhost:8000"):
    """Test health endpoint."""
    print("\n💊 Testing Health Endpoint...")
    
    response = requests.get(f"{base_url}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Server healthy!")
        print(f"   Status: {result['status']}")
        print(f"   Device: {result['device']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print("="*80)
    print("MM-Rec Inference Server - Test Client")
    print("="*80)
    print(f"Server: {base_url}\n")
    
    try:
        test_health(base_url)
        test_standard_request(base_url)
        test_streaming_request(base_url)
        
        print("\n" + "="*80)
        print("✅ All tests completed!")
        print("="*80)
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Could not connect to server at {base_url}")
        print("   Make sure the server is running:")
        print("   python mm_rec/scripts/serve_inference.py")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
