#!/usr/bin/env python3
"""
Chat Client for Phoenix Gateway
Interacts with the JAX Inference Server via HTTP.
"""

import argparse
import requests
import json
import time
import sys
import uuid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://phoenix:8090", help="Gateway URL")
    parser.add_argument("--session", type=str, default=None, help="Session ID (default: random)")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # Session Setup
    session_id = args.session if args.session else f"user_{uuid.uuid4().hex[:8]}"
    url = f"{args.server.rstrip('/')}/api/chat"

    print(f"🔌 Connected to {args.server}")
    print(f"🆔 Session: {session_id}")
    print("-" * 50)

    # Initial Health Check
    try:
        health = requests.get(f"{args.server.rstrip('/')}/gateway/health", timeout=2)
        if health.status_code == 200:
            h_data = health.json()
            inf_status = h_data.get('inference_server', {}).get('status', 'unknown')
            print(f"🧠 Inference Server Status: {inf_status.upper()}")
            if inf_status != 'up':
                print("⚠️  Warning: Inference Server seems down.")
    except Exception as e:
        print(f"⚠️  Health check failed: {e}")

    print("\n💬 Type your message (or 'exit'):")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Bye!")
                break
                
            if not user_input.strip():
                continue

            payload = {
                "session_id": session_id,
                "message": user_input,
                "temperature": args.temperature,
                "max_new_tokens": 100
            }

            print("   Thinking...", end="", flush=True)
            start_t = time.time()
            
            try:
                response = requests.post(url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    print("\r" + " " * 20 + "\rBot:", data['response'])  # Clear "Thinking..." fully
                    print(f"   (Speed: {data.get('speed_tok_sec', 0):.1f} tok/s)")
                else:
                    print(f"\r❌ Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                print("\r❌ Connection failed. Check server URL.")
            except requests.exceptions.Timeout:
                print("\r❌ Response timed out.")
                
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
