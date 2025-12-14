#!/usr/bin/env python3
import glob
import os
import json
import sys

print("="*40)
print("🔍 DEBUG: JOB CONFIGURATION & PATHS")
print(f"📂 CWD: {os.getcwd()}")
print("="*40)

# 1. Inspect Latest Config
print("\n[1] Checking Config File:")
files = glob.glob("workspace/*_config.json")
if not files:
    print("❌ No config files found in workspace/")
    sys.exit(1)

latest_file = max(files, key=os.path.getctime)
print(f"📄 Latest Config: {latest_file}")

try:
    with open(latest_file) as f:
        config = json.load(f)
    
    data_path = config.get('data_path')
    print(f"🔑 data_path value: '{data_path}'")
    
    if not data_path:
        print("❌ data_path is MISSING or EMPTY")
    else:
        # 2. Check Path Existence
        print("\n[2] Checking Path Existence:")
        exists = os.path.exists(data_path)
        status = "✅ FOUND" if exists else "❌ NOT FOUND"
        print(f"🔎 os.path.exists('{data_path}'): {status}")
        
        if not exists:
             print("\n[3] Deep Dive Diagnostics:")
             if os.path.isabs(data_path):
                 print("ℹ️ Path is Absolute")
                 parent = os.path.dirname(data_path)
                 if os.path.exists(parent):
                     print(f"✅ Parent dir exists: {parent}")
                     print(f"📂 Contents of parent:")
                     try:
                        print(os.listdir(parent))
                     except Exception as e:
                        print(f"❌ Error listing dir: {e}")
                 else:
                     print(f"❌ Parent dir MISSING: {parent}")
             else:
                 print("ℹ️ Path is Relative")
                 abs_path = os.path.abspath(data_path)
                 print(f"📍 Resolves to: {abs_path}")
                 print(f"❓ check abs: {os.path.exists(abs_path)}")

    # 3. Check Explicit Path
    print("\n[3] Checking Expected Explicit Path:")
    explicit = "workspace/data/combined_foundation.jsonl"
    print(f"🔎 Check '{explicit}': {os.path.exists(explicit)}")
    
    abs_explicit = os.path.abspath(explicit)
    print(f"📍 Abs Explicit: {abs_explicit}")

except Exception as e:
    print(f"❌ Error during debug: {e}")

print("="*40)
