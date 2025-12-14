
import requests
import json
import sys

def resume_specific_job(job_id, config_path="configs/baseline.json"):
    url = "http://phoenix:8090/api/train/submit"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Add the magic field
    config['resume_job_id'] = job_id
    
    print(f"🚀 Resubmitting Job {job_id} to {url}...")
    try:
        resp = requests.post(url, json=config)
        if resp.status_code == 200:
            print("✅ Success! Job resurrected.")
            print(resp.json())
        else:
            print(f"❌ Failed: {resp.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    resume_specific_job("a6208f8d")
