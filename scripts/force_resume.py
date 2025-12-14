
import requests
import json
import sys

def resume_specific_job(job_id, config_path="configs/baseline.json"):
    url = "http://phoenix:8090/api/train/submit"
    # Job ID to Resume (Based on LS output: fa5d0cb5 has Epoch 9 checkpoint)
    JOB_ID = "fa5d0cb5" 
    
    SUBMIT_URL = "http://phoenix:8090/api/train/submit"
    STATUS_URL = f"http://phoenix:8090/api/train/status/{JOB_ID}"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Add the magic field
    config['resume_job_id'] = JOB_ID # Changed to use JOB_ID
    
    print(f"🚀 Resubmitting Job {JOB_ID} to {SUBMIT_URL}...") # Changed to use JOB_ID and SUBMIT_URL
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
