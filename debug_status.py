import requests
import sys

try:
    job_id = "24571ba7"
    response = requests.get(f"http://phoenix:8090/api/train/status/{job_id}")
    print(response.text)
except Exception as e:
    print(e)
