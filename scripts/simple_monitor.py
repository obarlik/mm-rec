import requests
import time
import sys
import argparse

def tail_logs(server_url, job_id, poll_interval=5):
    url = f"{server_url}/api/logs/file/{job_id}"
    print(f"📡 Simple Monitor: Watching Job {job_id}")
    print(f"🔗 Server: {server_url}")
    print("="*60)
    
    seen_length = 0
    fail_count = 0

    while True:
        try:
            # Fetch full log (simple & robust)
            # If server supports Range, we could optimize, but this is safe
            resp = requests.get(url)
            
            if resp.status_code == 200:
                fail_count = 0
                text = resp.text
                current_length = len(text)
                
                if current_length > seen_length:
                    # Print ONLY the new part
                    new_chunk = text[seen_length:]
                    print(new_chunk, end='')
                    sys.stdout.flush()
                    seen_length = current_length
            elif resp.status_code == 404:
                print(f"\r⚠️ Job log not found yet...", end='')
            else:
                print(f"\r⚠️ Server returned {resp.status_code}", end='')
                
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped.")
            break
        except Exception as e:
            fail_count += 1
            print(f"\r❌ Connection error ({fail_count}): {e}", end='')
            if fail_count > 5:
                time.sleep(10)
            else:
                time.sleep(poll_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True, help="Job ID to monitor")
    parser.add_argument("--server", default="http://phoenix:8090", help="Gateway URL")
    args = parser.parse_args()
    
    tail_logs(args.server, args.job_id)
