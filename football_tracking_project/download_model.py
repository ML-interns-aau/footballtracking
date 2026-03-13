import requests
import sys
from pathlib import Path

def download_file(url, local_filename):
    print(f"Starting download of {local_filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = r.headers.get('content-length')
        
        with open(local_filename, 'wb') as f:
            if total_length is None: # no content length header
                f.write(r.content)
            else:
                dl = 0
                total_length = int(total_length)
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        dl += len(chunk)
                        f.write(chunk)
                        done = int(50 * dl / total_length)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl/(1024*1024):.1f} MB / {total_length/(1024*1024):.1f} MB")
                        sys.stdout.flush()
    print(f"\nDownload complete: {local_filename}")

if __name__ == "__main__":
    url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt"
    download_file(url, "yolov8m_fixed.pt")
