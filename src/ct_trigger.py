import os, time
import subprocess

WATCH_DIR = r"D:\XGBoost - Project\data\new_data"
PIPELINE_CMD = "python full_pipeline.py"

seen_files = set(os.listdir(WATCH_DIR)) if os.path.exists(WATCH_DIR) else set()

while True:
    if not os.path.exists(WATCH_DIR):
        time.sleep(10)
        continue
    current_files = set(os.listdir(WATCH_DIR))
    new_files = current_files - seen_files
    if new_files:
        print(f"New data detected: {new_files}")
        subprocess.run(PIPELINE_CMD, shell=True)
        seen_files = current_files
        # After training, optionally trigger deployment
    time.sleep(30)