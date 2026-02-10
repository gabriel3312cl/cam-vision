import os
import cv2
import json
from datetime import datetime

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_image(image, path):
    cv2.imwrite(path, image)

def log_metadata(metadata, log_file):
    # Load existing data if file exists
    data = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass # File might be empty or corrupted, start fresh

    data.append(metadata)

    with open(log_file, 'w') as f:
        json.dump(data, f, indent=4)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
