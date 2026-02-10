import os
import cv2
import json
import glob
from datetime import datetime

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_image(image, path):
    cv2.imwrite(path, image)

def log_metadata(metadata, log_file):
    # Load existing data if file exists
    data = {}
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass # File might be empty or corrupted, start fresh

    # Update with new metadata
    data.update(metadata)

    with open(log_file, 'w') as f:
        json.dump(data, f, indent=4)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def scan_output_directory(output_dir):
    """
    Scans the output directory to rebuild statistics and history.
    Returns a dictionary with totals, per-camera counts, and a list of all entries.
    """
    stats = {
        'total_faces': 0,
        'total_vehicles': 0,
        'faces_per_camera': {},
        'vehicles_per_camera': {},
        'vehicle_classes': {},
        'history': []
    }

    if not os.path.exists(output_dir):
        return stats

    # Walk through camera directories
    for cam_name in os.listdir(output_dir):
        cam_path = os.path.join(output_dir, cam_name)
        if not os.path.isdir(cam_path):
            continue

        # Check for person_* and vehicle_* folders
        for item_name in os.listdir(cam_path):
            item_path = os.path.join(cam_path, item_name)
            if not os.path.isdir(item_path):
                continue
            
            meta_path = os.path.join(item_path, "metadata.json")
            if not os.path.exists(meta_path):
                continue

            try:
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                
                # Normalize data (some might be list, some dict from old versions?)
                if isinstance(data, list):
                    data = data[0] if data else {}

                # Start collecting info
                entry = {
                    'camera': cam_name,
                    'timestamp': data.get('timestamp', ''),
                    'path': item_path,
                    'meta': data
                }

                if 'person_id' in data:
                    stats['total_faces'] += 1
                    stats['faces_per_camera'][cam_name] = stats['faces_per_camera'].get(cam_name, 0) + 1
                    entry['type'] = 'face'
                    entry['id'] = data['person_id']
                    # Check for image
                    if os.path.exists(os.path.join(item_path, "face.jpg")):
                        entry['image'] = os.path.join(item_path, "face.jpg")
                        entry['enhanced'] = os.path.exists(os.path.join(item_path, "face_enhanced.jpg"))
                    else:
                        entry['enhanced'] = False
                    
                elif 'vehicle_id' in data:
                    stats['total_vehicles'] += 1
                    stats['vehicles_per_camera'][cam_name] = stats['vehicles_per_camera'].get(cam_name, 0) + 1
                    cls = data.get('class', 'unknown')
                    stats['vehicle_classes'][cls] = stats['vehicle_classes'].get(cls, 0) + 1
                    entry['type'] = 'vehicle'
                    entry['id'] = data['vehicle_id']
                    entry['class'] = cls
                    # Check for image
                    if os.path.exists(os.path.join(item_path, "vehicle_crop.jpg")):
                        entry['image'] = os.path.join(item_path, "vehicle_crop.jpg")
                        entry['enhanced'] = os.path.exists(os.path.join(item_path, "vehicle_crop_enhanced.jpg"))
                    else:
                        entry['enhanced'] = False
                
                stats['history'].append(entry)
                
                stats['history'].append(entry)

            except Exception as e:
                print(f"[WARN] Failed to read metadata for {item_path}: {e}")

    # Sort history by timestamp descending
    stats['history'].sort(key=lambda x: x['timestamp'], reverse=True)
    return stats

def generate_history_html(history, output_file="history.html"):
    """Generates a standalone HTML gallery from the history list."""
    
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cam-Vision History</title>
        <style>
            body { background-color: #1a1a1a; color: #e0e0e0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }
            h1 { color: #00bcd4; text-align: center; margin-bottom: 30px; }
            .controls { text-align: center; margin-bottom: 20px; }
            .btn { background: #333; color: white; border: 1px solid #555; padding: 8px 16px; cursor: pointer; margin: 0 5px; border-radius: 4px; }
            .btn:hover, .btn.active { background: #00bcd4; border-color: #00bcd4; color: #000; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }
            .card { background: #2d2d2d; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.2s; }
            .card:hover { transform: translateY(-5px); }
            .card img { width: 100%; height: 200px; object-fit: cover; display: block; }
            .info { padding: 12px; }
            .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; margin-right: 5px; }
            .bg-face { background: #4caf50; color: white; }
            .bg-vehicle { background: #ff9800; color: black; }
            .bg-enhanced { background: #00bcd4; color: black; }
            .bg-pending { background: #555; color: #ccc; }
            .meta { font-size: 12px; color: #aaa; margin-top: 5px; }
            .id { font-size: 16px; font-weight: bold; color: #fff; }
            .timestamp { display: block; margin-top: 4px; font-size: 11px; color: #666; }
        </style>
        <script>
            function filter(type) {
                const cards = document.querySelectorAll('.card');
                const btns = document.querySelectorAll('.btn');
                btns.forEach(b => b.classList.remove('active'));
                document.getElementById('btn-' + type).classList.add('active');
                
                cards.forEach(card => {
                    if (type === 'all' || card.dataset.type === type) {
                        card.style.display = 'block';
                    } else {
                        card.style.display = 'none';
                    }
                });
            }
        </script>
    </head>
    <body>
        <h1>Detection History</h1>
        <div class="controls">
            <button id="btn-all" class="btn active" onclick="filter('all')">All</button>
            <button id="btn-face" class="btn" onclick="filter('face')">Faces</button>
            <button id="btn-vehicle" class="btn" onclick="filter('vehicle')">Vehicles</button>
        </div>
        <div class="grid">
    """

    for item in history:
        abs_img_path = item.get('image', '')
        rel_img_path = ""
        if abs_img_path and os.path.exists(abs_img_path):
            # Make path relative to the HTML file (which is in output_dir)
            try:
                rel_img_path = os.path.relpath(abs_img_path, os.path.dirname(output_file))
                # Ensure forward slashes for web compatibility
                rel_img_path = rel_img_path.replace(os.sep, '/')
            except ValueError:
                # Fallback if paths are on different drives
                rel_img_path = abs_img_path

        type_cls = "bg-face" if item['type'] == 'face' else "bg-vehicle"
        label = "Face" if item['type'] == 'face' else item.get('class', 'Vehicle').upper()
        
        status_cls = "bg-enhanced" if item.get('enhanced') else "bg-pending"
        status_label = "ENHANCED" if item.get('enhanced') else "PENDING"

        html += f"""
            <div class="card" data-type="{item['type']}">
                <a href="{rel_img_path}" target="_blank">
                    <img src="{rel_img_path}" alt="{item['type']}" loading="lazy">
                </a>
                <div class="info">
                    <div style="margin-bottom: 4px;">
                        <span class="badge {status_cls}">{status_label}</span>
                    </div>
                    <div>
                        <span class="badge {type_cls}">{label}</span>
                        <span class="id">#{item['id']}</span>
                    </div>
                    <div class="meta">Cam: {item['camera']}</div>
                    <span class="timestamp">{item['timestamp']}</span>
                </div>
            </div>
        """

    html += """
        </div>
    </body>
    </html>
    """

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html)
    
    return os.path.abspath(output_file)
