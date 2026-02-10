import cv2
import time
import os
import sys
import signal
import numpy as np
import threading
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from video_stream import VideoStream
from tracker import CentroidTracker
from face_detector import FaceDetectorGPU
from vehicle_detector import VehicleDetectorGPU
from enhancer import AsyncEnhancer
from utils import ensure_dir, save_image, log_metadata, get_timestamp, scan_output_directory, generate_history_html
import webbrowser

# Load environment variables
load_dotenv()

RTSP_URLS = os.getenv("RTSP_URLS", "").split(",")
CAMERA_NAMES = os.getenv("CAMERA_NAMES", "").split(",")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "captures")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.85))
TRACKING_DISTANCE = int(os.getenv("TRACKING_DISTANCE", 50))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", 40))
MAX_FACE_ASPECT_RATIO = float(os.getenv("MAX_FACE_ASPECT_RATIO", 2.0))
TARGET_FPS = int(os.getenv("TARGET_FPS", 30))
DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL", 3))

# Vehicle settings

# Vehicle settings
VEHICLE_CONFIDENCE = float(os.getenv("VEHICLE_CONFIDENCE", 0.5))
VEHICLE_MIN_SIZE = int(os.getenv("VEHICLE_MIN_SIZE", 60))
VEHICLE_TRACKING_DISTANCE = int(os.getenv("VEHICLE_TRACKING_DISTANCE", 80))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
VEHICLE_MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.onnx")

while len(CAMERA_NAMES) < len(RTSP_URLS):
    CAMERA_NAMES.append(f"Camera_{len(CAMERA_NAMES)}")

shutdown_event = threading.Event()

# -- Colors (BGR) --
COL_BG       = (30, 30, 30)
COL_PANEL    = (45, 45, 45)
COL_BORDER   = (80, 80, 80)
COL_TEXT     = (220, 220, 220)
COL_DIM      = (140, 140, 140)
COL_ACCENT   = (255, 200, 60)
COL_GREEN    = (80, 200, 80)
COL_CYAN     = (200, 200, 0)
COL_ORANGE   = (0, 140, 255)
COL_YELLOW   = (0, 220, 255)


def is_valid_face(x, y, w_box, h_box, landmarks):
    """Validates that a detection is likely a real face."""
    if w_box < MIN_FACE_SIZE or h_box < MIN_FACE_SIZE:
        return False
    aspect = max(w_box, h_box) / max(min(w_box, h_box), 1)
    if aspect > MAX_FACE_ASPECT_RATIO:
        return False
    if landmarks and len(landmarks) >= 10:
        for i in range(0, 10, 2):
            lx, ly = landmarks[i], landmarks[i + 1]
            if lx < x or lx > x + w_box or ly < y or ly > y + h_box:
                return False
        eye_dist = abs(landmarks[0] - landmarks[2])
        if eye_dist < w_box * 0.15:
            return False
    return True


class StatsTracker:
    """Thread-safe statistics collector shared across all cameras."""

    def __init__(self, max_recent=4):
        self.lock = threading.Lock()
        self.max_recent = max_recent
        # Face stats
        self.recent_faces = deque(maxlen=max_recent)
        self.total_faces = 0
        self.faces_per_camera = {}
        # Vehicle stats
        self.recent_vehicles = deque(maxlen=max_recent)
        self.total_vehicles = 0
        self.vehicles_per_camera = {}
        self.vehicle_classes_count = {}  # class_name -> count
        # Tracking
        self.active_face_tracks = {}
        self.active_vehicle_tracks = {}
        self.start_time = time.time()
        
        # Load persistence
        self.load_persistence()

    def load_persistence(self):
        print("[INFO] Scanning output directory for history...")
        data = scan_output_directory(OUTPUT_DIR)
        self.total_faces = data.get('total_faces', 0)
        self.total_vehicles = data.get('total_vehicles', 0)
        self.faces_per_camera = data.get('faces_per_camera', {})
        self.vehicles_per_camera = data.get('vehicles_per_camera', {})
        self.vehicle_classes_count = data.get('vehicle_classes', {})
        
        # Populate recent lists
        history = data.get('history', [])
        # Split history into faces and vehicles
        hist_faces = [h for h in history if h['type'] == 'face']
        hist_vehicles = [h for h in history if h['type'] == 'vehicle']

        # Add top 5 to recents (reversed because deque appends to right)
        for item in reversed(hist_faces[:5]):
            img_path = item.get('image')
            if img_path and os.path.exists(img_path):
                thumb = cv2.imread(img_path)
                if thumb is not None:
                    thumb = cv2.resize(thumb, (64, 64))
                    self.recent_faces.append({
                        'thumb': thumb,
                        'camera': item['camera'],
                        'id': item['id'],
                        'time': item['timestamp'], # Note: format might differ slightly if not careful, but okay for now
                        'face_path': img_path,
                        'rect': (0,0,0,0) # Placeholder
                    })

        for item in reversed(hist_vehicles[:5]):
            img_path = item.get('image')
            if img_path and os.path.exists(img_path):
                thumb = cv2.imread(img_path)
                if thumb is not None:
                    thumb = cv2.resize(thumb, (96, 64))
                     # Determine class if in meta, else 'vehicle'
                    cls = item.get('class', 'vehicle')
                    self.recent_vehicles.append({
                        'thumb': thumb,
                        'camera': item['camera'],
                        'id': item['id'],
                        'class': cls,
                        'time': item['timestamp'],
                        'vehicle_path': img_path,
                        'rect': (0,0,0,0)
                    })
        print(f"[INFO] Loaded stats: {self.total_faces} faces, {self.total_vehicles} vehicles.")

    def generate_report(self):
        """Generates and opens the history HTML report."""
        print("[INFO] Generating history report...")
        # Re-scan to get latest list including what's currently in memory (easiest way ensures consistency)
        data = scan_output_directory(OUTPUT_DIR)
        report_path = os.path.join(OUTPUT_DIR, "history.html")
        path = generate_history_html(data.get('history', []), report_path)
        print(f"[INFO] Report generated: {path}")
        webbrowser.open(f"file://{path}")

    def record_face(self, face_img, camera_name, person_id, path=None):
        with self.lock:
            thumb = cv2.resize(face_img, (64, 64)) if face_img.size > 0 else np.zeros((64, 64, 3), dtype=np.uint8)
            self.recent_faces.append({
                'thumb': thumb,
                'camera': camera_name,
                'id': person_id,
                'time': datetime.now().strftime("%H:%M:%S"),
                'face_path': path,
            })
            self.total_faces += 1
            self.faces_per_camera[camera_name] = self.faces_per_camera.get(camera_name, 0) + 1

    def record_vehicle(self, vehicle_img, camera_name, vehicle_id, class_name, path=None):
        with self.lock:
            thumb = cv2.resize(vehicle_img, (96, 64)) if vehicle_img.size > 0 else np.zeros((64, 96, 3), dtype=np.uint8)
            self.recent_vehicles.append({
                'thumb': thumb,
                'camera': camera_name,
                'id': vehicle_id,
                'class': class_name,
                'time': datetime.now().strftime("%H:%M:%S"),
                'vehicle_path': path,
            })
            self.total_vehicles += 1
            self.vehicles_per_camera[camera_name] = self.vehicles_per_camera.get(camera_name, 0) + 1
            self.vehicle_classes_count[class_name] = self.vehicle_classes_count.get(class_name, 0) + 1

    def update_active_tracks(self, camera_name, face_count, vehicle_count):
        with self.lock:
            self.active_face_tracks[camera_name] = face_count
            self.active_vehicle_tracks[camera_name] = vehicle_count

    def get_snapshot(self):
        with self.lock:
            return {
                'recent_faces': list(self.recent_faces),
                'recent_vehicles': list(self.recent_vehicles),
                'total_faces': self.total_faces,
                'total_vehicles': self.total_vehicles,
                'faces_per_cam': dict(self.faces_per_camera),
                'vehicles_per_cam': dict(self.vehicles_per_camera),
                'vehicle_classes': dict(self.vehicle_classes_count),
                'active_faces': dict(self.active_face_tracks),
                'active_vehicles': dict(self.active_vehicle_tracks),
                'uptime': time.time() - self.start_time,
            }

    def store_rendered(self, faces_list, vehicles_list):
        """Store the last rendered lists so click targets match what's displayed."""
        with self.lock:
            self._rendered_faces = faces_list
            self._rendered_vehicles = vehicles_list

    def update_recent_path(self, item_type, idx, path):
        """Update the file path for a recent item (for detail view)."""
        with self.lock:
            if item_type == 'face':
                if 0 <= idx < len(self.recent_faces):
                    self.recent_faces[idx]['face_path'] = path
            elif item_type == 'vehicle':
                if 0 <= idx < len(self.recent_vehicles):
                    self.recent_vehicles[idx]['vehicle_path'] = path

    def get_click_target(self, x, y):
        """Check if a click at (x,y) hits a thumbnail. Returns file path or None."""
        with self.lock:
            # Use rendered lists (matches what user sees)
            for item in getattr(self, '_rendered_faces', []):
                if 'rect' in item and item.get('face_path') and not item.get('fp'):
                    rx, ry, rw, rh = item['rect']
                    if rx <= x <= rx + rw and ry <= y <= ry + rh:
                        return item['face_path']
            for item in getattr(self, '_rendered_vehicles', []):
                if 'rect' in item and item.get('vehicle_path') and not item.get('fp'):
                    rx, ry, rw, rh = item['rect']
                    if rx <= x <= rx + rw and ry <= y <= ry + rh:
                        return item['vehicle_path']
        return None

    def get_fp_target(self, x, y):
        """Check if a right-click at (x,y) hits a thumbnail. Returns (type, index, item) or (None, None, None)."""
        with self.lock:
            for i, item in enumerate(getattr(self, '_rendered_faces', [])):
                if 'rect' in item:
                    rx, ry, rw, rh = item['rect']
                    if rx <= x <= rx + rw and ry <= y <= ry + rh:
                        return 'face', i, item
            for i, item in enumerate(getattr(self, '_rendered_vehicles', [])):
                if 'rect' in item:
                    rx, ry, rw, rh = item['rect']
                    if rx <= x <= rx + rw and ry <= y <= ry + rh:
                        return 'vehicle', i, item
        return None, None, None

    def mark_false_positive(self, item_type, item):
        """Mark a detection as false positive."""
        with self.lock:
            if item.get('fp'):
                return  # Already marked
            item['fp'] = True

            if item_type == 'face':
                self.total_faces = max(0, self.total_faces - 1)
                cam = item.get('camera', '')
                if cam in self.faces_per_camera:
                    self.faces_per_camera[cam] = max(0, self.faces_per_camera[cam] - 1)
                path_key = 'face_path'
            else:
                self.total_vehicles = max(0, self.total_vehicles - 1)
                cam = item.get('camera', '')
                if cam in self.vehicles_per_camera:
                    self.vehicles_per_camera[cam] = max(0, self.vehicles_per_camera[cam] - 1)
                cls = item.get('class', 'vehicle')
                if cls in self.vehicle_classes_count:
                    self.vehicle_classes_count[cls] = max(0, self.vehicle_classes_count[cls] - 1)
                path_key = 'vehicle_path'

            # Rename folder with _FP suffix
            file_path = item.get(path_key)
            if file_path:
                parent = os.path.dirname(file_path)
                if os.path.isdir(parent) and not parent.endswith('_FP'):
                    new_parent = parent + '_FP'
                    try:
                        os.rename(parent, new_parent)
                        item[path_key] = os.path.join(new_parent, os.path.basename(file_path))
                        fp_log = os.path.join(OUTPUT_DIR, 'false_positives.log')
                        with open(fp_log, 'a') as f:
                            f.write(f"{datetime.now().isoformat()} | {item_type} | {cam} | ID {item.get('id')} | {parent}\n")
                        print(f"[FP] Marked as false positive: {parent}")
                    except OSError as e:
                        print(f"[FP] Error renaming: {e}")



def _draw_text(img, text, pos, scale=0.42, color=COL_TEXT, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def render_dashboard(stats, width=420, height=720):
    """Render the stats dashboard as an image."""
    dash = np.full((height, width, 3), COL_BG, dtype=np.uint8)
    y = 0

    # -- Header --
    cv2.rectangle(dash, (0, 0), (width, 46), COL_PANEL, -1)
    _draw_text(dash, "DETECTION DASHBOARD", (10, 30), 0.5, COL_ACCENT, 2)
    # History button placeholder (visual only, click handled by coordinates)
    cv2.rectangle(dash, (width - 80, 10), (width - 10, 36), (60, 60, 60), -1)
    cv2.rectangle(dash, (width - 80, 10), (width - 10, 36), COL_BORDER, 1)
    _draw_text(dash, "HISTORY", (width - 72, 28), 0.4, COL_TEXT)
    
    # Enhancer Queue
    q_size = stats.get('queue_size', 0)
    col_q = COL_GREEN if q_size == 0 else (COL_ORANGE if q_size < 5 else (0, 0, 255))
    _draw_text(dash, f"Enhance Q: {q_size}", (width - 170, 28), 0.4, col_q)
    
    y = 56

    # Uptime
    up = int(stats['uptime'])
    _draw_text(dash, f"Uptime: {up // 3600:02d}:{(up % 3600) // 60:02d}:{up % 60:02d}", (15, y), 0.4, COL_DIM)
    y += 25

    # -- Totals --
    cv2.line(dash, (15, y), (width - 15, y), COL_BORDER, 1)
    y += 18
    _draw_text(dash, "TOTALS", (15, y), 0.4, COL_DIM)
    y += 22

    # Faces total
    _draw_text(dash, f"Faces:", (20, y), 0.5, COL_TEXT)
    _draw_text(dash, str(stats['total_faces']), (110, y), 0.5, COL_GREEN, 2)
    # Vehicles total
    _draw_text(dash, f"Vehicles:", (210, y), 0.5, COL_TEXT)
    _draw_text(dash, str(stats['total_vehicles']), (330, y), 0.5, COL_ORANGE, 2)
    y += 28

    # Vehicle class breakdown
    if stats['vehicle_classes']:
        classes_str = "  ".join([f"{name}: {cnt}" for name, cnt in sorted(stats['vehicle_classes'].items())])
        _draw_text(dash, classes_str, (20, y), 0.35, COL_DIM)
        y += 20

    # -- Per camera --
    cv2.line(dash, (15, y), (width - 15, y), COL_BORDER, 1)
    y += 18
    _draw_text(dash, "PER CAMERA", (15, y), 0.4, COL_DIM)
    y += 22

    all_cams = sorted(set(
        list(stats['faces_per_cam'].keys()) + list(stats['vehicles_per_cam'].keys()) +
        list(stats['active_faces'].keys()) + list(stats['active_vehicles'].keys())
    ))
    for cam in all_cams:
        f_saved = stats['faces_per_cam'].get(cam, 0)
        v_saved = stats['vehicles_per_cam'].get(cam, 0)
        f_track = stats['active_faces'].get(cam, 0)
        v_track = stats['active_vehicles'].get(cam, 0)

        _draw_text(dash, cam, (20, y + 12), 0.45, COL_TEXT)
        _draw_text(dash, f"F:{f_saved}/{f_track}t", (140, y + 12), 0.35, COL_GREEN)
        _draw_text(dash, f"V:{v_saved}/{v_track}t", (240, y + 12), 0.35, COL_ORANGE)
        y += 22
    y += 8

    # -- Recent faces --
    cv2.line(dash, (15, y), (width - 15, y), COL_BORDER, 1)
    y += 18
    _draw_text(dash, "RECENT FACES", (15, y), 0.4, COL_DIM)
    _draw_text(dash, "(Right-click = FP)", (140, y), 0.3, COL_DIM)
    y += 8

    thumb_h = 52
    for entry in reversed(stats['recent_faces']):
        if y + thumb_h + 6 > height - 160:
            break
        y += 4
        thumb = cv2.resize(entry['thumb'], (thumb_h, thumb_h))
        # Store rect for hit-testing
        entry['rect'] = (15, y, thumb_h, thumb_h)
        
        is_fp = entry.get('fp', False)
        border_color = (0, 0, 200) if is_fp else COL_BORDER
        cv2.rectangle(dash, (14, y - 1), (16 + thumb_h, y + thumb_h + 1), border_color, 1 if not is_fp else 2)
        dash[y:y + thumb_h, 15:15 + thumb_h] = thumb
        tx = 15 + thumb_h + 10
        if is_fp:
            _draw_text(dash, f"Person {entry['id']} [FP]", (tx, y + 16), 0.42, (0, 0, 200))
            _draw_text(dash, "FALSE POSITIVE", (tx, y + 32), 0.35, (0, 0, 200))
        else:
            _draw_text(dash, f"Person {entry['id']}", (tx, y + 16), 0.42, COL_TEXT)
            _draw_text(dash, entry['camera'], (tx, y + 32), 0.35, COL_GREEN)
        _draw_text(dash, entry['time'], (tx, y + 46), 0.32, COL_DIM)
        y += thumb_h + 4

    y += 8

    # -- Recent vehicles --
    cv2.line(dash, (15, y), (width - 15, y), COL_BORDER, 1)
    y += 18
    _draw_text(dash, "RECENT VEHICLES", (15, y), 0.4, COL_DIM)
    _draw_text(dash, "(Right-click = FP)", (160, y), 0.3, COL_DIM)
    y += 8

    vthumb_w, vthumb_h = 80, 52
    for entry in reversed(stats['recent_vehicles']):
        if y + vthumb_h + 6 > height - 10:
            break
        y += 4
        thumb = cv2.resize(entry['thumb'], (vthumb_w, vthumb_h))
        # Store rect for hit-testing
        entry['rect'] = (15, y, vthumb_w, vthumb_h)

        is_fp = entry.get('fp', False)
        border_color = (0, 0, 200) if is_fp else COL_BORDER
        cv2.rectangle(dash, (14, y - 1), (16 + vthumb_w, y + vthumb_h + 1), border_color, 1 if not is_fp else 2)
        dash[y:y + vthumb_h, 15:15 + vthumb_w] = thumb
        tx = 15 + vthumb_w + 10
        if is_fp:
            _draw_text(dash, f"Vehicle {entry['id']} [FP]", (tx, y + 16), 0.42, (0, 0, 200))
            _draw_text(dash, "FALSE POSITIVE", (tx, y + 32), 0.35, (0, 0, 200))
        else:
            _draw_text(dash, f"Vehicle {entry['id']} ({entry['class']})", (tx, y + 16), 0.42, COL_TEXT)
            _draw_text(dash, entry['camera'], (tx, y + 32), 0.35, COL_ORANGE)
        _draw_text(dash, entry['time'], (tx, y + 46), 0.32, COL_DIM)
        y += vthumb_h + 4

    return dash


class CameraProcessor:
    """Encapsulates one camera's processing loop."""

    def __init__(self, cam_id, name, url, face_detector, vehicle_detector, output_dir, stats, enhancer):
        self.cam_id = cam_id
        self.name = name
        self.url = url
        self.face_detector = face_detector
        self.vehicle_detector = vehicle_detector
        self.stats = stats
        self.enhancer = enhancer
        self.fix_1080n = os.getenv("FIX_1080N_ASPECT_RATIO", "False").lower() == "true"
        self.output_dir = os.path.join(output_dir, name)
        ensure_dir(self.output_dir)

        # Face tracking
        self.face_tracker = CentroidTracker(maxDisappeared=40, maxDistance=TRACKING_DISTANCE)
        self.face_captures = {}

        # Vehicle tracking
        self.vehicle_tracker = CentroidTracker(maxDisappeared=30, maxDistance=VEHICLE_TRACKING_DISTANCE)
        self.vehicle_captures = {}

        self.display_frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.frame_interval = 1.0 / TARGET_FPS
        self.frame_count = 0
        self.detection_interval = DETECTION_INTERVAL
        
        # Cache for drawing results between detection frames
        self.last_face_results = [] # list of (x, y, x2, y2, id, label, color)
        self.last_vehicle_results = []
        self.last_face_count = 0
        self.last_vehicle_count = 0

        print(f"[CAM {cam_id}] Initializing '{name}'...")
        self.vs = VideoStream(src=url)
        if self.vs.stopped:
            print(f"[CAM {cam_id}] ERROR: Failed to connect to '{name}'")
            self.stopped = True
            return

        frame = self.vs.read()
        self.frame_h, self.frame_w = frame.shape[:2]
        print(f"[CAM {cam_id}] '{name}' ready. Resolution: {self.frame_w}x{self.frame_h}")

        with self.lock:
            self.display_frame = frame.copy()

        self.vs.start()
        self.thread = threading.Thread(target=self._process_loop, daemon=True)

    def start(self):
        if not self.stopped:
            self.thread.start()

    def _process_faces(self, frame, run_detection=True):
        """Detect and process faces. Appends face rects to rects_out."""
        if run_detection:
            rects_out = []
            detections = self.face_detector.detect(frame)
            
            # Clear previous cache
            self.last_face_results = []

            for det in detections:
                x, y, fw, fh = det['bbox']
                landmarks = det['landmarks']
                x = max(0, x)
                y = max(0, y)
                fw = min(self.frame_w - x, fw)
                fh = min(self.frame_h - y, fh)

                if det['confidence'] < CONFIDENCE_THRESHOLD:
                    continue
                if not is_valid_face(x, y, fw, fh, landmarks):
                    # Cache invalid/rejected boxes too if we want to visualize them, but maybe not necessary for smooth video
                    # cv2.rectangle(frame, (x, y), (x + fw, y + fh), (128, 128, 128), 1)
                    continue

                rects_out.append((x, y, x + fw, y + fh))

            objects = self.face_tracker.update(rects_out)
            self.last_face_count = len(objects)

            for (objectID, centroid) in objects.items():
                best_rect = None
                min_dist = float('inf')
                for rect in rects_out:
                    rx, ry, rx2, ry2 = rect
                    d = np.linalg.norm(np.array([(rx + rx2) // 2, (ry + ry2) // 2]) - centroid)
                    if d < min_dist:
                        min_dist = d
                        best_rect = rect

                if best_rect:
                    x, y, x2, y2 = best_rect
                    w_curr, h_curr = x2 - x, y2 - y
                    
                    # Store result for drawing
                    color = (0, 255, 0)
                    text = f"ID {objectID}"
                    saved = False

                    if objectID in self.face_captures:
                        color = COL_CYAN
                        text = f"ID {objectID} SAVED"
                        saved = True
                    else:
                        person_dir = os.path.join(self.output_dir, f"person_{objectID}")
                        ensure_dir(person_dir)
                        timestamp = get_timestamp()

                        face_img = frame[y:y2, x:x2]
                        if face_img.size > 0:
                            face_path = os.path.join(person_dir, "face.jpg")
                            save_image(face_img, face_path)
                            # Queue for enhancement
                            self.enhancer.process(face_path)
                            # Update stats with image content AND path
                            self.stats.record_face(face_img, self.name, objectID, path=face_path)

                        bx_center = x + w_curr // 2
                        by_start = max(0, y - int(h_curr * 0.5))
                        by_end = min(self.frame_h, y + int(h_curr * 5))
                        bx_start = max(0, bx_center - int(w_curr * 1.5))
                        bx_end = min(self.frame_w, bx_center + int(w_curr * 1.5))
                        body_img = frame[by_start:by_end, bx_start:bx_end]
                        if body_img.size > 0:
                            save_image(body_img, os.path.join(person_dir, "body.jpg"))

                        log_metadata({
                            "person_id": objectID, "camera": self.name,
                            "timestamp": timestamp, "face_bbox": [x, y, w_curr, h_curr],
                        }, os.path.join(person_dir, "metadata.json"))
                        self.face_captures[objectID] = True
                        print(f"[CAM {self.cam_id}] Captured Person {objectID}")
                        
                        # Update drawing for this frame immediately after capture
                        color = (0, 0, 255) # Red flash for capture frame

                    self.last_face_results.append({
                        'rect': (x, y, x2, y2),
                        'text': text,
                        'color': color,
                        'thickness': 2 if not saved else 2, # can create visual diff
                        'saved': saved
                    })
                else:
                     # Tracking only (dot)
                     self.last_face_results.append({
                         'centroid': (centroid[0], centroid[1]),
                         'color': (0, 165, 255)
                     })

        # --- Draw phase ---
        for item in self.last_face_results:
            if 'rect' in item:
                x, y, x2, y2 = item['rect']
                cv2.rectangle(frame, (x, y), (x2, y2), item['color'], item.get('thickness', 2))
                cv2.putText(frame, item['text'], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, item['color'], 2)
            elif 'centroid' in item:
                cx, cy = item['centroid']
                cv2.circle(frame, (cx, cy), 4, item['color'], -1)

        return self.last_face_count

    def _process_vehicles(self, frame, run_detection=True):
        """Detect and process vehicles."""
        if run_detection:
            detections = self.vehicle_detector.detect(frame)
            vrects = []
            det_info = {}  # map rect -> detection info
            
            # Clear cache
            self.last_vehicle_results = []

            for det in detections:
                x, y, fw, fh = det['bbox']
                x = max(0, x)
                y = max(0, y)
                fw = min(self.frame_w - x, fw)
                fh = min(self.frame_h - y, fh)
                rect = (x, y, x + fw, y + fh)
                vrects.append(rect)
                det_info[rect] = det

            objects = self.vehicle_tracker.update(vrects)
            self.last_vehicle_count = len(objects)

            for (objectID, centroid) in objects.items():
                best_rect = None
                min_dist = float('inf')
                for rect in vrects:
                    rx, ry, rx2, ry2 = rect
                    d = np.linalg.norm(np.array([(rx + rx2) // 2, (ry + ry2) // 2]) - centroid)
                    if d < min_dist:
                        min_dist = d
                        best_rect = rect

                if best_rect:
                    x, y, x2, y2 = best_rect
                    det = det_info.get(best_rect, {})
                    class_name = det.get('class_name', 'vehicle')
                    
                    color = COL_ORANGE
                    text = f"V{objectID} {class_name}"
                    saved = False

                    if objectID in self.vehicle_captures:
                        color = COL_YELLOW
                        text = f"V{objectID} {class_name} SAVED"
                        saved = True
                    else:
                        vehicle_dir = os.path.join(self.output_dir, f"vehicle_{objectID}")
                        ensure_dir(vehicle_dir)
                        timestamp = get_timestamp()

                        vehicle_img = frame[y:y2, x:x2]
                        if vehicle_img.size > 0:
                            vehicle_path = os.path.join(vehicle_dir, "vehicle_crop.jpg")
                            save_image(vehicle_img, vehicle_path)
                            # Queue for enhancement
                            self.enhancer.process(vehicle_path)
                            self.stats.record_vehicle(vehicle_img, self.name, objectID, class_name, path=vehicle_path)

                        log_metadata({
                            "vehicle_id": objectID, "class": class_name,
                            "camera": self.name, "timestamp": timestamp,
                            "bbox": [x, y, x2 - x, y2 - y],
                            "confidence": det.get('confidence', 0),
                        }, os.path.join(vehicle_dir, "metadata.json"))
                        self.vehicle_captures[objectID] = True
                        print(f"[CAM {self.cam_id}] Captured Vehicle {objectID} ({class_name})")
                        
                        color = (0, 0, 255) # Red capture flash

                    self.last_vehicle_results.append({
                        'rect': (x, y, x2, y2),
                        'text': text,
                        'color': color,
                        'thickness': 2 if not saved else 2
                    })
                else:
                    self.last_vehicle_results.append({
                        'centroid': (centroid[0], centroid[1]),
                        'color': COL_ORANGE
                    })
        
        # --- Draw phase ---
        for item in self.last_vehicle_results:
            if 'rect' in item:
                x, y, x2, y2 = item['rect']
                cv2.rectangle(frame, (x, y), (x2, y2), item['color'], item.get('thickness', 2))
                cv2.putText(frame, item['text'], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, item['color'], 2)
            elif 'centroid' in item:
                cx, cy = item['centroid']
                cv2.circle(frame, (cx, cy), 4, item['color'], -1)

        return self.last_vehicle_count

    def _process_loop(self):
        while not self.stopped and not shutdown_event.is_set():
            loop_start = time.time()

            frame = self.vs.read()
            if frame is None:
                time.sleep(0.05)
                continue
            
            # Fix 1080N aspect ratio (960x1080 -> 1920x1080)
            if self.fix_1080n and frame.shape[1] == 960 and frame.shape[0] == 1080:
                frame = cv2.resize(frame, (1920, 1080))
            
            # Determine if we run detection this frame
            run_detection = (self.frame_count % self.detection_interval == 0)

            face_count = self._process_faces(frame, run_detection=run_detection)
            vehicle_count = self._process_vehicles(frame, run_detection=run_detection)
            
            self.frame_count += 1

            self.stats.update_active_tracks(self.name, face_count, vehicle_count)

            cv2.putText(frame, self.name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            with self.lock:
                self.display_frame = frame

            elapsed = time.time() - loop_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_display_frame(self):
        with self.lock:
            return self.display_frame.copy() if self.display_frame is not None else None

    def stop(self):
        self.stopped = True
        self.vs.stop()


def build_grid(frames, cols=2, cell_size=(640, 480)):
    """Arrange multiple camera frames into a grid."""
    n = len(frames)
    if n == 0:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * cell_size[1], cols * cell_size[0], 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        if frame is None:
            continue
        r = idx // cols
        c = idx % cols
        resized = cv2.resize(frame, cell_size)
        y1 = r * cell_size[1]
        x1 = c * cell_size[0]
        grid[y1:y1 + cell_size[1], x1:x1 + cell_size[0]] = resized

    return grid


def main():
    global CONFIDENCE_THRESHOLD, MIN_FACE_SIZE, MAX_FACE_ASPECT_RATIO, VEHICLE_CONFIDENCE
    ensure_dir(OUTPUT_DIR)

    urls = [u.strip() for u in RTSP_URLS if u.strip()]
    names = [n.strip() for n in CAMERA_NAMES]

    if not urls:
        print("[ERROR] No RTSP URLs configured. Check .env file.")
        return

    print(f"[INFO] Loading {len(urls)} camera(s)...")

    # Start background enhancer
    enhancer = AsyncEnhancer()
    enhancer.start()
    print("[INFO] Enhancer thread started.")

    stats = StatsTracker(max_recent=5)

    # Shared lock to serialize GPU inference (prevents DirectML conflicts)
    gpu_lock = threading.Lock()

    face_detector = FaceDetectorGPU(
        model_path=FACE_MODEL_PATH,
        input_size=(640, 640),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=0.3,
    )

    vehicle_detector = VehicleDetectorGPU(
        model_path=VEHICLE_MODEL_PATH,
        input_size=(640, 640),
        confidence_threshold=VEHICLE_CONFIDENCE,
        nms_threshold=0.45,
        min_size=VEHICLE_MIN_SIZE,
        gpu_lock=gpu_lock,
    )

    processors = []
    for i, url in enumerate(urls):
        name = names[i] if i < len(names) else f"Camera_{i}"
        proc = CameraProcessor(
            cam_id=i, name=name, url=url,
            face_detector=face_detector,
            vehicle_detector=vehicle_detector,
            output_dir=OUTPUT_DIR,
            stats=stats,
            enhancer=enhancer,
        )
        if not proc.stopped:
            processors.append(proc)

    if not processors:
        print("[ERROR] No cameras could be initialized.")
        return

    for proc in processors:
        proc.start()

    print(f"[INFO] All {len(processors)} camera(s) running. Press 'q' to quit.")

    WINDOW_CAMERAS = "Face Tracker"
    WINDOW_DASHBOARD = "Dashboard"
    WINDOW_SETTINGS = "Settings"

    cv2.namedWindow(WINDOW_CAMERAS, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_DASHBOARD, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_SETTINGS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_SETTINGS, 420, 1)

    # -- Trackbars for real-time tuning --
    conf_init = int(CONFIDENCE_THRESHOLD * 100)
    cv2.createTrackbar("Face Conf %", WINDOW_SETTINGS, conf_init, 95, lambda v: None)
    cv2.setTrackbarMin("Face Conf %", WINDOW_SETTINGS, 10)
    cv2.createTrackbar("Min Face px", WINDOW_SETTINGS, MIN_FACE_SIZE, 200, lambda v: None)
    cv2.setTrackbarMin("Min Face px", WINDOW_SETTINGS, 10)
    ar_init = int(MAX_FACE_ASPECT_RATIO * 10)
    cv2.createTrackbar("Aspect x10", WINDOW_SETTINGS, ar_init, 50, lambda v: None)
    cv2.setTrackbarMin("Aspect x10", WINDOW_SETTINGS, 10)
    vconf_init = int(VEHICLE_CONFIDENCE * 100)
    cv2.createTrackbar("Veh Conf %", WINDOW_SETTINGS, vconf_init, 95, lambda v: None)
    cv2.setTrackbarMin("Veh Conf %", WINDOW_SETTINGS, 10)

    # Mouse callback for dashboard interaction
    def on_dashboard_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check for History button click
            if 340 <= x <= 410 and 10 <= y <= 36:
                stats.generate_report()
                return

            target_path = stats.get_click_target(x, y)
            if target_path:
                print(f"[UI] Opening detail view for: {target_path}")
                show_detail_window(target_path)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click = mark false positive
            item_type, idx, item = stats.get_fp_target(x, y)
            if item_type is not None:
                stats.mark_false_positive(item_type, item)

    cv2.setMouseCallback(WINDOW_DASHBOARD, on_dashboard_click)

    try:
        while not shutdown_event.is_set():
            frames = [proc.get_display_frame() for proc in processors]

            # Responsive grid logic
            if len(processors) == 1 and frames[0] is not None:
                display = frames[0]
            else:
                cols = 2
                display = build_grid(frames, cols=cols)

            if display is not None:
                # Resize to fit screen if too large
                h, w = display.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    display = cv2.resize(display, (0, 0), fx=scale, fy=scale)
                cv2.imshow(WINDOW_CAMERAS, display)

            # -- Read trackbar values and update globals --
            try:
                new_conf = cv2.getTrackbarPos("Face Conf %", WINDOW_SETTINGS) / 100.0
                new_min = cv2.getTrackbarPos("Min Face px", WINDOW_SETTINGS)
                new_ar = cv2.getTrackbarPos("Aspect x10", WINDOW_SETTINGS) / 10.0
                new_vconf = cv2.getTrackbarPos("Veh Conf %", WINDOW_SETTINGS) / 100.0

                if new_conf != CONFIDENCE_THRESHOLD:
                    CONFIDENCE_THRESHOLD = new_conf
                    face_detector.detector.setScoreThreshold(new_conf)
                MIN_FACE_SIZE = new_min
                MAX_FACE_ASPECT_RATIO = new_ar
                if new_vconf != VEHICLE_CONFIDENCE:
                    VEHICLE_CONFIDENCE = new_vconf

                # Render settings info bar
                settings_bar = np.zeros((30, 420, 3), dtype=np.uint8)
                settings_bar[:] = COL_BG
                info = f"Conf:{CONFIDENCE_THRESHOLD:.0%}  Size:{MIN_FACE_SIZE}px  AR:{MAX_FACE_ASPECT_RATIO:.1f}  VehConf:{VEHICLE_CONFIDENCE:.0%}"
                cv2.putText(settings_bar, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_ACCENT, 1)
                cv2.imshow(WINDOW_SETTINGS, settings_bar)
            except cv2.error:
                pass  # Settings window was closed

            # Render dashboard
            snapshot = stats.get_snapshot()
            snapshot['queue_size'] = enhancer.queue.qsize()
            dashboard = render_dashboard(snapshot, width=420, height=720)
            cv2.imshow(WINDOW_DASHBOARD, dashboard)

            # Store rendered lists for accurate click targeting
            stats.store_rendered(snapshot['recent_faces'], snapshot['recent_vehicles'])

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                stats.generate_report()
    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] Shutting down...")
        shutdown_event.set()
        for proc in processors:
            proc.stop()
        enhancer.stop()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


def show_detail_window(original_path):
    """
    Opens a window showing Original vs Enhanced image side-by-side.
    Reads metadata to display info.
    """
    if not os.path.exists(original_path):
        print(f"[WARN] File not found: {original_path}")
        return

    base, ext = os.path.splitext(original_path)
    enhanced_path = f"{base}_enhanced{ext}"
    metadata_path = os.path.join(os.path.dirname(original_path), "metadata.json")

    # Load images
    img_orig = cv2.imread(original_path)
    if img_orig is None:
        return

    img_enh = cv2.imread(enhanced_path)
    # If enhanced doesn't exist yet (processing), show placeholder
    if img_enh is None:
        img_enh = np.zeros_like(img_orig)
        cv2.putText(img_enh, "Enhancing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Resize for display comfortably (max height 600)
    h, w = img_orig.shape[:2]
    disp_h = 400
    scale = disp_h / float(h)
    disp_w = int(w * scale)
    
    view_orig = cv2.resize(img_orig, (disp_w, disp_h))
    view_enh = cv2.resize(img_enh, (disp_w, disp_h))

    # Create side-by-side view
    margin = 20
    info_h = 100
    total_w = disp_w * 2 + margin
    total_h = disp_h + info_h

    canvas = np.full((total_h, total_w, 3), (30, 30, 30), dtype=np.uint8)

    # Paste images
    canvas[0:disp_h, 0:disp_w] = view_orig
    canvas[0:disp_h, disp_w + margin:disp_w * 2 + margin] = view_enh

    # Labels
    cv2.putText(canvas, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, "ENHANCED", (disp_w + margin + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Metadata info
    import json
    info_txt = "Metadata not found"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                info_txt = f"ID: {data.get('vehicle_id', data.get('person_id'))}   Cam: {data.get('camera')}   Time: {data.get('timestamp')}"
                if 'class' in data:
                    info_txt += f"   Class: {data['class']}"
                if 'confidence' in data:
                    info_txt += f"   Conf: {data['confidence']:.2f}"
        except:
            pass
    
    cv2.putText(canvas, info_txt, (10, disp_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, "Press any key to close this view", (10, disp_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    win_name = f"Detail View - {os.path.basename(original_path)}"
    cv2.imshow(win_name, canvas)


if __name__ == "__main__":
    main()
