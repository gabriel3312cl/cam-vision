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
from utils import ensure_dir, save_image, log_metadata, get_timestamp

# Load environment variables
load_dotenv()

RTSP_URLS = os.getenv("RTSP_URLS", "").split(",")
CAMERA_NAMES = os.getenv("CAMERA_NAMES", "").split(",")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "captures")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.85))
TRACKING_DISTANCE = int(os.getenv("TRACKING_DISTANCE", 50))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", 40))
MAX_FACE_ASPECT_RATIO = float(os.getenv("MAX_FACE_ASPECT_RATIO", 2.0))
TARGET_FPS = int(os.getenv("TARGET_FPS", 10))

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

    def __init__(self, max_recent=5):
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

    def record_face(self, face_img, camera_name, person_id):
        with self.lock:
            thumb = cv2.resize(face_img, (64, 64)) if face_img.size > 0 else np.zeros((64, 64, 3), dtype=np.uint8)
            self.recent_faces.append({
                'thumb': thumb,
                'camera': camera_name,
                'id': person_id,
                'time': datetime.now().strftime("%H:%M:%S"),
            })
            self.total_faces += 1
            self.faces_per_camera[camera_name] = self.faces_per_camera.get(camera_name, 0) + 1

    def record_vehicle(self, vehicle_img, camera_name, vehicle_id, class_name):
        with self.lock:
            thumb = cv2.resize(vehicle_img, (96, 64)) if vehicle_img.size > 0 else np.zeros((64, 96, 3), dtype=np.uint8)
            self.recent_vehicles.append({
                'thumb': thumb,
                'camera': camera_name,
                'id': vehicle_id,
                'class': class_name,
                'time': datetime.now().strftime("%H:%M:%S"),
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


def _draw_text(img, text, pos, scale=0.42, color=COL_TEXT, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def render_dashboard(stats, width=420, height=720):
    """Render the stats dashboard as an image."""
    dash = np.full((height, width, 3), COL_BG, dtype=np.uint8)
    y = 0

    # -- Header --
    cv2.rectangle(dash, (0, 0), (width, 46), COL_PANEL, -1)
    _draw_text(dash, "DETECTION DASHBOARD", (15, 30), 0.6, COL_ACCENT, 2)
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
    y += 8

    thumb_h = 52
    for entry in reversed(stats['recent_faces']):
        if y + thumb_h + 6 > height - 160:
            break
        y += 4
        thumb = cv2.resize(entry['thumb'], (thumb_h, thumb_h))
        cv2.rectangle(dash, (14, y - 1), (16 + thumb_h, y + thumb_h + 1), COL_BORDER, 1)
        dash[y:y + thumb_h, 15:15 + thumb_h] = thumb
        tx = 15 + thumb_h + 10
        _draw_text(dash, f"Person {entry['id']}", (tx, y + 16), 0.42, COL_TEXT)
        _draw_text(dash, entry['camera'], (tx, y + 32), 0.35, COL_GREEN)
        _draw_text(dash, entry['time'], (tx, y + 46), 0.32, COL_DIM)
        y += thumb_h + 4

    y += 8

    # -- Recent vehicles --
    cv2.line(dash, (15, y), (width - 15, y), COL_BORDER, 1)
    y += 18
    _draw_text(dash, "RECENT VEHICLES", (15, y), 0.4, COL_DIM)
    y += 8

    vthumb_w, vthumb_h = 80, 52
    for entry in reversed(stats['recent_vehicles']):
        if y + vthumb_h + 6 > height - 10:
            break
        y += 4
        thumb = cv2.resize(entry['thumb'], (vthumb_w, vthumb_h))
        cv2.rectangle(dash, (14, y - 1), (16 + vthumb_w, y + vthumb_h + 1), COL_BORDER, 1)
        dash[y:y + vthumb_h, 15:15 + vthumb_w] = thumb
        tx = 15 + vthumb_w + 10
        _draw_text(dash, f"Vehicle {entry['id']} ({entry['class']})", (tx, y + 16), 0.42, COL_TEXT)
        _draw_text(dash, entry['camera'], (tx, y + 32), 0.35, COL_ORANGE)
        _draw_text(dash, entry['time'], (tx, y + 46), 0.32, COL_DIM)
        y += vthumb_h + 4

    return dash


class CameraProcessor:
    """Encapsulates one camera's processing loop."""

    def __init__(self, cam_id, name, url, face_detector, vehicle_detector, output_dir, stats):
        self.cam_id = cam_id
        self.name = name
        self.url = url
        self.face_detector = face_detector
        self.vehicle_detector = vehicle_detector
        self.stats = stats
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

    def _process_faces(self, frame, rects_out):
        """Detect and process faces. Appends face rects to rects_out."""
        detections = self.face_detector.detect(frame)

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
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (128, 128, 128), 1)
                continue

            rects_out.append((x, y, x + fw, y + fh))

        objects = self.face_tracker.update(rects_out)

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

                if objectID in self.face_captures:
                    cv2.rectangle(frame, (x, y), (x2, y2), COL_CYAN, 2)
                    cv2.putText(frame, f"ID {objectID} SAVED", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CYAN, 2)
                else:
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {objectID}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    person_dir = os.path.join(self.output_dir, f"person_{objectID}")
                    ensure_dir(person_dir)
                    timestamp = get_timestamp()

                    face_img = frame[y:y2, x:x2]
                    if face_img.size > 0:
                        save_image(face_img, os.path.join(person_dir, "face.jpg"))
                        self.stats.record_face(face_img, self.name, objectID)

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
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 3)
                    print(f"[CAM {self.cam_id}] Captured Person {objectID}")
            else:
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 165, 255), -1)

        return len(objects)

    def _process_vehicles(self, frame):
        """Detect and process vehicles."""
        detections = self.vehicle_detector.detect(frame)
        vrects = []
        det_info = {}  # map rect -> detection info

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

                if objectID in self.vehicle_captures:
                    cv2.rectangle(frame, (x, y), (x2, y2), COL_YELLOW, 2)
                    cv2.putText(frame, f"V{objectID} {class_name} SAVED", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_YELLOW, 2)
                else:
                    cv2.rectangle(frame, (x, y), (x2, y2), COL_ORANGE, 2)
                    cv2.putText(frame, f"V{objectID} {class_name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_ORANGE, 2)

                    vehicle_dir = os.path.join(self.output_dir, f"vehicle_{objectID}")
                    ensure_dir(vehicle_dir)
                    timestamp = get_timestamp()

                    vehicle_img = frame[y:y2, x:x2]
                    if vehicle_img.size > 0:
                        save_image(vehicle_img, os.path.join(vehicle_dir, "vehicle_crop.jpg"))
                        self.stats.record_vehicle(vehicle_img, self.name, objectID, class_name)

                    log_metadata({
                        "vehicle_id": objectID, "class": class_name,
                        "camera": self.name, "timestamp": timestamp,
                        "bbox": [x, y, x2 - x, y2 - y],
                        "confidence": det.get('confidence', 0),
                    }, os.path.join(vehicle_dir, "metadata.json"))
                    self.vehicle_captures[objectID] = True
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 3)
                    print(f"[CAM {self.cam_id}] Captured Vehicle {objectID} ({class_name})")
            else:
                cv2.circle(frame, (centroid[0], centroid[1]), 4, COL_ORANGE, -1)

        return len(objects)

    def _process_loop(self):
        while not self.stopped and not shutdown_event.is_set():
            loop_start = time.time()

            frame = self.vs.read()
            if frame is None:
                time.sleep(0.05)
                continue

            face_rects = []
            face_count = self._process_faces(frame, face_rects)
            vehicle_count = self._process_vehicles(frame)

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
    ensure_dir(OUTPUT_DIR)

    urls = [u.strip() for u in RTSP_URLS if u.strip()]
    names = [n.strip() for n in CAMERA_NAMES]

    if not urls:
        print("[ERROR] No RTSP URLs configured. Check .env file.")
        return

    print(f"[INFO] Loading {len(urls)} camera(s)...")

    stats = StatsTracker(max_recent=5)

    # Shared lock to serialize GPU inference (prevents DirectML conflicts)
    gpu_lock = threading.Lock()

    face_detector = FaceDetectorGPU(
        model_path=FACE_MODEL_PATH,
        input_size=(640, 640),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=0.3,
        gpu_lock=gpu_lock,
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
    grid_cols = min(len(processors), 3)

    try:
        while not shutdown_event.is_set():
            frames = [proc.get_display_frame() for proc in processors]

            if len(processors) == 1 and frames[0] is not None:
                display = cv2.resize(frames[0], (960, 720))
            elif any(f is not None for f in frames):
                display = build_grid(frames, cols=grid_cols)
            else:
                time.sleep(0.05)
                continue

            cv2.imshow(WINDOW_CAMERAS, display)

            snapshot = stats.get_snapshot()
            dashboard = render_dashboard(snapshot, width=420, height=720)
            cv2.imshow(WINDOW_DASHBOARD, dashboard)

            key = cv2.waitKey(33) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("[INFO] Shutting down...")
        shutdown_event.set()
        for proc in processors:
            proc.stop()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
