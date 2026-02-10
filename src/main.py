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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")

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
        self.recent_captures = deque(maxlen=max_recent)  # (face_img, camera, person_id, timestamp)
        self.total_captures = 0
        self.captures_per_camera = {}
        self.start_time = time.time()
        self.active_tracks = {}  # camera -> count of currently tracked objects

    def record_capture(self, face_img, camera_name, person_id):
        """Record a new capture event."""
        with self.lock:
            thumb = cv2.resize(face_img, (80, 80)) if face_img.size > 0 else np.zeros((80,80,3), dtype=np.uint8)
            self.recent_captures.append({
                'thumb': thumb,
                'camera': camera_name,
                'person_id': person_id,
                'time': datetime.now().strftime("%H:%M:%S"),
            })
            self.total_captures += 1
            self.captures_per_camera[camera_name] = self.captures_per_camera.get(camera_name, 0) + 1

    def update_active_tracks(self, camera_name, count):
        """Update the number of currently active tracks for a camera."""
        with self.lock:
            self.active_tracks[camera_name] = count

    def get_snapshot(self):
        """Return a copy of current stats."""
        with self.lock:
            return {
                'recent': list(self.recent_captures),
                'total': self.total_captures,
                'per_camera': dict(self.captures_per_camera),
                'active_tracks': dict(self.active_tracks),
                'uptime': time.time() - self.start_time,
            }


def render_dashboard(stats, width=400, height=600):
    """Render the stats dashboard as an image."""
    dash = np.full((height, width, 3), COL_BG, dtype=np.uint8)
    y = 0

    # -- Header --
    header_h = 50
    cv2.rectangle(dash, (0, 0), (width, header_h), COL_PANEL, -1)
    cv2.putText(dash, "DETECTION DASHBOARD", (15, 33),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_ACCENT, 2)
    y = header_h + 10

    # -- Uptime --
    uptime_s = int(stats['uptime'])
    hours, remainder = divmod(uptime_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    cv2.putText(dash, f"Uptime: {uptime_str}", (15, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1)
    y += 30

    # -- Separator --
    cv2.line(dash, (15, y), (width - 15, y), COL_BORDER, 1)
    y += 15

    # -- Total captures --
    cv2.putText(dash, "TOTAL CAPTURES", (15, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1)
    y += 25
    cv2.putText(dash, str(stats['total']), (15, y + 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1.3, COL_ACCENT, 2)
    y += 50

    # -- Per camera stats --
    cv2.line(dash, (15, y), (width - 15, y), COL_BORDER, 1)
    y += 15
    cv2.putText(dash, "PER CAMERA", (15, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1)
    y += 28

    for cam_name in sorted(set(list(stats['per_camera'].keys()) + list(stats['active_tracks'].keys()))):
        captures = stats['per_camera'].get(cam_name, 0)
        active = stats['active_tracks'].get(cam_name, 0)

        cv2.putText(dash, cam_name, (20, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1)
        cv2.putText(dash, f"Saved: {captures}", (160, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_GREEN, 1)
        cv2.putText(dash, f"Tracking: {active}", (280, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_CYAN, 1)
        y += 25

    y += 10

    # -- Recent captures --
    cv2.line(dash, (15, y), (width - 15, y), COL_BORDER, 1)
    y += 15
    cv2.putText(dash, "RECENT CAPTURES", (15, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1)
    y += 30

    recent = stats['recent']
    if not recent:
        cv2.putText(dash, "No captures yet", (20, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_DIM, 1)
    else:
        thumb_size = 64
        padding = 8
        for entry in reversed(recent):
            if y + thumb_size + padding > height - 10:
                break

            thumb = cv2.resize(entry['thumb'], (thumb_size, thumb_size))

            # Border
            cv2.rectangle(dash, (14, y - 1), (15 + thumb_size + 1, y + thumb_size + 1), COL_BORDER, 1)
            # Thumbnail
            dash[y:y + thumb_size, 15:15 + thumb_size] = thumb

            # Info text
            tx = 15 + thumb_size + 12
            cv2.putText(dash, f"Person {entry['person_id']}", (tx, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_TEXT, 1)
            cv2.putText(dash, entry['camera'], (tx, y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_GREEN, 1)
            cv2.putText(dash, entry['time'], (tx, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_DIM, 1)

            y += thumb_size + padding + 4

    return dash


class CameraProcessor:
    """Encapsulates one camera's processing loop."""

    def __init__(self, cam_id, name, url, detector, output_dir, stats):
        self.cam_id = cam_id
        self.name = name
        self.url = url
        self.detector = detector
        self.stats = stats
        self.output_dir = os.path.join(output_dir, name)
        ensure_dir(self.output_dir)

        self.tracker = CentroidTracker(maxDisappeared=40, maxDistance=TRACKING_DISTANCE)
        self.object_captures = {}
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

    def _process_loop(self):
        while not self.stopped and not shutdown_event.is_set():
            loop_start = time.time()

            frame = self.vs.read()
            if frame is None:
                time.sleep(0.05)
                continue

            detections = self.detector.detect(frame)
            rects = []

            for det in detections:
                x, y, fw, fh = det['bbox']
                confidence = det['confidence']
                landmarks = det['landmarks']

                x = max(0, x)
                y = max(0, y)
                fw = min(self.frame_w - x, fw)
                fh = min(self.frame_h - y, fh)

                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                if not is_valid_face(x, y, fw, fh, landmarks):
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (128, 128, 128), 1)
                    continue

                rects.append((x, y, x + fw, y + fh))

            objects = self.tracker.update(rects)

            # Update active track count for stats
            self.stats.update_active_tracks(self.name, len(objects))

            for (objectID, centroid) in objects.items():
                best_rect = None
                min_dist = float('inf')

                for rect in rects:
                    rx, ry, rx2, ry2 = rect
                    rcx = (rx + rx2) // 2
                    rcy = (ry + ry2) // 2
                    d = np.linalg.norm(np.array([rcx, rcy]) - centroid)
                    if d < min_dist:
                        min_dist = d
                        best_rect = rect

                already_captured = objectID in self.object_captures

                if best_rect:
                    x, y, x2, y2 = best_rect
                    w_curr = x2 - x
                    h_curr = y2 - y

                    if already_captured:
                        color = (255, 255, 0)
                        label = f"ID {objectID} SAVED"
                        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        color = (0, 255, 0)
                        label = f"ID {objectID}"
                        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Capture once
                        person_dir = os.path.join(self.output_dir, f"person_{objectID}")
                        ensure_dir(person_dir)
                        timestamp = get_timestamp()

                        face_img = frame[y:y2, x:x2]
                        if face_img.size > 0:
                            save_image(face_img, os.path.join(person_dir, "face.jpg"))
                            # Push to stats dashboard
                            self.stats.record_capture(face_img, self.name, objectID)

                        bx_center = x + w_curr // 2
                        by_start = max(0, y - int(h_curr * 0.5))
                        by_end = min(self.frame_h, y + int(h_curr * 5))
                        bx_start = max(0, bx_center - int(w_curr * 1.5))
                        bx_end = min(self.frame_w, bx_center + int(w_curr * 1.5))

                        body_img = frame[by_start:by_end, bx_start:bx_end]
                        if body_img.size > 0:
                            save_image(body_img, os.path.join(person_dir, "body.jpg"))

                        metadata = {
                            "person_id": objectID,
                            "camera": self.name,
                            "timestamp": timestamp,
                            "face_bbox": [x, y, w_curr, h_curr],
                            "score": w_curr * h_curr,
                        }
                        log_metadata(metadata, os.path.join(person_dir, "metadata.json"))
                        self.object_captures[objectID] = True
                        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 3)
                        print(f"[CAM {self.cam_id}] Captured Person {objectID}")
                else:
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 165, 255), -1)

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

    # Shared stats tracker
    stats = StatsTracker(max_recent=5)

    detector_input_size = (640, 640)
    detector = FaceDetectorGPU(
        model_path=MODEL_PATH,
        input_size=detector_input_size,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=0.3,
    )

    processors = []
    for i, url in enumerate(urls):
        name = names[i] if i < len(names) else f"Camera_{i}"
        proc = CameraProcessor(
            cam_id=i, name=name, url=url,
            detector=detector, output_dir=OUTPUT_DIR,
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
            # Camera view
            frames = [proc.get_display_frame() for proc in processors]

            if len(processors) == 1 and frames[0] is not None:
                display = cv2.resize(frames[0], (960, 720))
            elif any(f is not None for f in frames):
                display = build_grid(frames, cols=grid_cols)
            else:
                time.sleep(0.05)
                continue

            cv2.imshow(WINDOW_CAMERAS, display)

            # Dashboard view
            snapshot = stats.get_snapshot()
            dashboard = render_dashboard(snapshot, width=400, height=600)
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
