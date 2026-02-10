import cv2
import time
import os
import numpy as np
from dotenv import load_dotenv
from video_stream import VideoStream
from tracker import CentroidTracker
from utils import ensure_dir, save_image, log_metadata, get_timestamp

# Load environment variables
load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "captures")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.85))
TRACKING_DISTANCE = int(os.getenv("TRACKING_DISTANCE", 50))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", 40))
MAX_FACE_ASPECT_RATIO = float(os.getenv("MAX_FACE_ASPECT_RATIO", 2.0))

# Model Path (YuNet)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")


def is_valid_face(x, y, w_box, h_box, landmarks, frame_shape):
    """Validates that a detection is likely a real face, not a leaf or noise."""
    # 1. Minimum size
    if w_box < MIN_FACE_SIZE or h_box < MIN_FACE_SIZE:
        return False
    
    # 2. Aspect ratio (real faces are roughly square, 1:1 to 1:1.5)
    aspect = max(w_box, h_box) / max(min(w_box, h_box), 1)
    if aspect > MAX_FACE_ASPECT_RATIO:
        return False
    
    # 3. Landmark sanity check: eyes, nose, mouth should be INSIDE the bounding box
    # landmarks: [right_eye_x, right_eye_y, left_eye_x, left_eye_y, nose_x, nose_y, ...]
    if landmarks is not None and len(landmarks) >= 10:
        for i in range(0, 10, 2):
            lx, ly = int(landmarks[i]), int(landmarks[i + 1])
            if lx < x or lx > x + w_box or ly < y or ly > y + h_box:
                return False
        
        # 4. Eye distance should be reasonable (at least 20% of face width)
        eye_dist = abs(landmarks[0] - landmarks[2])
        if eye_dist < w_box * 0.15:
            return False
    
    return True


def setup_detector(w, h):
    """Create YuNet detector, trying CUDA first, falling back to CPU."""
    # Try CUDA backend first
    try:
        detector = cv2.FaceDetectorYN.create(
            model=MODEL_PATH,
            config="",
            input_size=(w, h),
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_CUDA,
            target_id=cv2.dnn.DNN_TARGET_CUDA
        )
        # Test if it actually works
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        detector.detect(dummy)
        print("[INFO] Using CUDA (GPU) backend")
        return detector
    except Exception:
        pass
    
    # Fallback to CPU
    detector = cv2.FaceDetectorYN.create(
        model=MODEL_PATH,
        config="",
        input_size=(w, h),
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=0.3,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU
    )
    print("[INFO] Using CPU backend (install opencv-contrib-python with CUDA for GPU)")
    return detector


def main():
    # Ensure output directory exists
    ensure_dir(OUTPUT_DIR)

    # Initialize tracker
    ct = CentroidTracker(maxDisappeared=40, maxDistance=TRACKING_DISTANCE)
    
    # Keep track of which object IDs we have already captured
    object_captures = {} 

    print("[INFO] Starting video stream...")
    vs = VideoStream(src=RTSP_URL)
    
    if vs.stopped:
        print("[ERROR] Stream failed to initialize. Exiting.")
        return
    
    # Get the first frame (already guaranteed by VideoStream constructor)
    frame = vs.read()
    h, w, _ = frame.shape
    print(f"[INFO] Stream ready. Resolution: {w}x{h}")
    
    # NOW start the background thread for continuous capture
    vs.start()

    # Initialize YuNet Face Detector (with GPU fallback)
    detector = setup_detector(w, h)
    
    print(f"[INFO] Detection config: confidence={CONFIDENCE_THRESHOLD}, min_face={MIN_FACE_SIZE}px, max_aspect={MAX_FACE_ASPECT_RATIO}")

    frame_count = 0
    fps_start = time.time()

    while True:
        frame = vs.read()
        if frame is None:
            continue

        frame_count += 1

        # Detect faces
        results = detector.detect(frame)
        faces = results[1]

        rects = []
        
        if faces is not None:
            for face in faces:
                x, y, w_box, h_box = map(int, face[0:4])
                
                # Clamp to frame bounds
                x = max(0, x)
                y = max(0, y)
                w_box = min(w - x, w_box)
                h_box = min(h - y, h_box)

                confidence = face[14]
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # Landmark-based validation
                landmarks = face[4:14]
                if not is_valid_face(x, y, w_box, h_box, landmarks, frame.shape):
                    # Draw rejected detection in gray (for debug)
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (128, 128, 128), 1)
                    cv2.putText(frame, f"REJECTED {confidence:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 128, 128), 1)
                    continue

                rects.append((x, y, x + w_box, y + h_box))

        # Update tracker
        objects = ct.update(rects)

        # Loop over tracked objects
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
            
            already_captured = objectID in object_captures
            
            if best_rect:
                x, y, x2, y2 = best_rect
                w_curr = x2 - x
                h_curr = y2 - y
                current_score = w_curr * h_curr

                if already_captured:
                    # Already captured — show cyan box
                    color = (255, 255, 0)  # Cyan
                    label = f"ID {objectID} SAVED"
                    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    # Not yet captured — show green box, attempt capture
                    color = (0, 255, 0)  # Green
                    label = f"ID {objectID}"
                    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Save once per person
                    person_dir = os.path.join(OUTPUT_DIR, f"person_{objectID}")
                    ensure_dir(person_dir)
                    
                    timestamp = get_timestamp()
                    
                    # Crop Face
                    face_img = frame[y:y2, x:x2]
                    if face_img.size > 0:
                        save_image(face_img, os.path.join(person_dir, "face.jpg"))
                    
                    # Crop Extended Body
                    bx_center = x + w_curr // 2
                    by_start = max(0, y - int(h_curr * 0.5))
                    by_end = min(h, y + int(h_curr * 5))
                    bx_start = max(0, bx_center - int(w_curr * 1.5))
                    bx_end = min(w, bx_center + int(w_curr * 1.5))
                    
                    body_img = frame[by_start:by_end, bx_start:bx_end]
                    if body_img.size > 0:
                        save_image(body_img, os.path.join(person_dir, "body.jpg"))

                    # Log Metadata
                    metadata = {
                        "person_id": objectID,
                        "timestamp": timestamp,
                        "face_bbox": [x, y, w_curr, h_curr],
                        "body_bbox": [bx_start, by_start, bx_end - bx_start, by_end - by_start],
                        "score": current_score,
                    }
                    log_metadata(metadata, os.path.join(person_dir, "metadata.json"))
                    
                    object_captures[objectID] = {'score': current_score, 'time': time.time()}
                    
                    # Flash red to indicate capture moment
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 3)
                    print(f"[INFO] Captured Person {objectID} -> {person_dir}")
            else:
                # No matching rect — tracked but face not detected this frame
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 165, 255), -1)
                label = f"ID {objectID}" + (" SAVED" if already_captured else "")
                cv2.putText(frame, label, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # Draw FPS counter
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            frame_count = 0
            fps_start = time.time()
        else:
            fps = 0
        
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("[INFO] Cleaning up...")
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
