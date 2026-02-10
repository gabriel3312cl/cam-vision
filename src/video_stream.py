import cv2
import threading
import queue
import time
import os

class VideoStream:
    def __init__(self, src=0):
        # Force RTSP to use TCP (more reliable than UDP)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        
        self.src = src
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        
        print(f"[DEBUG] Connecting to stream...")
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        
        if not self.stream.isOpened():
            print("[ERROR] Could not open video stream. Check URL/Network.")
            self.stopped = True
            return
        
        # Wait for first frame (blocking, with retries)
        print("[DEBUG] Waiting for first frame...")
        for attempt in range(30):  # Try for up to 15 seconds
            ret, frame = self.stream.read()
            if ret and frame is not None:
                self.frame = frame
                print(f"[DEBUG] First frame received on attempt {attempt + 1}. Size: {frame.shape[1]}x{frame.shape[0]}")
                return
            time.sleep(0.5)
        
        print("[ERROR] Could not read first frame after 30 attempts. Check stream.")
        self.stopped = True

    def start(self):
        if not self.stopped:
            self.thread.start()
        return self

    def _update(self):
        consecutive_failures = 0
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print("[ERROR] Too many consecutive read failures. Stopping.")
                    self.stopped = True
                    return
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=3)
        if self.stream.isOpened():
            self.stream.release()

    def is_opened(self):
        return not self.stopped
