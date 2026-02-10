import cv2
import threading
import queue
import time
import os
import numpy as np
from utils import ensure_dir, save_image

class AsyncEnhancer:
    """
    Background worker that enhances captured images.
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and unsharp masking.
    """

    def __init__(self):
        self.queue = queue.Queue()
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        
        # Load config from env or defaults
        self.clahe_clip = float(os.getenv("ENHANCE_CLAHE_CLIP", 2.0))
        self.sharpen_amount = float(os.getenv("ENHANCE_SHARPEN_AMOUNT", 1.5))
        print(f"[INFO] Enhancer initialized. CLAHE: {self.clahe_clip}, Sharpen: {self.sharpen_amount}")

    def start(self):
        self.thread.start()

    def process(self, image_path):
        """Add image path to enhancement queue."""
        self.queue.put(image_path)

    def stop(self):
        self.stopped = True
        self.queue.put(None)  # Sentinel to unblock queue

    def _worker(self):
        while not self.stopped:
            try:
                path = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if path is None:
                break

            try:
                self._enhance_image(path)
            except Exception as e:
                print(f"[WARN] Enhancer failed for {path}: {e}")
            finally:
                self.queue.task_done()

    def _enhance_image(self, path):
        if not os.path.exists(path):
            return

        img = cv2.imread(path)
        if img is None:
            return

        # 1. Upscale (2x using Bicubic)
        h, w = img.shape[:2]
        upscaled = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        # 2. Convert to LAB color space
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 3. Apply CLAHE to Lightness channel
        # Use configurable clip limit
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # 4. Merge and convert back to BGR
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 5. Unsharp Masking for sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, self.sharpen_amount, gaussian, 1.0 - self.sharpen_amount, 0)

        # Save as *_enhanced.jpg
        base, ext = os.path.splitext(path)
        save_path = f"{base}_enhanced{ext}"
        cv2.imwrite(save_path, enhanced)
        # print(f"[INFO] Enhanced image saved: {save_path}")
