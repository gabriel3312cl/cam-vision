import cv2
import multiprocessing
import queue
import time
import os
import numpy as np

# Note regarding multiprocessing on Windows:
# The worker function needs to be picklable.
# We'll use a standalone function or keep it simple.

def enhance_worker(input_queue, clahe_clip, sharpen_amount):
    """Worker process for image enhancement."""
    # Re-import cv2 here to be safe across processes, though usually fine.
    import cv2
    
    print(f"[INFO] Enhancer process started. PID: {os.getpid()}")
    
    while True:
        try:
            path = input_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if path is None:
            break

        try:
            if not os.path.exists(path):
                continue

            img = cv2.imread(path)
            if img is None:
                continue

            # 1. Upscale (2x using Bicubic)
            h, w = img.shape[:2]
            upscaled = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

            # 2. Convert to LAB color space
            lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # 3. Apply CLAHE to Lightness channel
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            cl = clahe.apply(l)

            # 4. Merge and convert back to BGR
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # 5. Unsharp Masking for sharpening
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, sharpen_amount, gaussian, 1.0 - sharpen_amount, 0)

            # Save as *_enhanced.jpg
            base, ext = os.path.splitext(path)
            save_path = f"{base}_enhanced{ext}"
            cv2.imwrite(save_path, enhanced)
            
        except Exception as e:
            print(f"[WARN] Enhancer failed for {path}: {e}")

class AsyncEnhancer:
    """
    Background worker that enhances captured images using a separate PROCESS.
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and unsharp masking.
    """

    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.process_worker = None
        
        # Load config from env or defaults
        self.clahe_clip = float(os.getenv("ENHANCE_CLAHE_CLIP", 2.0))
        self.sharpen_amount = float(os.getenv("ENHANCE_SHARPEN_AMOUNT", 1.5))
        print(f"[INFO] Enhancer initialized (Multiprocessing). CLAHE: {self.clahe_clip}, Sharpen: {self.sharpen_amount}")

    def start(self):
        self.process_worker = multiprocessing.Process(
            target=enhance_worker, 
            args=(self.queue, self.clahe_clip, self.sharpen_amount),
            daemon=True
        )
        self.process_worker.start()

    def process(self, image_path):
        """Add image path to enhancement queue."""
        self.queue.put(image_path)

    def stop(self):
        self.queue.put(None)  # Sentinel to unblock queue
        if self.process_worker:
            self.process_worker.join(timeout=2.0)
            if self.process_worker.is_alive():
                self.process_worker.terminate()
