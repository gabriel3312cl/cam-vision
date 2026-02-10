import cv2
import threading


class FaceDetectorGPU:
    """
    Face detector using OpenCV's built-in FaceDetectorYN (YuNet).
    Supports dynamic input sizes for better detection of small faces.
    Uses the same ONNX model but with OpenCV's optimized inference pipeline.
    """

    def __init__(self, model_path, input_size=(640, 640), confidence_threshold=0.5,
                 nms_threshold=0.3, gpu_lock=None):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.lock = gpu_lock or threading.Lock()
        self.model_path = model_path

        # Create detector with default size (will be updated per-frame)
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            input_size,
            confidence_threshold,
            nms_threshold,
        )

        # Try to set backend to GPU
        # OpenCV DNN backends: DNN_BACKEND_DEFAULT=0, DNN_BACKEND_OPENCV=3
        # OpenCV DNN targets: DNN_TARGET_CPU=0, DNN_TARGET_OPENCL=1
        try:
            self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            print("[INFO] FaceDetector: Using OpenCL (GPU)")
        except Exception:
            try:
                self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[INFO] FaceDetector: Using CPU")
            except Exception:
                print("[INFO] FaceDetector: Using default backend")

    def detect(self, frame):
        """
        Detect faces in frame. Thread-safe.
        Returns list of dicts with 'bbox', 'confidence', 'landmarks'.
        """
        h, w = frame.shape[:2]

        with self.lock:
            # Update input size to match actual frame dimensions
            self.detector.setInputSize((w, h))
            retval, faces = self.detector.detect(frame)

        results = []
        if faces is not None:
            for face in faces:
                x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                confidence = float(face[-1])

                # Extract landmarks (5 points: right eye, left eye, nose, right mouth, left mouth)
                landmarks = []
                for i in range(5):
                    lx = int(face[4 + i * 2])
                    ly = int(face[5 + i * 2])
                    landmarks.extend([lx, ly])

                results.append({
                    'bbox': (x, y, fw, fh),
                    'confidence': confidence,
                    'landmarks': landmarks,
                })

        return results
