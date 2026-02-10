import numpy as np
import onnxruntime as ort
import cv2
import threading


class FaceDetectorGPU:
    """
    YuNet face detector using ONNX Runtime with GPU acceleration.
    Uses DirectML (Windows) or CUDA as backend.
    """

    STRIDES = [8, 16, 32]

    def __init__(self, model_path, input_size=(640, 640), confidence_threshold=0.85,
                 nms_threshold=0.3, gpu_lock=None):
        self.input_w, self.input_h = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.lock = gpu_lock or threading.Lock()

        # Try GPU providers in order of preference
        available = ort.get_available_providers()
        
        gpu_provider = None
        for provider in ['CUDAExecutionProvider', 'DmlExecutionProvider']:
            if provider in available:
                gpu_provider = provider
                break

        if gpu_provider:
            try:
                self.session = ort.InferenceSession(
                    model_path,
                    providers=[gpu_provider, 'CPUExecutionProvider'],
                )
                active = self.session.get_providers()
                if gpu_provider in active:
                    label = "CUDA (GPU)" if gpu_provider == 'CUDAExecutionProvider' else "DirectML (GPU)"
                    print(f"[INFO] FaceDetector: Using {label}")
                else:
                    raise RuntimeError("GPU provider not active")
            except Exception as e:
                print(f"[WARN] GPU provider failed: {e}")
                self.session = ort.InferenceSession(
                    model_path, providers=['CPUExecutionProvider']
                )
                print("[INFO] FaceDetector: Using CPU (fallback)")
        else:
            self.session = ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider']
            )
            print("[INFO] FaceDetector: Using CPU")

        self.input_name = self.session.get_inputs()[0].name
        self.priors = self._generate_priors()

    def _generate_priors(self):
        """Generate anchor boxes for each feature map scale."""
        priors = []
        min_sizes_per_stride = {
            8: [10, 16, 24],
            16: [32, 48],
            32: [64, 96, 128],
        }

        for stride in self.STRIDES:
            feat_h = self.input_h // stride
            feat_w = self.input_w // stride
            min_sizes = min_sizes_per_stride[stride]

            for y in range(feat_h):
                for x in range(feat_w):
                    for min_size in min_sizes:
                        cx = (x + 0.5) * stride
                        cy = (y + 0.5) * stride
                        priors.append([cx, cy, min_size, min_size])

        return np.array(priors, dtype=np.float32)

    def _preprocess(self, frame):
        """Resize and convert to model input format."""
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        blob = resized.astype(np.float32)
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # Add batch dim
        return blob

    def _decode(self, cls_scores, obj_scores, bboxes, kps, scale_x, scale_y):
        """Decode raw model outputs into face detections."""
        scores = (cls_scores * obj_scores).flatten()

        mask = scores > self.confidence_threshold
        if not np.any(mask):
            return []

        scores = scores[mask]
        filtered_bboxes = bboxes[mask]
        filtered_kps = kps[mask]
        filtered_priors = self.priors[mask]

        # Decode bounding boxes
        cx = filtered_priors[:, 0] + filtered_bboxes[:, 0] * filtered_priors[:, 2]
        cy = filtered_priors[:, 1] + filtered_bboxes[:, 1] * filtered_priors[:, 3]
        w = filtered_priors[:, 2] * np.exp(filtered_bboxes[:, 2])
        h = filtered_priors[:, 3] * np.exp(filtered_bboxes[:, 3])

        x1 = (cx - w / 2) * scale_x
        y1 = (cy - h / 2) * scale_y
        x2 = (cx + w / 2) * scale_x
        y2 = (cy + h / 2) * scale_y

        # Decode landmarks
        decoded_kps = []
        for i in range(5):
            lx = (filtered_priors[:, 0] + filtered_kps[:, i * 2] * filtered_priors[:, 2]) * scale_x
            ly = (filtered_priors[:, 1] + filtered_kps[:, i * 2 + 1] * filtered_priors[:, 3]) * scale_y
            decoded_kps.append(np.stack([lx, ly], axis=1))

        landmarks = np.concatenate(decoded_kps, axis=1)
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.confidence_threshold, self.nms_threshold,
        )

        results = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                bx1, by1, bx2, by2 = boxes[i]
                fw = bx2 - bx1
                fh = by2 - by1
                results.append({
                    'bbox': (int(bx1), int(by1), int(fw), int(fh)),
                    'confidence': float(scores[i]),
                    'landmarks': landmarks[i].astype(int).tolist(),
                })

        return results

    def detect(self, frame):
        """
        Detect faces in frame. Thread-safe.
        Returns list of dicts with 'bbox', 'confidence', 'landmarks'.
        """
        h, w = frame.shape[:2]
        scale_x = w / self.input_w
        scale_y = h / self.input_h

        blob = self._preprocess(frame)

        with self.lock:
            outputs = self.session.run(None, {self.input_name: blob})

        # Concatenate outputs across scales
        cls = np.concatenate([outputs[0], outputs[1], outputs[2]], axis=1)
        obj = np.concatenate([outputs[3], outputs[4], outputs[5]], axis=1)
        bbox = np.concatenate([outputs[6], outputs[7], outputs[8]], axis=1)
        kps = np.concatenate([outputs[9], outputs[10], outputs[11]], axis=1)

        return self._decode(cls[0], obj[0], bbox[0], kps[0], scale_x, scale_y)
