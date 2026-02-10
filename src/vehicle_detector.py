import numpy as np
import onnxruntime as ort
import cv2
import threading


# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}


class VehicleDetectorGPU:
    """
    YOLOv8n vehicle detector using ONNX Runtime with GPU acceleration.
    Detects cars, motorcycles, buses, and trucks.
    """

    def __init__(self, model_path, input_size=(640, 640), confidence_threshold=0.5,
                 nms_threshold=0.45, min_size=60, gpu_lock=None):
        self.input_w, self.input_h = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_size = min_size
        self.lock = gpu_lock or threading.Lock()

        # Try GPU providers
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
                    print(f"[INFO] VehicleDetector: Using {label}")
                else:
                    raise RuntimeError("GPU provider not active")
            except Exception as e:
                print(f"[WARN] GPU provider failed for vehicle detector: {e}")
                self.session = ort.InferenceSession(
                    model_path, providers=['CPUExecutionProvider']
                )
                print("[INFO] VehicleDetector: Using CPU (fallback)")
        else:
            self.session = ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider']
            )
            print("[INFO] VehicleDetector: Using CPU")

        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, frame):
        """Letterbox resize and normalize to [0, 1]."""
        h, w = frame.shape[:2]

        # Scale factor to fit within input_size while maintaining aspect ratio
        scale = min(self.input_w / w, self.input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        # Pad to input_size (letterbox)
        pad_w = (self.input_w - new_w) // 2
        pad_h = (self.input_h - new_h) // 2
        padded = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Normalize and convert to CHW
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        return blob, scale, pad_w, pad_h

    def _postprocess(self, output, scale, pad_w, pad_h, orig_w, orig_h):
        """Decode YOLOv8 output [1, 84, 8400] into vehicle detections."""
        # output shape: [1, 84, 8400] -> transpose to [8400, 84]
        preds = output[0].T  # [8400, 84]

        # Split: first 4 = bbox (cx, cy, w, h), remaining 80 = class scores
        boxes_raw = preds[:, :4]
        class_scores = preds[:, 4:]

        # Filter to vehicle classes only
        vehicle_class_ids = list(VEHICLE_CLASSES.keys())
        vehicle_scores = class_scores[:, vehicle_class_ids]  # [8400, N_vehicle_classes]

        # Best vehicle class per prediction
        best_vehicle_idx = np.argmax(vehicle_scores, axis=1)
        best_vehicle_score = vehicle_scores[np.arange(len(vehicle_scores)), best_vehicle_idx]

        # Confidence filter
        mask = best_vehicle_score > self.confidence_threshold
        if not np.any(mask):
            return []

        filtered_boxes = boxes_raw[mask]
        filtered_scores = best_vehicle_score[mask]
        filtered_class_idx = best_vehicle_idx[mask]

        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        cx, cy, bw, bh = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Remove letterbox padding and rescale to original image
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w).astype(np.int32)
        y1 = np.clip(y1, 0, orig_h).astype(np.int32)
        x2 = np.clip(x2, 0, orig_w).astype(np.int32)
        y2 = np.clip(y2, 0, orig_h).astype(np.int32)

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        nms_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, filtered_scores.tolist(),
            self.confidence_threshold, self.nms_threshold,
        )

        results = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                bx1, by1, bx2, by2 = boxes[i]
                fw = bx2 - bx1
                fh = by2 - by1

                # Min size filter
                if fw < self.min_size or fh < self.min_size:
                    continue

                class_id = vehicle_class_ids[filtered_class_idx[i]]
                results.append({
                    'bbox': (int(bx1), int(by1), int(fw), int(fh)),
                    'confidence': float(filtered_scores[i]),
                    'class_id': class_id,
                    'class_name': VEHICLE_CLASSES[class_id],
                })

        return results

    def detect(self, frame):
        """
        Detect vehicles in frame. Thread-safe.
        Returns list of dicts with 'bbox', 'confidence', 'class_id', 'class_name'.
        """
        h, w = frame.shape[:2]
        blob, scale, pad_w, pad_h = self._preprocess(frame)

        with self.lock:
            outputs = self.session.run(None, {self.input_name: blob})

        return self._postprocess(outputs[0], scale, pad_w, pad_h, w, h)
