# Cam-Vision

Real-time multi-camera face and vehicle detection, tracking, and capture system built with Python. Processes RTSP streams from IP cameras, detects faces (YuNet) and vehicles (YOLOv8n), assigns persistent IDs to each tracked object, and saves cropped images with structured metadata.

## Features

- **Multi-Camera Support** -- Process N cameras simultaneously, each with independent tracking
- **GPU-Accelerated Inference** -- Both detectors run on GPU via ONNX Runtime with DirectML (NVIDIA, AMD, Intel)
- **Face Detection** -- YuNet deep learning model with landmark validation and false positive filtering
- **Vehicle Detection** -- YOLOv8n model detecting cars, motorcycles, buses, and trucks
- **Persistent ID Tracking** -- Independent centroid trackers for faces and vehicles
- **One-Shot Capture** -- Saves one image per detected person/vehicle, avoids duplicate captures
- **Live Dashboard** -- Real-time stats window with recent captures, totals, and per-camera breakdown
- **Image Enhancement** -- Automatic background enhancement (upscale + sharpening) for all captures
- **Interactive Detail View** -- Click any recent item on the dashboard to see Original vs Enhanced comparison
- **Configurable via Environment** -- All parameters tunable through `.env` without code changes
- **Threaded Architecture** -- Non-blocking video capture with per-camera processing threads

## Architecture

```
         +------------------+   +---------------------+
         | FaceDetectorGPU  |   | VehicleDetectorGPU  |
         | (YuNet ONNX)     |   | (YOLOv8n ONNX)      |
         +--------+---------+   +----------+----------+
                  |         Shared GPU Lock  |
                  +------------+-------------+
                               |
                +--------------+--------------+
                |              |              |
       +--------v---+  +------v-----+  +-----v------+
       | Camera 0   |  | Camera 1   |  | Camera N   |
       | VideoStream|  | VideoStream|  | VideoStream|
       | FaceTracker|  | FaceTracker|  | FaceTracker|
       | VehTracker |  | VehTracker |  | VehTracker |
       | Capture    |  | Capture    |  | Capture    |
       +------------+  +------------+  +------------+
```

Each camera runs in its own thread with dedicated face and vehicle trackers. Both detectors share a single GPU lock to serialize DirectML inference calls.

## Tech Stack

| Component | Technology |
|---|---|
| Face Detection | YuNet (ONNX, 2023) |
| Vehicle Detection | YOLOv8n (ONNX, COCO) |
| GPU Inference | ONNX Runtime + DirectML |
| Video Capture | OpenCV + FFmpeg (RTSP/TCP) |
| Tracking | Centroid-based tracker with scipy |
| Configuration | python-dotenv |

## Requirements

- Python 3.10+
- NVIDIA, AMD, or Intel GPU with DirectX 12 support (for GPU acceleration)
- IP cameras with RTSP stream support

## Installation

```bash
git clone https://github.com/yourusername/cam-vision.git
cd cam-vision
pip install -r requirements.txt
```

Download the YuNet ONNX model:

```bash
mkdir -p src/models
curl -L -o src/models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## Configuration

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

At minimum, set your camera RTSP URL:

```dotenv
RTSP_URLS=rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
CAMERA_NAMES=FrontDoor
```

For multiple cameras, use comma-separated values:

```dotenv
RTSP_URLS=rtsp://...channel=1&subtype=0,rtsp://...channel=2&subtype=0
CAMERA_NAMES=FrontDoor,Backyard
```

### Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `RTSP_URLS` | -- | Comma-separated RTSP stream URLs (required) |
| `CAMERA_NAMES` | `Camera_0` | Display names for each camera |
| `CONFIDENCE_THRESHOLD` | `0.85` | Minimum face detection confidence |
| `MIN_FACE_SIZE` | `40` | Minimum face size in pixels |
| `MAX_FACE_ASPECT_RATIO` | `2.0` | Maximum width/height ratio for valid faces |
| `TRACKING_DISTANCE` | `50` | Max centroid movement for face tracking |
| `VEHICLE_CONFIDENCE` | `0.5` | Minimum vehicle detection confidence |
| `VEHICLE_MIN_SIZE` | `60` | Minimum vehicle bbox size in pixels |
| `VEHICLE_TRACKING_DISTANCE` | `80` | Max centroid movement for vehicle tracking |
| `TARGET_FPS` | `10` | Detection processing rate per camera |
| `OUTPUT_DIR` | `rostros_detectados` | Root directory for captured images |

See [.env.example](.env.example) for detailed documentation on each parameter, including recommended ranges and usage guidelines.

## Usage

```bash
python src/main.py
```

Press `q` in the video window to quit.

### Video Overlay

The live video feed displays:

| Indicator | Meaning |
|---|---|
| Green box + `ID X` | Newly detected face, not yet captured |
| Cyan box + `ID X SAVED` | Face already captured |
| Orange box + `V0 car` | Newly detected vehicle with class |
| Yellow box + `V0 car SAVED` | Vehicle already captured |
| Gray box | Detection filtered out by validation |
| Red flash | Capture in progress |

### Output Structure

```
rostros_detectados/
  FrontDoor/
    person_0/
      face.jpg            # Cropped face image
      body.jpg            # Estimated full body crop
      metadata.json       # Detection metadata
    vehicle_0/
      vehicle_crop.jpg    # Cropped vehicle image
      metadata.json       # Vehicle metadata (class, bbox, confidence)
    person_1/
      ...
  Backyard/
    person_0/
      ...
    vehicle_0/
      ...
```

Each `metadata.json` contains:

```json
{
  "person_id": 0,
  "camera": "FrontDoor",
  "timestamp": "20260210_140532",
  "face_bbox": [120, 80, 45, 52],
  "score": 2340
}
```

## Project Structure

```
cam-vision/
  .env.example              # Documented configuration template
  requirements.txt          # Python dependencies
  src/
    main.py                 # Entry point, multi-camera orchestration
    face_detector.py        # YuNet face detector with GPU acceleration
    vehicle_detector.py     # YOLOv8n vehicle detector with GPU acceleration
    video_stream.py         # Threaded RTSP capture
    tracker.py              # Centroid-based ID tracker
    utils.py                # File I/O and metadata helpers
    models/
      face_detection_yunet_2023mar.onnx
      yolov8n.onnx
```

## How It Works

1. **Video Capture** -- Each camera URL spawns a `VideoStream` thread that continuously reads frames over RTSP (TCP transport for reliability). The main thread always gets the latest frame without blocking.

2. **Face Detection** -- Frames are resized to 640x640 and fed to the YuNet model via ONNX Runtime. The model outputs bounding boxes, confidence scores, and facial landmarks at three scales. Post-processing decodes anchor-based predictions and applies NMS.

3. **Vehicle Detection** -- The same frame is also processed by YOLOv8n, which detects cars, motorcycles, buses, and trucks. The model uses letterbox preprocessing and standard YOLOv8 post-processing with class filtering.

4. **Validation** -- Face detections pass through filters: minimum size, aspect ratio, and landmark consistency. Vehicle detections are filtered by confidence and minimum size.

5. **Tracking** -- Independent centroid trackers match face and vehicle detections across frames. Each object receives a unique ID that persists as long as it remains visible.

6. **Capture** -- On first detection, the system saves cropped images and JSON metadata. Faces get face + body crops; vehicles get a vehicle crop with class label. Each object is captured exactly once per session.

7. **Dashboard** -- A live stats window displays total captures, per-camera breakdown, active tracking counts, and thumbnails of the last 5 captured faces and vehicles.

## Performance

Tested with 2 simultaneous Dahua RTSP cameras at 960x1080 resolution:

| Metric | Value |
|---|---|
| Detection input size | 640x640 |
| Processing rate | 10 FPS per camera |
| GPU utilization | DirectML on NVIDIA RTX 4070 |
| CPU usage | ~15-25% (with FPS throttling) |
| Memory | ~400 MB |

## License

MIT
