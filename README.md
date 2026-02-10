# Cam-Vision

Real-time multi-camera face detection, tracking, and capture system built with Python. Processes RTSP streams from IP cameras, detects faces using YuNet (deep learning), assigns persistent IDs to tracked individuals, and saves cropped face and body images with structured metadata.

## Features

- **Multi-Camera Support** -- Process N cameras simultaneously, each with independent tracking
- **GPU-Accelerated Inference** -- Face detection runs on GPU via ONNX Runtime with DirectML (NVIDIA, AMD, Intel)
- **Persistent ID Tracking** -- Centroid-based tracker assigns unique IDs to individuals across frames
- **One-Shot Capture** -- Saves one face and body crop per person, avoids duplicate captures
- **False Positive Filtering** -- Landmark validation, aspect ratio checks, and minimum size thresholds
- **Configurable via Environment** -- All parameters tunable through `.env` without code changes
- **Threaded Architecture** -- Non-blocking video capture with per-camera processing threads

## Architecture

```
                    +-------------------+
                    |   Shared GPU      |
                    |   FaceDetectorGPU |
                    |   (ONNX Runtime)  |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v-----+  +-----v------+
     | Camera 0   |  | Camera 1   |  | Camera N   |
     | VideoStream|  | VideoStream|  | VideoStream|
     | Tracker    |  | Tracker    |  | Tracker    |
     | Capture    |  | Capture    |  | Capture    |
     +------------+  +------------+  +------------+
```

Each camera runs in its own thread with a dedicated tracker and capture pipeline. All cameras share a single GPU-backed face detector instance (thread-safe).

## Tech Stack

| Component | Technology |
|---|---|
| Face Detection | YuNet (ONNX, 2023) |
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
| `CONFIDENCE_THRESHOLD` | `0.85` | Minimum detection confidence (0.0 - 1.0) |
| `MIN_FACE_SIZE` | `40` | Minimum face size in pixels |
| `MAX_FACE_ASPECT_RATIO` | `2.0` | Maximum width/height ratio for valid faces |
| `TRACKING_DISTANCE` | `50` | Max centroid movement in pixels between frames |
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
| Green box + `ID X` | Newly detected person, not yet captured |
| Cyan box + `ID X SAVED` | Person already captured |
| Gray box + `REJECTED` | Detection filtered out by validation |
| Red flash | Capture in progress |

### Output Structure

```
rostros_detectados/
  FrontDoor/
    person_0/
      face.jpg            # Cropped face image
      body.jpg            # Estimated full body crop
      metadata.json       # Detection metadata
    person_1/
      ...
  Backyard/
    person_0/
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
    face_detector.py        # YuNet detector with GPU acceleration
    video_stream.py         # Threaded RTSP capture
    tracker.py              # Centroid-based ID tracker
    utils.py                # File I/O and metadata helpers
    models/
      face_detection_yunet_2023mar.onnx
```

## How It Works

1. **Video Capture** -- Each camera URL spawns a `VideoStream` thread that continuously reads frames over RTSP (TCP transport for reliability). The main thread always gets the latest frame without blocking.

2. **Face Detection** -- Frames are resized to 640x640 and fed to the YuNet model via ONNX Runtime. The model outputs bounding boxes, confidence scores, and facial landmarks (eyes, nose, mouth) at three scales. A post-processing pipeline decodes anchor-based predictions and applies non-maximum suppression.

3. **Validation** -- Each detection passes through filters: minimum size, aspect ratio, and landmark consistency (all five landmarks must fall within the bounding box, and eye distance must be proportional to face width). This eliminates false positives from foliage, shadows, and textures.

4. **Tracking** -- A centroid tracker matches detections across frames using euclidean distance. Each person receives a unique ID that persists as long as they remain visible. IDs are released after a configurable number of frames without a match.

5. **Capture** -- On first detection, the system saves a cropped face image, an estimated body crop (heuristic extrapolation from face position), and a JSON metadata file. Each person is captured exactly once per session.

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
