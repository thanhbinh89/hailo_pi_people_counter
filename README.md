# People Counter (C++)

Real-time people counting in a configurable zone using Hailo inference on Raspberry Pi.

## Features

| Feature | Description |
|---|---|
| Zone counting | Count people currently inside a configurable rectangular zone |
| Entry counting | Cumulative count of people who entered the zone |
| Exit counting | Cumulative count of people who left the zone |
| Dwell time | Per-person time (seconds) spent inside the zone |
| Config-driven | Zone geometry, thresholds, and display options via `config.yaml` |

## Requirements

- Hailo-8 / Hailo-8L / Hailo-10H accelerator
- Raspberry Pi camera module (CSI) or USB camera
- YOLO object detection HEF model (e.g. `yolov8m.hef`)
- OpenCV 4+, HailoRT SDK, CMake 3.16+

## Build

```bash
cd hailo_apps/cpp/people_counter
./build.sh
```

The executable is placed at `build/aarch64/people_counter`.

## Run

### RPi camera (CSI)
```bash
./build/aarch64/people_counter --net yolov8m.hef --input rpi
```

### USB camera
```bash
./build/aarch64/people_counter --net yolov8m.hef --input usb
```

### Video file
```bash
./build/aarch64/people_counter --net yolov8m.hef --input /path/to/video.mp4
```

### Custom config path
```bash
./build/aarch64/people_counter --net yolov8m.hef --input rpi --config /path/to/config.yaml
```

### No display (headless)
```bash
./build/aarch64/people_counter --net yolov8m.hef --input rpi --no-display
```

### Save output video
```bash
./build/aarch64/people_counter --net yolov8m.hef --input rpi --save-stream-output
```

## Configuration (`config.yaml`)

The zone and tracker are fully configurable without recompiling.

```yaml
# Zone: normalized coordinates (0.0 – 1.0) relative to frame size
zone:
  x1: 0.2    # left edge
  y1: 0.2    # top edge
  x2: 0.8    # right edge
  y2: 0.8    # bottom edge

detection:
  confidence_threshold: 0.5   # minimum score for a valid person detection

tracker:
  max_disappeared: 30    # frames before a track is dropped
  max_distance: 100.0    # max pixel distance for centroid matching

display:
  show_zone: true          # draw zone rectangle
  show_tracking_ids: true  # draw person IDs and dwell time labels
  show_stats: true         # draw entry/exit/current-count panel
```

### Zone examples

| Use case | x1 | y1 | x2 | y2 |
|---|---|---|---|---|
| Full frame | 0.0 | 0.0 | 1.0 | 1.0 |
| Center 60% | 0.2 | 0.2 | 0.8 | 0.8 |
| Left half | 0.0 | 0.0 | 0.5 | 1.0 |
| Bottom strip (exit line) | 0.0 | 0.85 | 1.0 | 1.0 |

## Display Overlay

```
┌──────────────────────────────────┐
│ In zone :      2                 │  ← people currently inside zone
│ Entries :      7                 │  ← cumulative entries
│ Exits   :      5                 │  ← cumulative exits
├──────────────────────────────────┤
│   [orange box] = detected person │
│   [green dot]  = person in zone  │
│   [grey dot]   = person outside  │
│   ID:3  12.4s  = ID + dwell time │
│   [cyan rect]  = count zone      │
└──────────────────────────────────┘
```

## Architecture

```
┌─────────────┐     ┌───────────────┐     ┌──────────────────────┐
│  Preprocess │ --> │   Inference   │ --> │     Postprocess      │
│  (thread 1) │     │   (thread 2)  │     │     (thread 3)       │
│  BGR → RGB  │     │  HailoInfer   │     │  parse NMS output    │
│  resize     │     │  async jobs   │     │  → ZoneTracker       │
└─────────────┘     └───────────────┘     │  → draw overlay      │
                                           └──────────────────────┘
```

Frames flow through bounded thread-safe queues. The `ZoneTracker` lives entirely
in thread 3 so no locking is needed for tracking state.

## CLI Arguments

All standard Hailo app flags are supported:

| Flag | Description |
|---|---|
| `--net <path>` | Path to HEF model file |
| `--input <src>` | Input: `rpi`, `usb`, `/dev/videoX`, or video file path |
| `--config <path>` | Path to `config.yaml` (default: next to executable) |
| `--no-display` | Disable OpenCV window (headless mode) |
| `--save-stream-output` | Save output frames to file |
| `--camera-resolution <res>` | `sd` (640×480), `hd` (1280×720), `fhd` (1920×1080) |
| `--batch-size <n>` | Inference batch size (default: 1) |
