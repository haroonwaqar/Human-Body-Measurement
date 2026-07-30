# AI Human Body Measurement System

**Real-time, single-camera 2D body proportion measurement powered by YOLOv11 Pose Estimation and Fixed-Distance Checkerboard Calibration.**

## Overview

Standard monocular (single-lens) cameras cannot natively perceive depth, which makes accurate real-world body measurement from 2D video a genuinely hard problem. This project solves that limitation without requiring stereo cameras, LiDAR, or depth sensors.

Instead, it combines:

- **A one-time Fixed-Distance Checkerboard Calibration step** — establishing a precise Pixels-Per-Centimeter (PPM) ratio at known floor distances.
- **YOLOv11 Pose Estimation** — instantly mapping 17 anatomical keypoints (COCO format) onto a person in the frame, with zero manual clicking.

The result: a subject simply stands on a marked spot, and the system outputs real-world measurements — height, shoulder width, arm length — in centimeters, live, at high FPS.


## Key Features

| Feature | Description |
|---|---|
| **AI Pose Estimation** | YOLOv11 automatically maps 17 COCO-format anatomical keypoints per frame. No manual clicking or marker placement on the body required. |
| **Fixed-Distance Calibration** | Pre-calculated PPM scales tied to specific floor markers (e.g., 4 ft, 6 ft) remove the need to hold a calibration marker during every measurement. |
| **Smudge-Resistant Detection** | The checkerboard detector applies Gaussian blurring and micro-grid rejection so smudges, glare, or lighting artifacts on the printed board don't break calibration. |
| **Decoupled Architecture** | Calibration logic is fully separated from the live measurement loop, keeping real-time inference fast and uncluttered. |
| **Apple Silicon Optimized** | Runs on the Metal Performance Shaders (`mps`) backend for accelerated, real-time inference on M-series Macs. |
| **Edge-Server Ready** | Designed to ingest wireless HTTP video streams from an ESP32-CAM module for a future low-cost, camera-as-edge-device setup. |

## How It Works: The PPM Math

The system's core is a **Pixels-Per-Centimeter (PPM)** ratio, calculated once per fixed distance:

```
PPM = pixel_width_of_checkerboard_square / physical_width_of_square_cm
```

**Example:**
A checkerboard square that is physically **2.84 cm** wide occupies **68 pixels** on camera at a 5-foot distance:

```
PPM = 68 px / 2.84 cm ≈ 23.94 px/cm
```

Once this ratio is known for a given distance, any pixel distance between two YOLO keypoints (e.g., left shoulder → right shoulder) can be converted into a real-world measurement:

```
real_world_cm = pixel_distance_between_keypoints / PPM
```

Because the PPM is pre-calculated and tied to a specific, marked floor distance, the subject never needs to hold a calibration target — they just need to stand on the mark.

## Project Structure

```
.
├── main.py                    # Core measurement engine (YOLO inference + PPM math)
├── calibration.py             # CheckerboardCalibrator class (detection & smudge protection)
├── make_preset.py             # One-time setup utility to generate PPM presets
├── generate_checkerboard.py   # Utility to generate a printable checkerboard PNG
└── README.md
```

## Usage Guide

### 1. Generate & Print the Checkerboard

Run the generator script to create a print-ready checkerboard:

```bash
python generate_checkerboard.py
```

Print the resulting PNG at **100% scale** (no "fit to page" scaling), then physically measure the width of **one black square** in centimeters. You'll need this exact value in Step 3.

### 2. Set Up Your Physical Space

- Mount or place your camera in a **fixed position**.
- Place a tape mark on the floor at a known distance (e.g., **5 feet**) from the camera.
- Have someone stand on the mark holding the printed checkerboard flat and facing the camera.
- Take a calibration photo.

### 3. Generate a Calibration Preset

Open `make_preset.py` and provide:
- The physical width of one checkerboard square (measured in Step 1)
- The path to your calibration photo (from Step 2)

Run it:

```bash
python make_preset.py
```

This outputs a precise **PPM (Pixels-Per-Centimeter)** value for that specific distance.

### 4. Configure the Main Engine

Open `main.py` and paste the new PPM value into the `CALIBRATION_PRESETS` dictionary, then set `ACTIVE_PRESET` to point at it:

```python
CALIBRATION_PRESETS = {
    "5ft": 23.94,
    "4ft": 28.10,   # add more distances as needed
}

ACTIVE_PRESET = "5ft"
```

### 5. Run a Measurement Session

Launch the measurement engine using a live webcam or a static test image:

```bash
python main.py
```

Have the subject stand on the calibrated tape mark, facing the camera. The system will:
1. Detect the person and draw their pose skeleton
2. Extract the 17 COCO keypoints
3. Convert relevant pixel distances into real-world centimeters using the active PPM
4. Overlay live measurements (height, shoulder width, arm length) on the video feed

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│generate_checkerboard│  ────▶  │  Printed Checkerboard│
│        .py          │         │      (10x7 grid)     │
└─────────────────────┘         └──────────┬───────────┘
                                            │
                                            ▼
┌──────────────────────┐         ┌──────────────────────┐
│      cal2.py         │ ◀────── │  Calibration Photo   │
│CheckerboardCalibrator│         │  (fixed floor mark)  │
└──────────┬───────────┘         └──────────────────────┘
           │
           ▼
┌─────────────────────┐
│    make_preset.py   │  ──▶  PPM Ratio (px/cm)
└──────────┬──────────┘
           │  
           ▼
┌──────────────────────┐         ┌────────────────────┐
│       main.py        │ ◀────── │ YOLOv11 Pose Model │
│  CALIBRATION_PRESETS │         │  (17 keypoints)    │
│  + Live/Static Input │         └────────────────────┘
└──────────┬───────────┘
           ▼
   Real-World Measurements
    (Height, Shoulders, Arms)
```

The calibration pipeline (checkerboard generation → photo → preset) is fully **decoupled** from the live measurement loop, so `main.py` never runs checkerboard detection during real-time inference — keeping frame rates high.

## Limitations

- **2D Projection Only:** The subject must stand perfectly parallel to the camera plane. Twisting or rotating the body reduces the effective 2D pixel distance between keypoints, skewing measurements.
- **Strict Depth Adherence:** The subject must stand exactly on the calibrated floor mark. Stepping forward or backward invalidates the active PPM ratio, since pixel-to-cm scale changes with distance from the camera.
- **Single-Camera Constraint:** No true depth sensing — accuracy is inherently tied to calibration precision and subject positioning discipline.