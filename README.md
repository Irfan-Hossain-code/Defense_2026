# Defence Hackathon 2026 — Camera Tracking Subsystem

Real-time human tracking via laptop webcam using OpenCV and MediaPipe Pose.
When the person is visible, a live stick figure is drawn. When they leave the
frame, the system switches to **RF FALLBACK / GHOST MODE** and shows a fading
ghost at the last known position, drifting with the last known movement vector.
The RF slot is a clean placeholder ready to receive coarse position estimates
from ESP32 nodes.

## Quick start

```
pip install opencv-python mediapipe numpy
python main.py
```

First run downloads `pose_landmarker_lite.task` (~5 MB) automatically.
Press **ESC** in the camera window to quit.

---

## File map

| File | What it does | Key inputs | Key outputs |
|------|-------------|------------|-------------|
| `main.py` | Entry point — opens the webcam, runs the frame loop, calls tracker and display functions | Webcam feed (index 0) | OpenCV window, terminal status |
| `model_setup.py` | Downloads the MediaPipe pose model on first run and saves it locally | Internet (first run only) | `pose_landmarker_lite.task` on disk |
| `tracker/constants.py` | All tuning numbers: visibility threshold, ghost fade rate, drift duration, skeleton edge list | — | Imported by tracker and display modules |
| `tracker/state.py` | `PersonState` dataclass — a snapshot of everything known about the person in one frame | — | Used as the return type of `PersonTracker.update()` |
| `tracker/tracker.py` | `PersonTracker` — wraps MediaPipe, scores confidence, switches TRACKING / GHOST modes, accumulates ghost drift | RGB frame per call; optional RF dx/dy via `update_rf_estimate()` | `PersonState` or `None`; ghost alpha and offset readable as properties |
| `display/skeleton.py` | `draw_skeleton()` and `draw_bbox()` — draw the stick figure and bounding box onto a frame | BGR frame + `PersonState`; optional color, alpha, pixel offset | Frame modified in-place |
| `display/hud.py` | `draw_hud()` — renders the mode label, live metrics, and ESC hint as an overlay | BGR frame + `PersonTracker` + current `PersonState` | Frame modified in-place |

---

## RF integration (future)

`PersonTracker.update_rf_estimate(dx, dy)` is the hook for the ESP32 module.
Call it once per frame with a coarse pixel-space movement estimate and the ghost
figure will drift accordingly. See the comment block in `main.py` for the exact
location to wire it in.

---

## Modes

| Mode | Trigger | Visual |
|------|---------|--------|
| **CAMERA TRACKING** | Pose detected with sufficient confidence | Green skeleton + bounding box; live metrics in corner |
| **RF FALLBACK / GHOST MODE** | 12+ consecutive frames with no confident pose | Orange fading skeleton at last known position, drifting with last velocity |
