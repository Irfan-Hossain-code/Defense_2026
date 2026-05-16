# Defence Hackathon 2026 — Camera Tracking Subsystem

Real-time human tracking via laptop webcam using OpenCV and MediaPipe Pose.
When the person is visible, a live stick figure is drawn. When they leave the
frame, the system switches to **RF FALLBACK / GHOST MODE** and shows a fading
ghost at the last known position, drifting with the last known movement vector.
The RF slot is a clean placeholder ready to receive coarse position estimates
from ESP32 nodes.

An optional **facial recognition overlay** identifies known individuals and
draws their name in gold on screen. It runs in a background thread and never
affects tracking performance.

## Quick start

```
pip install opencv-python mediapipe numpy
python main.py
```

First run downloads `pose_landmarker_lite.task` (~5 MB) automatically.
Press **ESC** in the camera window to quit.

### Facial recognition (optional)

```
pip install face_recognition
```

Add photos of each person under `known_faces/<name>/`:

```
known_faces/
  irfan/
    photo1.jpg
    photo2.jpg
```

The system loads these on startup and identifies faces automatically. If
`face_recognition` is not installed or the folder is empty, the overlay is
silently disabled and everything else works normally.

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
| `display/identity_overlay.py` | `draw_identity_overlay()` — draws a gold face box and name label for a recognised person | BGR frame + `FaceState` (or `None`) | Frame modified in-place; no-op if `None` |
| `identity/face_state.py` | `FaceState` dataclass — name, confidence, and face bounding box for one recognised face | — | Returned by `FaceIdentifier.update()` |
| `identity/database.py` | `FaceDatabase` — loads face encodings from `known_faces/` at startup | `known_faces/<name>/*.jpg` | List of encodings + names used by `FaceIdentifier` |
| `identity/face_identifier.py` | `FaceIdentifier` — runs HOG face detection and dlib recognition in a background thread | RGB frame per call | `FaceState` or `None`; never blocks the main loop |

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

---

## Facial recognition performance

Recognition runs in a **background thread** in parallel with the main tracking
loop, so it never causes frame drops. The main loop always returns the cached
result instantly.

The only performance knob is `_RECOGNITION_INTERVAL` in
`identity/face_identifier.py`. It controls how often a new recognition pass is
launched:

| Interval | Behaviour |
|----------|-----------|
| `0.5` | Twice per second — more responsive label updates |
| `1.0` | Once per second — default, smooth with no perceptible dips |
| `2.0` | Every two seconds — maximum CPU headroom |

Increase the interval if the system feels sluggish; decrease it if you want the
name label to react faster to face changes. The tracking system is unaffected
either way.
