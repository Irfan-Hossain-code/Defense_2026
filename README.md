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

### Tactical AI + React HUD (Jarvis layer)

See **[TACTICAL_SETUP.md](TACTICAL_SETUP.md)** for WebSocket HUD, ElevenLabs voice, PTT, and `.env` keys.

```
pip install -r requirements.txt
cp .env.example .env
python main.py          # terminal 1
cd hud && npm run dev   # terminal 2 → http://localhost:5173
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

---

## ESP32 RF / CSI Setup — 3-Node Zone Localizer

Three ESP32-WROOM boards act as Wi-Fi sensing nodes arranged LEFT / MIDDLE / RIGHT
along a wall. They detect motion behind the wall and estimate which zone the person
is in. The MIDDLE node can be on a second laptop (Mac) over the shared hotspot.

### Hardware

| Node | Board | Connected to | Port |
|------|-------|-------------|------|
| LEFT | ESP32-WROOM | Windows laptop USB | COM10 |
| RIGHT | ESP32-WROOM | Windows laptop USB | COM8 |
| MIDDLE | ESP32-WROOM | Mac USB | `/dev/cu.usbserial-1130` |

All three boards connect to the **same Wi-Fi hotspot** (OnePlus 8, `Oneplus8`).
The hotspot phone stays stationary at all times.

Physical layout (same side of wall as the ESPs):

```
[wall — person moves on the other side]

[LEFT esp]        [MIDDLE esp]        [RIGHT esp]
  COM10           Mac serial             COM8
   |___________________|___________________|
                Wi-Fi hotspot (phone)
```

---

### WINDOWS LAPTOP — copy-paste these in order

**Step 1 — activate the environment (do this once per terminal session)**
```powershell
.venv\Scripts\Activate.ps1
```

**Step 2 — flash LEFT board (COM10)**
```powershell
python combining_three_esps\run_wroom.py left
```
> If it hangs at `Connecting......` — hold the **BOOT** button on the ESP32 while dots appear, then release.

**Step 3 — flash RIGHT board (COM8)**
```powershell
python combining_three_esps\run_wroom.py right
```
> Same tip: hold BOOT if it hangs.

**Step 4 — wait for the Mac to start the bridge (see Mac instructions below), then run the localizer**
```powershell
python combining_three_esps\run_all_three.py --middle-host MAC_IP_HERE
```
Replace `MAC_IP_HERE` with the IP the Mac prints when it starts the bridge (e.g. `10.75.241.42`).

> **No Mac / MIDDLE node?** Run without it — zone inference still works using LEFT + RIGHT symmetry:
> ```powershell
> python combining_three_esps\run_all_three.py
> ```

**Step 5 — calibrate**

When the localizer starts, keep the sensing area (behind the wall) completely empty.
Wait for all connected nodes to show `CAL✓`. Then walk behind the wall.

**Step 6 — skip recalibration on future runs**
```powershell
python combining_three_esps\run_all_three.py --middle-host MAC_IP_HERE --load-cal
```

---

### MAC (friend's laptop) — copy-paste these in order

**Step 1 — open a terminal in the project folder**
```bash
cd /path/to/defence_2026
```

**Step 2 — install tools (once only)**
```bash
pip3 install platformio pyserial
```

**Step 3 — flash the MIDDLE board**
```bash
cd combining_three_esps/firmware/wroom
pio run -e MIDDLE --target upload
cd ../../..
```
> If it hangs at `Connecting......` — hold the **BOOT** button on the ESP32 while dots appear, then release.
> If upload fails, check the port: `python3 -c "import serial.tools.list_ports; [print(p.device, p.description) for p in serial.tools.list_ports.comports()]"`

**Step 4 — start the bridge**
```bash
python3 combining_three_esps/bridge_middle.py
```

It will print something like:
```
This Mac's hotspot IP appears to be: 10.75.241.42
On the Windows laptop run:
  python combining_three_esps/run_all_three.py --middle-host 10.75.241.42
```

Give that IP to the Windows operator. Keep this terminal open — closing it stops the bridge.

---

### What you should see (Windows terminal)

```
╔══════════════════════════════════════════════════════════════╗
║           3-Node ESP32 CSI Zone Localizer                   ║
╚══════════════════════════════════════════════════════════════╝

  LEFT    [████████░░░░░░░░░░░░░░░░░░░░]   4.2x  -42dBm  pkts= 312  CAL✓
  MIDDLE  [████░░░░░░░░░░░░░░░░░░░░░░░░]   2.1x  -38dBm  pkts= 308  CAL✓
  RIGHT   [██░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.9x  -44dBm  pkts= 310  CAL✓
  ────────────────────────────────────────────────────────────────────────
  Zone: LEFT_MIDDLE    Conf: 0.68   Dist: mid       ◀◀
```

Zones: `LEFT` `LEFT_MIDDLE` `MIDDLE` `RIGHT_MIDDLE` `RIGHT` `NO_MOTION`

CSV logs are saved automatically to `combining_three_esps/data/`.

---

### Troubleshooting

| Problem | Fix |
|---------|-----|
| Flash hangs at `Connecting......` | Hold BOOT button on ESP32 while dots appear |
| `No ports found` | Check USB cable (must be data cable, not charge-only) |
| `csi_packets=0` after connecting | Normal — packets appear once traffic is generated. Wait 10s. |
| MIDDLE shows `not connected` | Bridge not running on Mac, or wrong `--middle-host` IP |
| Score stays at 0 forever | Wi-Fi credentials wrong — check `WIFI_SSID`/`WIFI_PASS` in `platformio.ini` |
| Too many false positives | Increase sensitivity: `--sensitivity 3.5` |
| Not detecting motion | Decrease sensitivity: `--sensitivity 1.8` |
