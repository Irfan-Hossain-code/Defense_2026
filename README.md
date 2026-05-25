# Defence Hackathon 2026 — Tactical Tracking System

Real-time human tracking via laptop webcam using OpenCV and MediaPipe Pose.
When the person is visible, a live stick figure is drawn. When they leave the
frame, the system switches to **RF FALLBACK / GHOST MODE** and shows a fading
ghost at the last known position, drifting with the last known movement vector.

Optional layers add facial recognition, a tactical React HUD, AI narration,
and ESP32 Wi-Fi sensing. **None of these are required to run the core tracker.**

---

## Quickstart — no hardware needed

```bash
git clone <repo>
cd defence_2026

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

First run downloads `pose_landmarker_lite.task` (~5 MB) automatically.
A webcam window opens. Press **ESC** to quit.

> **No webcam?** The system will print `ERROR: No camera found.` and exit.
> Try `$env:CAMERA_INDEX=1` (Windows) or `CAMERA_INDEX=1 python main.py` (macOS/Linux)
> to try a different camera index.

---

## Modes

| Mode | Trigger | Visual |
|------|---------|--------|
| **CAMERA TRACKING** | Pose detected with sufficient confidence | Green skeleton + bounding box, live metrics |
| **RF FALLBACK / GHOST MODE** | 12+ consecutive frames with no confident pose | Orange fading skeleton at last known position, drifting with last velocity |

Without ESP32 nodes the RF zone column stays blank — everything else works normally.

---

## Optional layer 1 — Facial recognition

```bash
pip install face_recognition
```

Add photos under `known_faces/<name>/`:

```
known_faces/
  irfan/
    photo1.jpg
```

The system loads encodings on startup and identifies faces in a background thread.
If `face_recognition` is not installed or the folder is empty, the overlay is
silently disabled and nothing else changes.

---

## Optional layer 2 — Tactical HUD + AI narration

The HUD is a React overlay that shows zones, subtitles, and event feeds over
WebSocket. The AI layer uses ConfidentialMind and Gemini for narration and
push-to-talk Q&A.

**Step 1 — copy and fill `.env`**

```bash
cp .env.example .env
# Edit .env — see key table below
```

| Variable | Required for | Where to get |
|----------|-------------|--------------|
| `CONFIDENTIALMIND_BASE_URL` | Event narration + vision profile | Junction / ConfidentialMind handout |
| `CONFIDENTIALMIND_API_KEY` | Same | Same |
| `CM_MODEL_NARRATION` | e.g. `Gemma 4` | `list_models.py` in CM repo |
| `CM_MODEL_VISION` | e.g. `Qwen3-Omni-...` | Same |
| `GOOGLE_API_KEY` | Voice Q&A (Gemini Flash) | GCP API key |
| `GEMINI_MODEL` | Default `gemini-2.0-flash` | — |

**Without any keys:** the tracker still runs, HUD still updates, fallback text
prints to terminal, and TTS is silently skipped.

**Step 2 — start the tracker**

```bash
python main.py
```

**Step 3 — start the HUD (second terminal)**

```bash
cd hud
npm install   # first time only
npm run dev
```

Open **http://localhost:5173** and drag it over the OpenCV window.

### Controls

| Key | Action |
|-----|--------|
| **ESC** | Quit |
| **T** | Push-to-talk: press once to record, again to send |

---

## Optional layer 3 — ESP32 RF zone sensing

Three ESP32-WROOM boards (LEFT / MIDDLE / RIGHT) detect motion behind a wall
and feed a coarse zone estimate into the ghost tracker via
`PersonTracker.update_rf_estimate(dx, dy)`.

### Hardware layout

```
[wall — person moves on the other side]

[LEFT esp]        [MIDDLE esp]        [RIGHT esp]
  COM10           Mac serial             COM8
   |___________________|___________________|
                Wi-Fi hotspot (phone)
```

| Node | Board | Connected to | Port |
|------|-------|-------------|------|
| LEFT | ESP32-WROOM | Windows laptop USB | COM10 |
| RIGHT | ESP32-WROOM | Windows laptop USB | COM8 |
| MIDDLE | ESP32-WROOM | Mac USB | `/dev/cu.usbserial-1130` |

All three boards connect to the **same Wi-Fi hotspot** (phone stays stationary).

---

### Windows laptop — run these in order

**Step 1 — activate the environment**
```powershell
.venv\Scripts\Activate.ps1
```

**Step 2 — flash LEFT board (COM10)**
```powershell
python combining_three_esps\run_wroom.py left
```
> Hangs at `Connecting......`? Hold the **BOOT** button while dots appear, then release.

**Step 3 — flash RIGHT board (COM8)**
```powershell
python combining_three_esps\run_wroom.py right
```

**Step 4 — run the localizer** (wait for Mac bridge first if using MIDDLE node)
```powershell
python combining_three_esps\run_all_three.py --middle-host MAC_IP_HERE
```

> **No Mac / MIDDLE node?** Omit `--middle-host` — zone inference still works with LEFT + RIGHT:
> ```powershell
> python combining_three_esps\run_all_three.py
> ```

**Step 5 — calibrate** (keep the sensing area empty, wait for all nodes to show `CAL✓`)

**Step 6 — skip recalibration on future runs**
```powershell
python combining_three_esps\run_all_three.py --middle-host MAC_IP_HERE --load-cal
```

---

### Mac (MIDDLE node) — run these in order

```bash
pip3 install platformio pyserial

cd combining_three_esps/firmware/wroom
pio run -e MIDDLE --target upload
cd ../../..

python3 combining_three_esps/bridge_middle.py
```

The bridge prints its IP. Pass that IP to the Windows operator as `--middle-host`.

---

### What you should see (Windows terminal)

```
╔══════════════════════════════════════════════════════════════╗
║           3-Node ESP32 CSI Zone Localizer                   ║
╚══════════════════════════════════════════════════════════════╝

  LEFT    [████████░░░░░░░░░░░░░░░░░░░░]   4.2x  -42dBm  pkts= 312  CAL✓
  MIDDLE  [████░░░░░░░░░░░░░░░░░░░░░░░░]   2.1x  -38dBm  pkts= 308  CAL✓
  RIGHT   [██░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.9x  -44dBm  pkts= 310  CAL✓
  ───────────────────────────────────────────────────────────────────────
  Zone: LEFT_MIDDLE    Conf: 0.68   Dist: mid       ◀◀
```

Zones: `LEFT` `LEFT_MIDDLE` `MIDDLE` `RIGHT_MIDDLE` `RIGHT` `NO_MOTION`

CSV logs are saved automatically to `combining_three_esps/data/`.

### RF model options

```bash
python main.py                       # RandomForest (default)
python main.py --model irfanin       # CNN-GRU
python main.py --model aalto         # GradientBoosting
python main.py --cal                 # calibration only, then exit
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ERROR: No camera found.` | Try `CAMERA_INDEX=1 python main.py` |
| Flash hangs at `Connecting......` | Hold BOOT button on ESP32 while dots appear |
| `No ports found` | Check USB cable (must be data cable, not charge-only) |
| `csi_packets=0` after connecting | Normal — wait ~10 s for traffic to appear |
| MIDDLE shows `not connected` | Bridge not running on Mac, or wrong `--middle-host` IP |
| Score stays at 0 forever | Wi-Fi credentials wrong — check `WIFI_SSID`/`WIFI_PASS` in `platformio.ini` |
| Too many false positives | `--sensitivity 3.5` |
| Not detecting motion | `--sensitivity 1.8` |
| `[tactical] disabled` in terminal | AI imports missing — check `pip install -r requirements.txt` and `.env` |

---

## File map

| File / folder | What it does |
|---------------|-------------|
| `main.py` | Entry point — webcam loop, calls tracker, display, RF, and tactical layers |
| `model_setup.py` | Downloads the MediaPipe pose model on first run |
| `tracker/` | `PersonTracker` — pose detection, TRACKING / GHOST mode switching, RF drift |
| `display/` | `draw_skeleton`, `draw_bbox`, `draw_hud`, `draw_identity_overlay` |
| `identity/` | Background face recognition thread (`face_recognition` optional) |
| `rf/csi_reader.py` | ESP32 CSI serial reader + zone ML model — silently disabled if no ESPs |
| `combining_three_esps/` | Firmware flash scripts and 3-node zone localizer |
| `ai/` | Event detector, ConfidentialMind narrator, Gemini PTT |
| `bridge/` | WebSocket server feeding the React HUD |
| `hud/` | React tactical overlay (Vite, runs on port 5173) |
