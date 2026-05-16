"""
rf/csi_reader.py — background CSI reader that feeds zone estimates to the ghost tracker.

Runs reader threads for LEFT (COM9), RIGHT (COM8), and optionally MIDDLE (COM7 or
Mac bridge over network).  Exposes get_rf_nudge(frame_w, frame_h) → (dx, dy) which
main.py calls once per frame and passes to tracker.update_rf_estimate().

Calibration:
    python main.py --cal
    → collects baseline variance for each node with an empty sensing area
    → saves to rf_calibration.json

Normal run:
    python main.py
    → loads rf_calibration.json automatically
    → if the file is missing, RF nudge is silently disabled (camera-only mode)
"""

from __future__ import annotations

import collections
import json
import math
import os
import socket
import sys
import threading
import time
from typing import Optional

import serial
import serial.tools.list_ports

# ── Port / network config ─────────────────────────────────────────────────────
LEFT_PORT   = "COM10"
RIGHT_PORT  = "COM8"
MIDDLE_PORT = "COM7"   # tried first; skipped if not connected
BRIDGE_PORT = 5555     # TCP port for Mac bridge

CAL_FILE = os.path.join(os.path.dirname(__file__), "..", "rf_calibration.json")
CAL_FILE = os.path.normpath(CAL_FILE)

# ── Algorithm constants ───────────────────────────────────────────────────────
CAL_PACKETS = 60     # packets per node during calibration
WINDOW_SIZE = 8      # rolling window for temporal variance
SENSITIVITY = 2.5    # variance ratio above baseline = motion detected

MODEL_FILES = {
    "rf":    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "rf_zone_model.pkl")),
    "knn":   os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "rf_zone_model_knn.pkl")),
    "aalto": os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "rf_zone_model_aalto.pkl")),
}
NODES_REQUIRED = {"left": True, "middle": False, "right": True}

INFERENCE_HZ = 20   # how often the inference thread runs (independent of serial rate)


def _load_model(model_name: str = "rf"):
    """Load a trained zone classifier. model_name: 'rf' or 'knn'."""
    path = MODEL_FILES.get(model_name, MODEL_FILES["rf"])
    if not os.path.isfile(path):
        return None
    try:
        import joblib
        model = joblib.load(path)
        print(f"[RF] {model_name.upper()} model loaded from {os.path.basename(path)}")
        return model
    except Exception as exc:
        print(f"[RF] Could not load {model_name} model ({exc}) — using threshold fallback.")
        return None

# How hard to push the ghost per frame (pixels).
# Confidence-weighted, so a 0.5-confidence LEFT = 2px/frame push.
RF_NUDGE_STRENGTH = 4.0

# Zone → normalised screen-X target (0 = left edge, 1 = right edge)
# Ghost is nudged toward this target while in GHOST mode.
_ZONE_TARGET_X: dict[str, float] = {
    "LEFT":   0.15,
    "MIDDLE": 0.50,
    "RIGHT":  0.85,
}


# ── CSI parsing ───────────────────────────────────────────────────────────────
def _build_features(left: float, middle: float, right: float) -> list[float]:
    """3-feature vector for rf/knn models (trained on ratio data)."""
    return [left, middle, right]


def _build_aalto_features(left_window: list, right_window: list) -> list[float]:
    """
    Extract the same 22 features that aalto_model.py uses during training.
    Imports the function directly from aalto_model so they always stay in sync.
    """
    try:
        import sys, os
        root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        from aalto_model import extract_features, N_SC
        import numpy as np

        def _mean_amps(window) -> list[float]:
            if not window:
                return [0.0] * N_SC
            arr = np.array([v[:N_SC] for v in window if len(v) >= 4], dtype=float)
            if arr.size == 0:
                return [0.0] * N_SC
            mean = arr.mean(axis=0)
            padded = list(mean) + [0.0] * max(0, N_SC - len(mean))
            return padded[:N_SC]

        return extract_features(_mean_amps(left_window), _mean_amps(right_window))
    except Exception:
        return [0.0] * 22


def _parse_csi(line: str) -> Optional[dict]:
    if not line.startswith("CSI:"):
        return None
    parts = line[4:].split(",")
    if len(parts) < 5:
        return None
    try:
        rssi    = int(parts[0])
        csi_len = int(parts[3])
        raw     = [int(x) for x in parts[4: 4 + csi_len]]
        if len(raw) < csi_len:
            return None
        amps = [math.sqrt(raw[i] ** 2 + raw[i + 1] ** 2)
                for i in range(0, len(raw) - 1, 2)]
        return {"rssi": rssi, "amplitudes": amps}
    except (ValueError, IndexError):
        return None


def _temporal_variance(window: list[list[float]], n_sc: int) -> float:
    n = len(window)
    if n < 2:
        return 0.0
    total, counted = 0.0, 0
    for i in range(n_sc):
        vals = [window[j][i] for j in range(n) if i < len(window[j])]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        total += sum((v - mean) ** 2 for v in vals) / len(vals)
        counted += 1
    return total / counted if counted else 0.0


# ── Zone inference ────────────────────────────────────────────────────────────
def _infer_zone(left: float, middle: float, right: float,
                middle_up: bool) -> tuple[str, float, bool]:
    """
    Returns (zone_str, confidence, middle_was_inferred).
    Uses 3-node logic when MIDDLE is connected, 2-node symmetry otherwise.
    """
    max_s = max(left, middle, right)

    if max_s < SENSITIVITY:
        return "NO_MOTION", round(1.0 - max_s / max(SENSITIVITY, 0.01), 2), False

    # ── 3-node ────────────────────────────────────────────────────────────────
    if middle_up and middle >= SENSITIVITY * 0.5:
        total = left + middle + right or 1e-9
        nl, nm, nr = left / total, middle / total, right / total
        if nl >= 0.55 and nr <= 0.25:      return "LEFT",         round(nl, 2), False
        if nr >= 0.55 and nl <= 0.25:      return "RIGHT",        round(nr, 2), False
        if nm >= 0.45 and nl < 0.35 and nr < 0.35:
                                           return "MIDDLE",       round(nm, 2), False
        if nl > nr and nl > nm:            return "LEFT_MIDDLE",  round((nl+nm)/2, 2), False
        if nr > nl and nr > nm:            return "RIGHT_MIDDLE", round((nr+nm)/2, 2), False
        if nl > nr:                        return "LEFT_MIDDLE",  round(nl, 2), False
        if nr > nl:                        return "RIGHT_MIDDLE", round(nr, 2), False
        return "MIDDLE", round(nm, 2), False

    # ── 2-node: infer MIDDLE from L/R symmetry ────────────────────────────────
    total_lr = left + right or 1e-9
    lr_ratio = left / total_lr
    symmetry = 1.0 - abs(left - right) / total_lr

    if   lr_ratio >= 0.65:               return "LEFT",         round(lr_ratio, 2),            True
    elif lr_ratio <= 0.35:               return "RIGHT",        round(1-lr_ratio, 2),           True
    elif symmetry >= 0.70:               return "MIDDLE",       round(min(symmetry*0.85, 0.75), 2), True
    elif lr_ratio > 0.5:                 return "LEFT_MIDDLE",  round((lr_ratio+symmetry)/2, 2), True
    else:                                return "RIGHT_MIDDLE", round((1-lr_ratio+symmetry)/2, 2), True


# ── Per-node state ────────────────────────────────────────────────────────────
def _fresh() -> dict:
    return {
        "connected":    False,
        "calibrated":   False,
        "cal_buffer":   [],
        "baseline_var": None,
        "n_sc":         0,
        "window":       collections.deque(maxlen=WINDOW_SIZE),
        "ratio":        0.0,
        "rssi":         0,
        "pkts":         0,
    }


# ── CsiReader ─────────────────────────────────────────────────────────────────
class CsiReader:
    """
    Background CSI reader.  Provides get_rf_nudge() for the main loop.

    Typical usage:
        rf = CsiReader(middle_host="10.75.241.42")  # or None if no Mac bridge
        rf.load_calibration()   # loads rf_calibration.json
        rf.start()              # kicks off background threads
        # ... in frame loop:
        dx, dy = rf.get_rf_nudge(frame_w, frame_h)
        tracker.update_rf_estimate(dx, dy)
        # ... at shutdown:
        rf.stop()
    """

    def __init__(self, middle_host: Optional[str] = None, model_name: str = "rf"):
        self._middle_host = middle_host
        self._lock        = threading.Lock()
        self._running     = False

        self._nodes: dict[str, dict] = {
            "left":   _fresh(),
            "middle": _fresh(),
            "right":  _fresh(),
        }

        self._zone       = "NO_MOTION"
        self._confidence = 0.0
        self._inferred   = False
        self._model      = _load_model(model_name)
        self._model_name = model_name

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, n_packets: int = CAL_PACKETS) -> None:
        """
        Blocking calibration.  Opens serial ports, collects n_packets per node
        with the sensing area EMPTY, saves baseline to rf_calibration.json.
        Call via:  python main.py --cal
        """
        print("\n=== RF Calibration Mode ===")
        print("Keep the ENTIRE sensing area (behind the wall) EMPTY.")
        print(f"Collecting {n_packets} packets per node...\n")

        calibrated = {}

        for name, port in [("left", LEFT_PORT), ("right", RIGHT_PORT),
                           ("middle", self._middle_host or MIDDLE_PORT)]:
            buf: list[list[float]] = []

            # Try opening source (serial or network)
            src = self._open_source(name, port)
            if src is None:
                print(f"  {name.upper():<7} — not connected, skipped.")
                continue

            print(f"  {name.upper():<7} — calibrating on {port} ...")
            bar_w = 30
            read_fn = self._make_readline(src)

            while len(buf) < n_packets:
                line = read_fn()
                if not line:
                    continue
                pkt = _parse_csi(line)
                if pkt and pkt["amplitudes"]:
                    buf.append(pkt["amplitudes"])
                    done = len(buf)
                    filled = int(bar_w * done / n_packets)
                    print(f"\r    [{'█'*filled}{'░'*(bar_w-filled)}] {done}/{n_packets}",
                          end="", flush=True)

            print(f"\r    [{'█'*bar_w}] {n_packets}/{n_packets}  done          ")

            n_sc = min(len(v) for v in buf)
            bvar = _temporal_variance(buf, n_sc)
            calibrated[name] = {
                "baseline_var": max(bvar, 0.5),
                "n_sc":         n_sc,
                "saved_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            self._close_source(src)

        if not calibrated:
            print("\nERROR: No nodes responded. Check USB cables and firmware.")
            sys.exit(1)

        with open(CAL_FILE, "w") as f:
            json.dump(calibrated, f, indent=2)

        print(f"\nCalibration saved to {CAL_FILE}")
        print("Run 'python main.py' to start tracking with RF support.\n")

    def load_calibration(self) -> bool:
        """
        Loads rf_calibration.json.  Returns True on success.
        If the file is missing, prints a warning and RF nudge is disabled.
        """
        if not os.path.isfile(CAL_FILE):
            print(f"[RF] No calibration file found ({CAL_FILE}).")
            print("[RF] Run 'python main.py --cal' first to calibrate.")
            print("[RF] Continuing in camera-only mode.\n")
            return False

        with open(CAL_FILE) as f:
            data = json.load(f)

        with self._lock:
            for name, entry in data.items():
                if name in self._nodes:
                    ns = self._nodes[name]
                    ns["baseline_var"] = entry["baseline_var"]
                    ns["n_sc"]         = entry["n_sc"]
                    ns["calibrated"]   = True

        nodes_cal = [n.upper() for n in data]
        print(f"[RF] Calibration loaded for: {', '.join(nodes_cal)}  "
              f"({os.path.basename(CAL_FILE)})")
        return True

    # ── Start / stop ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start serial reader threads + a dedicated inference thread."""
        self._running = True

        for name, port in [("left", LEFT_PORT), ("right", RIGHT_PORT)]:
            threading.Thread(target=self._serial_reader,
                             args=(name, port), daemon=True).start()

        if self._middle_host:
            threading.Thread(target=self._network_reader,
                             args=("middle", self._middle_host, BRIDGE_PORT),
                             daemon=True).start()
        else:
            threading.Thread(target=self._serial_reader,
                             args=("middle", MIDDLE_PORT), daemon=True).start()

        # Inference thread: ML runs here, never inside the serial read lock
        threading.Thread(target=self._inference_loop,
                         name="rf_inference", daemon=True).start()

    def stop(self) -> None:
        self._running = False

    # ── Public interface ──────────────────────────────────────────────────────

    def get_rf_nudge(self, frame_w: int, frame_h: int) -> tuple[float, float]:
        """
        No longer used for visual positioning (zone highlight handles that now).
        Kept for API compatibility — always returns (0, 0).
        """
        return 0.0, 0.0

    @property
    def latest_zone(self) -> str:
        with self._lock:
            return self._zone

    @property
    def latest_confidence(self) -> float:
        with self._lock:
            return self._confidence

    @property
    def middle_inferred(self) -> bool:
        with self._lock:
            return self._inferred

    @property
    def any_node_connected(self) -> bool:
        with self._lock:
            return any(n["connected"] for n in self._nodes.values())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_packet(self, name: str, pkt: dict) -> None:
        """Called under _lock. Updates node state ONLY — no ML inference here.
        Zone inference runs in its own thread so serial reading is never blocked."""
        ns   = self._nodes[name]
        amps = pkt["amplitudes"]
        ns["rssi"] = pkt["rssi"]
        ns["pkts"] += 1

        if not ns["calibrated"]:
            return

        ns["window"].append(amps[:ns["n_sc"]])
        if len(ns["window"]) >= 2:
            t_var       = _temporal_variance(list(ns["window"]), ns["n_sc"])
            ns["ratio"] = t_var / ns["baseline_var"]

    def _inference_loop(self) -> None:
        """
        Dedicated background thread for zone inference.
        Reads the latest ratios, runs the model, and writes back the result —
        all without blocking the serial reader threads.
        Runs at INFERENCE_HZ (20 Hz) which is plenty for display purposes.
        """
        import numpy as np
        interval = 1.0 / INFERENCE_HZ

        while self._running:
            time.sleep(interval)

            # Snapshot ratios under lock (fast — no ML here)
            with self._lock:
                left   = self._nodes["left"]["ratio"]
                middle = self._nodes["middle"]["ratio"]
                right  = self._nodes["right"]["ratio"]
                mid_up = self._nodes["middle"]["connected"]

            # Run inference OUTSIDE the lock
            if self._model is not None:
                max_r = max(left, middle, right)
                if max_r < 1.1:
                    zone, conf, inferred = "NO_MOTION", 0.9, False
                else:
                    try:
                        # aalto model needs raw amplitude features from window buffers;
                        # rf/knn models use the 3 aggregated ratio values.
                        if self._model_name == "aalto":
                            with self._lock:
                                lw = list(self._nodes["left"]["window"])
                                rw = list(self._nodes["right"]["window"])
                            features = _build_aalto_features(lw, rw)
                        else:
                            features = _build_features(left, middle, right)

                        proba    = self._model.predict_proba([features])[0]
                        best     = int(proba.argmax())
                        zone     = self._model.classes_[best]
                        conf     = round(float(proba[best]), 2)
                        inferred = not mid_up
                    except Exception:
                        zone, conf, inferred = _infer_zone(left, middle, right, mid_up)
            else:
                zone, conf, inferred = _infer_zone(left, middle, right, mid_up)

            # Write result back under lock (fast — just three assignments)
            with self._lock:
                self._zone       = zone
                self._confidence = conf
                self._inferred   = inferred

    def _serial_reader(self, name: str, port: str) -> None:
        try:
            ser = serial.Serial(port, 115200, timeout=2)
        except serial.SerialException:
            return  # port not available — silent for optional nodes

        with self._lock:
            self._nodes[name]["connected"] = True

        while self._running:
            try:
                raw = ser.readline()
            except serial.SerialException:
                break
            if not raw:
                continue
            try:
                line = raw.decode("ascii", errors="replace").strip()
            except Exception:
                continue
            pkt = _parse_csi(line)
            if pkt:
                with self._lock:
                    self._process_packet(name, pkt)

        ser.close()
        with self._lock:
            self._nodes[name]["connected"] = False

    def _network_reader(self, name: str, host: str, port: int) -> None:
        while self._running:
            try:
                sock = socket.create_connection((host, port), timeout=5)
                sock.settimeout(3)
                with self._lock:
                    self._nodes[name]["connected"] = True
                buf = b""
                while self._running:
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line_b, buf = buf.split(b"\n", 1)
                        try:
                            line = line_b.decode("ascii", errors="replace").strip()
                        except Exception:
                            continue
                        pkt = _parse_csi(line)
                        if pkt:
                            with self._lock:
                                self._process_packet(name, pkt)
                sock.close()
            except (OSError, socket.timeout):
                pass
            with self._lock:
                self._nodes[name]["connected"] = False
            if self._running:
                time.sleep(3)   # wait before reconnect attempt

    # ── Calibration helpers ───────────────────────────────────────────────────

    @staticmethod
    def _open_source(name: str, port: str):
        """Return an open serial.Serial or socket, or None if unavailable."""
        if name == "middle" and not port.startswith("/dev") and not port.startswith("COM"):
            # It's a hostname/IP
            try:
                sock = socket.create_connection((port, BRIDGE_PORT), timeout=5)
                sock.settimeout(3)
                return sock
            except OSError:
                return None
        try:
            return serial.Serial(port, 115200, timeout=2)
        except serial.SerialException:
            return None

    @staticmethod
    def _make_readline(src) -> callable:
        if isinstance(src, serial.Serial):
            def _read():
                raw = src.readline()
                return raw.decode("ascii", errors="replace").strip() if raw else ""
        else:
            buf = b""
            def _read():
                nonlocal buf
                try:
                    chunk = src.recv(4096)
                    if chunk:
                        buf += chunk
                except socket.timeout:
                    return ""
                if b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    return line.decode("ascii", errors="replace").strip()
                return ""
        return _read

    @staticmethod
    def _close_source(src) -> None:
        try:
            src.close()
        except Exception:
            pass
