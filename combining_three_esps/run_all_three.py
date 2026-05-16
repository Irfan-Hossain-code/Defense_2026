"""
run_all_three.py — live 3-node CSI localization.

Reads serial CSI data from LEFT (COM9), RIGHT (COM8), and optionally
MIDDLE.  MIDDLE can come from either:
  - A local COM port (COM7) if the board is USB-connected to this PC
  - A TCP socket from the Mac running bridge_middle.py over the hotspot

Usage:
    python combining_three_esps/run_all_three.py

    # MIDDLE from Mac over the network (most common setup):
    python combining_three_esps/run_all_three.py --middle-host 10.75.241.XXX

    # Reload saved calibration:
    python combining_three_esps/run_all_three.py --middle-host 10.75.241.XXX --load-cal

    # Adjust sensitivity (default 2.5):
    python combining_three_esps/run_all_three.py --sensitivity 2.0

Output dict (also printed and saved to CSV):
    {
        "zone":              "LEFT_MIDDLE",
        "confidence":        0.72,
        "distance_estimate": "near",
        "scores": {
            "left":   12.3,
            "middle":  7.1,
            "right":   1.8
        }
    }

This dict is designed to be consumed later by tracker.update_rf_estimate().

=== PHYSICAL SETUP ===

    [wall]
    |  (sensing area — person walks here)
    |
    |  distance ~0.5–2 m
    |
    [LEFT esp]   [MIDDLE esp]   [RIGHT esp]
         COM9         COM7           COM8
         |             |              |
         └─────────────┴──────────────┘ ← USB to laptop
                    Wi-Fi hotspot ↑
                   (phone, same room)

The three ESPs are on the SAME SIDE of the wall.
The person moves on the OTHER side.
LEFT/MIDDLE/RIGHT spacing: ~0.5–1 m apart.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import json
import math
import os
import socket
import sys
import threading
import time

import serial
import serial.tools.list_ports

# ── Node configuration ────────────────────────────────────────────────────────
# Change ports here if your boards are on different COM ports.
NODES: dict[str, dict] = {
    "left":   {"port": "COM9", "required": True},
    "middle": {"port": "COM7", "required": False},  # optional — battery-powered
    "right":  {"port": "COM8", "required": True},
}

BAUD             = 115200
CAL_PACKETS      = 60      # packets per node during calibration (stay STILL)
WINDOW_SIZE      = 8       # rolling window for temporal variance
DEFAULT_SENS     = 2.5     # variance ratio above baseline = motion
INFER_INTERVAL   = 0.5     # seconds between zone inference updates
DISPLAY_HISTORY  = 30      # sparkline width

CAL_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

BAR_WIDTH = 26
SPARK_CHARS = " ▁▂▃▄▅▆▇█"


# ── CSI parser ────────────────────────────────────────────────────────────────
def parse_csi_line(line: str) -> dict | None:
    """
    Parse: CSI:<rssi>,<noise>,<channel>,<len>,<byte0>,<byte1>,...
    Returns dict with rssi, noise_floor, channel, amplitudes list, or None.
    """
    if not line.startswith("CSI:"):
        return None
    parts = line[4:].split(",")
    if len(parts) < 5:
        return None
    try:
        rssi        = int(parts[0])
        noise_floor = int(parts[1])
        channel     = int(parts[2])
        csi_len     = int(parts[3])
        raw = [int(x) for x in parts[4: 4 + csi_len]]
        if len(raw) < csi_len:
            return None
        amps = [math.sqrt(raw[i]**2 + raw[i+1]**2)
                for i in range(0, len(raw)-1, 2)]
        return {"rssi": rssi, "noise_floor": noise_floor,
                "channel": channel, "amplitudes": amps}
    except (ValueError, IndexError):
        return None


# ── Temporal variance ─────────────────────────────────────────────────────────
def temporal_variance(window: list[list[float]], n_sc: int) -> float:
    """
    For each sub-carrier, compute variance of its amplitude across all
    packets in the window.  Return the mean variance across all sub-carriers.

    Low  → channel is stable → nothing moving
    High → channel is fluctuating → motion or presence
    """
    n = len(window)
    if n < 2:
        return 0.0
    total, counted = 0.0, 0
    for i in range(n_sc):
        vals = [window[j][i] for j in range(n) if i < len(window[j])]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        total += sum((v - mean)**2 for v in vals) / len(vals)
        counted += 1
    return total / counted if counted else 0.0


# ── Shared state ──────────────────────────────────────────────────────────────
_lock = threading.Lock()

def _fresh_node_state() -> dict:
    return {
        "connected":    False,
        "calibrated":   False,
        "cal_buffer":   [],         # raw amplitude lists during calibration
        "baseline_var": None,       # locked baseline variance after calibration
        "n_sc":         0,          # number of sub-carriers in baseline
        "window":       collections.deque(maxlen=WINDOW_SIZE),
        "latest_ratio": 0.0,        # current temporal_var / baseline_var
        "rssi":         0,
        "pkts":         0,
        "history":      collections.deque(maxlen=DISPLAY_HISTORY),
    }

node_state: dict[str, dict] = {name: _fresh_node_state() for name in NODES}


# ── Shared packet processor (used by both serial and network readers) ─────────
def _process_line(name: str, line: str) -> None:
    """Parse one CSI line and update node_state[name]. Called under _lock."""
    if not line.startswith("CSI:"):
        return
    pkt = parse_csi_line(line)
    if pkt is None or not pkt["amplitudes"]:
        return

    ns   = node_state[name]
    amps = pkt["amplitudes"]
    ns["pkts"] += 1
    ns["rssi"]  = pkt["rssi"]

    if not ns["calibrated"]:
        ns["cal_buffer"].append(amps)
        if len(ns["cal_buffer"]) >= CAL_PACKETS:
            n_sc = min(len(v) for v in ns["cal_buffer"])
            bvar = temporal_variance(ns["cal_buffer"], n_sc)
            ns["baseline_var"] = max(bvar, 0.5)
            ns["n_sc"]         = n_sc
            ns["calibrated"]   = True
    else:
        ns["window"].append(amps[:ns["n_sc"]])
        if len(ns["window"]) >= 2:
            t_var = temporal_variance(list(ns["window"]), ns["n_sc"])
            ns["latest_ratio"] = t_var / ns["baseline_var"]
            ns["history"].append(ns["latest_ratio"])


# ── Serial reader thread ──────────────────────────────────────────────────────
def reader_thread(name: str, port: str, sensitivity: float) -> None:
    """Reads from a local COM/serial port and updates node_state[name]."""
    try:
        ser = serial.Serial(port, BAUD, timeout=2)
    except serial.SerialException as exc:
        if NODES[name]["required"]:
            print(f"\n[{name.upper()}] ERROR: cannot open {port}: {exc}")
        return

    with _lock:
        node_state[name]["connected"] = True

    while True:
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
        with _lock:
            _process_line(name, line)

    ser.close()


# ── Network reader thread (MIDDLE node from Mac via bridge_middle.py) ─────────
def network_reader_thread(name: str, host: str, port: int = 5555) -> None:
    """
    Connects to bridge_middle.py running on the Mac over the hotspot network.
    Reads CSI lines from the TCP socket and updates node_state[name].
    Reconnects automatically if the connection drops.
    """
    print(f"[{name.upper()}] Connecting to bridge at {host}:{port} ...")
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=10)
            sock.settimeout(3)
            with _lock:
                node_state[name]["connected"] = True
            print(f"[{name.upper()}] Connected to Mac bridge.")
            buf = b""
            while True:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    try:
                        line = line_bytes.decode("ascii", errors="replace").strip()
                    except Exception:
                        continue
                    with _lock:
                        _process_line(name, line)
            sock.close()
        except (ConnectionRefusedError, socket.timeout, OSError) as exc:
            with _lock:
                node_state[name]["connected"] = False
            print(f"[{name.upper()}] Bridge disconnected ({exc}). Retrying in 3s...")
            time.sleep(3)
            ns["history"].append(ratio)

    ser.close()


# ── Zone inference ────────────────────────────────────────────────────────────
def infer_zone(scores: dict[str, float], sensitivity: float) -> dict:
    """
    Given a dict of variance-ratio scores per node, return a zone estimate.

    scores = {"left": 3.2, "middle": 1.1, "right": 0.8}

    Returns:
    {
        "zone":              "LEFT_MIDDLE",
        "confidence":        0.72,
        "distance_estimate": "near",   # near | mid | far | unknown
        "scores":            scores,
        "middle_inferred":   False,    # True when MIDDLE node absent, zone estimated from L+R
    }
    """
    left   = scores.get("left",   0.0)
    middle = scores.get("middle", 0.0)
    right  = scores.get("right",  0.0)

    middle_connected = node_state["middle"]["connected"]

    # ── Distance helper (shared) ───────────────────────────────────────────────
    def _dist(max_s: float) -> str:
        if max_s > 8.0:  return "near"
        if max_s > 3.5:  return "mid"
        return "far"

    # ── No-motion check ───────────────────────────────────────────────────────
    max_score = max(left, middle, right)
    if max_score < sensitivity:
        return {
            "zone":             "NO_MOTION",
            "confidence":       round(1.0 - max_score / max(sensitivity, 0.01), 2),
            "distance_estimate":"unknown",
            "scores":           scores,
            "middle_inferred":  not middle_connected,
        }

    # ── 2-node path: MIDDLE absent or not contributing ────────────────────────
    # When MIDDLE is offline, infer centre from LEFT/RIGHT symmetry.
    # Key insight: if someone is in the MIDDLE, both LEFT and RIGHT see
    # similar elevated changes.  Compute how symmetric they are.
    if not middle_connected or middle < sensitivity * 0.5:
        total_lr = left + right
        if total_lr == 0:
            return {"zone": "UNKNOWN", "confidence": 0.0,
                    "distance_estimate": "unknown", "scores": scores,
                    "middle_inferred": True}

        # lr_ratio: 1.0 = all LEFT, 0.5 = symmetric, 0.0 = all RIGHT
        lr_ratio  = left / total_lr
        # symmetry: 1.0 = both equal (middle), 0.0 = completely one-sided
        symmetry  = 1.0 - abs(left - right) / total_lr

        if lr_ratio >= 0.65:
            zone, conf = "LEFT",         lr_ratio
        elif lr_ratio <= 0.35:
            zone, conf = "RIGHT",        1.0 - lr_ratio
        elif symmetry >= 0.70:
            # Both nodes see similar strong change → infer MIDDLE
            # Confidence is capped at 0.75 because we're estimating without a sensor
            zone, conf = "MIDDLE",       min(symmetry * 0.85, 0.75)
        elif lr_ratio > 0.5:
            zone, conf = "LEFT_MIDDLE",  (lr_ratio + symmetry) / 2
        else:
            zone, conf = "RIGHT_MIDDLE", (1.0 - lr_ratio + symmetry) / 2

        return {
            "zone":             zone,
            "confidence":       round(conf, 2),
            "distance_estimate":_dist(max(left, right)),
            "scores":           scores,
            "middle_inferred":  True,
        }

    # ── 3-node path: MIDDLE is connected and contributing ─────────────────────
    total = left + middle + right
    nl = left   / total if total > 0 else 0.0
    nm = middle / total if total > 0 else 0.0
    nr = right  / total if total > 0 else 0.0

    if nl >= 0.55 and nr <= 0.25:
        zone, conf = "LEFT",         nl
    elif nr >= 0.55 and nl <= 0.25:
        zone, conf = "RIGHT",        nr
    elif nm >= 0.45 and nl < 0.35 and nr < 0.35:
        zone, conf = "MIDDLE",       nm
    elif nl > nr and nl > nm:
        zone, conf = "LEFT_MIDDLE",  (nl + nm) / 2
    elif nr > nl and nr > nm:
        zone, conf = "RIGHT_MIDDLE", (nr + nm) / 2
    elif nl > nr:
        zone, conf = "LEFT_MIDDLE",  nl
    elif nr > nl:
        zone, conf = "RIGHT_MIDDLE", nr
    else:
        zone, conf = "MIDDLE",       nm

    return {
        "zone":             zone,
        "confidence":       round(conf, 2),
        "distance_estimate":_dist(max_score),
        "scores":           scores,
        "middle_inferred":  False,
    }


# ── Display helpers ───────────────────────────────────────────────────────────
def bar(ratio: float, max_ratio: float, width: int) -> str:
    filled = min(int(width * ratio / max_ratio), width)
    return "█" * filled + "░" * (width - filled)

def sparkline(history: collections.deque, max_val: float) -> str:
    out = []
    for v in history:
        idx = min(int(8 * v / max(max_val, 0.01)), 8)
        out.append(SPARK_CHARS[idx])
    return "".join(out)

def node_status_line(name: str, ns: dict, max_ratio: float,
                     inferred: bool = False) -> str:
    label = name.upper().ljust(6)
    if not ns["connected"]:
        if name == "middle" and inferred:
            return f"  {label}  [{'·'*BAR_WIDTH}]  ─────  not connected (zone inferred from L+R)"
        return f"  {label}  [{'─'*BAR_WIDTH}]  ─────  not connected"
    ratio = ns["latest_ratio"]
    pkts  = ns["pkts"]
    rssi  = ns["rssi"]
    cal   = "CAL✓" if ns["calibrated"] else f"cal {len(ns['cal_buffer'])}/{CAL_PACKETS}"
    b     = bar(ratio, max_ratio, BAR_WIDTH)
    spark = sparkline(ns["history"], max_ratio)
    return (f"  {label}  [{b}]  {ratio:5.2f}x  "
            f"{rssi:4}dBm  pkts={pkts:4d}  {cal}  {spark}")

ZONE_ICONS = {
    "LEFT":        "◀◀◀          ",
    "LEFT_MIDDLE": " ◀◀          ",
    "MIDDLE":      "    ◆◆◆      ",
    "RIGHT_MIDDLE":"         ▶▶  ",
    "RIGHT":       "          ▶▶▶",
    "NO_MOTION":   "─── still ───",
    "UNKNOWN":     "   ???       ",
}

# Number of display lines we print each update (used for cursor-up overwrite)
_DISPLAY_LINES = 9


def print_display(result: dict, all_calibrated: bool) -> None:
    scores    = result["scores"]
    max_ratio = max(max(scores.values()), 3.0)  # floor so bar isn't tiny

    inferred = result.get("middle_inferred", False)

    lines = [
        "─" * 72,
        node_status_line("left",   node_state["left"],   max_ratio),
        node_status_line("middle", node_state["middle"], max_ratio, inferred),
        node_status_line("right",  node_state["right"],  max_ratio),
        "─" * 72,
    ]

    if not all_calibrated:
        lines.append("  Calibrating — keep sensing area EMPTY until all nodes show CAL✓")
    else:
        icon       = ZONE_ICONS.get(result["zone"], "")
        inf_marker = " [L+R]" if inferred else ""
        lines.append(
            f"  Zone: {result['zone']:<14}  "
            f"Conf: {result['confidence']:.2f}{inf_marker}   "
            f"Dist: {result['distance_estimate']:<8}  "
            f"{icon}"
        )

    lines.append("─" * 72)
    lines.append(f"  {datetime.datetime.now().strftime('%H:%M:%S')}  "
                 f"(Ctrl+C to stop)")

    # Move cursor up to overwrite previous block after the first print
    if getattr(print_display, "_first", True):
        print_display._first = False
    else:
        # Move cursor up _DISPLAY_LINES lines
        print(f"\033[{_DISPLAY_LINES}A", end="")

    for line in lines:
        print(f"\033[K{line}")   # \033[K clears to end of line before printing


# ── Calibration persistence ───────────────────────────────────────────────────
def save_calibration() -> None:
    data = {}
    with _lock:
        for name, ns in node_state.items():
            if ns["calibrated"]:
                data[name] = {
                    "baseline_var": ns["baseline_var"],
                    "n_sc":         ns["n_sc"],
                    "saved_at":     datetime.datetime.now().isoformat(),
                }
    with open(CAL_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nCalibration saved to {CAL_FILE}")


def load_calibration() -> bool:
    if not os.path.isfile(CAL_FILE):
        return False
    with open(CAL_FILE) as f:
        data = json.load(f)
    with _lock:
        for name, entry in data.items():
            if name in node_state:
                ns = node_state[name]
                ns["baseline_var"] = entry["baseline_var"]
                ns["n_sc"]         = entry["n_sc"]
                ns["calibrated"]   = True
                print(f"  Loaded calibration for {name.upper()}: "
                      f"baseline={entry['baseline_var']:.2f} "
                      f"n_sc={entry['n_sc']} "
                      f"(from {entry['saved_at'][:16]})")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-node CSI zone inference."
    )
    parser.add_argument("--sensitivity", "-s", type=float, default=DEFAULT_SENS,
                        help=f"Variance ratio threshold (default {DEFAULT_SENS}). "
                             "Lower = more sensitive.")
    parser.add_argument("--load-cal", action="store_true",
                        help="Load calibration from calibration.json instead of re-calibrating.")
    parser.add_argument("--middle-host", default=None, metavar="IP",
                        help="Mac's hotspot IP for MIDDLE node via bridge_middle.py "
                             "(e.g. 10.75.241.42).  If omitted, tries COM7 locally.")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(DATA_DIR, f"localization_{stamp}.csv")

    middle_src = f"Mac bridge @ {args.middle_host}:5555" if args.middle_host else "COM7 (local)"

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           3-Node ESP32 CSI Zone Localizer                   ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  LEFT   : COM9  (ESP32-WROOM, USB)                          ║")
    print(f"║  MIDDLE : {middle_src:<51}║")
    print(f"║  RIGHT  : COM8  (ESP32-WROOM, USB)                          ║")
    print(f"║  Sensitivity: {args.sensitivity:<3}  CSV: {os.path.basename(csv_path):<29}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Load calibration from file if requested ───────────────────────────────
    if args.load_cal:
        print("Loading calibration from file...")
        if load_calibration():
            print("Calibration loaded. Skipping live calibration phase.\n")
        else:
            print(f"No calibration file found at {CAL_FILE}. Will calibrate live.\n")
    else:
        print(f"Calibration: {CAL_PACKETS} still packets per node.")
        print("Keep the sensing area EMPTY during calibration.\n")

    # ── Start reader threads ──────────────────────────────────────────────────
    threads = []
    for name, cfg in NODES.items():
        if name == "middle" and args.middle_host:
            # MIDDLE comes from Mac over network instead of a local COM port
            t = threading.Thread(
                target=network_reader_thread,
                args=(name, args.middle_host, 5555),
                daemon=True,
                name=f"reader_middle_net",
            )
            t.start()
            threads.append(t)
            print(f"  Started network reader for MIDDLE → {args.middle_host}:5555")
        else:
            t = threading.Thread(
                target=reader_thread,
                args=(name, cfg["port"], args.sensitivity),
                daemon=True,
                name=f"reader_{name}",
            )
            t.start()
            threads.append(t)
            print(f"  Started serial reader for {name.upper()} on {cfg['port']}"
                  + (" (optional)" if not cfg["required"] else ""))

    # Give threads a moment to try connecting before we start printing
    time.sleep(1.5)

    # Check required nodes connected
    with _lock:
        for name, cfg in NODES.items():
            if cfg["required"] and not node_state[name]["connected"]:
                print(f"\nERROR: Required node {name.upper()} ({cfg['port']}) is not connected.")
                print("  Check the USB cable and that the firmware has been flashed.")
                sys.exit(1)

    print(f"\nCSV log: {csv_path}")
    print("Press Ctrl+C to stop and save calibration.\n")
    # Print blank lines so the first overwrite has something to replace
    for _ in range(_DISPLAY_LINES):
        print()

    # ── CSV setup ─────────────────────────────────────────────────────────────
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "zone", "confidence", "distance_estimate",
        "left_ratio", "middle_ratio", "right_ratio",
        "left_rssi",  "middle_rssi",  "right_rssi",
    ])

    # ── Main inference loop ───────────────────────────────────────────────────
    try:
        while True:
            time.sleep(INFER_INTERVAL)

            with _lock:
                scores = {
                    "left":   round(node_state["left"]["latest_ratio"],   2),
                    "middle": round(node_state["middle"]["latest_ratio"], 2),
                    "right":  round(node_state["right"]["latest_ratio"],  2),
                }
                all_cal = all(
                    node_state[n]["calibrated"]
                    for n in NODES
                    if node_state[n]["connected"]
                )
                rssi = {n: node_state[n]["rssi"] for n in NODES}

            result = infer_zone(scores, args.sensitivity)
            print_display(result, all_cal)

            # CSV row
            ts = datetime.datetime.now().isoformat(timespec="milliseconds")
            csv_writer.writerow([
                ts,
                result["zone"],
                result["confidence"],
                result["distance_estimate"],
                scores["left"],
                scores["middle"],
                scores["right"],
                rssi["left"],
                rssi["middle"],
                rssi["right"],
            ])
            csv_file.flush()

    except KeyboardInterrupt:
        pass

    csv_file.close()
    print(f"\n\nStopped. Log saved to {csv_path}")
    save_calibration()


if __name__ == "__main__":
    main()
