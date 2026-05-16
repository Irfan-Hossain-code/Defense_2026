"""
ml/csi/collect.py — real-time CNN training data collector.

Reads LEFT + RIGHT serial ports simultaneously, builds sliding windows
of shape (T=20, S=64, N=2), saves them to an NPZ file.

Usage:
    python -m ml.csi.collect --per-zone 300 --out data/csi_windows.npz
    python -m ml.csi.collect --per-zone 300 --out data/csi_windows.npz \\
                              --left COM10 --right COM8 --baud 115200

Output NPZ arrays:
    X            (N_samples, 20, 64, 2)  float32, z-score normalised
    y            (N_samples,)             int64  0=LEFT 1=MIDDLE 2=RIGHT
    label_names  ["LEFT", "MIDDLE", "RIGHT"]
    mean         scalar — use for inference normalisation
    std          scalar

Inference normalisation:
    data = np.load("data/csi_windows.npz", allow_pickle=True)
    mean, std = float(data["mean"]), float(data["std"])
    window_normalised = (raw_window - mean) / std
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

# Allow running as  python -m ml.csi.collect  from repo root
_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from ml.csi.windows import (
    WindowBuilder, parse_amplitudes,
    LABEL_NAMES, LABEL_TO_IDX, T, S,
)

try:
    import serial
except ImportError:
    print("pyserial not installed.  Run:  pip install pyserial")
    sys.exit(1)

ZONES      = ["LEFT", "MIDDLE", "RIGHT"]
BREAK_SECS = 10
WARMUP_SECS = 5
BAR_W       = 30


# ── Serial reader thread ──────────────────────────────────────────────────────

def _reader(port: str, baud: int, node_idx: int,
            builder: WindowBuilder, builder_lock: threading.Lock,
            windows: list, labels: list, zone_ref: list,
            running: list) -> None:
    """Thread: reads one serial port, pushes amplitudes into the WindowBuilder."""
    try:
        ser = serial.Serial(port, baud, timeout=2)
    except serial.SerialException as exc:
        print(f"\n  [node {node_idx}] Cannot open {port}: {exc}")
        return

    while running[0]:
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

        amps = parse_amplitudes(line)
        if amps is None:
            continue

        with builder_lock:
            builder.push(node_idx, amps)
            if builder.ready():
                window = builder.get_window()           # (T, S, 2)
                windows.append(window)
                labels.append(LABEL_TO_IDX[zone_ref[0]])

    try:
        ser.close()
    except Exception:
        pass


# ── Countdown ─────────────────────────────────────────────────────────────────

def countdown(label: str, seconds: int) -> None:
    for i in range(seconds, 0, -1):
        print(f"\r  {label} in {i}s ...  ", end="", flush=True)
        time.sleep(1)
    print(f"\r  {'':40}", end="", flush=True)


# ── Per-zone collection ───────────────────────────────────────────────────────

def collect_zone(zone: str, per_zone: int,
                 left_port: str, right_port: str, baud: int) -> tuple[list, list]:
    """Collect per_zone windows for one zone. Returns (windows, labels)."""
    windows:      list[np.ndarray] = []
    labels:       list[int]        = []
    zone_ref:     list[str]        = [zone]   # mutable ref for reader threads
    builder       = WindowBuilder()
    builder_lock  = threading.Lock()
    running       = [True]

    threads = [
        threading.Thread(target=_reader,
                         args=(left_port,  baud, 0, builder, builder_lock,
                               windows, labels, zone_ref, running),
                         daemon=True),
        threading.Thread(target=_reader,
                         args=(right_port, baud, 1, builder, builder_lock,
                               windows, labels, zone_ref, running),
                         daemon=True),
    ]
    for t in threads:
        t.start()

    deadline = time.time() + 600   # 10-min safety cap
    while len(windows) < per_zone and time.time() < deadline:
        time.sleep(0.2)
        n     = len(windows)
        filled = min(int(BAR_W * n / per_zone), BAR_W)
        print(f"\r  [{zone:<8}]  [{'█'*filled}{'░'*(BAR_W-filled)}]  "
              f"{n:>4}/{per_zone}  WALK BACK AND FORTH",
              end="", flush=True)

    running[0] = False
    for t in threads:
        t.join(timeout=3)

    print()
    return windows, labels


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect CNN training windows from ESP32 CSI serial output."
    )
    parser.add_argument("--per-zone",  type=int,   default=300,
                        help="Target windows per zone (default 300)")
    parser.add_argument("--out",       default="data/csi_windows.npz",
                        help="Output NPZ path (default data/csi_windows.npz)")
    parser.add_argument("--left",      default="COM10",  help="LEFT serial port")
    parser.add_argument("--right",     default="COM8",   help="RIGHT serial port")
    parser.add_argument("--baud",      type=int, default=115200)
    parser.add_argument("--append",    action="store_true",
                        help="Append to existing NPZ instead of overwriting")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  CSI CNN Data Collector")
    print("=" * 60)
    print(f"  Zones      : {' | '.join(ZONES)}")
    print(f"  Per zone   : {args.per_zone} windows  (window = {T} packets × {S} subcarriers × 2 nodes)")
    print(f"  LEFT port  : {args.left}")
    print(f"  RIGHT port : {args.right}")
    print(f"  Output     : {args.out}")
    print("=" * 60 + "\n")

    all_windows: list[np.ndarray] = []
    all_labels:  list[int]        = []

    for idx, zone in enumerate(ZONES):
        print(f"  Zone {idx+1}/{len(ZONES)}: {zone}")
        print(f"    → Walk to {zone.replace('_','-')} behind the wall.")
        print(f"    → Keep walking BACK AND FORTH for the full duration.")
        countdown("Recording starts", WARMUP_SECS)

        windows, labels = collect_zone(
            zone, args.per_zone, args.left, args.right, args.baud)
        all_windows.extend(windows)
        all_labels.extend(labels)
        print(f"  ✓ {len(windows)} windows for {zone}")

        if zone != ZONES[-1]:
            print(f"    Return to neutral — next zone in {BREAK_SECS}s.")
            for i in range(BREAK_SECS, 0, -1):
                print(f"\r    Next zone in {i}s ...  ", end="", flush=True)
                time.sleep(1)
            print()

    # ── Build dataset ─────────────────────────────────────────────────────────
    X = np.array(all_windows, dtype=np.float32)   # (N, T, S, 2)
    y = np.array(all_labels,  dtype=np.int64)

    # Load existing NPZ if appending
    if args.append and os.path.isfile(args.out):
        existing = np.load(args.out, allow_pickle=True)
        X = np.concatenate([existing["X"], X], axis=0)
        y = np.concatenate([existing["y"], y], axis=0)
        print(f"\n  Appended to existing NPZ (now {len(X)} total windows)")

    # Z-score normalise over the whole dataset
    mean = float(X.mean())
    std  = float(X.std()) + 1e-6
    X    = ((X - mean) / std).astype(np.float32)

    np.savez(args.out,
             X=X, y=y,
             label_names=np.array(LABEL_NAMES),
             mean=np.float32(mean),
             std=np.float32(std))

    print(f"\n  Saved {len(X)} windows → {args.out}")
    print(f"  X shape : {X.shape}  dtype={X.dtype}")
    print(f"  Norm    : mean={mean:.4f}  std={std:.4f}")
    for zone in ZONES:
        idx = LABEL_TO_IDX[zone]
        print(f"    {zone:<8} {int((y == idx).sum()):>5} windows")
    print("\n  Ready for:  python -m ml.csi.train --data", args.out)


if __name__ == "__main__":
    main()
