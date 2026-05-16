#!/usr/bin/env python3
"""
Collect labelled CSI windows for 1D-CNN training (LEFT + RIGHT ESP32s).

Usage:
  python -m ml.csi.collect --per-zone 300 --out data/csi_windows.npz
  python -m ml.csi.collect --left COM10 --right COM8
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from ml.csi.windows import LABEL_NAMES, LABEL_TO_IDX, S, T, WindowBuilder, parse_amplitudes

try:
    import serial
except ImportError:
    print("pyserial not installed.  Run:  pip install pyserial")
    sys.exit(1)

ZONES = ["LEFT", "MIDDLE", "RIGHT"]
BREAK_SECS = 10
WARMUP_SECS = 5
BAR_W = 30


def _reader(
    port: str,
    baud: int,
    node_idx: int,
    builder: WindowBuilder,
    builder_lock: threading.Lock,
    windows: list,
    labels: list,
    zone_ref: list,
    running: list,
) -> None:
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
                windows.append(builder.get_window())
                labels.append(LABEL_TO_IDX[zone_ref[0]])

    try:
        ser.close()
    except Exception:
        pass


def countdown(label: str, seconds: int) -> None:
    for i in range(seconds, 0, -1):
        print(f"\r  {label} in {i}s ...  ", end="", flush=True)
        time.sleep(1)
    print(f"\r  {'':40}", end="", flush=True)


def collect_zone(
    zone: str, per_zone: int, left_port: str, right_port: str, baud: int
) -> tuple[list, list]:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    zone_ref: list[str] = [zone]
    builder = WindowBuilder()
    builder_lock = threading.Lock()
    running = [True]

    threads = [
        threading.Thread(
            target=_reader,
            args=(left_port, baud, 0, builder, builder_lock, windows, labels, zone_ref, running),
            daemon=True,
        ),
        threading.Thread(
            target=_reader,
            args=(right_port, baud, 1, builder, builder_lock, windows, labels, zone_ref, running),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    deadline = time.time() + 600
    while len(windows) < per_zone and time.time() < deadline:
        time.sleep(0.2)
        n = len(windows)
        filled = min(int(BAR_W * n / per_zone), BAR_W)
        print(
            f"\r  [{zone:<8}]  [{'█' * filled}{'░' * (BAR_W - filled)}]  "
            f"{n:>4}/{per_zone}  WALK BACK AND FORTH",
            end="",
            flush=True,
        )

    running[0] = False
    for t in threads:
        t.join(timeout=3)
    print()
    return windows, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CNN CSI windows (2 ESP32s).")
    parser.add_argument("--per-zone", type=int, default=300)
    parser.add_argument("--out", default="data/csi_windows.npz")
    parser.add_argument("--left", default="COM10")
    parser.add_argument("--right", default="COM8")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print("\n=== CSI CNN collector ===")
    print(f"Window: ({T}, {S}, 2)  |  ports: {args.left}, {args.right}\n")

    all_windows: list[np.ndarray] = []
    all_labels: list[int] = []

    for idx, zone in enumerate(ZONES):
        print(f"Zone {idx + 1}/{len(ZONES)}: {zone} — walk in that zone behind the wall.")
        countdown("Recording starts", WARMUP_SECS)
        windows, labels = collect_zone(zone, args.per_zone, args.left, args.right, args.baud)
        all_windows.extend(windows)
        all_labels.extend(labels)
        print(f"  ✓ {len(windows)} windows for {zone}")
        if zone != ZONES[-1]:
            for i in range(BREAK_SECS, 0, -1):
                print(f"\r  Break — next zone in {i}s ...  ", end="", flush=True)
                time.sleep(1)
            print()

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    if args.append and os.path.isfile(args.out):
        existing = np.load(args.out, allow_pickle=True)
        X = np.concatenate([existing["X"], X], axis=0)
        y = np.concatenate([existing["y"], y], axis=0)

    mean = float(X.mean())
    std = float(X.std()) + 1e-6
    X = ((X - mean) / std).astype(np.float32)

    np.savez(
        args.out,
        X=X,
        y=y,
        label_names=np.array(LABEL_NAMES),
        mean=np.float32(mean),
        std=np.float32(std),
    )
    print(f"\nSaved {len(X)} windows → {args.out}")
    print("Train:  python -m ml.csi.train --data", args.out)


if __name__ == "__main__":
    main()
