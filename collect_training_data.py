"""
collect_training_data.py — calibrate then collect truly raw CSI training data.

Saves one row per received packet per node — NO averaging, NO preprocessing.
Each row: zone, node, ts_ms, amp_0 .. amp_63

Processing (feature engineering, windowing, normalization) happens only in
aalto_model.py so the same raw data can be reused by any future model.

Usage:
    python collect_training_data.py              # calibrate + collect + train
    python collect_training_data.py --append     # skip calibration, add more data
    python collect_training_data.py --middle-host 10.75.241.42
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time

from rf.csi_reader import LEFT_PORT, RIGHT_PORT, MIDDLE_PORT

RAW_DATA_FILE = os.path.join(os.path.dirname(__file__), "rf_raw_csi_data.csv")
N_SC          = 64        # subcarrier columns per node (padded / truncated)
ZONES         = ["LEFT", "MIDDLE", "RIGHT"]
TARGET        = 1000       # packets per NODE per zone  (both nodes collected in parallel)
BREAK_SECS    = 10
WARMUP_SECS   = 5
BAR_W         = 30
RAW_FIELDS    = ["zone", "node", "ts_ms"] + [f"amp_{i}" for i in range(N_SC)]


# ── CSI line parser ───────────────────────────────────────────────────────────

def _parse_amplitudes(line: str) -> list[float] | None:
    """Parse a CSI: serial line into a list of per-subcarrier amplitudes."""
    if not line.startswith("CSI:"):
        return None
    parts = line[4:].split(",")
    if len(parts) < 5:
        return None
    try:
        csi_len = int(parts[3])
        raw     = [int(x) for x in parts[4: 4 + csi_len]]
        if len(raw) < csi_len:
            return None
        return [math.sqrt(raw[i] ** 2 + raw[i + 1] ** 2)
                for i in range(0, len(raw) - 1, 2)]
    except (ValueError, IndexError):
        return None


def _amp_row(zone: str, node: str, amps: list[float]) -> dict:
    """Build one CSV row from raw amplitude list (pad/truncate to N_SC)."""
    row = {"zone": zone, "node": node, "ts_ms": int(time.time() * 1000)}
    for i in range(N_SC):
        row[f"amp_{i}"] = round(float(amps[i]), 4) if i < len(amps) else 0.0
    return row


# ── Per-zone direct serial collection ────────────────────────────────────────

def _serial_reader(port: str, node: str, zone: str,
                   target: int, rows: list, lock: threading.Lock,
                   counts: dict, running: list) -> None:
    """Thread: read raw packets from one serial port until target reached."""
    import serial
    try:
        ser = serial.Serial(port, 115200, timeout=2)
    except Exception as exc:
        print(f"\n  [{node.upper()}] Cannot open {port}: {exc}")
        return

    while running[0]:
        try:
            raw = ser.readline()
        except Exception:
            break
        if not raw:
            continue
        try:
            line = raw.decode("ascii", errors="replace").strip()
        except Exception:
            continue

        amps = _parse_amplitudes(line)
        if amps is None:
            continue

        with lock:
            if counts[node] < target:
                rows.append(_amp_row(zone, node, amps))
                counts[node] += 1
            if counts[node] >= target:
                break

    try:
        ser.close()
    except Exception:
        pass


def collect_zone(zone: str, target: int,
                 left_port: str, right_port: str) -> list[dict]:
    """
    Collect `target` raw packets from EACH of LEFT and RIGHT simultaneously.
    Returns all rows (2 × target at most).
    """
    rows:    list[dict]   = []
    lock     = threading.Lock()
    counts   = {"left": 0, "right": 0}
    running  = [True]

    ports = [("left", left_port), ("right", right_port)]
    threads = []
    for node, port in ports:
        t = threading.Thread(
            target=_serial_reader,
            args=(port, node, zone, target, rows, lock, counts, running),
            daemon=True,
        )
        t.start()
        threads.append(t)

    deadline = time.time() + 360   # 6-minute safety cap
    while time.time() < deadline:
        time.sleep(0.15)
        with lock:
            lc = counts["left"]
            rc = counts["right"]
        done = min(lc, rc)
        filled = min(int(BAR_W * done / target), BAR_W)
        print(f"\r  [{zone:<8}]  [{'█'*filled}{'░'*(BAR_W-filled)}]  "
              f"L:{lc:>4}  R:{rc:>4} / {target}   WALK BACK AND FORTH",
              end="", flush=True)
        if lc >= target and rc >= target:
            break

    running[0] = False
    for t in threads:
        t.join(timeout=3)

    print()
    with lock:
        return list(rows)


# ── Countdown helper ──────────────────────────────────────────────────────────

def countdown(label: str, seconds: int) -> None:
    for i in range(seconds, 0, -1):
        print(f"\r  {label} in {i}s ...  ", end="", flush=True)
        time.sleep(1)
    print(f"\r  {'':40}", end="", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--append",      "-a", action="store_true")
    parser.add_argument("--middle-host", default=None, metavar="IP")
    parser.add_argument("--target",      "-t", type=int, default=TARGET,
                        help=f"Packets per node per zone (default {TARGET})")
    args = parser.parse_args()

    from rf.csi_reader import CsiReader

    print("\n" + "=" * 60)
    print("  CSI Raw Data Collection")
    print("=" * 60)
    print(f"  Zones        : {' | '.join(ZONES)}")
    print(f"  Per zone     : {args.target} packets per node  (~{args.target//6}s walking)")
    print(f"  Break        : {BREAK_SECS}s between zones")
    print(f"  Raw data     : {RAW_DATA_FILE}")
    print(f"  Note         : data saved as-is — no averaging, no ratios")
    print("=" * 60 + "\n")

    # ── Step 1: Calibrate (uses CsiReader, closes ports after) ───────────────
    if args.append:
        print("--append: skipping calibration.")
        rf_check = CsiReader(middle_host=args.middle_host)
        if not rf_check.load_calibration():
            print("No calibration found. Run without --append first.")
            sys.exit(1)
    else:
        print("STEP 1 / 2 — Calibration")
        print("  Keep sensing area EMPTY. Stand well clear.\n")
        rf_cal = CsiReader(middle_host=args.middle_host)
        rf_cal.calibrate()
        print()

    # ── Step 2: Collect raw packets directly from serial ─────────────────────
    print("STEP 2 / 2 — Data Collection")
    print("  Walk BACK AND FORTH in each zone for the full duration.\n")

    all_rows: list[dict] = []

    for idx, zone in enumerate(ZONES):
        print(f"  Zone {idx+1}/{len(ZONES)}: {zone}")
        print(f"    → Walk to {zone.replace('_','-')} behind the wall.")
        countdown("Recording starts", WARMUP_SECS)

        zone_rows = collect_zone(zone, args.target, LEFT_PORT, RIGHT_PORT)
        all_rows.extend(zone_rows)

        lc = sum(1 for r in zone_rows if r["node"] == "left")
        rc = sum(1 for r in zone_rows if r["node"] == "right")
        print(f"  ✓ {zone}: {lc} LEFT packets, {rc} RIGHT packets")

        if zone != ZONES[-1]:
            print(f"    Return to neutral — next zone in {BREAK_SECS}s.")
            for i in range(BREAK_SECS, 0, -1):
                print(f"\r    Next zone in {i}s ...  ", end="", flush=True)
                time.sleep(1)
            print()

    # ── Write CSV ─────────────────────────────────────────────────────────────
    mode = "a" if args.append and os.path.isfile(RAW_DATA_FILE) else "w"
    with open(RAW_DATA_FILE, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        if mode == "w":
            writer.writeheader()
        writer.writerows(all_rows)

    total = len(all_rows)
    print(f"\n  {total} raw packets saved → {RAW_DATA_FILE}")
    for zone in ZONES:
        for node in ("left", "right"):
            n = sum(1 for r in all_rows if r["zone"] == zone and r["node"] == node)
            print(f"    {zone:<8} {node:<6} {n:>5} packets")

    # ── Also generate sliding-window NPZ for CNN training ─────────────────────
    npz_path = os.path.join(os.path.dirname(__file__), "data", "csi_windows.npz")
    print(f"\n  Building CNN windows from raw packets → {npz_path}")
    try:
        from ml.csi.windows import csv_to_npz
        n_windows = csv_to_npz(RAW_DATA_FILE, npz_path)
        if n_windows > 0:
            print(f"  ✓ {n_windows} windows saved.")
        else:
            print("  ⚠ No windows generated — check packet count per zone.")
    except Exception as exc:
        print(f"  ⚠ NPZ generation failed: {exc}")
        print("    The raw CSV is still intact — run: python -m ml.csi.collect separately.")

    print("\nOutputs:")
    print(f"  {RAW_DATA_FILE}   (raw CSV for aalto_model.py)")
    print(f"  {npz_path}  (CNN windows for ml/csi/train.py)")
    print("\nNext:")
    print("  python aalto_model.py         ← fast RF-based model")
    print("  python -m ml.csi.train ...    ← CNN on cluster")


if __name__ == "__main__":
    main()
