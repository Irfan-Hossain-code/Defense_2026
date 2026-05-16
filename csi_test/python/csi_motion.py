"""
csi_motion.py — through-wall presence and motion detection via ESP32 CSI.

=== HARDWARE SETUP ===

Best arrangement for through-wall detection:

    [Phone hotspot]  ──Wi-Fi──  [ESP32]  ──wall──  [sensing area]

The ESP32 and phone form a Wi-Fi link.  A person in the sensing area
alters the reflections on that link, even through a wall.

Keep the phone stationary.  Any movement of the phone = false signal.

=== HOW TO USE ===

1. Place the ESP32 near the wall, phone 1-2 m away on the same side.
2. Make sure the sensing area on the OTHER side is EMPTY.
3. Run this script — it will calibrate automatically.
4. Walk into the sensing area through the wall.

=== ALGORITHM ===

Single-packet comparison (old) is too noisy.
This script uses TEMPORAL VARIANCE across a rolling window:

  For each sub-carrier, compute the variance of its amplitude
  over the last WINDOW_SIZE packets.  Average those variances.

  Empty room  → stable channel → low temporal variance
  Person moving → reflections shift per packet → high variance
  Person still but breathing → slow oscillation → medium variance

The score is expressed as a RATIO vs the locked empty-room baseline.
  ratio 1.0 = same variation as empty room
  ratio 2.5+ = something is significantly disrupting the channel

=== TUNING ===
--sensitivity  Default 2.5.  Lower = more sensitive.
               Start at 2.5 for walking detection.
               Try 1.8 for slow/small movements.
               Try 3.5+ if you get too many false positives from other sources.

--window       Rolling window size in packets (default 8).
               Larger = smoother but slower to react.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import os
import sys
import time

import serial
import serial.tools.list_ports

sys.path.insert(0, os.path.dirname(__file__))
import csi_parser

# ── Constants ─────────────────────────────────────────────────────────────────
CAL_PACKETS       = 60     # packets collected in empty room for baseline
DEFAULT_WINDOW    = 8      # rolling window size for temporal variance
DEFAULT_SENSITIVITY = 2.5  # variance ratio that triggers PRESENCE
DISPLAY_HISTORY   = 40     # sparkline history length
BAR_WIDTH         = 32     # ASCII bar width

# How many consecutive windows above threshold before we call it PRESENCE
# (reduces single-packet false positives)
CONFIRM_COUNT = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_port() -> str | None:
    for p in serial.tools.list_ports.comports():
        d = (p.description or "").lower()
        h = (p.hwid or "").lower()
        if any(x in d or x in h for x in ["cp210", "ch340", "ftdi", "esp32"]):
            return p.device
    return None


def temporal_variance(window: list[list[float]], n_sc: int) -> float:
    """
    For each of n_sc sub-carriers, compute the variance of its amplitude
    across all packets in the window, then return the mean variance.

    High score → amplitudes are changing between packets → motion/presence.
    Low score  → amplitudes are stable → empty room.
    """
    n = len(window)
    if n < 2:
        return 0.0
    total = 0.0
    counted = 0
    for i in range(n_sc):
        vals = [window[j][i] for j in range(n) if i < len(window[j])]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        var  = sum((v - mean) ** 2 for v in vals) / len(vals)
        total   += var
        counted += 1
    return total / counted if counted else 0.0


def ascii_bar(ratio: float, max_ratio: float, width: int) -> str:
    filled = min(int(width * ratio / max_ratio), width)
    return "█" * filled + "░" * (width - filled)


SPARK = " ▁▂▃▄▅▆▇█"

def sparkline(history: list[float], max_val: float) -> str:
    out = []
    for v in history:
        idx = min(int(8 * v / max(max_val, 1e-6)), 8)
        out.append(SPARK[idx])
    return "".join(out)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(port: str, baud: int, out_dir: str,
        sensitivity: float, window_size: int) -> None:

    stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"csi_presence_{stamp}.csv")
    os.makedirs(out_dir, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║       ESP32 Through-Wall Presence Detector       ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Port      : {port:<36}║")
    print(f"║  Sensitivity: {sensitivity:<35}║")
    print(f"║  Window    : {window_size} packets{' ':<28}║")
    print(f"║  CSV       : {os.path.basename(out_path):<36}║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("  SETUP REMINDER:")
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  [Phone] ──Wi-Fi── [ESP32] ──wall── [you here]  │")
    print("  │                                                  │")
    print("  │  Keep the sensing area EMPTY during calibration. │")
    print("  │  Keep the phone STATIONARY at all times.         │")
    print("  └─────────────────────────────────────────────────┘")
    print()

    try:
        ser = serial.Serial(port, baud, timeout=2)
    except serial.SerialException as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # State
    cal_window:   list[list[float]] = []   # raw packets during calibration
    baseline_var: float = 0.0              # locked empty-room variance
    n_sc:         int   = 0               # number of sub-carriers in baseline
    calibrated:   bool  = False

    window: collections.deque = collections.deque(maxlen=window_size)
    ratio_history: collections.deque = collections.deque(maxlen=DISPLAY_HISTORY)

    total_pkts    = 0
    confirm_above = 0          # consecutive windows above threshold
    state         = "EMPTY"    # EMPTY | PRESENCE
    state_since   = time.time()

    rate_t0  = time.time()
    rate_pkts = 0
    rate      = 0.0

    print(f"  Calibrating — sensing area must be EMPTY.  "
          f"Collecting {CAL_PACKETS} packets...\n")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "rssi", "noise", "channel",
                         "temporal_var", "ratio", "state"])

        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                try:
                    line = raw.decode("ascii", errors="replace").strip()
                except Exception:
                    continue

                if line.startswith(("INFO:", "STATUS:", "ERROR:")):
                    if "ERROR" in line or "Connected" in line:
                        print(f"\r  [ESP32] {line}")
                    continue

                pkt = csi_parser.parse_line(line)
                if pkt is None or not pkt.amplitudes:
                    continue

                total_pkts += 1
                rate_pkts  += 1
                now = time.time()
                if now - rate_t0 >= 1.0:
                    rate   = rate_pkts / (now - rate_t0)
                    rate_t0 = now
                    rate_pkts = 0

                amps = pkt.amplitudes

                # ── Calibration ───────────────────────────────────────────────
                if not calibrated:
                    cal_window.append(amps)
                    prog = int(BAR_WIDTH * len(cal_window) / CAL_PACKETS)
                    print(f"\r  Calibrating  [{'█'*prog}{'░'*(BAR_WIDTH-prog)}]"
                          f"  {len(cal_window)}/{CAL_PACKETS}",
                          end="", flush=True)

                    if len(cal_window) >= CAL_PACKETS:
                        n_sc         = min(len(v) for v in cal_window)
                        baseline_var = temporal_variance(cal_window, n_sc)
                        # Guard: if baseline_var is near zero (perfect lab
                        # conditions), set a small floor so division works.
                        baseline_var = max(baseline_var, 0.5)
                        calibrated   = True
                        print(f"\r  Calibration done."
                              f"  Baseline variance={baseline_var:.2f}"
                              f"  ({n_sc} sub-carriers){'':20}")
                        print()
                        print(f"  {'Pkts':>6}  {'Rate':>6}  {'RSSI':>6}  "
                              f"{'Ratio':>6}  {'State':<14}  History (last {DISPLAY_HISTORY} windows)")
                        print("  " + "─" * 80)
                    continue

                # ── Detection ─────────────────────────────────────────────────
                window.append(amps[:n_sc])

                if len(window) < 2:
                    continue

                t_var = temporal_variance(list(window), n_sc)
                ratio = t_var / baseline_var
                ratio_history.append(ratio)

                # Hysteresis state machine
                if ratio >= sensitivity:
                    confirm_above += 1
                else:
                    confirm_above = max(0, confirm_above - 1)

                new_state = "PRESENCE" if confirm_above >= CONFIRM_COUNT else "EMPTY"
                if new_state != state:
                    state       = new_state
                    state_since = now

                elapsed = int(now - state_since)

                # Display
                max_ratio = max(sensitivity * 2, max(ratio_history) if ratio_history else 1)
                bar  = ascii_bar(ratio, max_ratio, BAR_WIDTH)
                spark = sparkline(list(ratio_history), max_ratio)

                if state == "PRESENCE":
                    label = f"PRESENCE {elapsed:>3}s"
                else:
                    label = f"empty    {elapsed:>3}s"

                print(f"\r  {total_pkts:>6}  {rate:>5.1f}/s  "
                      f"{pkt.rssi:>4}dBm  "
                      f"{ratio:>6.2f}  "
                      f"{label:<14}  {spark}",
                      end="", flush=True)

                writer.writerow([
                    datetime.datetime.now().isoformat(timespec="milliseconds"),
                    pkt.rssi, pkt.noise_floor, pkt.channel,
                    round(t_var, 4), round(ratio, 4), state,
                ])
                f.flush()

        except KeyboardInterrupt:
            pass

    print(f"\n\nStopped. {total_pkts} packets saved to {out_path}")
    ser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Through-wall presence detection via ESP32 CSI temporal variance."
    )
    parser.add_argument("--port",        "-p", default=None)
    parser.add_argument("--baud",        "-b", type=int, default=115200)
    parser.add_argument("--out",         "-o",
                        default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--sensitivity", "-s", type=float, default=DEFAULT_SENSITIVITY,
                        help="Variance ratio threshold (default 2.5). Lower = more sensitive.")
    parser.add_argument("--window",      "-w", type=int, default=DEFAULT_WINDOW,
                        help=f"Rolling window in packets (default {DEFAULT_WINDOW}).")
    args = parser.parse_args()

    port = args.port or find_port()
    if not port:
        print("No ESP32 port found.  Use --port COM9")
        sys.exit(1)

    run(port, args.baud, args.out, args.sensitivity, args.window)


if __name__ == "__main__":
    main()
