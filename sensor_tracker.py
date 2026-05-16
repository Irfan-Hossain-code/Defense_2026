"""
sensor_tracker.py — live terminal monitor for the 3-node CSI zone inference.

Run this in a second terminal while main.py is running to see the raw
sensor scores, zone decisions, and a consistency metric so you can judge
how reliable the inference actually is.

Usage:
    python sensor_tracker.py
    python sensor_tracker.py --middle-host 10.75.241.42
    python sensor_tracker.py --sensitivity 2.0
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

from rf.csi_reader import CsiReader, LEFT_PORT, RIGHT_PORT, MIDDLE_PORT, CAL_FILE

REFRESH_HZ   = 2          # display updates per second
HISTORY_LEN  = 20         # zones kept for consistency calculation
BAR_W        = 28
SPARK        = " ▁▂▃▄▅▆▇█"


# ── Display helpers ───────────────────────────────────────────────────────────
def bar(ratio: float, max_r: float, width: int) -> str:
    filled = min(int(width * ratio / max(max_r, 0.01)), width)
    return "█" * filled + "░" * (width - filled)

def spark(history: list[float], max_v: float) -> str:
    return "".join(SPARK[min(int(8 * v / max(max_v, 0.01)), 8)] for v in history)

def consistency(history: list[str]) -> tuple[float, str]:
    """Return (fraction, dominant_zone) of the most common zone in history."""
    if not history:
        return 0.0, "—"
    counts: dict[str, int] = {}
    for z in history:
        counts[z] = counts.get(z, 0) + 1
    top_zone  = max(counts, key=counts.get)
    top_frac  = counts[top_zone] / len(history)
    return top_frac, top_zone

def conf_bar(frac: float, width: int = 20) -> str:
    filled = int(width * frac)
    return "█" * filled + "░" * (width - filled)

def clear_lines(n: int) -> None:
    print(f"\033[{n}A", end="")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="CSI sensor live monitor")
    parser.add_argument("--middle-host", default=None, metavar="IP")
    parser.add_argument("--sensitivity", "-s", type=float, default=2.5)
    args = parser.parse_args()

    print("\n=== CSI Sensor Tracker ===\n")

    if not os.path.isfile(CAL_FILE):
        print(f"No calibration file found ({CAL_FILE}).")
        print("Run:  python main.py --cal  first.\n")
        sys.exit(1)

    rf = CsiReader(middle_host=args.middle_host)
    rf.load_calibration()
    rf.start()

    print(f"Nodes:  LEFT={LEFT_PORT}  MIDDLE={'bridge@'+args.middle_host if args.middle_host else MIDDLE_PORT}  RIGHT={RIGHT_PORT}")
    print(f"Sensitivity threshold: {args.sensitivity}  (same as rf/csi_reader.py default)\n")
    print("Press Ctrl+C to stop.\n")

    zone_history: collections.deque[str] = collections.deque(maxlen=HISTORY_LEN)
    ratio_hist: dict[str, collections.deque] = {
        n: collections.deque(maxlen=BAR_W) for n in ("left", "middle", "right")
    }
    pkt_counts_prev = {"left": 0, "middle": 0, "right": 0}
    pkt_rates       = {"left": 0.0, "middle": 0.0, "right": 0.0}
    last_rate_t     = time.time()
    first_print     = True
    N_LINES         = 17   # number of lines printed per refresh

    try:
        while True:
            time.sleep(1.0 / REFRESH_HZ)

            # ── Read state under rf's lock ─────────────────────────────────
            with rf._lock:
                nodes    = rf._nodes
                zone     = rf._zone
                conf     = rf._confidence
                inferred = rf._inferred
                ratios   = {n: nodes[n]["ratio"]   for n in ("left","middle","right")}
                rssis    = {n: nodes[n]["rssi"]    for n in ("left","middle","right")}
                conns    = {n: nodes[n]["connected"] for n in ("left","middle","right")}
                cals     = {n: nodes[n]["calibrated"] for n in ("left","middle","right")}
                pkts     = {n: nodes[n]["pkts"]    for n in ("left","middle","right")}

            # ── Packet rates ───────────────────────────────────────────────
            now = time.time()
            dt  = now - last_rate_t
            if dt >= 1.0:
                for n in ("left", "middle", "right"):
                    pkt_rates[n] = (pkts[n] - pkt_counts_prev[n]) / dt
                    pkt_counts_prev[n] = pkts[n]
                last_rate_t = now

            # ── History ────────────────────────────────────────────────────
            for n in ("left", "middle", "right"):
                ratio_hist[n].append(ratios[n])
            if zone != "NO_MOTION":
                zone_history.append(zone)

            consist_frac, consist_zone = consistency(list(zone_history))
            max_ratio = max(max(ratios.values()), args.sensitivity * 1.5, 1.0)

            # ── Build display lines ────────────────────────────────────────
            def node_line(name: str) -> str:
                conn = "●" if conns[name] else "○"
                cal  = "CAL✓" if cals[name] else "cal?"
                b    = bar(ratios[name], max_ratio, BAR_W)
                sp   = spark(list(ratio_hist[name]), max_ratio)
                r    = ratios[name]
                rate_s = f"{pkt_rates[name]:4.1f}/s"
                rssi_s = f"{rssis[name]:4}dBm" if conns[name] else "  n/a  "
                return (f"  {conn} {name.upper():<7}"
                        f"[{b}] {r:5.2f}x  "
                        f"{rssi_s}  {rate_s}  {cal}  {sp}")

            mid_note = "(inferred from L+R)" if inferred else ""
            zone_color_char = "●" if zone not in ("NO_MOTION","UNKNOWN") else "○"

            lines = [
                "─" * 74,
                node_line("left"),
                node_line("middle") + f"  {mid_note}",
                node_line("right"),
                "─" * 74,
                f"  Zone now :  {zone_color_char} {zone:<14}  conf={conf:.2f}",
                f"  Dominant :    {consist_zone:<14}  [{conf_bar(consist_frac)}]  {consist_frac*100:.0f}% of last {len(zone_history)}",
                "─" * 74,
                f"  Zone log (newest →):  {' '.join(list(zone_history)[-10:]) or '(calibrating)'}",
                "─" * 74,
                "  Interpretation:",
                f"    score > {args.sensitivity:.1f}×  →  motion detected",
                f"    score ≈ 1.0   →  stable (person still / no one there)",
                f"    symmetry      →  LEFT≈RIGHT score means MIDDLE zone",
                "─" * 74,
                f"  Ctrl+C to stop                  {time.strftime('%H:%M:%S')}",
                "",   # trailing blank so cursor-up lands cleanly
            ]

            # ── Overwrite previous block ───────────────────────────────────
            if not first_print:
                clear_lines(N_LINES)
            first_print = False

            for line in lines:
                print(f"\033[K{line}")

    except KeyboardInterrupt:
        pass

    rf.stop()
    print("\nSensor tracker stopped.")


if __name__ == "__main__":
    main()
