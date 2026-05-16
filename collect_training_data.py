"""
collect_training_data.py — calibrate, collect labelled CSI data, and train.

One script does everything in order:
  1. Calibrate baseline (sensing area EMPTY)
  2. Collect labelled samples per zone (YOU WALK BACK AND FORTH in each zone)
  3. Train the RandomForest model and save it

Usage:
    python collect_training_data.py
    python collect_training_data.py --seconds 25   # longer per zone (default 20)
    python collect_training_data.py --middle-host 10.75.241.42
    python collect_training_data.py --append       # skip calibration, add more data

IMPORTANT during collection:
  Walk back and forth in the zone the whole time.
  Do NOT stand still. Varied movement = better model.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

DATA_FILE  = os.path.join(os.path.dirname(__file__), "rf_training_data.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "rf_zone_model.pkl")

ZONES          = ["LEFT", "MIDDLE", "RIGHT"]
TARGET_SAMPLES = 200   # samples to collect per zone
MAX_SECS       = 360    # safety timeout per zone (6 min) in case sensor is slow
BREAK_SECS     = 10     # break between zones — walk back to neutral position
WARMUP_SECS    = 5      # countdown before each zone starts recording
BAR_W          = 30


# ── Helpers ───────────────────────────────────────────────────────────────────

def countdown(label: str, seconds: int) -> None:
    for i in range(seconds, 0, -1):
        print(f"\r  {label} in {i}s ...  ", end="", flush=True)
        time.sleep(1)
    print(f"\r  {' '*40}", end="", flush=True)


def collect_zone(rf, zone: str, target: int) -> list[dict]:
    """Collect exactly `target` samples for one zone (or until MAX_SECS timeout)."""
    samples: list[dict] = []
    deadline = time.time() + MAX_SECS
    prev_pkts = {n: rf._nodes[n]["pkts"] for n in ("left", "middle", "right")}

    while len(samples) < target and time.time() < deadline:
        time.sleep(0.10)
        with rf._lock:
            ratios = {n: rf._nodes[n]["ratio"] for n in ("left", "middle", "right")}
            pkts   = {n: rf._nodes[n]["pkts"]  for n in ("left", "middle", "right")}

        new = sum(pkts[n] - prev_pkts[n] for n in pkts)
        if new > 0:
            samples.append({
                "zone":         zone,
                "left_ratio":   round(ratios["left"],   4),
                "middle_ratio": round(ratios["middle"], 4),
                "right_ratio":  round(ratios["right"],  4),
                "timestamp":    time.strftime("%H:%M:%S"),
            })
            prev_pkts = pkts

        filled = min(int(BAR_W * len(samples) / target), BAR_W)
        print(f"\r  [{zone:<8}]  [{'█'*filled}{'░'*(BAR_W-filled)}]  "
              f"{len(samples):>4}/{target}   WALK BACK AND FORTH",
              end="", flush=True)

    print()
    return samples


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_file: str) -> None:
    """Train RandomForest on collected data and save model."""
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import classification_report
    except ImportError:
        print("\nERROR: scikit-learn not installed.")
        print("Run:  pip install scikit-learn  then re-run this script.")
        return

    rows = []
    with open(data_file, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if len(rows) < 30:
        print(f"\nOnly {len(rows)} samples — collect more data first.")
        return

    X = np.array([[float(r["left_ratio"]),
                   float(r["middle_ratio"]),
                   float(r["right_ratio"])] for r in rows])
    y = np.array([r["zone"] for r in rows])

    present_zones = sorted(set(y), key=lambda z: ZONES.index(z) if z in ZONES else 99)

    print(f"\n  Samples per zone:")
    for zone in present_zones:
        n   = int((y == zone).sum())
        bar = "█" * (n // 3)
        print(f"    {zone:<14} {n:>4}  {bar}")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    n_splits = min(5, min(int((y == z).sum()) for z in present_zones))
    n_splits = max(n_splits, 2)
    cv      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores  = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\n  Cross-validation accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    model.fit(X, y)
    print(f"  Training accuracy (in-sample): {(model.predict(X) == y).mean():.1%}")

    names = ["left_ratio", "middle_ratio", "right_ratio"]
    print(f"\n  Feature importances:")
    for name, imp in sorted(zip(names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"    {name:<14} {imp:.3f}  {'█' * int(imp * 30)}")

    joblib.dump(model, MODEL_FILE)
    print(f"\n  Model saved → {MODEL_FILE}")
    print("  Run 'python main.py' — it loads the model automatically.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", "-s", type=int, default=TARGET_SAMPLES,
                        help=f"Samples to collect per zone (default {TARGET_SAMPLES})")
    parser.add_argument("--append",  "-a", action="store_true",
                        help="Skip calibration, append to existing training data")
    parser.add_argument("--middle-host", default=None, metavar="IP")
    args = parser.parse_args()

    from rf.csi_reader import CsiReader

    print("\n" + "=" * 56)
    print("  CSI Data Collection + Training Pipeline")
    print("=" * 56)
    print(f"  Zones       : {' | '.join(ZONES)}")
    print(f"  Per zone    : {TARGET_SAMPLES} samples  (~{TARGET_SAMPLES//6} seconds of walking)")
    print(f"  Break       : {BREAK_SECS}s between zones to reposition")
    print(f"  Output      : {DATA_FILE}")
    print("=" * 56)
    print()

    rf = CsiReader(middle_host=args.middle_host)

    # ── Step 1: Calibrate ─────────────────────────────────────────────────────
    if args.append:
        print("--append mode: skipping calibration, loading existing baseline.")
        if not rf.load_calibration():
            print("No calibration found — cannot append. Run without --append first.")
            sys.exit(1)
    else:
        print("STEP 1 / 3 — Calibration")
        print("  Keep the ENTIRE sensing area EMPTY.")
        print("  Stand well clear of the wall and the ESPs.\n")
        rf.calibrate()
        rf.load_calibration()

    rf.start()
    print("\nWarming up sensors (3s)...")
    time.sleep(3)

    # ── Step 2: Collect per zone ──────────────────────────────────────────────
    print("\nSTEP 2 / 3 — Data Collection")
    print("  For each zone you will have a countdown then need to")
    print("  WALK BACK AND FORTH in that zone for the full duration.")
    print("  More movement = better model.\n")

    all_samples: list[dict] = []

    for i, zone in enumerate(ZONES):
        print(f"  Zone {i+1}/{len(ZONES)}: {zone}")
        if zone == "NO_MOTION":
            print("    Stay well away from the sensing area (behind wall should be empty).")
        else:
            zone_label = zone.replace("_", "-")
            print(f"    Walk to the {zone_label} area behind the wall.")
            print(f"    Keep walking back and forth the ENTIRE time.")
        countdown("Recording starts", WARMUP_SECS)
        samples = collect_zone(rf, zone, TARGET_SAMPLES)
        all_samples.extend(samples)
        print(f"  ✓ {len(samples)} samples collected for {zone}.")
        if zone != ZONES[-1]:
            print(f"    Walk back to neutral position — next zone in {BREAK_SECS}s.")
            for i in range(BREAK_SECS, 0, -1):
                print(f"\r    Next zone in {i}s ...  ", end="", flush=True)
                time.sleep(1)
            print()

    rf.stop()

    # ── Write CSV ──────────────────────────────────────────────────────────────
    mode = "a" if args.append and os.path.isfile(DATA_FILE) else "w"
    write_header = (mode == "w")
    with open(DATA_FILE, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "zone", "left_ratio", "middle_ratio", "right_ratio", "timestamp"])
        if write_header:
            writer.writeheader()
        writer.writerows(all_samples)

    total = len(all_samples)
    print(f"\n  {total} new samples saved to {DATA_FILE}")

    # Count all rows in file (including previous if appended)
    with open(DATA_FILE, newline="") as f:
        all_rows = sum(1 for _ in csv.DictReader(f))
    print(f"  Total in file: {all_rows} samples")

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("\nSTEP 3 / 3 — Training Model")
    train(DATA_FILE)

    print("\nDone. Run:  python main.py")


if __name__ == "__main__":
    main()
