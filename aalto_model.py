"""
aalto_model.py — zone classifier trained on raw per-subcarrier CSI amplitudes.

Uses rf_raw_csi_data.csv which is produced by collect_training_data.py.
Each sample contains the mean amplitude of all 64 subcarriers from both
the LEFT and RIGHT sensors, giving the model 128+ values to work with
instead of the 3 aggregated ratios used by the standard model.

From those 128 raw values we derive ~20 discriminative features capturing:
  - Per-node signal shape (mean, spread, balance across subcarriers)
  - Cross-node asymmetry (which sensor sees more disturbance and by how much)
  - Spectral features (low vs high subcarrier response difference)

Model: GradientBoosting with calibrated probabilities.

Usage:
    python aalto_model.py
    python main.py --model aalto
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

RAW_DATA_FILE  = os.path.join(os.path.dirname(__file__), "rf_raw_csi_data.csv")
MODEL_FILE     = os.path.join(os.path.dirname(__file__), "rf_zone_model_aalto.pkl")
ZONE_ORDER     = ["LEFT", "MIDDLE", "RIGHT"]
N_SC           = 64
WINDOW_MS      = 300   # milliseconds — pair LEFT+RIGHT packets within this window


# ── Feature extraction (shared with inference in csi_reader.py) ──────────────

def extract_features(left_amps: list[float], right_amps: list[float]) -> list[float]:
    """
    Compute ~22 discriminative features from two N_SC-length amplitude vectors.

    Raw subcarrier amplitudes vary with noise.  The derived features capture
    the RELATIONSHIP between sensors and the SHAPE of each sensor's response —
    both of which are more stable across different walking speeds.
    """
    import numpy as np

    L = np.array(left_amps,  dtype=float)
    R = np.array(right_amps, dtype=float)
    eps = 1e-6

    def node_stats(v: np.ndarray) -> list[float]:
        """8 statistics describing one sensor's amplitude distribution."""
        return [
            float(v.mean()),                          # overall mean
            float(v.std()),                           # spread across subcarriers
            float(np.percentile(v, 25)),              # lower quartile
            float(np.percentile(v, 75)),              # upper quartile
            float(v.max() - v.min()),                 # dynamic range
            float(v[:N_SC//2].mean()),                # low-subcarrier mean
            float(v[N_SC//2:].mean()),                # high-subcarrier mean
            float(v[:N_SC//2].mean() - v[N_SC//2:].mean()),  # spectral tilt
        ]

    l_stats = node_stats(L)
    r_stats = node_stats(R)

    # Cross-node features — these are the most important for L/R discrimination
    l_mean   = float(L.mean()) + eps
    r_mean   = float(R.mean()) + eps
    total    = l_mean + r_mean
    lr_diff  = l_mean - r_mean
    lr_norm  = lr_diff / total                    # -1 (right) to +1 (left)
    lr_ratio = l_mean / r_mean

    # Per-subcarrier cross features
    diff_per_sc = L - R                           # which subcarriers lean left
    cross = [
        lr_diff,
        lr_norm,
        lr_ratio,
        float(diff_per_sc.mean()),                # mean per-subcarrier difference
        float(diff_per_sc.std()),                 # consistency of difference
        float((diff_per_sc > 0).mean()),          # fraction of SCs where left > right
    ]

    return l_stats + r_stats + cross              # 8 + 8 + 6 = 22 features


FEATURE_NAMES = (
    [f"L_{n}" for n in ["mean","std","p25","p75","range","lo","hi","tilt"]] +
    [f"R_{n}" for n in ["mean","std","p25","p75","range","lo","hi","tilt"]] +
    ["lr_diff","lr_norm","lr_ratio","diff_mean","diff_std","frac_l_wins"]
)


# ── Main ─────────────────────────────────────────────────────────────────────

def _load_and_pair(data_file: str) -> tuple:
    """
    Load per-packet raw CSV and pair LEFT+RIGHT packets within WINDOW_MS.

    Each training sample = one LEFT packet + nearest RIGHT packet (same zone).
    Returns (X, y) where X is the feature matrix and y is zone labels.
    """
    import numpy as np

    rows = []
    with open(data_file, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Separate by node
    left_pkts  = [r for r in rows if r["node"] == "left"]
    right_pkts = [r for r in rows if r["node"] == "right"]

    def amps(row: dict) -> list[float]:
        return [float(row.get(f"amp_{i}", 0.0)) for i in range(N_SC)]

    # Sort by timestamp
    left_pkts.sort(key=lambda r: int(r["ts_ms"]))
    right_pkts.sort(key=lambda r: int(r["ts_ms"]))

    X_list, y_list = [], []
    ri = 0   # pointer into right_pkts

    for lp in left_pkts:
        lt = int(lp["ts_ms"])
        zone = lp["zone"]

        # Advance right pointer to closest packet in time
        while ri + 1 < len(right_pkts) and \
              abs(int(right_pkts[ri+1]["ts_ms"]) - lt) < abs(int(right_pkts[ri]["ts_ms"]) - lt):
            ri += 1

        if ri >= len(right_pkts):
            continue

        rp = right_pkts[ri]
        rt = int(rp["ts_ms"])

        # Only pair if zones match and timestamps are close
        if rp["zone"] != zone:
            continue
        if abs(lt - rt) > WINDOW_MS:
            continue

        feat = extract_features(amps(lp), amps(rp))
        X_list.append(feat)
        y_list.append(zone)

    return np.array(X_list), np.array(y_list)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=RAW_DATA_FILE)
    args = parser.parse_args()

    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import confusion_matrix, classification_report
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Run:  pip install scikit-learn")
        sys.exit(1)

    if not os.path.isfile(args.data):
        print(f"Raw data not found: {args.data}")
        print("Run:  python collect_training_data.py  first.")
        sys.exit(1)

    # ── Load and pair LEFT+RIGHT packets ─────────────────────────────────────
    print(f"\n=== Aalto Model (raw CSI + GradientBoosting) ===\n")
    print(f"Loading {args.data} ...")
    X, y = _load_and_pair(args.data)

    if len(X) < 30:
        print(f"Only {len(X)} paired samples after alignment — collect more data.")
        sys.exit(1)

    print(f"Paired samples: {len(X)}")
    for zone in ZONE_ORDER:
        n = int((y == zone).sum())
        print(f"  {zone:<8} {n:>5}  {'█' * max(1, n // 8)}")

    print(f"\nFeatures : {X.shape[1]}  ({', '.join(FEATURE_NAMES[:4])}, ...)")

    # ── Model: GradientBoosting in a scaling pipeline ─────────────────────────
    # Shallow trees (depth 4) with slow learning (0.05) work best for GB.
    # CalibratedClassifierCV gives accurate probability estimates which
    # directly drive the opacity of the zone highlight in main.py.
    base = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gb",     base),
    ])

    # Cross-validation — gives honest accuracy before we fit on all data
    cv     = StratifiedKFold(n_splits=min(5, len(set(y))), shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\nCV accuracy  : {scores.mean():.1%} ± {scores.std():.1%}  "
          f"folds={[f'{s:.0%}' for s in scores]}")

    # Fit on full dataset
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"Train acc    : {(y_pred == y).mean():.1%}")

    print("\nClassification report (train):")
    print(classification_report(y, y_pred, target_names=ZONE_ORDER,
                                labels=ZONE_ORDER, zero_division=0))

    # Feature importances from the GB step
    gb_step = model.named_steps["gb"]
    print("Feature importances (top 10):")
    pairs = sorted(zip(FEATURE_NAMES, gb_step.feature_importances_),
                   key=lambda x: -x[1])
    for name, imp in pairs[:10]:
        print(f"  {name:<20} {imp:.4f}  {'█' * int(imp * 60)}")

    cm     = confusion_matrix(y, y_pred, labels=ZONE_ORDER)
    header = "".join(f"{z[:4]:>7}" for z in ZONE_ORDER)
    print(f"\nConfusion matrix:")
    print(f"  {'':>10}{header}")
    for i, zone in enumerate(ZONE_ORDER):
        row = "".join(f"{cm[i][j]:>7}" for j in range(len(ZONE_ORDER)))
        print(f"  {zone:<10}{row}")

    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved → {MODEL_FILE}")
    print("Run:  python main.py --model aalto")


if __name__ == "__main__":
    main()
