"""
train_zone_model.py — train a simple RandomForest zone classifier on CSI data.

Reads rf_training_data.csv, trains the model, reports accuracy, saves it.
The trained model is automatically picked up by main.py on next run.

Usage:
    python train_zone_model.py
    python train_zone_model.py --data rf_training_data.csv
"""

from __future__ import annotations

import argparse
import os
import sys

DATA_FILE  = os.path.join(os.path.dirname(__file__), "rf_training_data.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "rf_zone_model.pkl")

ZONE_ORDER = ["LEFT", "MIDDLE", "RIGHT"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_FILE)
    args = parser.parse_args()

    # ── Imports (keep sklearn import local so rest of codebase doesn't need it) ──
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import classification_report, confusion_matrix
        import csv
    except ImportError:
        print("Missing dependency.  Run:  pip install scikit-learn")
        sys.exit(1)

    if not os.path.isfile(args.data):
        print(f"Training data not found: {args.data}")
        print("Run:  python collect_training_data.py  first.")
        sys.exit(1)

    # ── Load CSV ───────────────────────────────────────────────────────────────
    rows = []
    with open(args.data, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if len(rows) < 20:
        print(f"Only {len(rows)} samples — collect more data first.")
        sys.exit(1)

    X = np.array([[float(r["left_ratio"]),
                   float(r["middle_ratio"]),
                   float(r["right_ratio"])] for r in rows])
    y = np.array([r["zone"] for r in rows])

    print(f"\n=== Training Zone Classifier ===\n")
    print(f"Samples loaded : {len(rows)}")
    for zone in ZONE_ORDER:
        n = int((y == zone).sum())
        bar = "█" * (n // 2)
        print(f"  {zone:<14} {n:>4} samples  {bar}")

    # ── Train ──────────────────────────────────────────────────────────────────
    # RandomForest: works well on small datasets, no feature scaling needed,
    # handles class imbalance via class_weight, gives feature importances.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",   # handles unequal samples per zone
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validate to get honest accuracy estimate before fitting on all data
    print(f"\nCross-validating (5-fold) ...")
    cv = StratifiedKFold(n_splits=min(5, len(set(y))), shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"  CV accuracy: {scores.mean():.1%} ± {scores.std():.1%}")
    print(f"  Per-fold   : {[f'{s:.0%}' for s in scores]}")

    # Fit on all data for the final model
    model.fit(X, y)

    # ── Report ─────────────────────────────────────────────────────────────────
    y_pred = model.predict(X)
    print(f"\nTraining accuracy (in-sample): {(y_pred == y).mean():.1%}")
    print("\nClassification report:")
    print(classification_report(y, y_pred, target_names=ZONE_ORDER,
                                labels=ZONE_ORDER, zero_division=0))

    print("Feature importances:")
    names = ["left_ratio", "middle_ratio", "right_ratio"]
    for name, imp in sorted(zip(names, model.feature_importances_),
                            key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:<14} {imp:.3f}  {bar}")

    # ── Confusion matrix (text) ────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred, labels=ZONE_ORDER)
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    header = "".join(f"{z[:4]:>8}" for z in ZONE_ORDER)
    print(f"{'':>14}{header}")
    for i, zone in enumerate(ZONE_ORDER):
        row = "".join(f"{cm[i][j]:>8}" for j in range(len(ZONE_ORDER)))
        print(f"  {zone:<12}{row}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    print("Run main.py — it will load this model automatically.")


if __name__ == "__main__":
    main()
