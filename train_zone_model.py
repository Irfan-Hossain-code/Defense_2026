"""
train_zone_model.py — train both RandomForest and KNN zone classifiers.

Trains two models from the same data and saves them separately.
main.py picks which one to load via --model flag.

Usage:
    python train_zone_model.py
    python train_zone_model.py --data rf_training_data.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

DATA_FILE = os.path.join(os.path.dirname(__file__), "rf_training_data.csv")

MODEL_FILES = {
    "rf":  os.path.join(os.path.dirname(__file__), "rf_zone_model.pkl"),
    "knn": os.path.join(os.path.dirname(__file__), "rf_zone_model_knn.pkl"),
}

ZONE_ORDER = ["LEFT", "MIDDLE", "RIGHT"]


def _cv_and_fit(model, X, y, name: str, cv):
    from sklearn.model_selection import cross_val_score
    print(f"\n── {name} ──────────────────────────────────")
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"  CV accuracy : {scores.mean():.1%} ± {scores.std():.1%}  "
          f"{[f'{s:.0%}' for s in scores]}")
    model.fit(X, y)
    train_acc = (model.predict(X) == y).mean()
    print(f"  Train acc   : {train_acc:.1%}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_FILE)
    args = parser.parse_args()

    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError:
        print("Missing dependency.  Run:  pip install scikit-learn")
        sys.exit(1)

    if not os.path.isfile(args.data):
        print(f"Training data not found: {args.data}")
        print("Run:  python collect_training_data.py  first.")
        sys.exit(1)

    rows = []
    with open(args.data, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if len(rows) < 20:
        print(f"Only {len(rows)} samples — collect more first.")
        sys.exit(1)

    X = np.array([[float(r["left_ratio"]),
                   float(r["middle_ratio"]),
                   float(r["right_ratio"])] for r in rows])
    y = np.array([r["zone"] for r in rows])

    print(f"\n=== Training Zone Classifiers ===\n")
    print(f"Samples: {len(rows)}")
    for zone in ZONE_ORDER:
        n   = int((y == zone).sum())
        bar = "█" * max(1, n // 5)
        print(f"  {zone:<8} {n:>5}  {bar}")

    cv = StratifiedKFold(n_splits=min(5, len(set(y))), shuffle=True, random_state=42)

    # ── RandomForest ──────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf = _cv_and_fit(rf, X, y, "RandomForest", cv)

    names = ["left_ratio", "middle_ratio", "right_ratio"]
    print("  Importances:", {n: f"{v:.3f}" for n, v in zip(names, rf.feature_importances_)})

    # ── KNN (with scaling — KNN is distance-based so scale matters) ───────────
    # Wrap in a Pipeline so scaling is baked into the saved model.
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",   # closer training points count more
            metric="euclidean",
            n_jobs=-1,
        )),
    ])
    knn = _cv_and_fit(knn, X, y, "KNN (k=7, distance-weighted, scaled)", cv)

    # ── Confusion matrices ────────────────────────────────────────────────────
    for label, model in [("RF", rf), ("KNN", knn)]:
        cm     = confusion_matrix(y, model.predict(X), labels=ZONE_ORDER)
        header = "".join(f"{z[:4]:>7}" for z in ZONE_ORDER)
        print(f"\n  {label} confusion (rows=actual, cols=predicted):")
        print(f"  {'':>10}{header}")
        for i, zone in enumerate(ZONE_ORDER):
            row = "".join(f"{cm[i][j]:>7}" for j in range(len(ZONE_ORDER)))
            print(f"  {zone:<10}{row}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(rf,  MODEL_FILES["rf"])
    joblib.dump(knn, MODEL_FILES["knn"])
    print(f"\nSaved:")
    print(f"  RF  → {MODEL_FILES['rf']}")
    print(f"  KNN → {MODEL_FILES['knn']}")
    print()
    print("Usage:")
    print("  python main.py              # RandomForest (default)")
    print("  python main.py --model knn  # KNN")


if __name__ == "__main__":
    main()
