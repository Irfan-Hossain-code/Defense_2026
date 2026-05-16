"""
ml/csi/windows.py — shared utilities for building the CNN window dataset.

Tensor shape per sample: (T=20, S=64, N=2)
  T = 20 consecutive packets (~2 s at ~10 Hz)
  S = 64 subcarrier amplitudes (padded / truncated)
  N = 2 nodes: index 0 = LEFT, index 1 = RIGHT
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Optional

import numpy as np

T = 20
S = 64
N = 2
STRIDE = 1

LABEL_NAMES = ["LEFT", "MIDDLE", "RIGHT"]
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_NAMES)}


def parse_amplitudes(line: str) -> Optional[list[float]]:
    if not line.startswith("CSI:"):
        return None
    parts = line[4:].split(",")
    if len(parts) < 5:
        return None
    try:
        csi_len = int(parts[3])
        raw = [int(x) for x in parts[4 : 4 + csi_len]]
        if len(raw) < csi_len:
            return None
        return [
            math.sqrt(raw[i] ** 2 + raw[i + 1] ** 2)
            for i in range(0, len(raw) - 1, 2)
        ]
    except (ValueError, IndexError):
        return None


def pad_amplitudes(amps: list[float], n_sc: int = S) -> np.ndarray:
    a = np.array(amps[:n_sc], dtype=np.float32)
    if len(a) < n_sc:
        a = np.pad(a, (0, n_sc - len(a)))
    return a


class WindowBuilder:
    """Rolling buffer → (T, S, 2) when both nodes have contributed STRIDE new packets."""

    def __init__(self, t: int = T, s: int = S):
        self.t = t
        self.s = s
        self._bufs: list[deque] = [deque(maxlen=t), deque(maxlen=t)]
        self._new = [0, 0]

    def push(self, node_idx: int, amps: list[float]) -> None:
        self._bufs[node_idx].append(pad_amplitudes(amps, self.s))
        self._new[node_idx] += 1

    def ready(self) -> bool:
        return (
            len(self._bufs[0]) == self.t
            and len(self._bufs[1]) == self.t
            and self._new[0] >= STRIDE
            and self._new[1] >= STRIDE
        )

    def get_window(self) -> np.ndarray:
        left = np.stack(list(self._bufs[0]), axis=0)
        right = np.stack(list(self._bufs[1]), axis=0)
        self._new = [0, 0]
        return np.stack([left, right], axis=-1)


def csv_to_npz(csv_file: str, npz_file: str, t: int = T, s: int = S, stride: int = STRIDE) -> int:
    import csv as _csv

    rows: dict[str, dict[str, list]] = defaultdict(lambda: {"left": [], "right": []})

    with open(csv_file, newline="") as f:
        for row in _csv.DictReader(f):
            zone = row["zone"]
            node = row["node"]
            ts_ms = int(row["ts_ms"])
            amps = [float(row.get(f"amp_{i}", 0.0)) for i in range(s)]
            rows[zone][node].append((ts_ms, amps))

    X_list, y_list = [], []

    for zone, node_data in rows.items():
        if zone not in LABEL_TO_IDX:
            continue
        label = LABEL_TO_IDX[zone]
        left_pkts = sorted(node_data["left"], key=lambda x: x[0])
        right_pkts = sorted(node_data["right"], key=lambda x: x[0])

        if len(left_pkts) < t or len(right_pkts) < t:
            print(
                f"  [warn] {zone}: not enough packets "
                f"(L={len(left_pkts)}, R={len(right_pkts)}) — skipped"
            )
            continue

        ri_base = 0
        for i in range(0, len(left_pkts) - t + 1, stride):
            left_win = left_pkts[i : i + t]
            right_win = []
            ri = ri_base
            for lt_ms, _ in left_win:
                while (
                    ri + 1 < len(right_pkts)
                    and abs(right_pkts[ri + 1][0] - lt_ms) < abs(right_pkts[ri][0] - lt_ms)
                ):
                    ri += 1
                right_win.append(right_pkts[ri])

            max_gap = max(abs(lp[0] - rp[0]) for lp, rp in zip(left_win, right_win))
            if max_gap > 500:
                continue

            left_amps = np.array([pad_amplitudes(lp[1], s) for lp in left_win])
            right_amps = np.array([pad_amplitudes(rp[1], s) for rp in right_win])
            window = np.stack([left_amps, right_amps], axis=-1)
            X_list.append(window)
            y_list.append(label)

    if not X_list:
        print("  [warn] No windows generated — check CSV contents.")
        return 0

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    mean = float(X.mean())
    std = float(X.std()) + 1e-6
    X = (X - mean) / std

    np.savez(
        npz_file,
        X=X,
        y=y,
        label_names=np.array(LABEL_NAMES),
        mean=np.float32(mean),
        std=np.float32(std),
    )
    print(f"  NPZ: {npz_file}  shape={X.shape}  norm mean={mean:.4f} std={std:.4f}")
    return len(X_list)
