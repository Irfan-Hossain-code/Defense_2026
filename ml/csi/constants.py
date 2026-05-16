"""CSI ML constants — 2-node (LEFT + RIGHT) windows, 3 zone labels."""

from __future__ import annotations

from .windows import LABEL_NAMES, LABEL_TO_IDX, N, S, T

N_SUBCARRIERS = S
WINDOW_SIZE = T
N_NODES = N
NODE_ORDER = ("left", "right")

ZONE_LABELS = tuple(LABEL_NAMES)
ID_TO_LABEL = {i: z for z, i in LABEL_TO_IDX.items()}

NO_MOTION_THRESHOLD = 1.1
