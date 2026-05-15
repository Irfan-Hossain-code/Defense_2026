from dataclasses import dataclass


@dataclass
class PersonState:
    """All derived data about a tracked person, captured once per frame."""
    landmarks: list        # list of (x_norm, y_norm, z_norm, visibility) per landmark
    center: tuple          # (cx, cy) in pixels — midpoint of bounding box
    bbox: tuple            # (x1, y1, x2, y2) in pixels
    body_height_px: float  # nose-to-ankle distance in pixels
    confidence: float      # mean visibility of torso/leg landmarks [0.0 – 1.0]
    movement_vec: tuple    # (dx, dy) pixels moved since last frame
