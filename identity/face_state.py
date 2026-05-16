from dataclasses import dataclass


@dataclass
class FaceState:
    """Identity recognition result for a single detected face."""
    name: str           # Recognised name; never "Unknown" — None result means unknown
    confidence: float   # Match confidence [0.0 – 1.0] (1.0 - face_distance)
    bbox: tuple         # (top, right, bottom, left) in original-resolution pixels
