import os
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = "pose_landmarker_lite.task"


def ensure_model():
    """Download the MediaPipe pose model if not already present (~5 MB, one-time)."""
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading pose model (~5 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_progress)
    print()


def _progress(count, block_size, total):
    pct = min(100, int(count * block_size * 100 / total))
    print(f"\r  {pct}%", end="", flush=True)
