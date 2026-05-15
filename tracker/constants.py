# Tracking thresholds
VISIBILITY_THRESHOLD = 0.45   # mean landmark visibility below this = person "lost"
LOST_CONFIRM_FRAMES  = 12     # consecutive bad frames before switching to ghost mode
GHOST_FADE_RATE      = 0.012  # ghost alpha reduction per frame (1.0 → 0.15 over ~70 frames)
GHOST_MIN_ALPHA      = 0.15   # ghost never fades completely away
GHOST_DRIFT_SECONDS  = 2.5    # how long to apply the last movement vector as drift

# Body skeleton: pairs of MediaPipe landmark indices to draw as bones
# Indices follow MediaPipe's 33-point pose model (shoulders=11/12, hips=23/24, etc.)
SKELETON_EDGES = [
    (11, 12),            # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),            # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]

# Landmark indices used to score confidence (torso + legs only, skips face/hands)
KEY_LANDMARKS = list(range(11, 29))
