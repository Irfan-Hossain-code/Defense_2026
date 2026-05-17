#!/usr/bin/env python3
"""List camera indices — find MacBook cam vs iPhone Continuity Camera."""

import cv2

print("Scanning camera indices 0-4...\n")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"  [{i}] not available")
        continue
    ret, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    status = "OK" if ret and frame is not None else "open but no frame"
    print(f"  [{i}] {status}  ({w}x{h})")

print("\nIf iPhone is index 0, set in .env:  CAMERA_INDEX=1")
print("Then restart: python main.py")
