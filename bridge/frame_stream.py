"""MJPEG HTTP stream — OpenCV composite (skeleton + face) in the browser."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import cv2
import numpy as np


class FrameStream:
    """Serves /video as multipart MJPEG for <img src='http://127.0.0.1:8766/video'>."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8766) -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._serve, daemon=True, name="mjpeg")
        self._thread.start()
        print(f"[video] MJPEG stream http://{self.host}:{self.port}/video")

    def push(self, frame_bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def _serve(self) -> None:
        stream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # noqa: ARG001
                pass

            def _send_jpeg(self, jpeg: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(jpeg)

            def do_GET(self):  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/frame.jpg":
                    jpeg = stream.get_jpeg()
                    if not jpeg:
                        self.send_error(503, "No frame yet")
                        return
                    try:
                        self._send_jpeg(jpeg)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return

                if path not in ("/video", "/"):
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    while True:
                        jpeg = stream.get_jpeg()
                        if jpeg:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        time.sleep(0.033)  # ~30 fps cap
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = HTTPServer((self.host, self.port), Handler)
        server.serve_forever()
