"""macOS `say` TTS — sequential playback (not rushed)."""

from __future__ import annotations

import os
import platform
import queue
import subprocess
import threading
import time

_MIN_GAP_SEC = float(os.getenv("TTS_MIN_GAP_SEC", "2.0"))
_SAY_RATE = os.getenv("SAY_RATE", "175")
_SAY_VOICE = os.getenv("SAY_VOICE", "").strip()  # e.g. Samantha, Daniel — empty = system default

_q: queue.Queue[str | None] = queue.Queue()
_worker_started = False
_lock = threading.Lock()
_last_spoken: str = ""


def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if _worker_started:
            return
        threading.Thread(target=_speech_worker, daemon=True, name="tts-queue").start()
        _worker_started = True


def speak_line(text: str) -> None:
    global _last_spoken
    line = text.strip()
    if not line:
        return
    key = line.lower()
    with _lock:
        if key == _last_spoken:
            return
    _ensure_worker()
    _q.put(line)


def _speech_worker() -> None:
    while True:
        text = _q.get()
        if text is None:
            break
        _speak_once(text)
        time.sleep(_MIN_GAP_SEC)
        _q.task_done()


def _speak_once(text: str) -> None:
    global _last_spoken
    key = text.strip().lower()
    if not key or key == _last_spoken:
        return

    if platform.system() != "Darwin":
        print(f"[tts] {text}")
        _last_spoken = key
        return

    cmd = ["say", "-r", _SAY_RATE]
    if _SAY_VOICE:
        cmd.extend(["-v", _SAY_VOICE])
    cmd.append(text)
    try:
        subprocess.run(cmd, check=False)
        _last_spoken = key
    except Exception as exc:
        print(f"[tts] {exc}")
