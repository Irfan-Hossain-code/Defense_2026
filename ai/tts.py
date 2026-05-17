"""
TTS — tries in order:
  1. ElevenLabs API  (ELEVENLABS_API_KEY set)  → sounddevice PCM playback
  2. macOS `say`     (Darwin only)
  3. Windows native  (PowerShell MediaPlayer)
  4. Print-only      (silent fallback)
"""

from __future__ import annotations

import os
import platform
import queue
import subprocess
import tempfile
import threading
import time

_MIN_GAP_SEC = float(os.getenv("TTS_MIN_GAP_SEC", "2.0"))
_SAY_RATE    = os.getenv("SAY_RATE", "175")
_SAY_VOICE   = os.getenv("SAY_VOICE", "").strip()
_EL_API_KEY  = os.getenv("ELEVENLABS_API_KEY",  "").strip()
_EL_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL").strip()
_EL_MODEL    = os.getenv("ELEVENLABS_MODEL",    "eleven_turbo_v2_5").strip()

_q: queue.Queue[str | None] = queue.Queue()
_started = False
_lock    = threading.Lock()
_last: str = ""


def _ensure_worker() -> None:
    global _started
    with _lock:
        if _started:
            return
        threading.Thread(target=_worker, daemon=True, name="tts").start()
        _started = True


def speak_line(text: str) -> None:
    global _last
    line = text.strip()
    if not line:
        return
    key = line.lower()
    with _lock:
        if key == _last:
            return
    _ensure_worker()
    _q.put(line)


def _worker() -> None:
    while True:
        text = _q.get()
        if text is None:
            break
        _speak_once(text)
        time.sleep(_MIN_GAP_SEC)
        _q.task_done()


def _speak_once(text: str) -> None:
    global _last
    key = text.strip().lower()
    if not key or key == _last:
        return
    _last = key

    if _EL_API_KEY and _speak_elevenlabs(text):
        return
    if platform.system() == "Darwin" and _speak_macos(text):
        return
    if platform.system() == "Windows" and _speak_windows(text):
        return
    print(f"[tts] {text}")


# ── ElevenLabs ────────────────────────────────────────────────────────────────

def _speak_elevenlabs(text: str) -> bool:
    try:
        import numpy as np
        import sounddevice as sd
        import requests as req
    except ImportError:
        return False

    try:
        resp = req.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{_EL_VOICE_ID}",
            headers={
                "xi-api-key":   _EL_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "text":           text,
                "model_id":       _EL_MODEL,
                "output_format":  "pcm_22050",   # raw signed-int16, no decoder needed
                "voice_settings": {
                    "stability":        0.45,
                    "similarity_boost": 0.80,
                    "style":            0.30,
                    "use_speaker_boost": True,
                },
            },
            timeout=12,
        )
    except Exception as exc:
        print(f"[tts] ElevenLabs request failed: {exc}")
        return False

    if resp.status_code != 200:
        print(f"[tts] ElevenLabs {resp.status_code}: {resp.text[:120]}")
        return False

    try:
        pcm = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(pcm, samplerate=22050, blocking=True)
        return True
    except Exception as exc:
        print(f"[tts] ElevenLabs playback failed: {exc}")
        return False


# ── macOS `say` ───────────────────────────────────────────────────────────────

def _speak_macos(text: str) -> bool:
    cmd = ["say", "-r", _SAY_RATE]
    if _SAY_VOICE:
        cmd += ["-v", _SAY_VOICE]
    cmd.append(text)
    try:
        subprocess.run(cmd, check=False)
        return True
    except Exception:
        return False


# ── Windows native audio via PowerShell ──────────────────────────────────────
# Uses the built-in Windows Media.SoundPlayer via a temp WAV file generated
# by the SAPI (Speech API) text-to-speech engine — no extra installs needed.

def _speak_windows(text: str) -> bool:
    try:
        # Use PowerShell's built-in SAPI TTS to generate + play immediately
        escaped = text.replace("'", "''").replace('"', '""')
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Speak('{escaped}');"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            check=False,
            timeout=30,
        )
        return True
    except Exception as exc:
        print(f"[tts] Windows SAPI failed: {exc}")
        return False
