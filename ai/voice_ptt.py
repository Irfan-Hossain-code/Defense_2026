"""Push-to-talk: T toggles record → STT → Q&A → TTS."""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import client
from .store import TargetStore
from .tts import speak_line

SAMPLE_RATE = 16_000
CHANNELS = 1


class VoicePTT:
    def __init__(
        self,
        store: TargetStore,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.store = store
        self._on_line = on_line
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._busy = False
        self._stream = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def toggle(self) -> None:
        if self._busy:
            return
        if not self._recording:
            self._start()
        else:
            self._stop_and_process()

    def _start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            print("[voice] pip install sounddevice")
            return

        self._frames = []
        self._recording = True
        print("[voice] Recording… press T again to send.")

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            if self._recording:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def _stop_and_process(self) -> None:
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            print("[voice] No audio.")
            return

        audio = np.concatenate(self._frames, axis=0).flatten()
        self._busy = True
        threading.Thread(target=self._process_audio, args=(audio,), daemon=True).start()

    def _process_audio(self, audio: np.ndarray) -> None:
        try:
            question = self._transcribe(audio)
            if not question:
                return
            print(f"[voice] Heard: {question}")
            q_lower = question.lower()
            if any(
                p in q_lower
                for p in ("clear dossier", "dismiss target", "remove target", "clear target")
            ):
                self.store.clear_dossier()
                msg = "Target dossier cleared."
                self.store.set_last_narration(msg, show_subtitle=True)
                if self._on_line:
                    self._on_line(msg)
                speak_line(msg)
                return

            answer = client.answer_question(question, self.store.snapshot())
            self.store.set_last_narration(answer, show_subtitle=True)
            if self._on_line:
                time.sleep(0.2)
                self._on_line(answer)
            speak_line(answer)
        finally:
            self._busy = False

    def _transcribe(self, audio: np.ndarray) -> str:
        wav_path = Path("data/ptt_last.wav")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

        try:
            import speech_recognition as sr

            r = sr.Recognizer()
            with sr.AudioFile(str(wav_path)) as source:
                audio_data = r.record(source)
            return r.recognize_google(audio_data).strip()
        except Exception as exc:
            print(f"[voice] STT: {exc}")
            return ""
