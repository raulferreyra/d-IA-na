import threading
from typing import Optional

import pyttsx3


class TTSEngine:
    """Persistent TTS engine with live volume updates and immediate stop()."""

    def __init__(self):
        self._engine = pyttsx3.init()
        self._lock = threading.Lock()
        self._speaking = False

    def speak(self, text: str, volume: float) -> None:
        if not text:
            return

        volume = max(0.0, min(1.0, volume))

        with self._lock:
            self._engine.setProperty("volume", volume)
            self._engine.say(text)

        self._speaking = True
        try:
            self._engine.runAndWait()
        finally:
            self._speaking = False

    def stop(self) -> None:
        with self._lock:
            self._engine.stop()
