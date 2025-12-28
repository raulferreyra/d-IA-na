from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd


class AudioCapture:
    """Push-to-talk audio capture into RAM buffers."""

    def __init__(self, samplerate: int, channels: int):
        self.samplerate = samplerate
        self.channels = channels
        self.frames: List[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None
        self.is_recording = False

    def start(self, on_error: Optional[Callable[[str], None]] = None) -> None:
        if self.is_recording:
            return

        self.is_recording = True
        self.frames = []

        def callback(indata, frames, time_info, status):
            if status and on_error:
                on_error(str(status))
            if self.is_recording:
                self.frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
        )
        self.stream.start()

    def stop(self) -> np.ndarray:
        self.is_recording = False
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None

        if not self.frames:
            return np.array([], dtype=np.float32)

        return np.concatenate(self.frames, axis=0)
