import io
import wave
import numpy as np


class NamedBytesIO(io.BytesIO):
    """BytesIO with a .name attribute (some clients expect it)."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def float32_to_wav_bytes(audio_float32: np.ndarray, samplerate: int) -> bytes:
    """float32 PCM (-1..1) -> WAV 16-bit PCM in memory (no files)."""
    if audio_float32.ndim == 2:
        audio_float32 = audio_float32[:, 0]

    audio_float32 = np.clip(audio_float32, -1.0, 1.0)
    audio_int16 = (audio_float32 * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(samplerate))
        wf.writeframes(audio_int16.tobytes())

    return buf.getvalue()
