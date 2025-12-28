import sys
import os
import io
import wave
import numpy as np
import sounddevice as sd
import re

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

from PySide6.QtCore import Qt, QPoint, QSize, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
)


@dataclass
class Intent:
    kind: str                 # "command" | "question" | "dictation"
    # e.g. "summarize", "next_steps", "explain", "translate"
    command: Optional[str]
    format: Optional[str]     # e.g. "markdown", "plain"
    style: Optional[str]      # e.g. "bullets", "numbered", "short", "detailed"
    payload: str              # remaining text to process
    raw: str                  # original transcript


def parse_intent(transcript: str) -> Intent:
    """
    Classifies transcript into command/question/dictation and extracts simple preferences.
    """
    raw = (transcript or "").strip()
    t = raw

    # --- Normalize spaces
    t = re.sub(r"\s+", " ", t).strip()

    # --- Detect activation prefix (Diana, Diana:, D_IA_NA, etc.)
    activation = False
    activation_pattern = r"^(diana|d_ia_na|diana\.|diana:|diana,)\s*"
    if re.match(activation_pattern, t, flags=re.IGNORECASE):
        activation = True
        t = re.sub(activation_pattern, "", t, flags=re.IGNORECASE).strip()

    # --- Detect format preferences
    fmt = None
    if re.search(r"\b(markdown|md|\.md)\b", t, flags=re.IGNORECASE):
        fmt = "markdown"

    # --- Detect style preferences
    style = None
    if re.search(r"\b(viñetas|bullets|bullet points)\b", t, flags=re.IGNORECASE):
        style = "bullets"
    elif re.search(r"\b(pasos|step by step|paso a paso)\b", t, flags=re.IGNORECASE):
        style = "numbered"

    if re.search(r"\b(corto|breve|short)\b", t, flags=re.IGNORECASE):
        style = (style + "+short") if style else "short"
    if re.search(r"\b(detallado|largo|detailed)\b", t, flags=re.IGNORECASE):
        style = (style + "+detailed") if style else "detailed"

    # --- Command keyword detection (Spanish-first, simple)
    # You can extend this list safely later.
    cmd = None
    command_map = [
        (r"\bresume\b|\bresumen\b|\bresum(e|e)\b", "summarize"),
        (r"\bexplica\b|\bexplain\b", "explain"),
        (r"\bpasos\b|\bsiguientes pasos\b|\bnext steps\b", "next_steps"),
        (r"\btraduce\b|\btranslate\b", "translate"),
        (r"\bmejora\b|\brefactoriza\b|\brefactor\b", "refactor"),
        (r"\bdebug\b|\barregla\b|\bfix\b", "debug"),
    ]

    for pattern, name in command_map:
        if re.search(pattern, t, flags=re.IGNORECASE):
            cmd = name
            break

    # --- Question detection
    # If there is a question mark or typical question words, treat as question unless activation+command.
    looks_like_question = (
        "?" in t
        or re.match(r"^(qué|que|cómo|como|por qué|porque|cuál|cual|dónde|donde|para qué|cuando)\b", t, flags=re.IGNORECASE)
    )

    # --- Decide kind
    if activation and cmd:
        kind = "command"
    elif looks_like_question and not cmd:
        kind = "question"
    elif activation and not cmd:
        # If user calls Diana but no recognized command, treat as question by default
        kind = "question" if looks_like_question else "command"
        # but keep cmd = None (router decides later)
    elif cmd and not looks_like_question:
        # Command without saying "Diana"
        kind = "command"
    else:
        kind = "dictation"

    payload = t.strip()

    return Intent(
        kind=kind,
        command=cmd,
        format=fmt,
        style=style,
        payload=payload,
        raw=raw,
    )


@dataclass
class Turn:
    role: str   # "user" | "assistant"
    content: str


class ConversationContext:
    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.turns: Deque[Turn] = deque(maxlen=max_turns)

    def add_user(self, text: str):
        self.turns.append(Turn(role="user", content=text))

    def add_assistant(self, text: str):
        self.turns.append(Turn(role="assistant", content=text))

    def as_messages(self) -> List[Dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def last_user_text(self) -> Optional[str]:
        for t in reversed(self.turns):
            if t.role == "user":
                return t.content
        return None

    def last_assistant_text(self) -> Optional[str]:
        for t in reversed(self.turns):
            if t.role == "assistant":
                return t.content
        return None


class ChatWorker(QThread):
    done = Signal(str)
    error = Signal(str)

    def __init__(self, api_key: str, messages: List[Dict[str, str]], model: str = "gpt-4o-mini", parent=None):
        super().__init__(parent)
        self.api_key = api_key
        self.messages = messages
        self.model = model

    def run(self):
        try:
            # SDK nuevo (OpenAI >= 1.x)
            try:
                from openai import OpenAI  # type: ignore
                client = OpenAI(api_key=self.api_key)
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                )
                text = resp.choices[0].message.content or ""
                self.done.emit(text.strip())
                return
            except Exception:
                pass

            # Fallback SDK viejo (openai 0.x)
            import openai  # type: ignore
            openai.api_key = self.api_key
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
            )
            text = resp["choices"][0]["message"]["content"] or ""
            self.done.emit(text.strip())

        except Exception as e:
            self.error.emit(str(e))


# ----------------------------
# Canvas: night sky with stars + moon
# ----------------------------


class NightSky(QWidget):
    def __init__(self):
        super().__init__()

        rng = np.random.default_rng(42)

        self.base_w = 1000
        self.base_h = 600

        self.star_colors = [
            QColor(255, 255, 255),
            QColor(255, 244, 214),
            QColor(180, 220, 255),
            QColor(255, 120, 120),
        ]

        self.stars = []
        for _ in range(220):
            x = int(rng.integers(0, self.base_w))
            y = int(rng.integers(0, self.base_h))
            c = self.star_colors[int(rng.integers(0, len(self.star_colors)))]
            size = int(rng.integers(1, 3))
            self.stars.append((x, y, c, size))

        self.moon_center = QPoint(780, 120)
        self.moon_radius = 52

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#061225"))

        sx = self.width() / self.base_w
        sy = self.height() / self.base_h

        # Stars
        for x, y, color, size in self.stars:
            painter.setPen(QPen(color))
            rx = int(x * sx)
            ry = int(y * sy)
            painter.drawPoint(rx, ry)
            if size > 1:
                painter.drawPoint(rx + 1, ry)
                painter.drawPoint(rx, ry + 1)

        # Moon
        cx = int(self.moon_center.x() * sx)
        cy = int(self.moon_center.y() * sy)
        r = int(self.moon_radius * min(sx, sy))

        painter.setPen(Qt.NoPen)

        painter.setBrush(QBrush(QColor(255, 248, 220)))
        painter.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)

        painter.setBrush(QBrush(QColor(255, 255, 255, 120)))
        painter.drawEllipse(
            cx - int(r * 0.85),
            cy - int(r * 0.85),
            int(2 * r * 0.85),
            int(2 * r * 0.85),
        )

        painter.setBrush(QBrush(QColor(255, 255, 255, 70)))
        painter.drawEllipse(
            cx - int(r * 0.65),
            cy - int(r * 0.65),
            int(2 * r * 0.65),
            int(2 * r * 0.65),
        )


# ----------------------------
# Helpers: audio -> WAV bytes (in memory)
# ----------------------------
def float32_to_wav_bytes(audio_float32: np.ndarray, samplerate: int) -> bytes:
    """
    Convert float32 PCM (-1..1) to WAV (16-bit PCM) in memory.
    No files are written to disk.
    """
    if audio_float32.ndim == 2:
        # take first channel if needed
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


class NamedBytesIO(io.BytesIO):
    """A BytesIO that mimics a file with a name (some clients expect it)."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


# ----------------------------
# Transcription worker (runs off the UI thread)
# ----------------------------
class TranscriptionWorker(QThread):
    done = Signal(str)
    error = Signal(str)

    def __init__(self, wav_bytes: bytes, api_key: str, parent=None):
        super().__init__(parent)
        self.wav_bytes = wav_bytes
        self.api_key = api_key

    def run(self):
        try:
            client = OpenAI(api_key=self.api_key)

            file_obj = NamedBytesIO(self.wav_bytes, "mic.wav")
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=file_obj,
                language="es",
            )

            text = getattr(result, "text", "").strip()
            if not text:
                text = "[Transcription] Empty result."
            self.done.emit(text)

        except Exception as e:
            self.error.emit(str(e))


# ----------------------------
# Main window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load env and OpenAI key
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
        self.ctx = ConversationContext(max_turns=8)
        self.last_intent = None
        self.chat_worker = None

        # Audio capture state (RAM)
        self.is_recording = False
        self.audio_frames: list[np.ndarray] = []
        self.samplerate = 16000
        self.channels = 1
        self.stream = None

        # Window
        self.setWindowTitle("D_IA_NA")
        self.setWindowIcon(QIcon("assets/icons/moon.ico"))
        self.setMinimumSize(820, 520)

        # Background canvas
        self.sky = NightSky()
        self.setCentralWidget(self.sky)

        # Overlay panel
        self.overlay = QWidget(self.sky)
        self.overlay.setStyleSheet("""
            QWidget {
                background: rgba(0, 0, 0, 140);
                border-radius: 12px;
            }
            QLabel {
                color: #E9F2FF;
                font-size: 15px;
                font-weight: 600;
            }
            QTextEdit {
                background: rgba(10, 20, 40, 180);
                color: #E9F2FF;
                border: 1px solid rgba(233, 242, 255, 120);
                border-radius: 10px;
                padding: 8px;
                font-family: Consolas, "Cascadia Mono", monospace;
                font-size: 12px;
            }
        """)

        layout = QVBoxLayout(self.overlay)
        layout.setContentsMargins(22, 16, 22, 16)
        layout.setSpacing(10)

        title = QLabel("Resultado")
        layout.addWidget(title)

        # Text box (main log)
        self.text_box = QTextEdit()
        self.text_box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.text_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_box.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text_box.setReadOnly(True)
        self.text_box.setText(
            "Hola 🌙\n\n"
            "d_IA_na está conectada.\n"
            "Presiona para hablar.\n"
        )
        layout.addWidget(self.text_box)

        # Top row: mic button + mic info + status
        top_row = QHBoxLayout()

        self.mic_btn = QPushButton("")
        self.mic_btn.setToolTip("Mantener para hablar")
        self.mic_btn.setCursor(Qt.PointingHandCursor)

        BTN_SIZE = 46
        self.mic_btn.setFixedSize(BTN_SIZE, BTN_SIZE)

        mic_icon = QIcon("assets/images/microphone.png")
        self.mic_btn.setIcon(mic_icon)
        self.mic_btn.setIconSize(QSize(22, 22))

        self.mic_info_label = QLabel("Audio: —")
        self.mic_info_label.setStyleSheet(
            "color: rgba(233, 242, 255, 180); font-size: 12px;")

        self.status_label = QLabel("Estado: Listo.")
        self.status_label.setStyleSheet("color: #E9F2FF; font-size: 12px;")

        top_row.addWidget(self.mic_btn)
        top_row.addStretch(1)
        top_row.addWidget(self.mic_info_label)
        top_row.addSpacing(12)
        top_row.addWidget(self.status_label)
        layout.addLayout(top_row)

        # Wiring
        self.mic_btn.pressed.connect(self.start_recording)
        self.mic_btn.released.connect(self.stop_recording)
        self.set_mic_idle_style()

        # Transcription worker placeholder
        self.tx_worker = None

        # Friendly warning if key missing
        if not self.openai_key:
            self.text_box.append(
                "[Config] OPENAI_API_KEY no encontrada em .env. La transcripción fallará hasta que la establezcas.")

    def route_intent(self, intent):
        self.last_intent = intent

        payload_lower = intent.payload.lower().strip()

        if payload_lower in ("continúa", "continua", "sigue", "continue"):
            last = self.ctx.last_assistant_text() or ""
            user_text = f"Continúa desde aquí:\n\n{last}"
            self._ask_chat(user_text, intent)
            return

        if payload_lower in ("resume lo anterior", "resumen de lo anterior", "resume lo último", "resume lo ultimo"):
            last = self.ctx.last_assistant_text() or ""
            user_text = f"Resume lo anterior en pocas líneas:\n\n{last}"
            self._ask_chat(user_text, intent)
            return

        if intent.kind == "dictation":
            self.ctx.add_user(f"[Dictado] {intent.raw}")
            self.text_box.append(f"[Dictado] {intent.raw}")
            return

        if intent.kind == "question":
            self._ask_chat(intent.raw, intent)
            return

        if intent.kind == "command":
            cmd = intent.command or "general_command"

            if cmd == "summarize":
                self._ask_chat(
                    f"Resume lo siguiente:\n\n{intent.payload}", intent)
                return

            if cmd == "next_steps":
                self._ask_chat(
                    f"Dame los siguientes pasos para:\n\n{intent.payload}", intent)
                return

            if cmd == "explain":
                self._ask_chat(
                    f"Explica lo siguiente:\n\n{intent.payload}", intent)
                return

            if cmd == "translate":
                self._ask_chat(
                    f"Traduce lo siguiente:\n\n{intent.payload}", intent)
                return

            if cmd in ("debug", "refactor"):
                self._ask_chat(f"{intent.payload}", intent)
                return

            self._ask_chat(intent.payload, intent)
            return

    def _build_style_instructions(self, intent) -> str:
        """
        Convierte format/style detectados (Paso 9) en instrucciones.
        """
        instr = []
        if intent.format == "markdown":
            instr.append("Responde en Markdown.")

        if intent.style:
            s = intent.style.lower()
            if "bullets" in s:
                instr.append("Usa viñetas.")
            if "numbered" in s:
                instr.append("Usa pasos numerados.")
            if "short" in s:
                instr.append("Sé breve.")
            if "detailed" in s:
                instr.append("Sé detallado.")

        instr.append(
            "Mantén un tono amoroso, claro y directo. No uses emojis.")

        return " ".join(instr).strip()

    def _ask_chat(self, user_text: str, intent):
        if not self.openai_key:
            self.text_box.append("[Chat] Falta OPENAI_API_KEY.")
            return

        style_instr = self._build_style_instructions(intent)

        system_prompt = (
            "Eres Diana, una asistente para programación y tareas técnicas. "
            "Responde con precisión. "
            + (style_instr if style_instr else "")
        ).strip()

        self.ctx.add_user(user_text)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.ctx.as_messages())

        # UI status
        self.status_label.setText("Estado: Pensando...")
        self.mic_btn.setEnabled(False)

        self.chat_worker = ChatWorker(
            api_key=self.openai_key,
            messages=messages,
            model=self.chat_model,
        )
        self.chat_worker.done.connect(self.on_chat_done)
        self.chat_worker.error.connect(self.on_chat_error)
        self.chat_worker.start()

    def on_chat_done(self, answer: str):
        self.ctx.add_assistant(answer)
        self.text_box.append("\n" + answer + "\n")
        self.status_label.setText("Estado: Listo.")
        self.mic_btn.setEnabled(True)

    def on_chat_error(self, msg: str):
        self.text_box.append(f"[Chat ERROR] {msg}")
        self.status_label.setText("Estado: Listo.")
        self.mic_btn.setEnabled(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        w = self.sky.width()
        h = self.sky.height()

        ow = int(w * 0.72)
        oh = int(h * 0.45)

        ox = int((w - ow) / 2)
        oy = int(h * 0.42)

        self.overlay.setGeometry(ox, oy, ow, oh)

    # ----------------------------
    # Start/stop recording
    # ----------------------------
    def start_recording(self):
        if self.is_recording:
            return

        self.set_mic_recording_style()
        self.is_recording = True
        self.audio_frames = []

        self.status_label.setText("Estado: Grabando...")
        self.mic_info_label.setText("Audio: grabando...")

        def callback(indata, frames, time_info, status):
            if self.is_recording:
                self.audio_frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
        )
        self.stream.start()

    def stop_recording(self):
        # Always re-enable button and restore style even if something goes wrong
        if not self.is_recording:
            return

        self.mic_btn.setEnabled(False)
        self.status_label.setText("Estado: Procesando...")

        self.is_recording = False

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None

        # If nothing recorded, exit gracefully
        if not self.audio_frames:
            self.mic_info_label.setText("Audio: —")
            self.status_label.setText("Estado: Listo.")
            self.set_mic_idle_style()
            self.mic_btn.setEnabled(True)
            return

        # Concatenate and compute quick stats
        audio = np.concatenate(self.audio_frames, axis=0)  # (N, 1)
        duration = len(audio) / float(self.samplerate)
        peak = float(np.max(np.abs(audio)))

        self.mic_info_label.setText(
            f"Audio: {duration:.2f}s | pico={peak:.3f}")

        # ----------------------------
        # Transcribe audio (in memory WAV)
        # ----------------------------
        wav_bytes = float32_to_wav_bytes(audio, self.samplerate)

        if not self.openai_key:
            self.text_box.append(
                "[Transcripción] Falta OPENAI_API_KEY. No se puede transcribir.")
            self.status_label.setText("Estado: Listo.")
            self.set_mic_idle_style()
            self.mic_btn.setEnabled(True)
            return

        self.status_label.setText("Estado: Transcribiendo...")

        self.tx_worker = TranscriptionWorker(
            wav_bytes=wav_bytes, api_key=self.openai_key)
        self.tx_worker.done.connect(self.on_transcription_done)
        self.tx_worker.error.connect(self.on_transcription_error)
        self.tx_worker.start()

    # ----------------------------
    # Transcription callbacks
    # ----------------------------
    def on_transcription_done(self, text: str):
        intent = parse_intent(text)
        self.route_intent(intent)

        self.text_box.append("—" * 36)
        self.text_box.append(f"Transcrito: {intent.raw}")

        self.text_box.append(
            f"Intent(kind={intent.kind}, command={intent.command}, format={intent.format}, style={intent.style})"
        )
        self.text_box.append(f"Payload: {intent.payload}")

        self.status_label.setText("Estado: Listo.")
        self.set_mic_idle_style()
        self.mic_btn.setEnabled(True)

    def on_transcription_error(self, msg: str):
        self.text_box.append(f"[ERROR de Transcripción] {msg}")
        self.status_label.setText("Estado: Listo.")
        self.set_mic_idle_style()
        self.mic_btn.setEnabled(True)

    # ----------------------------
    # Mic button styles
    # ----------------------------
    def set_mic_idle_style(self):
        self.mic_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(190, 70, 70, 210);
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: 23px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: rgba(210, 85, 85, 220);
            }
        """)

    def set_mic_recording_style(self):
        self.mic_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(80, 170, 110, 230);
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: 23px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: rgba(95, 190, 125, 240);
            }
        """)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
