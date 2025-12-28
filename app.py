import io
import os
import re
import sys
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from PySide6.QtCore import (
    QPoint,
    QSize,
    Qt,
    QThread,
    Signal,
    QPropertyAnimation,
    QEasingCurve,
    QRect,
    QTimer,
)
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QSlider,
)

# ----------------------------
# Config / constants
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent

ICON_PATH = BASE_DIR / "assets" / "icons" / "moon.ico"
MIC_ICON_PATH = BASE_DIR / "assets" / "images" / "microphone.png"

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

BTN_SIZE = 46
BTN_ICON_SIZE = 22

OVERLAY_MARGIN_LR = 22
OVERLAY_MARGIN_TB = 16
OVERLAY_WIDTH_RATIO = 0.72
OVERLAY_HEIGHT_RATIO = 0.45
OVERLAY_TOP_RATIO = 0.42

CONTEXT_MAX_TURNS = 8

# set it to False when you no longer want to see “Intent(...)”
DEBUG_INTENT_TO_TEXTBOX = True


DEV_TEMPLATES = {
    "general": (
        "Eres Diana, una asistente de ingeniería de software. "
        "Da respuestas técnicas, correctas y accionables."
    ),
    "explain": (
        "Explica como senior developer. "
        "Si hay código, describe qué hace, riesgos, y un ejemplo mínimo si aporta. "
        "Si el usuario pide Markdown, usa encabezados claros."
    ),
    "developer": (
        "Muestra código como senior developer. "
        "aporta código correcto, seguro y eficiente. "
        "Si el usuario pide un lenguaje, úsalo. "
    ),
    "refactor": (
        "Actúa como revisor de código. "
        "Propón refactor con enfoque en legibilidad, separación de responsabilidades, y testabilidad. "
        "Entrega un plan y luego el código propuesto. "
        "No inventes dependencias inexistentes."
    ),
    "tests": (
        "Actúa como QA/Dev. "
        "Genera tests (unitarios primero) con casos borde. "
        "Si falta contexto, asume lo mínimo y dilo explícitamente."
    ),
    "debug": (
        "Actúa como debugger. "
        "Primero hipótesis, luego verificación, luego fix. "
        "Pide logs/trace solo si es estrictamente necesario."
    ),
}


# ----------------------------
# Intent parsing
# ----------------------------
@dataclass
class Intent:
    kind: str                 # "command" | "question" | "dictation"
    # e.g. "summarize", "next_steps", "explain", "translate"
    command: Optional[str]
    format: Optional[str]     # e.g. "markdown"
    style: Optional[str]      # e.g. "bullets", "numbered", "short", "detailed"
    payload: str              # remaining text to process
    raw: str                  # original transcript


def parse_intent(transcript: str) -> Intent:
    raw = (transcript or "").strip()
    t = re.sub(r"\s+", " ", raw).strip()

    # Activation prefix (Diana, Diana:, D_IA_NA...)
    activation = False
    activation_pattern = r"^(diana|d_ia_na|diana\.|diana:|diana,)\s*"
    if re.match(activation_pattern, t, flags=re.IGNORECASE):
        activation = True
        t = re.sub(activation_pattern, "", t, flags=re.IGNORECASE).strip()

    # Format preference
    fmt = "markdown" if re.search(
        r"\b(markdown|md|\.md)\b", t, flags=re.IGNORECASE) else None

    # Style preference
    style = None
    if re.search(r"\b(viñetas|bullets|bullet points)\b", t, flags=re.IGNORECASE):
        style = "bullets"
    elif re.search(r"\b(pasos|step by step|paso a paso)\b", t, flags=re.IGNORECASE):
        style = "numbered"

    if re.search(r"\b(corto|breve|short)\b", t, flags=re.IGNORECASE):
        style = (style + "+short") if style else "short"
    if re.search(r"\b(detallado|largo|detailed)\b", t, flags=re.IGNORECASE):
        style = (style + "+detailed") if style else "detailed"

    # Command detection
    cmd = None
    command_map = [
        (r"\b(resume|resumen)\b", "summarize"),
        (r"\b(explica|explain)\b", "explain"),
        (r"\b(siguientes pasos|next steps|pasos)\b", "next_steps"),
        (r"\b(traduce|translate)\b", "translate"),
        (r"\b(mejora|refactoriza|refactor)\b", "refactor"),
        (r"\b(debug|arregla|fix)\b", "debug"),
        (r"\b(corrige|corregir)\b", "correct"),
    ]
    for pattern, name in command_map:
        if re.search(pattern, t, flags=re.IGNORECASE):
            cmd = name
            break

    # Question detection
    looks_like_question = (
        "?" in t
        or re.match(
            r"^(qué|que|cómo|como|por qué|porque|cuál|cual|dónde|donde|para qué|cuando)\b",
            t,
            flags=re.IGNORECASE,
        )
    )

    # Decide kind
    if activation and cmd:
        kind = "command"
    elif looks_like_question and not cmd:
        kind = "question"
    elif activation and not cmd:
        kind = "question" if looks_like_question else "command"
    elif cmd and not looks_like_question:
        kind = "command"
    else:
        kind = "dictation"

    return Intent(
        kind=kind,
        command=cmd,
        format=fmt,
        style=style,
        payload=t.strip(),
        raw=raw,
    )


# ----------------------------
# Short context
# ----------------------------
@dataclass
class Turn:
    role: str   # "user" | "assistant"
    content: str


class ConversationContext:
    def __init__(self, max_turns: int = CONTEXT_MAX_TURNS):
        self.turns: Deque[Turn] = deque(maxlen=max_turns)

    def add_user(self, text: str) -> None:
        self.turns.append(Turn(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.turns.append(Turn(role="assistant", content=text))

    def as_messages(self) -> List[Dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def last_assistant_text(self) -> str:
        for t in reversed(self.turns):
            if t.role == "assistant":
                return t.content
        return ""


# ----------------------------
# UI background: night sky
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
        painter.drawEllipse(cx - int(r * 0.85), cy -
                            int(r * 0.85), int(2 * r * 0.85), int(2 * r * 0.85))

        painter.setBrush(QBrush(QColor(255, 255, 255, 70)))
        painter.drawEllipse(cx - int(r * 0.65), cy -
                            int(r * 0.65), int(2 * r * 0.65), int(2 * r * 0.65))


# ----------------------------
# Audio helpers
# ----------------------------
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


class NamedBytesIO(io.BytesIO):
    """BytesIO with a .name attribute (some clients expect it)."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


# ----------------------------
# Workers (threads)
# ----------------------------
class TTSWorker(QThread):
    error = Signal(str)

    def __init__(self, text: str, volume: float, parent=None):
        super().__init__(parent)
        self.text = text
        self.volume = volume

    def run(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("volume", self.volume)  # 0.0 - 1.0
            engine.say(self.text)
            engine.runAndWait()
        except Exception as e:
            self.error.emit(str(e))


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
            self.done.emit(text if text else "[Transcription] Empty result.")
        except Exception as e:
            self.error.emit(str(e))


class ChatWorker(QThread):
    done = Signal(str)
    error = Signal(str)

    def __init__(self, api_key: str, messages: List[Dict[str, str]], model: str, parent=None):
        super().__init__(parent)
        self.api_key = api_key
        self.messages = messages
        self.model = model

    def run(self):
        try:
            client = OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
            )
            text = resp.choices[0].message.content or ""
            self.done.emit(text.strip())
        except Exception as e:
            self.error.emit(str(e))


# ----------------------------
# Main window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        load_dotenv()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()

        self.ctx = ConversationContext()
        self.last_intent: Optional[Intent] = None

        # Audio capture state
        self.is_recording = False
        self.audio_frames: List[np.ndarray] = []
        self.stream = None
        self.mic_pulse_anim = None

        self.thinking_timer = QTimer()
        self.thinking_dots = 0
        self.thinking_timer.timeout.connect(self.update_thinking_status)

        self.tx_worker: Optional[TranscriptionWorker] = None
        self.chat_worker: Optional[ChatWorker] = None

        # Window
        self.setWindowTitle("D_IA_NA")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.setMinimumSize(820, 520)

        # Background
        self.sky = NightSky()
        self.setCentralWidget(self.sky)

        # Overlay
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
        layout.setContentsMargins(
            OVERLAY_MARGIN_LR, OVERLAY_MARGIN_TB, OVERLAY_MARGIN_LR, OVERLAY_MARGIN_TB)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Resultado"))

        # Text box
        self.text_box = QTextEdit()
        self.text_box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.text_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_box.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text_box.setReadOnly(True)
        self.text_box.setText(
            "Hola 🌙\n\nd_IA_na está conectada.\nPresiona para hablar.\n")
        layout.addWidget(self.text_box)

        # Bottom row
        row = QHBoxLayout()

        self.mic_btn = QPushButton("")
        self.mic_btn.setToolTip("Mantener para hablar")
        self.mic_btn.setCursor(Qt.PointingHandCursor)
        self.mic_btn.setFixedSize(BTN_SIZE, BTN_SIZE)

        if MIC_ICON_PATH.exists():
            self.mic_btn.setIcon(QIcon(str(MIC_ICON_PATH)))
        self.mic_btn.setIconSize(QSize(BTN_ICON_SIZE, BTN_ICON_SIZE))

        self.mic_info_label = QLabel("Audio: —")
        self.mic_info_label.setStyleSheet(
            "color: rgba(233, 242, 255, 180); font-size: 12px;")

        self.status_label = QLabel("Estado: Listo.")
        self.status_label.setStyleSheet("color: #E9F2FF; font-size: 12px;")

        self.tts_enabled = QCheckBox("🔊")
        self.tts_enabled.setChecked(False)
        self.tts_enabled.setToolTip("Activar/desactivar voz")

        self.tts_volume = QSlider(Qt.Horizontal)
        self.tts_volume.setRange(0, 100)
        self.tts_volume.setValue(35)
        self.tts_volume.setFixedWidth(120)
        self.tts_volume.setToolTip("Volumen de voz")

        row.addSpacing(18)
        row.addWidget(self.tts_enabled)
        row.addWidget(self.tts_volume)
        row.addWidget(self.mic_btn)
        row.addStretch(1)
        row.addWidget(self.mic_info_label)
        row.addSpacing(12)
        row.addWidget(self.status_label)

        layout.addLayout(row)

        # Wire mic
        self.mic_btn.pressed.connect(self.start_recording)
        self.mic_btn.released.connect(self.stop_recording)
        self.set_mic_idle_style()

        if not self.openai_key:
            self.text_box.append(
                "[Config] OPENAI_API_KEY no encontrada en .env.")

    def start_thinking(self):
        self.thinking_dots = 0
        self.status_label.setText("Estado: Pensando")
        self.thinking_timer.start(450)

    def stop_thinking(self):
        self.thinking_timer.stop()
        self.status_label.setText("Estado: Listo.")

    def update_thinking_status(self):
        self.thinking_dots = (self.thinking_dots + 1) % 4
        dots = "." * self.thinking_dots
        self.status_label.setText(f"Estado: Pensando{dots}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if not self.is_recording:
                self.start_recording()
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if self.is_recording:
                self.stop_recording()
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def start_mic_pulse(self):
        if self.mic_pulse_anim:
            return

        rect = self.mic_btn.geometry()
        grow = QRect(
            rect.x() - 3,
            rect.y() - 3,
            rect.width() + 6,
            rect.height() + 6,
        )

        anim = QPropertyAnimation(self.mic_btn, b"geometry")
        anim.setDuration(900)
        anim.setLoopCount(-1)
        anim.setStartValue(rect)
        anim.setKeyValueAt(0.5, grow)
        anim.setEndValue(rect)
        anim.setEasingCurve(QEasingCurve.InOutSine)

        self.mic_pulse_anim = anim
        anim.start()

    def stop_mic_pulse(self):
        if self.mic_pulse_anim:
            self.mic_pulse_anim.stop()
            self.mic_pulse_anim = None

    def resizeEvent(self, event):
        super().resizeEvent(event)

        w = self.sky.width()
        h = self.sky.height()

        ow = int(w * OVERLAY_WIDTH_RATIO)
        oh = int(h * OVERLAY_HEIGHT_RATIO)

        ox = int((w - ow) / 2)
        oy = int(h * OVERLAY_TOP_RATIO)

        self.overlay.setGeometry(ox, oy, ow, oh)

    # ----------------------------
    # Recording
    # ----------------------------
    def start_recording(self):
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_frames = []

        self.set_mic_recording_style()
        self.status_label.setText("Estado: Grabando...")
        self.mic_info_label.setText("Audio: grabando...")

        def callback(indata, frames, time_info, status):
            if self.is_recording:
                self.audio_frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype="float32",
            callback=callback,
        )
        self.stream.start()

    def stop_recording(self):
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

        if not self.audio_frames:
            self.mic_info_label.setText("Audio: —")
            self.finish_interaction()
            return

        audio = np.concatenate(self.audio_frames, axis=0)
        duration = len(audio) / float(AUDIO_SAMPLE_RATE)
        peak = float(np.max(np.abs(audio)))
        self.mic_info_label.setText(
            f"Audio: {duration:.2f}s | pico={peak:.3f}")

        if not self.openai_key:
            self.text_box.append("[Transcripción] Falta OPENAI_API_KEY.")
            self.finish_interaction()
            return

        self.status_label.setText("Estado: Transcribiendo...")

        wav_bytes = float32_to_wav_bytes(audio, AUDIO_SAMPLE_RATE)
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
        self.last_intent = intent

        if DEBUG_INTENT_TO_TEXTBOX:
            self.text_box.append("—" * 36)
            self.text_box.append(f"Transcrito: {intent.raw}")
            self.text_box.append(
                f"Intent(kind={intent.kind}, command={intent.command}, format={intent.format}, style={intent.style})"
            )
            self.text_box.append(f"Payload: {intent.payload}")

        self.route_intent(intent)

    def on_transcription_error(self, msg: str):
        self.text_box.append(f"[ERROR de Transcripción] {msg}")
        self.finish_interaction()

    # ----------------------------
    # Router + Context
    # ----------------------------
    def route_intent(self, intent: Intent):
        payload_lower = intent.payload.lower().strip()

        # Context commands
        if payload_lower in ("continúa", "continua", "sigue", "continue"):
            last = self.ctx.last_assistant_text()
            self.ask_chat(f"Continúa desde aquí:\n\n{last}", intent)
            return

        if payload_lower in (
            "resume lo anterior",
            "resumen de lo anterior",
            "resume lo último",
            "resume lo ultimo",
        ):
            last = self.ctx.last_assistant_text()
            self.ask_chat(
                f"Resume lo anterior en pocas líneas:\n\n{last}", intent)
            return

        # Dictation: only log for now
        if intent.kind == "dictation":
            self.ctx.add_user(f"[Dictado] {intent.raw}")
            self.text_box.append(f"[Dictado] {intent.raw}")
            self.finish_interaction()
            return

        # Question
        if intent.kind == "question":
            self.ask_chat(intent.raw, intent)
            return

        # Command
        if intent.kind == "command":
            cmd = intent.command or "general_command"

            if cmd == "summarize":
                self.ask_chat(
                    f"Resume lo siguiente:\n\n{intent.payload}", intent)
                return

            if cmd == "next_steps":
                self.ask_chat(
                    f"Dame los siguientes pasos para:\n\n{intent.payload}", intent)
                return

            if cmd == "explain":
                self.ask_chat(
                    f"Explica lo siguiente:\n\n{intent.payload}", intent)
                return

            if cmd == "translate":
                self.ask_chat(
                    f"Traduce lo siguiente:\n\n{intent.payload}", intent)
                return

            if cmd == "correct":
                self.ask_chat(
                    f"Corrige el siguiente texto:\n\n{intent.payload}", intent)
                return

            self.ask_chat(intent.payload, intent)
            return

        # Fallback
        self.finish_interaction()

    def build_style_instructions(self, intent: Intent) -> str:
        instr = []

        # Output format
        if intent.format == "markdown":
            instr.append(
                "Responde en Markdown. "
                "Usa encabezados (##) si corresponde. "
                "No uses listas anidadas raras: si hay pasos numerados, mantén subpuntos con '-' bien indentados. "
                "No mezcles 1) con bullets sin necesidad."
            )

        # Style hints
        if intent.style:
            s = intent.style.lower()
            if "bullets" in s:
                instr.append("Usa viñetas '-' como lista principal.")
            if "numbered" in s:
                instr.append(
                    "Usa lista numerada '1.' '2.' como lista principal.")
            if "short" in s:
                instr.append("Sé breve.")
            if "detailed" in s:
                instr.append("Sé detallado.")

        instr.append(
            "Tono técnico, claro y directo. No uses emojis.")
        return " ".join(instr).strip()

    def maybe_speak(self, answer: str):
        # Only read the final response, not logs or intent debug
        if not self.tts_enabled.isChecked():
            return

        # Controlled Volume
        vol = max(0.0, min(1.0, self.tts_volume.value() / 100.0))

        # Minimal cleanup: if the markdown is very long, you can trim it.
        text = answer.strip()
        if not text:
            return

        # Optional: Avoid reading entire blocks of code
        # (To avoid "dictating" 200 lines)
        text = self.strip_code_blocks(text)

        self.tts_worker = TTSWorker(text=text, volume=vol)
        self.tts_worker.error.connect(
            lambda msg: self.text_box.append(f"[TTS ERROR] {msg}"))
        self.tts_worker.start()

    def strip_code_blocks(self, md: str) -> str:
        # Remove content within ``` ``` for easier reading
        return re.sub(r"```.*?```", "[código omitido]", md, flags=re.DOTALL)

    def ask_chat(self, user_text: str, intent: Intent):
        if not self.openai_key:
            self.text_box.append("[Chat] Falta OPENAI_API_KEY.")
            self.finish_interaction()
            return

        dev_template = self.pick_dev_template(intent)
        style_instr = self.build_style_instructions(intent)
        system_prompt = (dev_template + " " + style_instr).strip()

        self.ctx.add_user(user_text)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.ctx.as_messages())

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
        self.maybe_speak(answer)
        self.finish_interaction()

    def on_chat_error(self, msg: str):
        self.text_box.append(f"[Chat ERROR] {msg}")
        self.finish_interaction()

    def pick_dev_template(self, intent: Intent) -> str:
        # Default
        if intent.kind == "question":
            return DEV_TEMPLATES["general"]

        if intent.kind == "command":
            cmd = intent.command or "general"
            if cmd in DEV_TEMPLATES:
                return DEV_TEMPLATES[cmd]
            # alias: correct -> explain (o general)
            if cmd == "correct":
                return DEV_TEMPLATES["general"]
            return DEV_TEMPLATES["general"]

        return DEV_TEMPLATES["general"]

    # ----------------------------
    # UI helpers
    # ----------------------------

    def finish_interaction(self):
        self.status_label.setText("Estado: Listo.")
        self.set_mic_idle_style()
        self.mic_btn.setEnabled(True)

    def set_mic_idle_style(self):
        self.mic_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(190, 70, 70, 210);
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: {BTN_SIZE // 2}px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: rgba(210, 85, 85, 220);
            }}
        """)

    def set_mic_recording_style(self):
        self.mic_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(80, 170, 110, 230);
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: {BTN_SIZE // 2}px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: rgba(95, 190, 125, 240);
            }}
        """)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
