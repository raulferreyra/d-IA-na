import numpy as np
from dotenv import load_dotenv
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config import load_config
from core.context import ConversationContext
from core.intent import parse_intent, Intent
from audio.capture import AudioCapture
from audio.wav import float32_to_wav_bytes, NamedBytesIO
from ai.templates import DEV_TEMPLATES
from ai.router import build_style_instructions, normalize_user_text, tts_text_from_answer
from ai.client import AIClient
from tts.engine import TTSEngine
from ui.styles import overlay_stylesheet, mic_idle_style, mic_recording_style
from ui.widgets.night_sky import NightSky


class Worker(QThread):
    done = Signal(object)
    error = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.done.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        load_dotenv()
        self.cfg = load_config()

        self.ctx = ConversationContext(max_turns=self.cfg.context_max_turns)
        self.capture = AudioCapture(
            self.cfg.audio_sample_rate, self.cfg.audio_channels)

        self.ai = AIClient(api_key=self.cfg.openai_key,
                           chat_model=self.cfg.chat_model)
        self.tts = TTSEngine()

        self._active_tx_worker = None
        self._active_chat_worker = None
        self._active_tts_worker = None

        self._build_ui()
        self._wire_events()

        if not self.cfg.openai_key:
            self.append_text("[Config] OPENAI_API_KEY missing in .env.")

    def _build_ui(self):
        self.setWindowTitle("D_IA_NA")
        if self.cfg.icon_path.exists():
            self.setWindowIcon(QIcon(str(self.cfg.icon_path)))
        self.setMinimumSize(820, 520)

        self.sky = NightSky()
        self.setCentralWidget(self.sky)

        self.overlay = QWidget(self.sky)
        self.overlay.setStyleSheet(overlay_stylesheet())

        layout = QVBoxLayout(self.overlay)
        layout.setContentsMargins(self.cfg.overlay_margin_lr, self.cfg.overlay_margin_tb,
                                  self.cfg.overlay_margin_lr, self.cfg.overlay_margin_tb)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Resultado"))

        self.text_box = QTextEdit()
        self.text_box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.text_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_box.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text_box.setReadOnly(True)
        self.text_box.setText("Hola 🌙\n\nd_IA_na is ready.\nHold to talk.\n")
        layout.addWidget(self.text_box)

        row = QHBoxLayout()

        self.tts_enabled = QCheckBox("🔊")
        self.tts_enabled.setChecked(False)
        self.tts_enabled.setToolTip("Enable/disable voice")

        self.tts_volume = QSlider(Qt.Horizontal)
        self.tts_volume.setRange(0, 100)
        self.tts_volume.setValue(35)
        self.tts_volume.setFixedWidth(120)
        self.tts_volume.setToolTip("Voice volume")

        self.mic_btn = QPushButton("")
        self.mic_btn.setToolTip("Hold to talk (SPACE)")
        self.mic_btn.setCursor(Qt.PointingHandCursor)
        self.mic_btn.setFixedSize(self.cfg.btn_size, self.cfg.btn_size)

        if self.cfg.mic_icon_path.exists():
            self.mic_btn.setIcon(QIcon(str(self.cfg.mic_icon_path)))
        self.mic_btn.setIconSize(
            QSize(self.cfg.btn_icon_size, self.cfg.btn_icon_size))
        self.mic_btn.setStyleSheet(mic_idle_style(self.cfg.btn_size))

        self.mic_info_label = QLabel("Audio: —")
        self.mic_info_label.setStyleSheet(
            "color: rgba(233, 242, 255, 180); font-size: 12px;")

        self.status_label = QLabel("Estado: Listo.")
        self.status_label.setStyleSheet("color: #E9F2FF; font-size: 12px;")

        row.addWidget(self.tts_enabled)
        row.addWidget(self.tts_volume)
        row.addSpacing(10)
        row.addWidget(self.mic_btn)
        row.addStretch(1)
        row.addWidget(self.mic_info_label)
        row.addSpacing(12)
        row.addWidget(self.status_label)

        layout.addLayout(row)

        self._layout_overlay()

    def _layout_overlay(self):
        w = self.sky.width()
        h = self.sky.height()

        ow = int(w * self.cfg.overlay_width_ratio)
        oh = int(h * self.cfg.overlay_height_ratio)

        ox = int((w - ow) / 2)
        oy = int(h * self.cfg.overlay_top_ratio)

        self.overlay.setGeometry(ox, oy, ow, oh)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_overlay()

    def _wire_events(self):
        self.mic_btn.pressed.connect(self.on_mic_pressed)
        self.mic_btn.released.connect(self.on_mic_released)

        self.tts_enabled.stateChanged.connect(self.on_tts_toggle)

    def on_tts_toggle(self):
        if not self.tts_enabled.isChecked():
            self.tts.stop()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.on_mic_pressed()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.on_mic_released()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def on_mic_pressed(self):
        self.mic_btn.setStyleSheet(mic_recording_style(self.cfg.btn_size))
        self.status_label.setText("Estado: Grabando...")
        self.mic_info_label.setText("Audio: recording...")
        self.mic_btn.setToolTip("Release to send (SPACE)")

        self.capture.start()

    def on_mic_released(self):
        audio = self.capture.stop()
        self.mic_btn.setEnabled(False)
        self.status_label.setText("Estado: Procesando...")
        self.mic_btn.setToolTip("Hold to talk (SPACE)")

        if audio.size == 0:
            self.mic_info_label.setText("Audio: —")
            self._finish_interaction()
            return

        duration = len(audio) / float(self.cfg.audio_sample_rate)
        peak = float(np.max(np.abs(audio)))
        self.mic_info_label.setText(
            f"Audio: {duration:.2f}s | peak={peak:.3f}")

        wav_bytes = float32_to_wav_bytes(audio, self.cfg.audio_sample_rate)
        file_obj = NamedBytesIO(wav_bytes, "mic.wav")

        self.status_label.setText("Estado: Transcribiendo...")
        self._active_tx_worker = Worker(self.ai.transcribe_wav, file_obj, "es")
        self._active_tx_worker.done.connect(self.on_transcription_done)
        self._active_tx_worker.error.connect(self.on_transcription_error)
        self._active_tx_worker.start()

    def on_transcription_done(self, text: str):
        intent = parse_intent(text)

        if self.cfg.debug_intent_to_textbox:
            self.append_text("—" * 36)
            self.append_text(f"Transcribed: {intent.raw}")
            self.append_text(
                f"Intent(kind={intent.kind}, command={intent.command}, format={intent.format}, style={intent.style})")
            self.append_text(f"Payload: {intent.payload}")

        self._route_intent(intent)

    def on_transcription_error(self, msg: str):
        self.append_text(f"[Transcription ERROR] {msg}")
        self._finish_interaction()

    def _pick_template(self, intent: Intent) -> str:
        if intent.kind == "question":
            return DEV_TEMPLATES["general"]
        if intent.kind == "command":
            cmd = intent.command or "general"
            return DEV_TEMPLATES.get(cmd, DEV_TEMPLATES["general"])
        return DEV_TEMPLATES["general"]

    def _route_intent(self, intent: Intent):
        if intent.kind == "dictation":
            self.ctx.add_user(f"[Dictation] {intent.raw}")
            self.append_text(f"[Dictation] {intent.raw}")
            self._finish_interaction()
            return

        system_prompt = (self._pick_template(intent) + " " +
                         build_style_instructions(intent)).strip()
        user_text = normalize_user_text(intent, self.ctx.last_assistant_text())

        self.ctx.add_user(user_text)
        messages = [{"role": "system", "content": system_prompt}
                    ] + self.ctx.as_messages()

        self.status_label.setText("Estado: Pensando...")
        self._active_chat_worker = Worker(self.ai.chat, messages)
        self._active_chat_worker.done.connect(self.on_chat_done)
        self._active_chat_worker.error.connect(self.on_chat_error)
        self._active_chat_worker.start()

    def on_chat_done(self, answer: str):
        self.ctx.add_assistant(answer)
        self.append_text("")
        self.append_text(answer)
        self.append_text("")

        self._speak_answer(answer)
        self._finish_interaction()

    def on_chat_error(self, msg: str):
        self.append_text(f"[Chat ERROR] {msg}")
        self._finish_interaction()

    def _speak_answer(self, answer: str):
        if not self.tts_enabled.isChecked():
            return

        text = tts_text_from_answer(answer)
        if not text:
            return

        volume = max(0.0, min(1.0, self.tts_volume.value() / 100.0))

        self._active_tts_worker = Worker(self.tts.speak, text, volume)
        self._active_tts_worker.error.connect(
            lambda msg: self.append_text(f"[TTS ERROR] {msg}"))
        self._active_tts_worker.start()

    def append_text(self, s: str):
        self.text_box.append(s)
        sb = self.text_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _finish_interaction(self):
        self.status_label.setText("Estado: Listo.")
        self.mic_btn.setStyleSheet(mic_idle_style(self.cfg.btn_size))
        self.mic_btn.setEnabled(True)
