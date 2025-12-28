import sys
import os
import openai
import numpy as np
import sounddevice as sd

from dotenv import load_dotenv
from PySide6.QtCore import (
    Qt,
    QPoint,
    QSize,
)
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QIcon,
)
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

# Class for drawing the night sky with stars and a moon


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
        painter.drawEllipse(cx - int(r * 0.85), cy - int(r * 0.85),
                            int(2 * r * 0.85), int(2 * r * 0.85))

        painter.setBrush(QBrush(QColor(255, 255, 255, 70)))
        painter.drawEllipse(cx - int(r * 0.65), cy - int(r * 0.65),
                            int(2 * r * 0.65), int(2 * r * 0.65))


# ----------------------------
# Main window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- buffers for RAM ---
        self.is_recording = False
        self.audio_frames = []
        self.samplerate = 16000
        self.channels = 1
        self.stream = None

        self.setWindowTitle("D_IA_NA")
        self.setWindowIcon(QIcon("assets/icons/moon.ico"))
        self.setMinimumSize(820, 520)

        self.sky = NightSky()
        self.setCentralWidget(self.sky)

        # Overlay (panel over the sky)
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
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("Resultado")
        layout.addWidget(title)

        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

        audio_file = open("audio.mp3", "rb")
        transcribed = openai.Audio.transcribe("whisper-1", audio_file)

        self.text_box = QTextEdit()
        self.text_box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.text_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_box.setLineWrapMode(QTextEdit.WidgetWidth)

        self.text_box.setReadOnly(True)
        self.text_box.setText(
            "Hola 🌙\n\n"
            "Diana está enchufada ( ͡° ͜ʖ ͡°).\n"
            "Resultado: " + transcribed.text
        )
        layout.addWidget(self.text_box)

        top_row = QHBoxLayout()

        self.mic_btn = QPushButton("")
        self.mic_btn.setToolTip("Mantener para hablar")
        self.mic_btn.setCursor(Qt.PointingHandCursor)

        # Fixed size for the button
        BTN_SIZE = 46
        self.mic_btn.setFixedSize(BTN_SIZE, BTN_SIZE)

        # Centered button
        mic_icon = QIcon("assets/images/microphone.png")
        self.mic_btn.setIcon(mic_icon)
        self.mic_btn.setIconSize(QSize(22, 22))

        # Based styles
        self.mic_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(190, 70, 70, 210);   /* rojo suave */
                border: 1px solid rgba(255, 255, 255, 120);
                border-radius: 23px; /* la mitad de 46 */
                padding: 8px;
            }
            QPushButton:hover {
                background-color: rgba(210, 85, 85, 220);
            }
            QPushButton:pressed {
                background-color: rgba(80, 170, 110, 230);  /* verde suave mientras presionas */
            }
        """)

        self.status_label = QLabel("Estado: Listo.")
        self.status_label.setStyleSheet("color: #E9F2FF; font-size: 12px;")
        self.mic_info_label = QLabel("")
        self.mic_info_label.setStyleSheet("color: rgba(233, 242, 255, 180); font-size: 12px;")

        top_row.addWidget(self.mic_btn)
        top_row.addStretch(1)
        top_row.addWidget(self.mic_info_label)
        top_row.addSpacing(12)
        top_row.addWidget(self.status_label)

        layout.addLayout(top_row)

        # Connections (per now, just for demo)
        self.mic_btn.pressed.connect(self.start_recording)
        self.mic_btn.released.connect(self.stop_recording)
        self.set_mic_idle_style()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        w = self.sky.width()
        h = self.sky.height()

        ow = int(w * 0.75)
        oh = int(h * 0.45)

        ox = int((w - ow) / 2)
        oy = int(h * 0.42)

        self.overlay.setGeometry(ox, oy, ow, oh)

    def start_recording(self):
        self.set_mic_recording_style()

        if self.is_recording:
            return

        self.is_recording = True
        self.audio_frames = []
        self.status_label.setText("Estado: Grabando...")

        def callback(indata, frames, time_info, status):
            if self.is_recording:
                self.audio_frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=callback
        )
        self.stream.start()

    def stop_recording(self):
        self.mic_btn.setEnabled(False)
        self.status_label.setText("Estado: Procesando...")

        if not self.is_recording:
            return

        self.is_recording = False

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None

        # Process the captured audio (no yet implemented)
        if not self.audio_frames:
            self.mic_info_label.setText("Audio: —")
            self.status_label.setText("Estado: Listo.")
            return

        audio = np.concatenate(self.audio_frames, axis=0)
        duration = len(audio) / float(self.samplerate)
        peak = float(np.max(np.abs(audio)))

        self.mic_info_label.setText(f"Audio: {duration:.2f}s | pico={peak:.3f}")
        self.status_label.setText("Estado: Listo.")
        self.set_mic_idle_style()

        self.mic_btn.setEnabled(True)
        self.status_label.setText("Estado: Listo.")


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
