import sys
import numpy as np

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
)

'''import os
import openai

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

audio_file = open("audio.mp3", "rb")
transcribed = openai.Audio.transcribe("whisper-1", audio_file)
print("Resultado: ", transcribed.text)'''

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
# Ventana principal
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("D_IA_NA")
        self.setMinimumSize(820, 520)

        self.sky = NightSky()
        self.setCentralWidget(self.sky)

        # Overlay (panel encima del cielo)
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

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setText(
            "Hola 🌙\n\n"
            "Diana está conectada ( ͡° ͜ʖ ͡°).\n"
            "Esta caja mostrará el texto transcrito del audio."
        )
        layout.addWidget(self.text_box)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        w = self.sky.width()
        h = self.sky.height()

        ow = int(w * 0.75)
        oh = int(h * 0.45)

        ox = int((w - ow) / 2)
        oy = int(h * 0.42)

        self.overlay.setGeometry(ox, oy, ow, oh)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()