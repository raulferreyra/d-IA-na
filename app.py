import sys
from PySide6.QtGui import QPainter, QColor
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

'''import os
import openai

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

audio_file = open("audio.mp3", "rb")
transcribed = openai.Audio.transcribe("whisper-1", audio_file)
print("Resultado: ", transcribed.text)'''

class NightSky(QWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#061225"))

def main():
    app = QApplication(sys.argv)

    win = QMainWindow()
    win.setWindowTitle("D_IA_NA")
    win.setMinimumSize(820, 520)

    win.setCentralWidget(NightSky())

    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
