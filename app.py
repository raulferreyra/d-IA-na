import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QMainWindow

'''import os
import openai

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

audio_file = open("audio.mp3", "rb")
transcribed = openai.Audio.transcribe("whisper-1", audio_file)
print("Resultado: ", transcribed.text)'''

def main():
    app = QApplication(sys.argv)

    win = QMainWindow()
    win.setWindowTitle("D_IA_NA")
    win.setMinimumSize(820, 520)

    palette = win.palette()
    palette.setColor(QPalette.Window, QColor("#061225"))
    win.setAutoFillBackground(True)
    win.setPalette(palette)

    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
