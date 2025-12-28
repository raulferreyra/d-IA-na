import sys
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
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
