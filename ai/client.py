from typing import Dict, List

from openai import OpenAI


class AIClient:
    """OpenAI wrapper for transcription and chat."""

    def __init__(self, api_key: str, chat_model: str):
        self.api_key = api_key
        self.chat_model = chat_model
        self.client = OpenAI(api_key=api_key)

    def transcribe_wav(self, file_obj, language: str = "es") -> str:
        result = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=file_obj,
            language=language,
        )
        return (getattr(result, "text", "") or "").strip()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
        )
        return (resp.choices[0].message.content or "").strip()
