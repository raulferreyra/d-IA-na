import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    icon_path: Path
    mic_icon_path: Path

    openai_key: str
    chat_model: str

    audio_sample_rate: int
    audio_channels: int

    btn_size: int
    btn_icon_size: int

    overlay_margin_lr: int
    overlay_margin_tb: int
    overlay_width_ratio: float
    overlay_height_ratio: float
    overlay_top_ratio: float

    context_max_turns: int
    debug_intent_to_textbox: bool


def load_config() -> AppConfig:
    base_dir = Path(__file__).resolve().parents[1]

    return AppConfig(
        base_dir=base_dir,
        icon_path=base_dir / "assets" / "icons" / "moon.ico",
        mic_icon_path=base_dir / "assets" / "images" / "microphone.png",
        openai_key=os.getenv("OPENAI_API_KEY", "").strip(),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
        audio_sample_rate=16000,
        audio_channels=1,
        btn_size=46,
        btn_icon_size=22,
        overlay_margin_lr=22,
        overlay_margin_tb=16,
        overlay_width_ratio=0.72,
        overlay_height_ratio=0.45,
        overlay_top_ratio=0.42,
        context_max_turns=8,
        debug_intent_to_textbox=True,
    )
