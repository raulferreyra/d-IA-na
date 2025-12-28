from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class UserSettings:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    tts_enabled: bool = False
    tts_volume: int = 35
    language: str = "es"
    persona: str = "técnico,directo"

    templates: Dict[str, str] = field(default_factory=dict)
    persona_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "tts_enabled": self.tts_enabled,
            "tts_volume": int(self.tts_volume),
            "language": self.language,
            "persona": self.persona,
            "templates": dict(self.templates),
            "persona_tags": list(self.persona_tags),
        }

    @staticmethod
    def from_dict(d: dict) -> "UserSettings":
        d = d or {}
        return UserSettings(
            provider=d.get("provider", "openai"),
            model=d.get("model", "gpt-4o-mini"),
            api_key=d.get("api_key", ""),
            tts_enabled=bool(d.get("tts_enabled", False)),
            tts_volume=int(d.get("tts_volume", 35)),
            language=d.get("language", "es"),
            persona=d.get("persona", "técnico,directo"),
            templates=d.get("templates", {}) or {},
            persona_tags=d.get("persona_tags", []) or [],
        )
