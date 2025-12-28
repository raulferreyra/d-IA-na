import json
from pathlib import Path

from core.settings_model import UserSettings


class JsonSettingsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> UserSettings:
        if not self.path.exists():
            return UserSettings()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return UserSettings.from_dict(data)
        except Exception:
            return UserSettings()

    def save(self, settings: UserSettings) -> None:
        self.path.write_text(
            json.dumps(settings.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
