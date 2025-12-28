from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class ProviderSpec:
    name: str
    chat_models: list[str]


class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, ProviderSpec] = {
            "openai": ProviderSpec(name="openai", chat_models=["gpt-4o-mini", "gpt-4.1-mini", "gpt-5.2"]),
            "anthropic": ProviderSpec(name="anthropic", chat_models=["claude-3.5-sonnet"]),
            "deepseek": ProviderSpec(name="deepseek", chat_models=["deepseek-chat"]),
            "local": ProviderSpec(name="local", chat_models=["local-default"]),
        }

    def get(self, name: str) -> Optional[ProviderSpec]:
        return self._providers.get((name or "").strip().lower())

    def list_names(self) -> list[str]:
        return list(self._providers.keys())
