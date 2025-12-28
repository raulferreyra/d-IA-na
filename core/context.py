from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List


@dataclass
class Turn:
    role: str
    content: str


class ConversationContext:
    def __init__(self, max_turns: int):
        self.turns: Deque[Turn] = deque(maxlen=max_turns)

    def add_user(self, text: str) -> None:
        self.turns.append(Turn(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.turns.append(Turn(role="assistant", content=text))

    def as_messages(self) -> List[Dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def last_assistant_text(self) -> str:
        for t in reversed(self.turns):
            if t.role == "assistant":
                return t.content
        return ""
