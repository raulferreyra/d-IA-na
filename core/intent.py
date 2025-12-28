import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Intent:
    kind: str
    command: Optional[str]
    format: Optional[str]
    style: Optional[str]
    payload: str
    raw: str


def parse_intent(transcript: str) -> Intent:
    raw = (transcript or "").strip()
    t = re.sub(r"\s+", " ", raw).strip()

    activation_pattern = r"^(diana|d_ia_na|diana\.|diana:|diana,)\s*"
    activation = bool(re.match(activation_pattern, t, flags=re.IGNORECASE))
    if activation:
        t = re.sub(activation_pattern, "", t, flags=re.IGNORECASE).strip()

    fmt = "markdown" if re.search(
        r"\b(markdown|md|\.md)\b", t, flags=re.IGNORECASE) else None

    style = None
    if re.search(r"\b(viñetas|bullets|bullet points)\b", t, flags=re.IGNORECASE):
        style = "bullets"
    elif re.search(r"\b(pasos|step by step|paso a paso)\b", t, flags=re.IGNORECASE):
        style = "numbered"

    if re.search(r"\b(corto|breve|short)\b", t, flags=re.IGNORECASE):
        style = (style + "+short") if style else "short"
    if re.search(r"\b(detallado|largo|detailed)\b", t, flags=re.IGNORECASE):
        style = (style + "+detailed") if style else "detailed"

    cmd = None
    command_map = [
        (r"\b(resume|resumen)\b", "summarize"),
        (r"\b(explica|explain)\b", "explain"),
        (r"\b(siguientes pasos|next steps|pasos)\b", "next_steps"),
        (r"\b(traduce|translate)\b", "translate"),
        (r"\b(mejora|refactoriza|refactor)\b", "refactor"),
        (r"\b(debug|arregla|fix)\b", "debug"),
        (r"\b(corrige|corregir)\b", "correct"),
        (r"\b(código|codigo|code)\b", "developer"),
        (r"\b(tests|pruebas)\b", "tests"),
    ]
    for pattern, name in command_map:
        if re.search(pattern, t, flags=re.IGNORECASE):
            cmd = name
            break

    looks_like_question = (
        "?" in t
        or re.match(
            r"^(qué|que|cómo|como|por qué|porque|cuál|cual|dónde|donde|para qué|cuando)\b",
            t,
            flags=re.IGNORECASE,
        )
    )

    if activation and cmd:
        kind = "command"
    elif looks_like_question and not cmd:
        kind = "question"
    elif activation and not cmd:
        kind = "question" if looks_like_question else "command"
    elif cmd and not looks_like_question:
        kind = "command"
    else:
        kind = "dictation"

    return Intent(
        kind=kind,
        command=cmd,
        format=fmt,
        style=style,
        payload=t.strip(),
        raw=raw,
    )
