from core.intent import Intent
from core.utils import strip_code_blocks


def build_style_instructions(intent: Intent) -> str:
    parts = []

    if intent.format == "markdown":
        parts.append(
            "Respond in Markdown. "
            "Use '##' headings when relevant. "
            "Keep list indentation consistent and avoid awkward nested lists."
        )

    if intent.style:
        s = intent.style.lower()
        if "bullets" in s:
            parts.append("Use '-' bullet points as the main list format.")
        if "numbered" in s:
            parts.append("Use '1.' numbered steps as the main list format.")
        if "short" in s:
            parts.append("Be concise.")
        if "detailed" in s:
            parts.append("Be detailed.")

    parts.append("Use a technical, clear tone. No emojis.")
    return " ".join(parts).strip()


def normalize_user_text(intent: Intent, ctx_last_assistant: str) -> str:
    p = (intent.payload or "").strip().lower()

    if p in ("continúa", "continua", "sigue", "continue"):
        return f"Continue from here:\n\n{ctx_last_assistant}"

    if p in ("resume lo anterior", "resumen de lo anterior", "resume lo último", "resume lo ultimo"):
        return f"Summarize the following in a few lines:\n\n{ctx_last_assistant}"

    if intent.kind == "question":
        return intent.raw

    if intent.kind == "command":
        cmd = intent.command or "general_command"

        if cmd == "summarize":
            return f"Summarize the following:\n\n{intent.payload}"
        if cmd == "next_steps":
            return f"Provide next steps for:\n\n{intent.payload}"
        if cmd == "explain":
            return f"Explain the following:\n\n{intent.payload}"
        if cmd == "translate":
            return f"Translate the following:\n\n{intent.payload}"
        if cmd == "correct":
            return f"Correct the following text:\n\n{intent.payload}"

        return intent.payload

    return f"[Dictation] {intent.raw}"


def tts_text_from_answer(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    return strip_code_blocks(text)
