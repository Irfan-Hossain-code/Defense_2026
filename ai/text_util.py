"""Strip reasoning / think tags from model output."""

from __future__ import annotations

import re

_REASONING = re.compile(
    r"<\s*(?:redacted_reasoning|think)\b[^>]*>[\s\S]*?<\s*/\s*(?:redacted_reasoning|think)\s*>",
    re.IGNORECASE,
)
_THINKING_LINE = re.compile(
    r"^(?:thinking|reasoning|analysis|let me|i need to|the user wants|okay,? so)\b",
    re.IGNORECASE,
)


def clean_model_output(text: str, max_words: int = 24) -> str:
    if not text:
        return ""
    t = _REASONING.sub("", text.strip())
    t = re.sub(r"```[\s\S]*?```", "", t)
    t = re.sub(r"\*\*", "", t)
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    skip_starts = (
        "got it",
        "let's",
        "i need",
        "the user",
        "okay",
        "so ",
        "hmm",
        "thinking",
        "reasoning",
        "first,",
        "next,",
    )
    candidates = [
        ln
        for ln in lines
        if not any(ln.lower().startswith(s) for s in skip_starts)
        and not _THINKING_LINE.match(ln)
    ]
    t = (candidates[-1] if candidates else (lines[-1] if lines else t)).strip()
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    return t
