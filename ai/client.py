"""ConfidentialMind + Gemini clients."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional

from openai import OpenAI

from .prompts import EVENT_INTENTS, NARRATOR_SYSTEM, QA_SYSTEM, VISION_SYSTEM
from .text_util import clean_model_output


def _cm_client() -> Optional[OpenAI]:
    # Official hackathon .env uses BASE_URL + API_KEY (see junction-defence-2026)
    base = (
        os.getenv("CONFIDENTIALMIND_BASE_URL")
        or os.getenv("BASE_URL")
        or ""
    ).strip()
    key = (
        os.getenv("CONFIDENTIALMIND_API_KEY")
        or os.getenv("API_KEY")
        or ""
    ).strip()
    if not base or not key:
        return None
    return OpenAI(base_url=base, api_key=key)


def narrate_event(event_type: str, facts: dict[str, Any]) -> str:
    """Generate one tactical line for an event (ConfidentialMind)."""
    client = _cm_client()
    intent = EVENT_INTENTS.get(event_type, "Give a brief tactical situational update.")
    model = os.getenv("CM_MODEL_NARRATION", "Gemma 4-fovcnlriirydcgilvaix")

    user = (
        f"Event: {event_type}\n"
        f"Required meaning: {intent}\n"
        f"Facts: {json.dumps(facts, default=str)}\n"
        "Do not invent fields missing from Facts."
    )

    if client is None:
        return fallback_line(event_type, facts)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": NARRATOR_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=100,
            temperature=0.25,
        )
        text = clean_model_output(resp.choices[0].message.content or "")
        return text or fallback_line(event_type, facts)
    except Exception as exc:
        print(f"[narrator] CM API error: {exc}")
        return fallback_line(event_type, facts)


def describe_target_image(b64_jpeg: str, subject_id: str = "") -> str:
    """Clothing profile from still — Gemini preferred (no thinking leak)."""
    label = subject_id.replace("_", " ").title() if subject_id else "this person"
    gemini = _gemini_vision(b64_jpeg, label)
    if gemini:
        return gemini

    cm = _cm_client()
    model = os.getenv("CM_MODEL_VISION", "Qwen3-Omni-30B-A3B-Thinking-vcyznndokaubolqaavrr")
    if cm is None:
        return "Unknown attire"

    data_uri = f"data:image/jpeg;base64,{b64_jpeg}"
    try:
        resp = cm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VISION_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Clothing and build only. One line."},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            max_tokens=80,
            temperature=0.1,
        )
        cleaned = clean_model_output(resp.choices[0].message.content or "", max_words=15)
        return cleaned or "Attire not determined"
    except Exception as exc:
        print(f"[narrator] vision API error: {exc}")
        return "Unknown attire"


def _gemini_vision(b64_jpeg: str, subject_label: str = "this person") -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        return ""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        model = os.getenv("GEMINI_VISION_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
        prompt = (
            f"{VISION_SYSTEM}\n"
            f"Describe the person in the image (identified as {subject_label}). "
            "Reply with ONLY the description line. No thinking, no markdown."
        )
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=base64.b64decode(b64_jpeg), mime_type="image/jpeg"),
            ],
        )
        cleaned = clean_model_output(resp.text or "", max_words=15)
        return cleaned
    except Exception as exc:
        print(f"[narrator] Gemini vision: {exc}")
        return ""


def answer_question(question: str, snapshot: dict[str, Any]) -> str:
    """Operator Q&A — Gemini Flash preferred, CM fallback."""
    # Try JSON shortcuts first
    quick = _answer_from_store(question, snapshot)
    if quick is not None:
        return quick

    gemini = _gemini_answer(question, snapshot)
    if gemini:
        return gemini

    client = _cm_client()
    if client is None:
        return "No data on file."

    model = os.getenv("CM_MODEL_NARRATION", "Gemma 4-fovcnlriirydcgilvaix")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QA_SYSTEM},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nTarget Context:\n{json.dumps(snapshot, default=str)}",
                },
            ],
            max_tokens=120,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "No data on file.").strip()
    except Exception as exc:
        print(f"[voice] QA error: {exc}")
        return "Comms degraded. No response available."


def _gemini_answer(question: str, snapshot: dict[str, Any]) -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        return ""
    try:
        from google import genai

        client = genai.Client(api_key=key)
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        prompt = (
            f"{QA_SYSTEM}\n\nQuestion: {question}\n\n"
            f"Target Context:\n{json.dumps(snapshot, default=str)}"
        )
        resp = client.models.generate_content(model=model, contents=prompt)
        return clean_model_output(resp.text or "", max_words=30)
    except Exception as exc:
        print(f"[voice] Gemini error: {exc}")
        return ""


def _answer_from_store(question: str, snapshot: dict[str, Any]) -> Optional[str]:
    q = question.lower()
    t = snapshot.get("target", {})
    appearance = t.get("appearance") or {}
    summary = appearance.get("summary") or ""

    dossier = snapshot.get("dossier") or {}
    dossier_desc = (dossier.get("description") or "").strip()

    if any(w in q for w in ("wearing", "clothes", "attire", "dress", "outfit", "look like", "description")):
        if dossier_desc and dossier_desc != "Pending optical profile":
            return dossier_desc
        return summary if summary else "No optical profile on file."

    if any(w in q for w in ("who", "name", "identify", "target")):
        name = t.get("name", "UNKNOWN")
        conf = t.get("identity_confidence", 0)
        return f"Target is {name}, {conf:.0%} confidence."

    if any(w in q for w in ("rgb", "camera", "visual", "see")):
        if t.get("sensor") == "RGB":
            return "RGB visual tracking is active."
        return "RGB offline. RF fallback tracking general location."

    if any(w in q for w in ("rf", "radio", "behind", "wall", "sight", "occluded")):
        if t.get("sensor") == "RF":
            return "Line of sight lost. RF tracking general location."
        return "Visual line of sight is clear."

    if any(w in q for w in ("where", "position", "location")):
        p = t.get("position") or {}
        return f"Last fix ({p.get('cx', 0)}, {p.get('cy', 0)}). Sensor {t.get('sensor', '?')}."

    return None


def fallback_line(event_type: str, facts: dict[str, Any]) -> str:
    name = facts.get("name", "UNKNOWN")
    conf = facts.get("identity_confidence", 0)
    if event_type == "TARGET_ACQUIRED":
        return f"Target acquired — {name}, {conf:.0%}."
    if event_type == "LOS_LOST":
        return "Visual lost. RF track active."
    if event_type == "VISUAL_REACQUIRED":
        return f"Visual on {name}."
    if event_type == "PROFILE_LOGGED":
        appearance = (facts.get("appearance") or "").strip()
        label = facts.get("name") or "Target"
        if appearance:
            return f"{label} — {appearance}"
        return "Optical profile updated."
    if event_type == "RF_ACTIVE":
        return f"RF telemetry active. Variance {facts.get('rf_variance', 0):.2f}."
    if event_type == "SQUAD_URGENT":
        team = facts.get("team", "UNKNOWN")
        msg = facts.get("message", "urgent request")
        return f"{team}: {msg}."
    return "Standing by."
