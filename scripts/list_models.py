#!/usr/bin/env python3
"""List ConfidentialMind models on your endpoint. Run from repo root."""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def main() -> int:
    base_url = os.getenv("BASE_URL", "").strip()
    api_key = os.getenv("API_KEY", "").strip()
    if not base_url or not api_key:
        print("Set BASE_URL and API_KEY in .env", file=sys.stderr)
        return 1

    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    models = r.json().get("data", [])

    print(f"Found {len(models)} model(s):\n")
    for m in models:
        mid = m.get("id", "?")
        display = m.get("display_name") or mid
        print(f"  {display}")
        print(f"    id: {mid}\n")
    print("Copy the id values into .env as CM_MODEL_NARRATION and CM_MODEL_VISION")
    return 0


if __name__ == "__main__":
    sys.exit(main())
