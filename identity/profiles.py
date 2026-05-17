"""Static suspect profiles — loaded from data/profiles/<id>.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


class ProfileStore:
    def __init__(self, directory: Optional[str] = None) -> None:
        path = directory or os.getenv("PROFILE_DB_PATH", "data/profiles")
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        count = 0
        for fpath in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[profiles] skip {fpath.name}: {exc}")
                continue
            pid = (data.get("id") or fpath.stem).lower().replace(" ", "_")
            data["id"] = pid
            self._cache[pid] = data
            count += 1
        print(f"[profiles] Loaded {count} profile(s) from {self._dir}")

    def get(self, person_id: str) -> Optional[dict[str, Any]]:
        if not person_id:
            return None
        key = person_id.lower().replace(" ", "_")
        return self._cache.get(key)

    def display_name(self, person_id: str) -> str:
        p = self.get(person_id)
        if p:
            return p.get("display_name") or person_id.replace("_", " ").title()
        return person_id.replace("_", " ").title()

    def description(self, person_id: str) -> str:
        p = self.get(person_id)
        if not p:
            return ""
        return (p.get("description") or "").strip()
