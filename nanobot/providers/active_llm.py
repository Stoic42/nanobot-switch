"""Persist which LLM route last completed a request successfully (for ``nanobot status``)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanobot.config.paths import get_runtime_subdir


def _state_path() -> Path:
    return get_runtime_subdir("state") / "active_llm.json"


def write_active_llm_route(route: str) -> None:
    """Record the route label (e.g. ``minimax/MiniMax-M2.7``) after a successful model response."""
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"route": route, "ts": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_active_llm_state() -> dict[str, Any] | None:
    """Return parsed ``active_llm.json`` or ``None`` if missing/unreadable."""
    path = _state_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
