from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_profile(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Profile file must contain a mapping: {path}")
    return data
