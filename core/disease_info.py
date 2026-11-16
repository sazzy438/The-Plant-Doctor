"""Access to the structured disease information database."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import config


@lru_cache(maxsize=1)
def _load(path: str = str(config.DISEASE_INFO_PATH)) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "classes" not in data:
        raise ValueError("disease_information.json is missing the 'classes' section")
    return data


def disclaimer() -> str:
    return _load()["_meta"]["disclaimer"]


def sources_note() -> str:
    return _load()["_meta"]["sources_note"]


def all_classes() -> dict[str, dict]:
    return _load()["classes"]


def get_info(class_name: str) -> dict:
    """Return the entry for a model class. Raises ``KeyError`` if the class is unknown."""
    entry = _load()["classes"][class_name]
    return {**entry, "class_name": class_name}


def crops_summary() -> dict[str, list[str]]:
    """{crop: [condition, ...]} for the About page and landing page."""
    out: dict[str, list[str]] = {}
    for name, entry in _load()["classes"].items():
        out.setdefault(entry["crop"], []).append(entry["common_name"])
    return dict(sorted(out.items()))
