"""Per-browser-session analysis history (kept in memory via gr.State; nothing is written to disk)."""
from __future__ import annotations

import base64
import io
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from PIL import Image

import config


def _thumb_b64(image: Image.Image, size: int = 96) -> str:
    t = image.copy()
    t.thumbnail((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class HistoryEntry:
    id: str
    timestamp: str
    file_name: str
    prediction: str          # e.g. "Tomato — Early blight" or "No confident result"
    confidence: float | None
    abstained: bool
    thumbnail_b64: str
    payload: dict = field(default_factory=dict)  # everything needed to re-render the result


def new_entry(image: Image.Image, file_name: str, prediction: str, confidence: float | None, abstained: bool, payload: dict) -> HistoryEntry:
    return HistoryEntry(
        id=uuid.uuid4().hex[:10],
        timestamp=datetime.now().strftime("%H:%M:%S"),
        file_name=file_name,
        prediction=prediction,
        confidence=confidence,
        abstained=abstained,
        thumbnail_b64=_thumb_b64(image),
        payload=payload,
    )


def push(history: list[HistoryEntry] | None, entry: HistoryEntry, limit: int = config.HISTORY_LIMIT) -> list[HistoryEntry]:
    history = list(history or [])
    history.insert(0, entry)
    return history[:limit]


def find(history: list[HistoryEntry] | None, entry_id: str) -> HistoryEntry | None:
    for e in history or []:
        if e.id == entry_id:
            return e
    return None
