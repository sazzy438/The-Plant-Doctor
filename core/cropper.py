"""Server-side crop selection helpers: draw the selection on a preview and manage the box state."""
from __future__ import annotations

from PIL import Image, ImageDraw

import config

OUTLINE = (31, 77, 46)
HANDLE = (255, 255, 255)


def box_from_percent(image: Image.Image, left: float, top: float, right: float, bottom: float) -> tuple[int, int, int, int]:
    w, h = image.size
    x0, x1 = sorted((left, right)); y0, y1 = sorted((top, bottom))
    return (round(w * x0 / 100), round(h * y0 / 100), round(w * x1 / 100), round(h * y1 / 100))


def percent_from_box(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    w, h = image.size
    x0, y0, x1, y1 = box
    return (100 * x0 / w, 100 * y0 / h, 100 * x1 / w, 100 * y1 / h)


def is_full(image: Image.Image, box: tuple[int, int, int, int] | None) -> bool:
    if box is None:
        return True
    w, h = image.size
    return box == (0, 0, w, h)


def draw_selection(image: Image.Image, box: tuple[int, int, int, int] | None, pending: tuple[int, int] | None = None,
                   max_side: int = 1024) -> Image.Image:
    """Return a preview with the area outside ``box`` dimmed and the box outlined."""
    preview = image.copy()
    preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    sx, sy = preview.width / image.width, preview.height / image.height
    if box is None and pending is None:
        return preview
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    lw = max(2, round(min(preview.size) / 160))
    if box is not None:
        x0, y0, x1, y1 = [round(v * s) for v, s in zip(box, (sx, sy, sx, sy))]
        draw.rectangle((0, 0, preview.width, preview.height), fill=(10, 20, 12, 120))
        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 0))
        draw.rectangle((x0, y0, x1, y1), outline=OUTLINE + (255,), width=lw)
        r = lw * 3
        for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=HANDLE + (255,), outline=OUTLINE + (255,), width=max(1, lw // 2))
    if pending is not None:
        px, py = round(pending[0] * sx), round(pending[1] * sy)
        r = lw * 4
        draw.ellipse((px - r, py - r, px + r, py + r), outline=OUTLINE + (255,), width=lw)
        draw.line((px - r * 2, py, px + r * 2, py), fill=OUTLINE + (255,), width=max(1, lw // 2))
        draw.line((px, py - r * 2, px, py + r * 2), fill=OUTLINE + (255,), width=max(1, lw // 2))
    out = preview.convert("RGBA")
    out.alpha_composite(overlay)
    return out.convert("RGB")
