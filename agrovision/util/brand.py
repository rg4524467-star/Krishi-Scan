"""
Krishi Scan brand assets.

`make_logo(size)` draws the platform logo with pure PIL primitives only
(downsampled from a modest supersample for crisp edges), so the exact same mark
embeds in the web header and the offline/Pyodide-safe PDF report without
shipping external images and WITHOUT depending on numpy (the stlite/Pyodide
package set is reduced).  Every PIL call uses only APIs present in the Pillow
versions shipped by Pyodide; anything risky is wrapped so the logo can never
break the report.
"""
from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw

BRAND_NAME = "Krishi Scan"
BRAND_TAGLINE_EN = "Crop disease detection for Indian farmers"

GREEN = (11, 94, 59)
GREEN_HI = (28, 132, 90)
GOLD = (240, 180, 60)
WHITE = (255, 255, 255)
SOFT_WHITE = (220, 240, 230)

_LOGO_CACHE: dict = {}
_SS = 4                                    # supersample factor for crisp edges


def _antialias() -> int:
    """Resampling filter constant, works across Pillow versions."""
    for name in ("LANCZOS", "ANTIALIAS"):
        if hasattr(Image, name):
            return getattr(Image, name)
    return 1


def _rounded(img: Image.Image, radius: float) -> Image.Image:
    """Apply a rounded-corner alpha mask.  Falls back to a square crop."""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    try:
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=int(radius), fill=255)
    except Exception:
        d.rectangle([0, 0, w - 1, h - 1], fill=255)
    img.putalpha(mask)
    return img


def _gradient_tile(px: int) -> Image.Image:
    """Rounded green tile with a soft diagonal gradient (pure PIL)."""
    img = Image.new("RGB", (px, px))
    d = ImageDraw.Draw(img)
    # Interpolate GREEN_HI (top-left) -> GREEN (bottom-right) per row.
    for y in range(px):
        t = y / max(1, px - 1)
        row = tuple(round(GREEN_HI[i] + (GREEN[i] - GREEN_HI[i]) * t)
                    for i in range(3))
        d.line([(0, y), (px, y)], fill=row)
    # lighten towards the top-left corner for a subtle diagonal sheen
    sheen = Image.new("L", (px, px), 0)
    sd = ImageDraw.Draw(sheen)
    for x in range(px):
        t = 1 - x / max(1, px - 1)
        sd.line([(x, 0), (x, px)], fill=int(16 * t))
    img = Image.composite(img, img.point(lambda p: min(255, p + 16)),
                          sheen)
    return _rounded(img, 0.20 * px)


def _leaf_points(px: int) -> list:
    """Pointed-tip leaf silhouette: pointed top & bottom, bulging sides."""
    cx = cy = px / 2
    rx, ry = 0.24 * px, 0.34 * px
    n = 48
    pts = []
    for i in range(n + 1):
        u = -1 + 2 * i / n
        w = rx * max(0.0, 1 - u * u) ** 0.65
        pts.append((cx + w, cy + ry * u))
    for i in range(n - 1, 0, -1):
        u = -1 + 2 * i / n
        w = rx * max(0.0, 1 - u * u) ** 0.65
        pts.append((cx - w, cy + ry * u))
    return pts


def _draw_mark(px: int) -> Image.Image:
    """The leaf + scan-line mark drawn on a transparent canvas at final px."""
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = px / 2

    # leaf
    d.polygon(_leaf_points(px), fill=WHITE)

    # central vein
    vein_w = max(1, int(round(0.018 * px)))
    d.line([(cx, cy - 0.30 * px), (cx, cy + 0.30 * px)],
           fill=GREEN, width=vein_w)

    # scan line (horizontal beam above centre) with cursor ticks
    scan_y = cy - 0.20 * px
    half = 0.34 * px
    beam_w = max(2, int(round(0.03 * px)))
    d.line([(cx - half, scan_y), (cx + half, scan_y)], fill=GOLD, width=beam_w)
    tick_w = max(2, int(round(0.022 * px)))
    for sx in (cx - half, cx + half):
        d.line([(sx, scan_y - 0.10 * px), (sx, scan_y + 0.10 * px)],
               fill=GOLD, width=tick_w)
    return img


def _corner_brackets(img: Image.Image) -> None:
    """Viewfinder corner brackets that frame the tile (scan motif)."""
    px = img.width
    d = ImageDraw.Draw(img)
    inset = 0.11 * px
    arm = 0.13 * px
    t = max(2, int(round(0.028 * px)))
    for (bx, by, dx, dy) in (
        (inset, inset, 1, 1), (px - inset, inset, -1, 1),
        (inset, px - inset, 1, -1), (px - inset, px - inset, -1, -1),
    ):
        d.line([(bx, by), (bx + dx * arm, by)], fill=GOLD, width=t)
        d.line([(bx, by), (bx, by + dy * arm)], fill=GOLD, width=t)


def make_logo(size: int = 96) -> Image.Image:
    """
    Full logo tile: green rounded tile + white leaf with a gold scan beam and
    viewfinder brackets.  Rendered once per size and cached.
    """
    if size in _LOGO_CACHE:
        return _LOGO_CACHE[size].copy()
    px = size * _SS
    tile = _gradient_tile(px)
    mark = _draw_mark(px)
    _corner_brackets(mark)
    base = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    base.alpha_composite(tile)
    base.alpha_composite(mark)
    out = base.resize((size, size), _antialias())
    _LOGO_CACHE[size] = out.copy()
    return out


def logo_png_bytes(size: int = 96) -> bytes:
    buf = io.BytesIO()
    make_logo(size).save(buf, format="PNG")
    return buf.getvalue()


def logo_data_uri(size: int = 96) -> str:
    """base64 data URI for embedding in HTML/CSS."""
    b64 = base64.b64encode(logo_png_bytes(size)).decode("ascii")
    return f"data:image/png;base64,{b64}"


def logo_svg(size: int = 96) -> str:
    """Standalone inline SVG of the logo (same PNG bytes as web/PDF)."""
    b64 = base64.b64encode(logo_png_bytes(size)).decode("ascii")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
            f'height="{size}" viewBox="0 0 {size} {size}">'
            f'<image href="data:image/png;base64,{b64}" '
            f'width="{size}" height="{size}"/></svg>')