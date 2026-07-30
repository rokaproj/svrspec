"""Generate the app icon with the standard library only.

No Pillow, no checked-in binary blob whose provenance nobody can audit: the .ico
is drawn here in a few lines, so the mark is reviewable as code.

The mark is three descending bars — the three-tier recommendation (minimum,
recommended, comfortable) that the whole tool exists to produce. Drawn in the
design system's accent colour so the icon and the UI agree.
"""

from __future__ import annotations

import struct
from pathlib import Path

SIZES = (16, 24, 32, 48, 64, 128, 256)

#: --accent from the light theme, and its dark-theme counterpart for the mark.
ACCENT = (0x2F, 0x5F, 0xA8)
BAR = (0xF4, 0xF4, 0xF1)


def _rounded(x: int, y: int, n: int, radius: float) -> bool:
    """Inside a rounded square of side n?"""
    cx = min(x, n - 1 - x)
    cy = min(y, n - 1 - y)
    if cx >= radius or cy >= radius:
        return True
    dx, dy = radius - cx, radius - cy
    return dx * dx + dy * dy <= radius * radius


def _pixels(n: int) -> list[tuple[int, int, int, int]]:
    """BGRA rows, bottom-up, as a BMP inside an ICO expects."""
    radius = max(2.0, n * 0.22)
    # Three bars of descending width, evenly spaced in the middle band.
    unit = n / 16.0
    bar_h = max(1, round(unit * 1.9))
    gap = max(1, round(unit * 1.35))
    left = round(unit * 3.4)
    widths = [round(unit * 9.2), round(unit * 6.6), round(unit * 4.0)]
    block = len(widths) * bar_h + (len(widths) - 1) * gap
    top = round((n - block) / 2)

    bars = []
    for i, w in enumerate(widths):
        y0 = top + i * (bar_h + gap)
        bars.append((y0, y0 + bar_h, left, left + w))

    out: list[tuple[int, int, int, int]] = []
    for y in range(n - 1, -1, -1):  # bottom-up
        for x in range(n):
            if not _rounded(x, y, n, radius):
                out.append((0, 0, 0, 0))
                continue
            colour = ACCENT
            for y0, y1, x0, x1 in bars:
                if y0 <= y < y1 and x0 <= x < x1:
                    colour = BAR
                    break
            out.append((colour[2], colour[1], colour[0], 255))
    return out


def _png_free_bmp(n: int) -> bytes:
    """A 32bpp BITMAPINFOHEADER image, as embedded in classic .ico entries."""
    header = struct.pack(
        "<IiiHHIIiiII",
        40,          # biSize
        n,           # biWidth
        n * 2,       # biHeight: doubled, colour + (empty) mask
        1,           # biPlanes
        32,          # biBitCount
        0,           # biCompression = BI_RGB
        0, 0, 0, 0, 0,
    )
    body = b"".join(struct.pack("<4B", *px) for px in _pixels(n))
    # AND mask: one bit per pixel, rows padded to 4 bytes. Fully opaque, so
    # every bit is zero -- but the bytes must be present.
    stride = ((n + 31) // 32) * 4
    mask = b"\x00" * (stride * n)
    return header + body + mask


def build(path: Path, sizes: tuple[int, ...] = SIZES) -> Path:
    images = [(n, _png_free_bmp(n)) for n in sizes]

    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for n, blob in images:
        entries += struct.pack(
            "<BBBBHHII",
            0 if n >= 256 else n,   # width, 0 means 256
            0 if n >= 256 else n,   # height
            0,                      # palette count
            0,                      # reserved
            1,                      # colour planes
            32,                     # bits per pixel
            len(blob),
            offset,
        )
        blobs += blob
        offset += len(blob)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<HHH", 0, 1, len(images)) + entries + blobs)
    return path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "svrspec.ico"
    out = build(target)
    print(f"{out} ({out.stat().st_size:,} bytes, {len(SIZES)} sizes)")
