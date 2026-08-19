"""Swap vivid red garment pixels to #14B8A6 — avoids skin/face tones."""
from __future__ import annotations

import sys
from colorsys import hsv_to_rgb, rgb_to_hsv
from pathlib import Path

import numpy as np
from PIL import Image

TARGET_HEX = "#14B8A6"
TARGET_RGB = (0x14, 0xB8, 0xA6)


def rgb_to_hsv_array(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rf = rgb[..., 0].astype(np.float32) / 255.0
    gf = rgb[..., 1].astype(np.float32) / 255.0
    bf = rgb[..., 2].astype(np.float32) / 255.0
    maxc = np.maximum(np.maximum(rf, gf), bf)
    minc = np.minimum(np.minimum(rf, gf), bf)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    chroma = delta > 1e-6
    rmax = chroma & (maxc == rf)
    gmax = chroma & (maxc == gf)
    bmax = chroma & (maxc == bf)

    hue[rmax] = ((gf[rmax] - bf[rmax]) / delta[rmax]) % 6.0
    hue[gmax] = (bf[gmax] - rf[gmax]) / delta[gmax] + 2.0
    hue[bmax] = (rf[bmax] - gf[bmax]) / delta[bmax] + 4.0
    hue = hue / 6.0

    sat = np.divide(delta, maxc, out=np.zeros_like(maxc), where=maxc > 0)
    val = maxc
    return hue, sat, val


def hsv_to_rgb_array(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    out = np.zeros((*h.shape, 3), dtype=np.float32)
    for idx, (r, g, b) in enumerate([
        (v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q),
    ]):
        m = i == idx
        out[m, 0] = r[m]
        out[m, 1] = g[m]
        out[m, 2] = b[m]

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def red_garment_mask(rgb: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """Target saturated red fabric only — not skin, lips, or warm highlights."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)

    return (
        (r > 108)
        & (r - g > 58)
        & (r - b > 62)
        & (g < r * 0.58)
        & (b < r * 0.50)
        & (sat > 0.43)
    )


def swap_red_tones(img: Image.Image) -> tuple[Image.Image, int]:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    hue, sat, val = rgb_to_hsv_array(rgb)
    mask = red_garment_mask(rgb, sat)
    changed = int(np.sum(mask))

    target_h, target_s, _ = rgb_to_hsv(
        TARGET_RGB[0] / 255.0,
        TARGET_RGB[1] / 255.0,
        TARGET_RGB[2] / 255.0,
    )

    new_h = np.where(mask, target_h, hue)
    new_s = np.where(mask, np.maximum(target_s * 0.55, sat * 0.92), sat)
    new_v = val

    out_rgb = hsv_to_rgb_array(new_h, new_s, new_v)
    out = np.dstack([out_rgb, alpha])
    return Image.fromarray(out, "RGBA"), changed


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "src/holder/random-ceo-lady.webp"
    )
    if not src.is_file():
        print(f"Missing file: {src}", file=sys.stderr)
        return 1

    img = Image.open(src)
    result, changed = swap_red_tones(img)
    result.save(src, "WEBP", quality=92, method=6)
    print(f"Updated {src} ({changed} garment-red pixels remapped to {TARGET_HEX})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
