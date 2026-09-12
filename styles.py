"""Per-frame drawing looks. Names are the --style choices."""

from __future__ import annotations

import cv2
import numpy as np

NAMES = (
    "cartoon",
    "comic",
    "paint",
    "neon",
    "sketch",
    "pop",
    "mono",
    "duo",
    "anime",
    "watercolor",
    "oil",
    "halftone",
    "pixel",
    "noir",
    "thermal",
    "vapor",
    "ink",
    "chalk",
    "poster",
    "ice",
    "ember",
    "holo",
    "sepia",
    "blueprint",
    "matrix",
    "glass",
    "pastel",
    "scan",
    "riso",
    "sunset",
    "night",
    "manga",
)

BLURBS = {
    "cartoon": "Flat cel color, soft outlines",
    "comic": "Hard ink, punchy poster color",
    "paint": "Painterly oil-brush restyle",
    "neon": "Dark plate, glowing edge lines",
    "sketch": "Pencil wash over faint color",
    "pop": "Warhol posterize, thick ink",
    "mono": "High-contrast black and white",
    "duo": "Two-tone orange and navy",
    "anime": "Clean anime cel, few colors",
    "watercolor": "Soft wet-paper bleed",
    "oil": "Heavy brush, wet oil look",
    "halftone": "Newsprint comic dots",
    "pixel": "Chunky pixel art",
    "noir": "Ink noir, cold blue shadows",
    "thermal": "Heat-map camera",
    "vapor": "Magenta / cyan vaporwave",
    "ink": "Black ink on cream paper",
    "chalk": "White chalk on a dark board",
    "poster": "Extreme 4-color poster",
    "ice": "Frozen cyan grade",
    "ember": "Fire orange crush",
    "holo": "Iridescent scanline hologram",
    "sepia": "Old film brown",
    "blueprint": "Cyanotype technical print",
    "matrix": "Green phosphor digital",
    "glass": "Stained glass tiles",
    "pastel": "Washed candy colors",
    "scan": "Xerox photocopy",
    "riso": "Pink / teal risograph",
    "sunset": "Orange highlights, purple shade",
    "night": "Teal and orange night grade",
    "manga": "Screentone manga ink",
}


def list_styles() -> str:
    lines = ["Drawing styles (use --style NAME, or --style auto):", ""]
    for name in NAMES:
        lines.append(f"  {name:<12}  {BLURBS.get(name, '')}")
    lines.append("")
    lines.append("  auto          New random style every run (skips the last few).")
    return "\n".join(lines)


def quantize(img: np.ndarray, steps: int) -> np.ndarray:
    q = max(8, 256 // steps)
    return (img // q) * q


def _prep(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = frame.shape[:2]
    scale = 640.0 / max(h, w)
    if scale < 1.0:
        work = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        work = frame
    return work, h, w


def _finish(out: np.ndarray, h: int, w: int) -> np.ndarray:
    if out.shape[0] != h or out.shape[1] != w:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def _gray(work: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)


def _hsv_mul(bgr: np.ndarray, s: float = 1.0, v: float = 1.0, h_add: float = 0.0) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    if h_add:
        hsv[:, :, 0] = (hsv[:, :, 0] + h_add) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _ink_mask(work: np.ndarray, blk: int = 7, c: int = 2, erode: bool = False) -> np.ndarray:
    gray = cv2.medianBlur(_gray(work), 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk, c)
    if erode:
        edges = cv2.erode(edges, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def _toon(work: np.ndarray, steps: int, sat: float, blk: int, c: int, erode: bool) -> np.ndarray:
    color = work
    for _ in range(2):
        color = cv2.bilateralFilter(color, 7, 55, 55)
    color = quantize(color, steps)
    color = _hsv_mul(color, s=sat, v=1.10)
    return cv2.bitwise_and(color, _ink_mask(work, blk, c, erode))


def _s_cartoon(work: np.ndarray) -> np.ndarray:
    return _toon(work, 18, 1.38, 7, 2, False)


def _s_comic(work: np.ndarray) -> np.ndarray:
    return _toon(work, 12, 1.55, 5, 3, True)


def _s_paint(work: np.ndarray) -> np.ndarray:
    return cv2.stylization(work, sigma_s=45, sigma_r=0.35)


def _s_sketch(work: np.ndarray) -> np.ndarray:
    gray = _gray(work)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    dodge = cv2.divide(gray, np.clip(255 - blur, 1, 255), scale=256)
    wash = _hsv_mul(cv2.GaussianBlur(work, (7, 7), 0), s=0.38, v=1.06)
    pencil = cv2.cvtColor(dodge, cv2.COLOR_GRAY2BGR)
    out = cv2.bitwise_and(wash, _ink_mask(work, 9, 2, False))
    return cv2.addWeighted(out, 0.52, pencil, 0.48, 0)


def _s_pop(work: np.ndarray) -> np.ndarray:
    color = work
    for _ in range(2):
        color = cv2.bilateralFilter(color, 9, 80, 80)
    color = _hsv_mul(quantize(color, 8), s=1.85, v=1.16)
    return cv2.bitwise_and(color, _ink_mask(work, 5, 4, True))


def _s_mono(work: np.ndarray) -> np.ndarray:
    gray = cv2.medianBlur(_gray(work), 5)
    poster = quantize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 6)
    out = cv2.bitwise_and(poster, _ink_mask(work, 7, 3, False))
    return cv2.convertScaleAbs(out, alpha=1.25, beta=-12)


def _s_duo(work: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(_gray(work), (5, 5), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = np.zeros_like(work)
    out[bw > 0] = (40, 170, 255)
    out[bw == 0] = (28, 18, 12)
    edges = cv2.Canny(gray, 80, 160)
    out[edges > 0] = (255, 255, 255)
    return out


def _s_neon(work: np.ndarray) -> np.ndarray:
    gray = _gray(work)
    edges = cv2.dilate(cv2.Canny(gray, 70, 160), np.ones((2, 2), np.uint8), iterations=1)
    glow = cv2.GaussianBlur(edges, (0, 0), 2.4)
    dark = (work.astype(np.float32) * 0.22).astype(np.uint8)
    color_edge = np.zeros_like(work)
    color_edge[:, :, 0] = glow
    color_edge[:, :, 1] = (glow * 0.55).astype(np.uint8)
    color_edge[:, :, 2] = np.clip(edges.astype(np.uint16) + glow * 0.4, 0, 255).astype(np.uint8)
    return cv2.add(dark, color_edge)


def _s_anime(work: np.ndarray) -> np.ndarray:
    color = work
    for _ in range(3):
        color = cv2.bilateralFilter(color, 9, 70, 70)
    color = _hsv_mul(quantize(color, 10), s=1.28, v=1.08)
    gray = cv2.medianBlur(_gray(work), 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 4)
    edges = cv2.erode(edges, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))


def _s_watercolor(work: np.ndarray) -> np.ndarray:
    try:
        wet = cv2.edgePreservingFilter(work, flags=1, sigma_s=64, sigma_r=0.42)
    except cv2.error:
        wet = cv2.stylization(work, sigma_s=50, sigma_r=0.45)
    wet = cv2.GaussianBlur(wet, (5, 5), 0)
    return _hsv_mul(wet, s=0.85, v=1.08)


def _s_oil(work: np.ndarray) -> np.ndarray:
    heavy = cv2.stylization(work, sigma_s=70, sigma_r=0.48)
    heavy = cv2.bilateralFilter(heavy, 9, 90, 90)
    return _hsv_mul(heavy, s=1.18, v=0.98)


def _s_halftone(work: np.ndarray) -> np.ndarray:
    h, w = work.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dots = ((np.sin(xx * 0.95) * np.sin(yy * 0.95) + 1.0) * 0.5) * 255.0
    color = _hsv_mul(quantize(cv2.bilateralFilter(work, 7, 50, 50), 10), s=1.45, v=1.05)
    lum = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = (lum > dots).astype(np.uint8) * 255
    paper = np.full_like(color, 235)
    return np.where(mask[..., None] > 0, color, paper)


def _s_pixel(work: np.ndarray) -> np.ndarray:
    h, w = work.shape[:2]
    cell = 8
    small = cv2.resize(work, (max(8, w // cell), max(8, h // cell)), interpolation=cv2.INTER_AREA)
    chunk = cv2.resize(quantize(small, 12), (w, h), interpolation=cv2.INTER_NEAREST)
    return _hsv_mul(chunk, s=1.25, v=1.04)


def _s_noir(work: np.ndarray) -> np.ndarray:
    gray = cv2.convertScaleAbs(_gray(work), alpha=1.45, beta=-30)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out[:, :, 0] *= 1.25
    out[:, :, 2] *= 0.72
    edges = cv2.Canny(gray, 60, 140)
    out[edges > 0] = (210, 210, 220)
    return np.clip(out, 0, 255).astype(np.uint8)


def _s_thermal(work: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(_gray(work), (5, 5), 0)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def _s_vapor(work: np.ndarray) -> np.ndarray:
    img = work.astype(np.float32)
    img[:, :, 0] = np.clip(img[:, :, 0] * 1.35 + 18, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * 0.72, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * 1.28 + 22, 0, 255)
    color = quantize(img.astype(np.uint8), 10)
    return cv2.bitwise_and(_hsv_mul(color, s=1.55, v=1.08), _ink_mask(work, 7, 3, False))


def _s_ink(work: np.ndarray) -> np.ndarray:
    gray = cv2.medianBlur(_gray(work), 5)
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
    ink = 255 - cv2.erode(255 - ink, np.ones((2, 2), np.uint8), iterations=1)
    paper = np.full_like(work, (210, 228, 242))
    paper[ink == 0] = (18, 16, 14)
    return paper


def _s_chalk(work: np.ndarray) -> np.ndarray:
    gray = _gray(work)
    inv = 255 - gray
    dodge = cv2.divide(gray, np.clip(255 - cv2.GaussianBlur(inv, (17, 17), 0), 1, 255), scale=256)
    board = np.zeros_like(work)
    board[:] = (22, 28, 18)
    chalk = cv2.cvtColor(dodge, cv2.COLOR_GRAY2BGR)
    tint = _hsv_mul(cv2.GaussianBlur(work, (11, 11), 0), s=0.45, v=1.2)
    mix = cv2.addWeighted(chalk, 0.78, tint, 0.22, 0)
    return np.clip(board.astype(np.float32) + mix.astype(np.float32) * 0.95, 0, 255).astype(np.uint8)


def _s_poster(work: np.ndarray) -> np.ndarray:
    color = cv2.bilateralFilter(work, 9, 90, 90)
    return cv2.bitwise_and(_hsv_mul(quantize(color, 5), s=1.9, v=1.12), _ink_mask(work, 5, 4, True))


def _s_ice(work: np.ndarray) -> np.ndarray:
    img = work.astype(np.float32)
    img[:, :, 0] = np.clip(img[:, :, 0] * 1.45 + 20, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * 1.08, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * 0.72, 0, 255)
    color = quantize(cv2.bilateralFilter(img.astype(np.uint8), 7, 50, 50), 14)
    return cv2.bitwise_and(_hsv_mul(color, s=0.9, v=1.12), _ink_mask(work, 7, 2, False))


def _s_ember(work: np.ndarray) -> np.ndarray:
    img = work.astype(np.float32)
    img[:, :, 0] = np.clip(img[:, :, 0] * 0.45, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * 0.75, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * 1.45 + 18, 0, 255)
    color = quantize(cv2.bilateralFilter(img.astype(np.uint8), 7, 50, 50), 12)
    return cv2.bitwise_and(_hsv_mul(color, s=1.4, v=1.05), _ink_mask(work, 7, 2, False))


def _s_holo(work: np.ndarray) -> np.ndarray:
    h, w = work.shape[:2]
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.linspace(0, 50, w, dtype=np.float32)[None, :]) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    scan = np.ones((h, 1), dtype=np.float32)
    scan[::3] = 0.62
    out = np.clip(out.astype(np.float32) * scan[..., None], 0, 255).astype(np.uint8)
    return cv2.addWeighted(out, 0.82, _s_neon(work), 0.18, 0)


def _s_sepia(work: np.ndarray) -> np.ndarray:
    img = work.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    out = np.empty_like(img)
    out[:, :, 0] = np.clip(0.272 * r + 0.534 * g + 0.131 * b, 0, 255)
    out[:, :, 1] = np.clip(0.349 * r + 0.686 * g + 0.168 * b, 0, 255)
    out[:, :, 2] = np.clip(0.393 * r + 0.769 * g + 0.189 * b, 0, 255)
    grain = cv2.GaussianBlur(out.astype(np.uint8), (3, 3), 0)
    return _hsv_mul(grain, s=0.85, v=0.95)


def _s_blueprint(work: np.ndarray) -> np.ndarray:
    gray = cv2.convertScaleAbs(_gray(work), alpha=1.2, beta=-10)
    edges = cv2.Canny(gray, 50, 130)
    out = np.zeros_like(work)
    out[:] = (92, 42, 12)
    paper = (255 - gray).astype(np.float32) / 255.0
    out = (out.astype(np.float32) * (0.35 + 0.65 * paper[..., None])).astype(np.uint8)
    out[edges > 0] = (220, 210, 80)
    return out


def _s_matrix(work: np.ndarray) -> np.ndarray:
    gray = _gray(work)
    out = np.zeros_like(work)
    out[:, :, 1] = gray
    out[:, :, 0] = (gray.astype(np.uint16) * 0.25).astype(np.uint8)
    edges = cv2.Canny(gray, 70, 150)
    out[edges > 0] = (40, 255, 90)
    scan = np.ones((work.shape[0], 1), dtype=np.float32)
    scan[::2] = 0.78
    return np.clip(out.astype(np.float32) * scan[..., None], 0, 255).astype(np.uint8)


def _s_glass(work: np.ndarray) -> np.ndarray:
    h, w = work.shape[:2]
    tile = 14
    small = cv2.resize(work, (max(6, w // tile), max(6, h // tile)), interpolation=cv2.INTER_AREA)
    tiles = cv2.resize(quantize(small, 8), (w, h), interpolation=cv2.INTER_NEAREST)
    gray = _gray(work)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((2, 2), np.uint8), iterations=1)
    tiles[edges > 0] = (8, 8, 10)
    return _hsv_mul(tiles, s=1.35, v=1.05)


def _s_pastel(work: np.ndarray) -> np.ndarray:
    soft = cv2.bilateralFilter(work, 11, 70, 70)
    return _hsv_mul(quantize(soft, 14), s=0.55, v=1.22)


def _s_scan(work: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(_gray(work), (3, 3), 0)
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    paper = np.full_like(work, (200, 228, 236))
    paper[ink == 0] = (22, 22, 24)
    noise = (np.random.default_rng(3).random(paper.shape[:2]) > 0.97).astype(np.uint8)
    paper[noise > 0] = (40, 40, 42)
    return paper


def _s_riso(work: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(_gray(work), (5, 5), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = np.zeros_like(work)
    out[bw > 0] = (170, 70, 255)
    out[bw == 0] = (90, 110, 18)
    mid = (gray > 90) & (gray < 170)
    out[mid] = (40, 200, 220)
    return out


def _s_sunset(work: np.ndarray) -> np.ndarray:
    img = work.astype(np.float32)
    lum = (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]) / 255.0
    warm = img.copy()
    warm[:, :, 0] *= 0.55
    warm[:, :, 2] = np.clip(warm[:, :, 2] * 1.35 + 20, 0, 255)
    cool = img.copy()
    cool[:, :, 0] = np.clip(cool[:, :, 0] * 1.25 + 30, 0, 255)
    cool[:, :, 2] *= 0.7
    t = lum[..., None]
    out = cool * (1.0 - t) + warm * t
    return quantize(np.clip(out, 0, 255).astype(np.uint8), 14)


def _s_night(work: np.ndarray) -> np.ndarray:
    img = work.astype(np.float32)
    img[:, :, 0] = np.clip(img[:, :, 0] * 1.2 + 8, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * 0.9, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * 1.15, 0, 255)
    img *= 0.72
    color = quantize(img.astype(np.uint8), 16)
    return cv2.bitwise_and(_hsv_mul(color, s=1.2, v=0.92), _ink_mask(work, 9, 2, False))


def _s_manga(work: np.ndarray) -> np.ndarray:
    gray = cv2.medianBlur(_gray(work), 5)
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    tone = ((xx + yy) % 4 < 2).astype(np.uint8) * 255
    mid = (gray > 70) & (gray < 170)
    page = np.full_like(work, 245)
    page[ink == 0] = (12, 12, 12)
    page[mid & (tone == 0) & (ink > 0)] = (170, 170, 170)
    return page


_FNS = {
    "cartoon": _s_cartoon,
    "comic": _s_comic,
    "paint": _s_paint,
    "neon": _s_neon,
    "sketch": _s_sketch,
    "pop": _s_pop,
    "mono": _s_mono,
    "duo": _s_duo,
    "anime": _s_anime,
    "watercolor": _s_watercolor,
    "oil": _s_oil,
    "halftone": _s_halftone,
    "pixel": _s_pixel,
    "noir": _s_noir,
    "thermal": _s_thermal,
    "vapor": _s_vapor,
    "ink": _s_ink,
    "chalk": _s_chalk,
    "poster": _s_poster,
    "ice": _s_ice,
    "ember": _s_ember,
    "holo": _s_holo,
    "sepia": _s_sepia,
    "blueprint": _s_blueprint,
    "matrix": _s_matrix,
    "glass": _s_glass,
    "pastel": _s_pastel,
    "scan": _s_scan,
    "riso": _s_riso,
    "sunset": _s_sunset,
    "night": _s_night,
    "manga": _s_manga,
}


def apply(frame: np.ndarray, style: str) -> np.ndarray:
    work, h, w = _prep(frame)
    fn = _FNS.get(style, _s_cartoon)
    return _finish(fn(work), h, w)
