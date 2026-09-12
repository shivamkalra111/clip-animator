"""Motion packs: different recut recipes so clips do not all animate the same way."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import random

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "opencv-python-headless is required. From the venv run:\n"
        "  pip install opencv-python-headless\n"
    ) from exc

PACK_NAMES = (
    "live",
    "drift",
    "whip",
    "panels",
    "smash",
    "glitch",
    "impact",
    "trailer",
    "freeze",
    "replay",
    "strobe",
)
# --pack auto never picks time-warp (slow-mo / freeze / whoosh audio).
AUTO_PACKS = ("live", "drift", "whip", "panels", "smash", "glitch", "impact")


@dataclass(frozen=True)
class Pack:
    name: str
    kicker: str
    stamp: str
    blurb: str
    slow: float = 0.38
    fast: float = 1.38
    win: float = 0.85
    zoom_gain: float = 0.22
    peak_zoom: float = 0.0
    flash_gain: float = 0.55
    flash_win: float = 0.16
    shake: float = 0.0
    line_gain: float = 0.55
    freeze_hold: float = 0.0
    glitch_gain: float = 0.0
    chroma_gain: float = 0.0
    letterbox: float = 0.04
    title_u: float = 0.12
    stamp_mode: str = "always"
    audio: str = "boom"
    time_mode: str = "slowmo"
    grade: str = "warm"


PACKS: dict[str, Pack] = {
    "live": Pack(
        name="live",
        kicker="TOONCUT",
        stamp="",
        blurb="Real-time. Style only - no slow-mo, freeze, or whoosh.",
        zoom_gain=0.0,
        flash_gain=0.0,
        line_gain=0.0,
        letterbox=0.0,
        title_u=0.08,
        stamp_mode="never",
        audio="dry",
        time_mode="linear",
        grade="warm",
    ),
    "trailer": Pack(
        name="trailer",
        kicker="ANIMATED REPLAY",
        stamp="DRAMATIC CUT",
        blurb="Classic ToonCut: slow-mo on hits, punch-in zoom, radial speed lines.",
    ),
    "smash": Pack(
        name="smash",
        kicker="SMASH ZOOM",
        stamp="HIT",
        blurb="Aggressive peak zooms, camera shake, no speed lines.",
        slow=0.28,
        fast=1.52,
        win=0.70,
        zoom_gain=0.14,
        peak_zoom=0.42,
        flash_gain=0.72,
        flash_win=0.12,
        shake=1.0,
        line_gain=0.0,
        letterbox=0.07,
        title_u=0.08,
        stamp_mode="peak",
        audio="dry",
        time_mode="linear",
        grade="punch",
    ),
    "freeze": Pack(
        name="freeze",
        kicker="FREEZE FRAME",
        stamp="HOLD",
        blurb="Locks the peak in a comic freeze, then slams back into motion.",
        zoom_gain=0.08,
        peak_zoom=0.10,
        flash_gain=0.35,
        shake=0.15,
        line_gain=0.20,
        freeze_hold=0.42,
        letterbox=0.05,
        title_u=0.10,
        stamp_mode="freeze",
        audio="whoosh",
        time_mode="freeze",
        grade="warm",
    ),
    "replay": Pack(
        name="replay",
        kicker="INSTANT REPLAY",
        stamp="WATCH IT AGAIN",
        blurb="Sports replay: plays through, then rewinds the biggest hit in slow-mo.",
        slow=0.34,
        zoom_gain=0.16,
        peak_zoom=0.18,
        flash_gain=0.40,
        line_gain=0.30,
        title_u=0.10,
        stamp_mode="replay",
        audio="whoosh",
        time_mode="replay",
        grade="warm",
    ),
    "whip": Pack(
        name="whip",
        kicker="WHIP CUT",
        stamp="NEXT BEAT",
        blurb="Normal speed with horizontal whip-pan blurs between action beats.",
        zoom_gain=0.10,
        peak_zoom=0.08,
        flash_gain=0.25,
        shake=0.25,
        line_gain=0.0,
        letterbox=0.03,
        title_u=0.09,
        stamp_mode="whip",
        audio="dry",
        time_mode="linear",
        grade="punch",
    ),
    "panels": Pack(
        name="panels",
        kicker="COMIC PANEL",
        stamp="SPLIT",
        blurb="On peaks, splits the frame into a comic-book panel layout.",
        zoom_gain=0.06,
        peak_zoom=0.12,
        flash_gain=0.15,
        line_gain=0.0,
        letterbox=0.0,
        title_u=0.10,
        stamp_mode="never",
        audio="dry",
        time_mode="linear",
        grade="comic",
    ),
    "drift": Pack(
        name="drift",
        kicker="SLOW BURN",
        stamp="HOLD THE FRAME",
        blurb="No time-warp. Slow Ken Burns push-in that follows the action.",
        zoom_gain=0.20,
        flash_gain=0.0,
        line_gain=0.0,
        letterbox=0.06,
        title_u=0.20,
        stamp_mode="never",
        audio="dry",
        time_mode="linear",
        grade="soft",
    ),
    "strobe": Pack(
        name="strobe",
        kicker="STROBE HIT",
        stamp="IMPACT",
        blurb="Anime-style stutter frames on every hit, with hard flashes.",
        slow=0.45,
        fast=1.22,
        zoom_gain=0.12,
        peak_zoom=0.16,
        flash_gain=0.85,
        flash_win=0.20,
        shake=0.35,
        line_gain=0.25,
        title_u=0.08,
        stamp_mode="strobe",
        audio="stutter",
        time_mode="strobe",
        grade="punch",
    ),
    "glitch": Pack(
        name="glitch",
        kicker="GLITCH CUT",
        stamp="SIGNAL",
        blurb="RGB splits and slice glitches on hits. Pairs well with neon.",
        slow=0.40,
        fast=1.35,
        zoom_gain=0.10,
        peak_zoom=0.14,
        flash_gain=0.30,
        shake=0.20,
        line_gain=0.0,
        glitch_gain=1.0,
        chroma_gain=0.85,
        letterbox=0.02,
        title_u=0.10,
        stamp_mode="peak",
        audio="dry",
        time_mode="linear",
        grade="cool",
    ),
    "impact": Pack(
        name="impact",
        kicker="IMPACT FRAME",
        stamp="BOOM",
        blurb="Anime impact frames: contrast slam, radial burst, one inverted hit.",
        slow=0.32,
        fast=1.40,
        win=0.75,
        zoom_gain=0.12,
        peak_zoom=0.28,
        flash_gain=0.50,
        flash_win=0.10,
        shake=0.45,
        line_gain=0.85,
        letterbox=0.05,
        title_u=0.08,
        stamp_mode="impact",
        audio="dry",
        time_mode="linear",
        grade="punch",
    ),
}


@dataclass
class Beat:
    t: float
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    shake: float = 0.0
    flash: float = 0.0
    lines: float = 0.0
    freeze: bool = False
    replay: bool = False
    panel: bool = False
    whip: float = 0.0
    strobe: bool = False
    glitch: float = 0.0
    impact: float = 0.0
    chroma: float = 0.0


LINE_KINDS = ("radial", "diagonal", "cross", "none")
INKS = (
    (255, 106, 36),
    (255, 62, 92),
    (90, 210, 255),
    (255, 214, 48),
    (186, 92, 255),
    (64, 245, 150),
    (255, 255, 255),
)


@dataclass
class Look:
    """Full per-run recipe: drawing, motion, grade extras, overlay ink."""
    style: str
    pack: str
    hue: float
    sat: float
    grain: float
    line_kind: str
    ink: tuple[int, int, int]
    kicker: str
    stamp: str

    def summary(self) -> str:
        return (
            f"{self.style} + {self.pack}  "
            f"lines={self.line_kind}  hue={self.hue:+.0f}  grain={self.grain:.2f}"
        )

    def as_dict(self) -> dict:
        return {
            "style": self.style,
            "pack": self.pack,
            "hue": round(self.hue, 2),
            "sat": round(self.sat, 2),
            "grain": round(self.grain, 2),
            "line_kind": self.line_kind,
            "ink": list(self.ink),
            "kicker": self.kicker,
            "stamp": self.stamp,
        }


def list_packs() -> str:
    lines = ["Motion packs (use --pack NAME, or --pack auto):", ""]
    for name in PACK_NAMES:
        p = PACKS[name]
        lines.append(f"  {name:<8}  {p.blurb}")
    lines.append("")
    lines.append("  auto      Real-time packs only (no slow-mo / freeze / whoosh).")
    lines.append("  Time-warp (opt-in): trailer, freeze, replay, strobe")
    return "\n".join(lines)


def _load_state(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(path: str, state: dict) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f)


def _choose(rng: random.Random, options: tuple[str, ...] | list, banned: set) -> str:
    pool = [o for o in options if o not in banned]
    if not pool:
        pool = list(options)
    return rng.choice(pool)


def resolve(style_req: str, pack_req: str, src: str, state_path: str, styles: tuple[str, ...]) -> Look:
    """Pick a full look. `auto` is a fresh random every run, never the last few combos."""
    if style_req and style_req != "auto" and style_req not in styles:
        raise SystemExit(f"Unknown style: {style_req}. Choose: {', '.join(styles)}")
    if pack_req and pack_req != "auto" and pack_req not in PACK_NAMES:
        raise SystemExit(f"Unknown pack: {pack_req}. Choose: {', '.join(PACK_NAMES)}")

    rng = random.SystemRandom()
    state = _load_state(state_path)
    history = state.get("history") if isinstance(state.get("history"), list) else []
    recent = [h for h in history if isinstance(h, dict)][-8:]
    banned_styles = {h.get("style") for h in recent[-2:]} - {None}
    banned_packs = {h.get("pack") for h in recent[-4:]} - {None}
    banned_combos = {(h.get("style"), h.get("pack")) for h in recent}
    banned_lines = {h.get("line_kind") for h in recent[-1:]} - {None, "none"}
    banned_inks = {tuple(h["ink"]) for h in recent[-1:] if isinstance(h.get("ink"), list)}

    style = pack = None
    for _ in range(24):
        style = style_req if style_req and style_req != "auto" else _choose(rng, styles, banned_styles)
        pack = pack_req if pack_req and pack_req != "auto" else _choose(rng, AUTO_PACKS, banned_packs)
        if (style, pack) not in banned_combos:
            break
        if style_req != "auto" and pack_req != "auto":
            break
        banned_packs = banned_packs | {pack} if pack_req == "auto" else banned_packs
    assert style and pack
    recipe = get(pack)

    line_kind = _choose(rng, LINE_KINDS, banned_lines)
    ink = rng.choice([c for c in INKS if c not in banned_inks] or list(INKS))
    hue = rng.uniform(-24.0, 24.0)
    sat = rng.uniform(0.88, 1.38)
    grain = rng.choice((0.0, 0.12, 0.22, 0.34, 0.45))

    look = Look(
        style=style,
        pack=pack,
        hue=hue,
        sat=sat,
        grain=grain,
        line_kind=line_kind,
        ink=ink,
        kicker=recipe.kicker,
        stamp=recipe.stamp,
    )
    recent.append(look.as_dict())
    state["history"] = recent[-12:]
    state["last_style"] = style
    state["last_pack"] = pack
    _save_state(state_path, state)
    _ = os.path.basename(src)
    return look


def apply_look(bgr: np.ndarray, look: Look, rng: np.random.Generator) -> np.ndarray:
    img = bgr
    if abs(look.hue) > 0.4 or abs(look.sat - 1.0) > 0.04:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + look.hue) % 180.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * look.sat, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if look.grain > 0.04:
        noise = rng.normal(0.0, 10.0 * look.grain, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def get(name: str) -> Pack:
    if name not in PACKS:
        raise SystemExit(f"Unknown pack: {name}. Choose: {', '.join(PACK_NAMES)}")
    return PACKS[name]


def _drama_scale(drama: str) -> float:
    return {"low": 0.35, "medium": 0.72, "high": 1.0}.get(drama, 1.0)


def _near(t: float, peaks: list[float], win: float) -> float:
    if win <= 0 or not peaks:
        return 0.0
    best = 0.0
    for p in peaks:
        d = abs(t - p)
        if d < win:
            best = max(best, 1.0 - d / win)
    return best


def accents(peaks: list[float], times: np.ndarray, energy: np.ndarray, duration: float) -> list[float]:
    if peaks:
        return list(peaks)
    if times.size and energy.size:
        return [float(times[int(np.argmax(energy))])]
    return [max(duration * 0.45, 0.2)]


def best_peak(peaks: list[float], times: np.ndarray, energy: np.ndarray) -> float:
    if not peaks:
        return float(times[int(np.argmax(energy))]) if times.size else 0.5
    return max(peaks, key=lambda p: float(np.interp(p, times, energy)))


def _slowmo_times(duration: float, fps: float, peaks: list[float], pack: Pack,
                  drama: str, max_out: float) -> np.ndarray:
    if drama == "low":
        return np.arange(0, duration, 1.0 / fps, dtype=np.float32)
    dt = 1.0 / fps
    t = 0.0
    out: list[float] = []
    while t < duration - 0.02 and (len(out) / fps) < max_out:
        out.append(t)
        near = _near(t, peaks, pack.win)
        rate = pack.slow * near + pack.fast * (1.0 - near)
        t += dt * rate
    return np.array(out or [0.0], dtype=np.float32)


def _linear_times(duration: float, fps: float) -> np.ndarray:
    return np.arange(0, max(duration - 0.02, 0.001), 1.0 / fps, dtype=np.float32)


def _freeze_times(duration: float, fps: float, peaks: list[float], hold: float,
                  max_out: float) -> tuple[np.ndarray, list[bool]]:
    dt = 1.0 / fps
    t = 0.0
    used: set[float] = set()
    ts: list[float] = []
    flags: list[bool] = []
    n_hold = max(4, int(round(hold * fps)))
    while t < duration - 0.02 and (len(ts) / fps) < max_out:
        hit = None
        for p in peaks:
            if p in used:
                continue
            if abs(t - p) <= dt * 1.6:
                hit = p
                break
        if hit is not None:
            used.add(hit)
            for _ in range(n_hold):
                ts.append(hit)
                flags.append(True)
            t = hit + dt
        else:
            ts.append(t)
            flags.append(False)
            t += dt
    return np.array(ts or [0.0], dtype=np.float32), flags


def _replay_times(duration: float, fps: float, peak: float, pack: Pack,
                  max_out: float) -> tuple[np.ndarray, list[bool]]:
    dt = 1.0 / fps
    pre, post = 0.72, 0.38
    a = max(0.0, peak - pre)
    b = min(duration - 0.02, peak + post)
    live_end = min(duration - 0.02, peak + 0.10)
    ts: list[float] = []
    flags: list[bool] = []

    t = 0.0
    while t < live_end and (len(ts) / fps) < max_out:
        ts.append(t)
        flags.append(False)
        t += dt

    t = a
    while t < b and (len(ts) / fps) < max_out:
        ts.append(t)
        flags.append(True)
        t += dt * pack.slow

    t = live_end
    while t < duration - 0.02 and (len(ts) / fps) < max_out:
        ts.append(t)
        flags.append(False)
        t += dt
    return np.array(ts or [0.0], dtype=np.float32), flags


def _strobe_times(duration: float, fps: float, peaks: list[float],
                  max_out: float) -> tuple[np.ndarray, list[bool]]:
    dt = 1.0 / fps
    t = 0.0
    ts: list[float] = []
    flags: list[bool] = []
    armed = {round(p, 2) for p in peaks}
    while t < duration - 0.02 and (len(ts) / fps) < max_out:
        hit = None
        for p in list(armed):
            if abs(t - p) <= dt * 1.5:
                hit = p
                break
        if hit is not None:
            armed.discard(round(hit, 2))
            pattern = (0.0, 0.0, -2 * dt, 0.0, 0.0, -2 * dt, 0.0, dt, 0.0)
            for step in pattern:
                ts.append(max(0.0, min(duration - 0.02, hit + step)))
                flags.append(True)
            t = hit + dt * 3.0
        else:
            ts.append(t)
            flags.append(False)
            t += dt
    return np.array(ts or [0.0], dtype=np.float32), flags


def _whip_mids(peaks: list[float], duration: float) -> list[float]:
    pts = sorted(peaks)
    if len(pts) >= 2:
        return [0.5 * (a + b) for a, b in zip(pts, pts[1:])]
    return [duration * 0.33, duration * 0.66]


def build_beats(
    pack: Pack,
    duration: float,
    fps: float,
    peaks: list[float],
    times: np.ndarray,
    energy: np.ndarray,
    drama: str,
    keep_speed: bool,
) -> list[Beat]:
    hits = accents(peaks, times, energy, duration)
    max_out = duration * (2.5 if drama == "high" else 1.7) + 2.0
    scale = _drama_scale(drama)
    freeze_flags: list[bool] = []
    replay_flags: list[bool] = []
    strobe_flags: list[bool] = []

    mode = "linear" if keep_speed or drama == "low" else pack.time_mode
    if mode == "freeze":
        src, freeze_flags = _freeze_times(duration, fps, hits, pack.freeze_hold * scale, max_out)
    elif mode == "replay":
        src, replay_flags = _replay_times(
            duration, fps, best_peak(hits, times, energy), pack, max_out)
    elif mode == "strobe":
        src, strobe_flags = _strobe_times(duration, fps, hits, max_out)
    elif mode == "slowmo":
        src = _slowmo_times(duration, fps, hits, pack, drama, max_out)
    else:
        src = _linear_times(duration, fps)

    n = len(src)
    whip_at = _whip_mids(hits, duration) if pack.name == "whip" else []
    beats: list[Beat] = []
    for i, t in enumerate(src):
        e = float(np.interp(t, times, energy)) if times.size else 0.0
        near = _near(float(t), hits, pack.win)
        flash_near = _near(float(t), hits, pack.flash_win)
        u = i / max(n - 1, 1)
        freeze = bool(freeze_flags[i]) if freeze_flags else False
        replay = bool(replay_flags[i]) if replay_flags else False
        strobe = bool(strobe_flags[i]) if strobe_flags else False

        zoom = 1.0 + pack.zoom_gain * e * scale + pack.peak_zoom * near * scale
        pan_x = pan_y = 0.0
        if freeze:
            zoom = max(zoom, 1.10 + 0.08 * scale)
        if pack.name == "drift":
            zoom = 1.0 + pack.zoom_gain * u * scale
            pan_x = 0.045 * math.sin(2 * math.pi * u) * scale
            pan_y = 0.03 * math.cos(2 * math.pi * u * 0.7) * scale

        whip = _near(float(t), whip_at, 0.22) * scale if whip_at else 0.0
        panel = pack.name == "panels" and near > 0.28

        glitch = pack.glitch_gain * max(flash_near, near * 0.6) * scale
        chroma = pack.chroma_gain * max(flash_near, near * 0.5) * scale
        impact = (near * scale) if pack.name == "impact" else 0.0
        lines = pack.line_gain * max(0.0, e - 0.28) * scale
        if pack.name == "impact":
            lines = pack.line_gain * max(near, max(0.0, e - 0.25)) * scale
        flash = pack.flash_gain * flash_near * scale
        if strobe:
            flash = max(flash, 0.45 * scale)
        shake = pack.shake * max(near, flash_near) * scale

        beats.append(Beat(
            t=float(t),
            zoom=float(zoom),
            pan_x=float(pan_x),
            pan_y=float(pan_y),
            shake=float(shake),
            flash=float(flash),
            lines=float(lines),
            freeze=freeze,
            replay=replay,
            panel=panel,
            whip=float(whip),
            strobe=strobe,
            glitch=float(glitch),
            impact=float(impact),
            chroma=float(chroma),
        ))
    return beats


def apply_whip(bgr: np.ndarray, amt: float) -> np.ndarray:
    if amt < 0.08:
        return bgr
    sigma = 6.0 + 38.0 * amt
    return cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=0.6)


def apply_chroma(bgr: np.ndarray, amt: float) -> np.ndarray:
    if amt < 0.05:
        return bgr
    shift = max(1, int(round(14 * amt)))
    out = bgr.copy()
    out[:, :, 2] = np.roll(out[:, :, 2], shift, axis=1)
    out[:, :, 0] = np.roll(out[:, :, 0], -shift, axis=1)
    return out


def apply_glitch(bgr: np.ndarray, amt: float, rng: np.random.Generator) -> np.ndarray:
    if amt < 0.06:
        return bgr
    out = apply_chroma(bgr, 0.45 + 0.55 * amt)
    h, w = out.shape[:2]
    n = 2 + int(5 * amt)
    for _ in range(n):
        y = int(rng.integers(0, max(h - 10, 1)))
        hh = int(rng.integers(3, 16))
        dx = int(rng.integers(-48, 49))
        out[y:y + hh] = np.roll(out[y:y + hh], dx, axis=1)
    if amt > 0.75:
        band = int(h * 0.04)
        y = int(rng.integers(0, max(h - band, 1)))
        out[y:y + band] = np.clip(out[y:y + band].astype(np.int16) * 2 - 40, 0, 255).astype(np.uint8)
    return out


def apply_impact(bgr: np.ndarray, amt: float, lines: np.ndarray) -> np.ndarray:
    if amt < 0.08:
        return bgr
    img = bgr.astype(np.float32)
    img = (img - 128.0) * (1.0 + 0.65 * amt) + 128.0
    if lines is not None:
        img += lines[..., None] * amt * 230.0
    if amt > 0.90:
        img = 255.0 - img
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_shake(bgr: np.ndarray, amt: float, i: int) -> np.ndarray:
    if amt < 0.05:
        return bgr
    dx = int(round(18 * amt * math.sin(i * 2.31)))
    dy = int(round(14 * amt * math.cos(i * 1.73)))
    out = np.roll(bgr, dy, axis=0)
    out = np.roll(out, dx, axis=1)
    return out


def _panel_stroke(img: np.ndarray, t: int = 5) -> np.ndarray:
    out = img.copy()
    c = (36, 106, 255)
    out[:t] = c
    out[-t:] = c
    out[:, :t] = c
    out[:, -t:] = c
    return out


def compose_panels(main: np.ndarray, alt: np.ndarray, shorts: bool) -> np.ndarray:
    h, w = main.shape[:2]
    out = np.empty_like(main)
    out[:] = (22, 18, 16)
    g = max(8, w // 80)
    if shorts:
        top_h = int(h * 0.58)
        main_p = _panel_stroke(cv2.resize(main, (w - 2 * g, top_h - g), interpolation=cv2.INTER_LINEAR))
        out[g:g + main_p.shape[0], g:g + main_p.shape[1]] = main_p
        bot_h = h - top_h - 2 * g
        bot_w = (w - 3 * g) // 2
        left = _panel_stroke(cv2.resize(alt, (bot_w, bot_h), interpolation=cv2.INTER_LINEAR))
        hh, ww = main.shape[:2]
        crop = main[hh // 6: hh * 5 // 6, ww // 6: ww * 5 // 6]
        right = _panel_stroke(cv2.resize(crop, (bot_w, bot_h), interpolation=cv2.INTER_LINEAR))
        y = top_h + g
        out[y:y + bot_h, g:g + bot_w] = left
        out[y:y + bot_h, 2 * g + bot_w: 2 * g + 2 * bot_w] = right
    else:
        left_w = int(w * 0.58)
        left = _panel_stroke(cv2.resize(main, (left_w - g, h - 2 * g), interpolation=cv2.INTER_LINEAR))
        out[g:g + left.shape[0], g:g + left.shape[1]] = left
        rw = w - left_w - 2 * g
        rh = (h - 3 * g) // 2
        top = _panel_stroke(cv2.resize(alt, (rw, rh), interpolation=cv2.INTER_LINEAR))
        hh, ww = main.shape[:2]
        crop = main[hh // 5: hh * 4 // 5, ww // 5: ww * 4 // 5]
        bot = _panel_stroke(cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR))
        x = left_w + g
        out[g:g + rh, x:x + rw] = top
        out[2 * g + rh: 2 * g + 2 * rh, x:x + rw] = bot
    return out


def grade(bgr: np.ndarray, pack: Pack, vignette: np.ndarray, flash: float,
          lines: np.ndarray, line_amt: float) -> np.ndarray:
    img = bgr.astype(np.float32)
    mode = pack.grade
    if mode == "cool":
        img[:, :, 0] = np.clip(img[:, :, 0] * 1.18 + 8, 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] * 0.92, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * 0.96, 0, 255)
    elif mode == "punch":
        img = (img - 128.0) * 1.12 + 128.0
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.10 + 8, 0, 255)
        img[:, :, 0] = np.clip(img[:, :, 0] * 1.02, 0, 255)
    elif mode == "soft":
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.04 + 4, 0, 255)
        img = img * 0.92 + 14
    elif mode == "comic":
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.12 + 10, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * 1.05, 0, 255)
    else:
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.08 + 6, 0, 255)
        img[:, :, 0] = np.clip(img[:, :, 0] * 1.04 + 4, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * 0.97, 0, 255)
    vig = vignette
    if mode == "soft":
        vig = 0.55 + 0.45 * vignette
    img *= vig[..., None]
    if line_amt > 0.02 and lines is not None:
        img += (lines * line_amt * 210.0)[..., None]
    if flash > 0.01:
        img = img * (1.0 - 0.35 * flash) + 255.0 * flash
    return np.clip(img, 0, 255).astype(np.uint8)


def show_stamp(pack: Pack, beat: Beat, near_peak: bool) -> bool:
    mode = pack.stamp_mode
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode == "peak":
        return near_peak or beat.flash > 0.18 or beat.glitch > 0.35
    if mode == "freeze":
        return beat.freeze
    if mode == "replay":
        return beat.replay
    if mode == "whip":
        return beat.whip > 0.35
    if mode == "strobe":
        return beat.strobe
    if mode == "impact":
        return beat.impact > 0.40
    return False
