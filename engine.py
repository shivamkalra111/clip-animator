"""Video engine: cartoon / comic / paint / neon restyle + dramatic recut.

Used by animate.py. Do not run this file directly.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import wave

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "opencv-python-headless is required. From the venv run:\n"
        "  pip install opencv-python-headless\n"
    ) from exc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
ORANGE = (255, 106, 36)
WHITE = (255, 255, 255)
WIN_FONTS = [
    r"C:\Windows\Fonts\impact.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]
STYLES = ("cartoon", "comic", "paint", "neon")
DRAMA = ("low", "medium", "high")


def _font_path() -> str:
    for p in WIN_FONTS:
        if os.path.isfile(p):
            return p
    return WIN_FONTS[0]


def load_font(size: int) -> pygame.font.Font:
    try:
        return pygame.font.Font(_font_path(), size)
    except Exception:
        return pygame.font.Font(None, size)


def probe(path: str) -> dict:
    raw = subprocess.check_output(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        text=True,
    )
    info = json.loads(raw)
    vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    aus = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not vs:
        raise SystemExit(f"No video stream in {path}")
    fps_txt = vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "30/1"
    num, den = fps_txt.split("/") if "/" in fps_txt else (fps_txt, "1")
    fps = float(num) / max(float(den), 1e-6)
    if fps < 1 or fps > 120:
        fps = 30.0
    duration = float(info.get("format", {}).get("duration") or vs.get("duration") or 0)
    width = int(vs.get("width") or 1280)
    height = int(vs.get("height") or 720)
    return {
        "fps": fps, "duration": duration, "width": width, "height": height,
        "has_audio": aus is not None,
    }


def extract_audio(path: str, wav_path: str, max_seconds: float | None) -> bool:
    cmd = ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le"]
    if max_seconds:
        cmd.extend(["-t", f"{max_seconds:.3f}"])
    cmd.append(wav_path)
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and os.path.isfile(wav_path) and os.path.getsize(wav_path) > 1024


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "r") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            pcm = pcm.reshape(-1, 2).mean(axis=1)
        return pcm, sr


def write_wav(path: str, pcm: np.ndarray, sr: int) -> None:
    x = np.clip(pcm, -1.0, 1.0)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def motion_profile(path: str, src_fps: float, max_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsampled motion energy + a rough action centroid (nx, ny)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    step = max(1, int(round(src_fps / 12.0)))
    prev = None
    times, energy, cx, cy = [], [], [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / src_fps
        if t >= max_seconds:
            break
        if idx % step == 0:
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is None:
                times.append(t)
                energy.append(0.0)
                cx.append(0.5)
                cy.append(0.5)
            else:
                diff = cv2.absdiff(gray, prev)
                energy.append(float(diff.mean()))
                times.append(t)
                _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
                ys, xs = np.where(thresh > 0)
                if len(xs) > 40:
                    cx.append(float(xs.mean()) / 320.0)
                    cy.append(float(ys.mean()) / 180.0)
                else:
                    cx.append(cx[-1] if cx else 0.5)
                    cy.append(cy[-1] if cy else 0.5)
            prev = gray
        idx += 1
    cap.release()
    if not times:
        return np.array([0.0]), np.array([0.0]), np.array([[0.5, 0.5]])
    e = np.array(energy, dtype=np.float32)
    kernel = np.ones(5, dtype=np.float32) / 5.0
    e = np.convolve(e, kernel, mode="same")
    peak = float(e.max()) or 1.0
    e = e / peak
    centroids = np.stack([np.array(cx), np.array(cy)], axis=1)
    return np.array(times, dtype=np.float32), e, centroids


def find_peaks(times: np.ndarray, energy: np.ndarray, min_gap: float = 1.35) -> list[float]:
    peaks = []
    for i in range(1, len(energy) - 1):
        if energy[i] >= 0.52 and energy[i] >= energy[i - 1] and energy[i] >= energy[i + 1]:
            t = float(times[i])
            if not peaks or (t - peaks[-1]) >= min_gap:
                peaks.append(t)
    return peaks


def rate_at(t: float, peaks: list[float], drama: str) -> float:
    if drama == "low":
        return 1.0
    near = 0.0
    win = 0.85 if drama == "high" else 0.55
    slow = 0.38 if drama == "high" else 0.55
    fast = 1.38 if drama == "high" else 1.18
    for p in peaks:
        d = abs(t - p)
        if d < win:
            near = max(near, 1.0 - d / win)
    return slow * near + fast * (1.0 - near)


def build_src_times(duration: float, out_fps: float, peaks: list[float], drama: str) -> np.ndarray:
    times = []
    t = 0.0
    dt = 1.0 / out_fps
    # Hard cap so a long clip with lots of slow-mo cannot explode.
    max_out = duration * (2.4 if drama == "high" else 1.6) + 1.5
    while t < duration - 0.02 and (len(times) / out_fps) < max_out:
        times.append(t)
        t += dt * rate_at(t, peaks, drama)
    if not times:
        times = [0.0]
    return np.array(times, dtype=np.float32)


def quantize(img: np.ndarray, steps: int) -> np.ndarray:
    q = max(8, 256 // steps)
    return (img // q) * q


def stylize_bgr(frame: np.ndarray, style: str) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = 640.0 / max(h, w)
    if scale < 1.0:
        work = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        work = frame
    if style == "paint":
        painted = cv2.stylization(work, sigma_s=45, sigma_r=0.35)
        out = painted
    elif style == "neon":
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 160)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        glow = cv2.GaussianBlur(edges, (0, 0), 2.4)
        dark = (work.astype(np.float32) * 0.22).astype(np.uint8)
        color_edge = np.zeros_like(work)
        color_edge[:, :, 0] = glow  # blue-ish
        color_edge[:, :, 1] = (glow * 0.55).astype(np.uint8)
        color_edge[:, :, 2] = np.clip(edges.astype(np.uint16) + glow * 0.4, 0, 255).astype(np.uint8)
        out = cv2.add(dark, color_edge)
    else:
        color = work
        for _ in range(2):
            color = cv2.bilateralFilter(color, 7, 55, 55)
        steps = 12 if style == "comic" else 18
        color = quantize(color, steps)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        blk = 5 if style == "comic" else 7
        c = 3 if style == "comic" else 2
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk, c)
        if style == "comic":
            edges = cv2.erode(edges, np.ones((2, 2), np.uint8), iterations=1)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out = cv2.bitwise_and(color, edges_bgr)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.55 if style == "comic" else 1.38), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.10, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if out.shape[0] != h or out.shape[1] != w:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def cover_crop(img: np.ndarray, tw: int, th: int, cx: float, cy: float, zoom: float) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(tw / w, th / h) * zoom
    nw, nh = max(tw, int(round(w * scale))), max(th, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px = int(cx * nw) - tw // 2
    py = int(cy * nh) - th // 2
    px = max(0, min(nw - tw, px))
    py = max(0, min(nh - th, py))
    return resized[py:py + th, px:px + tw]


def make_vignette(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1) - 0.5) * 2.0
    ny = (yy / max(h - 1, 1) - 0.5) * 2.0
    r = np.sqrt(nx * nx + ny * ny * 1.15)
    v = np.clip(1.18 - 0.55 * r * r, 0.35, 1.0)
    return v.astype(np.float32)


def make_speed_lines(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.48
    dx, dy = xx - cx, yy - cy
    ang = np.arctan2(dy, dx)
    rad = np.sqrt(dx * dx + dy * dy)
    spokes = (np.sin(ang * 42.0) > 0.88).astype(np.float32)
    ring = ((rad > min(h, w) * 0.16) & (rad < min(h, w) * 0.72)).astype(np.float32)
    lines = spokes * ring
    lines = cv2.GaussianBlur(lines, (0, 0), 1.2)
    return np.clip(lines, 0, 1)


def grade(bgr: np.ndarray, vignette: np.ndarray, flash: float, lines: np.ndarray, line_amt: float) -> np.ndarray:
    img = bgr.astype(np.float32)
    img[:, :, 2] = np.clip(img[:, :, 2] * 1.08 + 6, 0, 255)  # red
    img[:, :, 0] = np.clip(img[:, :, 0] * 1.04 + 4, 0, 255)  # blue
    img[:, :, 1] = np.clip(img[:, :, 1] * 0.97, 0, 255)
    img *= vignette[..., None]
    if line_amt > 0.02:
        img += (lines * line_amt * 210.0)[..., None]
    if flash > 0.01:
        img = img * (1.0 - 0.35 * flash) + 255.0 * flash
    return np.clip(img, 0, 255).astype(np.uint8)


def overlay_rgba(bgr: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    h, w = bgr.shape[:2]
    rh, rw = rgba.shape[:2]
    if x >= w or y >= h or x + rw <= 0 or y + rh <= 0:
        return
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + rw), min(h, y + rh)
    sx0, sy0 = x0 - x, y0 - y
    patch = rgba[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    rgb = patch[:, :, :3][:, :, ::-1].astype(np.float32)
    dst = bgr[y0:y1, x0:x1].astype(np.float32)
    bgr[y0:y1, x0:x1] = (rgb * alpha + dst * (1 - alpha)).astype(np.uint8)


def surf_rgba(surf: pygame.Surface) -> np.ndarray:
    """pygame surface -> numpy HxWxRGBA."""
    rgb = pygame.surfarray.array3d(surf)
    if surf.get_alpha() is not None or surf.get_bitsize() == 32:
        try:
            a = pygame.surfarray.array_alpha(surf)
        except ValueError:
            a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    else:
        a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack([rgb, a])
    return np.transpose(rgba, (1, 0, 2))


def build_overlays(width: int, height: int, title: str, logo: pygame.Surface | None):
    pygame.font.init()
    font_big = load_font(int(height * 0.07))
    font_sub = load_font(int(height * 0.028))
    title_s = pygame.Surface((width, int(height * 0.22)), pygame.SRCALPHA)
    bar = pygame.Surface((width, int(height * 0.16)), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 150))
    title_s.blit(bar, (0, int(height * 0.03)))
    t = font_big.render("ANIMATED REPLAY", True, ORANGE)
    title_s.blit(t, t.get_rect(center=(width // 2, int(height * 0.085))))
    s = font_sub.render(title[:80], True, WHITE)
    title_s.blit(s, s.get_rect(center=(width // 2, int(height * 0.145))))
    caption_s = pygame.Surface((width, int(height * 0.08)), pygame.SRCALPHA)
    cap = font_sub.render("DRAMATIC CUT", True, (230, 230, 230))
    caption_s.blit(cap, cap.get_rect(center=(width // 2, int(height * 0.04))))
    logo_rgba = None
    if logo is not None:
        logo_rgba = surf_rgba(logo)
    return surf_rgba(title_s), surf_rgba(caption_s), logo_rgba


def warp_audio(pcm: np.ndarray, sr: int, src_times: np.ndarray, out_fps: float) -> np.ndarray:
    n_out = int(round(len(src_times) / out_fps * sr))
    t_out = np.arange(n_out) / sr
    frame_t = np.arange(len(src_times)) / out_fps
    t_src = np.interp(t_out, frame_t, src_times)
    idx = t_src * sr
    idx = np.clip(idx, 0, max(len(pcm) - 2, 0))
    i0 = np.floor(idx).astype(np.int64)
    frac = idx - i0
    return pcm[i0] * (1 - frac) + pcm[np.minimum(i0 + 1, len(pcm) - 1)] * frac


def dramatic_audio(pcm: np.ndarray, sr: int, src_times: np.ndarray, out_fps: float,
                   peaks: list[float], drama: str) -> np.ndarray:
    out = warp_audio(pcm, sr, src_times, out_fps) if len(pcm) > sr // 10 else np.zeros(
        int(len(src_times) / out_fps * sr), dtype=np.float32)
    n = len(out)
    t = np.arange(n) / sr
    rng = np.random.default_rng(11)
    # trailer rumble
    rumble = (np.sin(2 * math.pi * 48 * t) * 0.08 + np.sin(2 * math.pi * 73 * t) * 0.04)
    rumble *= 0.55 + 0.45 * np.sin(2 * math.pi * 0.35 * t)
    noise = rng.standard_normal(n).astype(np.float32)
    # crude lowpass
    k = np.ones(64, dtype=np.float32) / 64.0
    whoosh = np.convolve(noise, k, mode="same") * 0.12
    mixed = out * (0.78 if drama == "high" else 0.88) + rumble + whoosh
    # map output time -> nearest source peak, then boom at corresponding out t
    frame_t = np.arange(len(src_times)) / out_fps
    for p in peaks:
        # output time when source time is nearest p
        j = int(np.argmin(np.abs(src_times - p)))
        t0 = frame_t[j]
        i0 = int(t0 * sr)
        ln = min(int(0.55 * sr), n - i0)
        if i0 < 0 or ln <= 0:
            continue
        tt = np.arange(ln) / sr
        env = np.exp(-tt * 7.5)
        boom = np.sin(2 * math.pi * (70 * np.exp(-tt * 8) + 38) * tt) * env
        mixed[i0:i0 + ln] += boom * (0.85 if drama == "high" else 0.5)
        mixed[i0:i0 + ln] += rng.uniform(-1, 1, ln).astype(np.float32) * np.exp(-tt * 18) * 0.25
        # duck original slightly under the hit
        mixed[i0:i0 + ln] *= (0.72 + 0.28 * (1 - env))
    peak = float(np.max(np.abs(mixed))) or 1.0
    return np.tanh(mixed / peak * 1.15).astype(np.float32)


def render_clip(
    src: str,
    out_video: str,
    *,
    style: str = "cartoon",
    drama: str = "high",
    shorts: bool = False,
    max_seconds: float | None = None,
    keep_speed: bool = False,
    title: str = "",
) -> dict:
    meta = probe(src)
    max_s = float(max_seconds) if max_seconds else float(meta["duration"])
    max_s = min(max_s, float(meta["duration"]))
    out_fps = 30
    out_w, out_h = (1080, 1920) if shorts else (1920, 1080)

    print(f"Source: {meta['width']}x{meta['height']}  {meta['fps']:.2f}fps  {meta['duration']:.1f}s")
    print("Scanning motion...")
    times, energy, centroids = motion_profile(src, meta["fps"], max_s)
    peaks = [] if keep_speed or drama == "low" else find_peaks(times, energy)
    if keep_speed:
        src_times = np.arange(0, max_s, 1.0 / out_fps, dtype=np.float32)
    else:
        src_times = build_src_times(max_s, out_fps, peaks, drama)
    print(f"Style={style}  drama={drama}  peaks={len(peaks)}  out={len(src_times)/out_fps:.1f}s")

    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((1, 1))
    logo = None
    logo_path = os.path.join(HERE, "brand", "logo_circle.png")
    if os.path.isfile(logo_path):
        raw = pygame.image.load(logo_path).convert_alpha()
        logo = pygame.transform.smoothscale(raw, (72, 72))
    label = title or os.path.splitext(os.path.basename(src))[0].replace("_", " ")
    title_ov, cap_ov, logo_rgba = build_overlays(out_w, out_h, label, logo)
    vignette = make_vignette(out_h, out_w)
    lines = make_speed_lines(out_h, out_w)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {src}")
    src_fps = meta["fps"]
    buf_idx = -1
    buf_frame = None

    def frame_at(t: float) -> np.ndarray:
        nonlocal buf_idx, buf_frame
        want = int(min(max(t, 0.0) * src_fps, max_s * src_fps - 1))
        if buf_frame is None or want < buf_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, want)
            buf_idx = want - 1
        while buf_idx < want:
            ok, fr = cap.read()
            if not ok:
                break
            buf_idx += 1
            buf_frame = fr
        if buf_frame is None:
            raise SystemExit("Failed to read source frames")
        return buf_frame

    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    tmp = out_video + ".tmp.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_w}x{out_h}", "-r", str(out_fps), "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", "17",
         "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    n = len(src_times)
    for i, t_src in enumerate(src_times):
        fr = frame_at(float(t_src))
        styled = stylize_bgr(fr, style)
        e = float(np.interp(t_src, times, energy))
        cx = float(np.interp(t_src, times, centroids[:, 0]))
        cy = float(np.interp(t_src, times, centroids[:, 1]))
        zoom = 1.0
        flash = 0.0
        if drama != "low":
            zoom = 1.0 + (0.22 if drama == "high" else 0.12) * e
            near = 0.0
            for p in peaks:
                d = abs(float(t_src) - p)
                if d < 0.16:
                    near = max(near, 1.0 - d / 0.16)
            flash = near * (0.55 if drama == "high" else 0.3)
        canvas = cover_crop(styled, out_w, out_h, cx, cy, zoom)
        line_amt = (0.55 if drama == "high" else 0.28) * max(0.0, e - 0.35) if drama != "low" else 0.0
        canvas = grade(canvas, vignette, flash, lines, line_amt)
        # cinematic letterbox
        if drama != "low" and not shorts:
            bar = int(out_h * (0.09 if drama == "high" else 0.06))
            canvas[:bar] = (4, 4, 6)
            canvas[-bar:] = (4, 4, 6)
        elif shorts and drama != "low":
            bar = int(out_h * 0.04)
            canvas[:bar] = (4, 4, 6)
            canvas[-bar:] = (4, 4, 6)
        u = i / max(n - 1, 1)
        if u < 0.12:
            overlay_rgba(canvas, title_ov, 0, 0)
        overlay_rgba(canvas, cap_ov, 0, out_h - cap_ov.shape[0] - 12)
        if logo_rgba is not None:
            overlay_rgba(canvas, logo_rgba, out_w - logo_rgba.shape[1] - 18, out_h - logo_rgba.shape[0] - 18)
        ff.stdin.write(np.ascontiguousarray(canvas).tobytes())
        if i % 40 == 0:
            print(f"  {i+1}/{n} frames  ({100.0 * (i+1)/n:.0f}%)")

    ff.stdin.close()
    ff.wait()
    cap.release()

    wav_in = out_video + ".src.wav"
    wav_out = out_video + ".mix.wav"
    has_src_audio = extract_audio(src, wav_in, max_s)
    if has_src_audio:
        pcm, sr = read_wav(wav_in)
    else:
        sr = 44100
        pcm = np.zeros(int(max_s * sr), dtype=np.float32)
    mixed = dramatic_audio(pcm, sr, src_times, out_fps, peaks, drama)
    write_wav(wav_out, mixed, sr)
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp, "-i", wav_out,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", out_video],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    for p in (tmp, wav_in, wav_out):
        if os.path.isfile(p):
            os.remove(p)

    duration = round(len(src_times) / out_fps, 2)
    print(f"Rendered {duration:.1f}s  peaks={len(peaks)}")
    return {"video": out_video, "duration": duration, "peaks": [round(p, 2) for p in peaks]}
