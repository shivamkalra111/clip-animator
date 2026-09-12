"""CLI: turn a local clip into a dramatic animated recut + YouTube metadata.

Usage:
    python animate.py path\\to\\clip.mp4 --shorts --genre football --about "Manchester United, Cunha goal"
    python animate.py --list-packs
    python animate.py --list-styles
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import engine
import metadata
import packs
import styles


def _load_env() -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _unique_path(path: str) -> str:
    if not os.path.isfile(path):
        return path
    root, ext = os.path.splitext(path)
    n = 2
    while os.path.isfile(f"{root}_{n}{ext}"):
        n += 1
    return f"{root}_{n}{ext}"


def main() -> int:
    _load_env()
    ap = argparse.ArgumentParser(
        description="Animate a clip and write YouTube title / description / tags.")
    ap.add_argument("clip", nargs="?", help="path to a local video (mp4/mov/webm/…)")
    ap.add_argument("--style", default="auto",
                    choices=("auto",) + engine.STYLES,
                    help="drawing look. auto picks a new one every run")
    ap.add_argument("--pack", default="auto",
                    choices=("auto",) + packs.PACK_NAMES,
                    help="motion recipe. auto = real-time only (no slow-mo/whoosh)")
    ap.add_argument("--list-packs", action="store_true",
                    help="print motion packs and exit")
    ap.add_argument("--list-styles", action="store_true",
                    help="print drawing styles and exit")
    ap.add_argument("--drama", choices=engine.DRAMA, default="high")
    ap.add_argument("--shorts", action="store_true", help="9:16 1080x1920")
    ap.add_argument("--keep-speed", action="store_true")
    ap.add_argument("--sfx", action="store_true",
                    help="add trailer booms/rumble (off: keep the clip's own audio)")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--title", default="", help="on-screen label in the video")
    ap.add_argument("--out", default="", help="output mp4 path")
    ap.add_argument("--genre", default="", help="e.g. football, basketball, f1")
    ap.add_argument("--about", default="",
                    help='free-text facts, e.g. "Manchester United, Cunha goal"')
    ap.add_argument("--teams", default="")
    ap.add_argument("--player", default="")
    ap.add_argument("--moment", default="")
    ap.add_argument("--extra", default="", help="anything else the LLM should know")
    ap.add_argument("--require-llm", action="store_true",
                    help="fail if the LLM is not configured instead of using a local fallback")
    args = ap.parse_args()

    if args.list_packs:
        print(packs.list_packs())
        return 0
    if args.list_styles:
        print(styles.list_styles())
        return 0
    if not args.clip:
        ap.error("clip is required (or pass --list-packs / --list-styles)")

    src = args.clip
    if not os.path.isfile(src):
        print(f"File not found: {src}")
        return 1

    look = packs.resolve(
        args.style, args.pack, src, engine.STATE_PATH, engine.STYLES)
    audio = "sfx" if args.sfx else "source"
    print(f"This run: {look.summary()}  audio={audio}")

    os.makedirs(engine.OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(src))[0]
    suffix = "short" if args.shorts else "animated"
    out = args.out or os.path.join(
        engine.OUT_DIR, f"{base}_{look.style}_{look.pack}_{suffix}.mp4")
    if not args.out:
        out = _unique_path(out)

    result = engine.render_clip(
        src, out,
        look=look, drama=args.drama, shorts=args.shorts,
        max_seconds=args.max_seconds, keep_speed=args.keep_speed,
        title=args.title, audio=audio,
    )
    style = result.get("style") or look.style
    pack = result.get("pack") or look.pack

    facts = {
        "genre": args.genre,
        "about": args.about,
        "teams": args.teams,
        "player": args.player,
        "moment": args.moment,
        "extra": args.extra,
        "style": style,
        "pack": pack,
        "shorts": args.shorts,
        "duration": result["duration"],
        "on_screen_title": args.title,
        "source_name": os.path.basename(src),
    }
    yt = metadata.generate(facts, require_llm=args.require_llm)

    meta_path = os.path.splitext(out)[0] + ".json"
    payload = {
        "title": yt["title"],
        "description": yt["description"],
        "tags": yt["tags"],
        "tags_csv": yt["tags_csv"],
        "metadata_source": yt.get("source"),
        "facts": {k: v for k, v in facts.items() if v not in ("", None, False)},
        "video": os.path.abspath(out),
        "source_clip": os.path.abspath(src),
        "style": style,
        "pack": pack,
        "look": result.get("look") or look.as_dict(),
        "audio": audio,
        "drama": args.drama,
        "shorts": args.shorts,
        "duration": result["duration"],
        "peaks": result.get("peaks") or [],
        "format": "animated_clip",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Done -> {out}")
    print(f"Metadata -> {meta_path}")
    print(f"Look: {look.summary()}   audio={audio}")
    print(f"Title: {yt['title']}")
    print(f"Tags: {yt['tags_csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
