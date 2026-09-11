"""CLI: turn a local clip into a dramatic animated recut + YouTube metadata.

Usage:
    python animate.py path\\to\\clip.mp4 --shorts --genre football --about "Manchester United, Cunha goal"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import engine
import metadata


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


def main() -> int:
    _load_env()
    ap = argparse.ArgumentParser(
        description="Animate a clip and write YouTube title / description / tags.")
    ap.add_argument("clip", help="path to a local video (mp4/mov/webm/…)")
    ap.add_argument("--style", choices=engine.STYLES, default="cartoon")
    ap.add_argument("--drama", choices=engine.DRAMA, default="high")
    ap.add_argument("--shorts", action="store_true", help="9:16 1080x1920")
    ap.add_argument("--keep-speed", action="store_true")
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

    src = args.clip
    if not os.path.isfile(src):
        print(f"File not found: {src}")
        return 1

    os.makedirs(engine.OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(src))[0]
    suffix = "short" if args.shorts else "animated"
    out = args.out or os.path.join(engine.OUT_DIR, f"{base}_{args.style}_{suffix}.mp4")

    result = engine.render_clip(
        src, out,
        style=args.style, drama=args.drama, shorts=args.shorts,
        max_seconds=args.max_seconds, keep_speed=args.keep_speed, title=args.title,
    )

    facts = {
        "genre": args.genre,
        "about": args.about,
        "teams": args.teams,
        "player": args.player,
        "moment": args.moment,
        "extra": args.extra,
        "style": args.style,
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
        "style": args.style,
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
    print(f"Title: {yt['title']}")
    print(f"Tags: {yt['tags_csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
