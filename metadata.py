"""YouTube-ready title / description / tags from clip facts, via LLM."""

from __future__ import annotations

import json
import re

import llm

SYSTEM = """You write YouTube metadata for a stylized animated recut of a clip the uploader owns.
The video is NOT official broadcast footage — it is a cartoon / comic / paint / neon dramatic edit.

Return a JSON object with exactly these keys:
- title: string, max 100 characters, punchy, no clickbait spam, include the main subject
- description: string, 4–8 short lines. First line is the hook. Mention it is an animated recut.
  End with a blank line then 6–10 hashtags.
- tags: array of 12–18 short search tags (no #). Mix specific names and generic genre terms.
- tags_csv: the tags joined with commas

Do not invent match scores, dates, or quotes that were not in the facts.
Do not claim this is FIFA, Premier League, or club official content.
"""


def _facts_block(info: dict) -> str:
    lines = []
    for key in ("genre", "about", "teams", "player", "moment", "extra"):
        val = (info.get(key) or "").strip()
        if val:
            lines.append(f"{key}: {val}")
    lines.append(f"visual_style: {info.get('style') or 'cartoon'}")
    lines.append(f"edit: dramatic {'9:16 Shorts' if info.get('shorts') else '16:9'} recut")
    if info.get("duration"):
        lines.append(f"duration_seconds: {info['duration']}")
    if info.get("on_screen_title"):
        lines.append(f"on_screen_label: {info['on_screen_title']}")
    if info.get("source_name"):
        lines.append(f"source_filename: {info['source_name']}")
    return "\n".join(lines)


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    title = str(data.get("title") or "").strip()[:100]
    description = str(data.get("description") or "").strip()
    tags = data.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    tags = [str(t).strip().lstrip("#") for t in tags if str(t).strip()]
    tags = tags[:18]
    csv = str(data.get("tags_csv") or ", ".join(tags))
    if not title or not description or not tags:
        raise ValueError("LLM JSON missing title, description, or tags")
    return {"title": title, "description": description, "tags": tags, "tags_csv": csv}


def _fallback(info: dict) -> dict:
    genre = (info.get("genre") or "sports").strip()
    about = (info.get("about") or info.get("moment") or info.get("player") or "Highlight").strip()
    teams = (info.get("teams") or "").strip()
    player = (info.get("player") or "").strip()
    moment = (info.get("moment") or "").strip()
    bits = [b for b in (player, moment, teams, about) if b]
    hook = bits[0] if bits else about
    title = f"{hook} | Animated {genre.title()} Short"
    title = title[:100]
    desc_lines = [
        f"{about}.",
        "",
        f"Animated {genre} recut — cartoon look, dramatic slow-mo and punch-ins.",
        "Stylized edit of footage you provided. Not official broadcast.",
        "",
        "Like if you want more of these.",
        "",
    ]
    tags = []
    for t in (genre, player, teams, moment, about, "animated", "cartoon", "shorts",
              "dramatic", "highlight", "recut", "sports"):
        for piece in re.split(r"[,/|]", t or ""):
            p = piece.strip()
            if p and p.lower() not in {x.lower() for x in tags}:
                tags.append(p[:40])
    hashes = " ".join("#" + re.sub(r"[^A-Za-z0-9]", "", t)[:24] for t in tags[:8] if re.sub(r"[^A-Za-z0-9]", "", t))
    desc_lines.append(hashes)
    description = "\n".join(desc_lines)
    return {
        "title": title,
        "description": description,
        "tags": tags[:16],
        "tags_csv": ", ".join(tags[:16]),
        "source": "fallback",
    }


def generate(info: dict, *, require_llm: bool = False) -> dict:
    """Build title / description / tags. Uses LLM when configured."""
    user = (
        "Write YouTube metadata from these facts:\n\n"
        + _facts_block(info)
        + "\n\nJSON only."
    )
    try:
        raw, backend = llm.complete(SYSTEM, user)
        out = _parse(raw)
        out["source"] = backend
        return out
    except (llm.LLMError, json.JSONDecodeError, ValueError) as exc:
        if require_llm:
            raise
        print(f"LLM metadata skipped ({exc}). Writing from the facts you provided.")
        return _fallback(info)
