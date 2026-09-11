"""Minimal LLM client: OpenAI-compatible HTTP, Groq, or local Ollama.

No extra packages. Set one of:
  OPENAI_API_KEY      (+ optional OPENAI_BASE_URL, OPENAI_MODEL)
  GROQ_API_KEY        (https://console.groq.com — free tier)
  OLLAMA_MODEL        (default llama3.2, server at http://127.0.0.1:11434)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 90) -> dict:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=raw, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:800]
        raise LLMError(f"LLM HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"LLM connection failed: {exc.reason}") from exc


def _openai_compatible(messages: list[dict], api_key: str, base: str, model: str) -> str:
    url = base.rstrip("/") + "/chat/completions"
    data = _post_json(
        url,
        {
            "model": model,
            "temperature": 0.7,
            "messages": messages,
        },
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected LLM response: {data!r}"[:500]) from exc


def _ollama(messages: list[dict], model: str, host: str) -> str:
    url = host.rstrip("/") + "/api/chat"
    data = _post_json(
        url,
        {"model": model, "stream": False, "format": "json", "messages": messages},
        {"Content-Type": "application/json"},
        timeout=180,
    )
    content = data.get("message", {}).get("content")
    if not content:
        raise LLMError(f"Unexpected Ollama response: {data!r}"[:500])
    return content


def configured() -> str | None:
    """Which backend would be used, or None."""
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OLLAMA_MODEL") or os.environ.get("OLLAMA_HOST"):
        return "ollama"
    return None


def complete(system: str, user: str) -> tuple[str, str]:
    """Return (text, backend_name)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if os.environ.get("OPENAI_API_KEY"):
        base = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        return _openai_compatible(messages, os.environ["OPENAI_API_KEY"], base, model), "openai"
    if os.environ.get("GROQ_API_KEY"):
        model = os.environ.get("GROQ_MODEL") or "llama-3.3-70b-versatile"
        return _openai_compatible(
            messages, os.environ["GROQ_API_KEY"], "https://api.groq.com/openai/v1", model
        ), "groq"
    host = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    model = os.environ.get("OLLAMA_MODEL") or "llama3.2"
    # Only hit Ollama if explicitly opted in, or if OPENAI/GROQ missing and user set OLLAMA_*
    if os.environ.get("OLLAMA_MODEL") or os.environ.get("OLLAMA_HOST"):
        return _ollama(messages, model, host), "ollama"
    raise LLMError(
        "No LLM configured. Set OPENAI_API_KEY, GROQ_API_KEY, or OLLAMA_MODEL "
        "(see .env.example)."
    )
