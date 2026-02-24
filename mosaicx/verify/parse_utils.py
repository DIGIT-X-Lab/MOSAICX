"""Utilities for parsing LLM outputs that are almost-JSON."""

from __future__ import annotations

import ast
import json
from typing import Any


def _strip_markdown_fences(text: str) -> str:
    """Remove optional markdown code fences."""
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_bracket_block(text: str, opener: str, closer: str) -> str | None:
    """Return the substring from first opener to last closer, if present."""
    start = text.find(opener)
    end = text.rfind(closer)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1].strip()


def _try_parse_candidate(text: str) -> Any | None:
    """Try JSON parse first, then safe Python literal parse."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def parse_json_like(raw: str) -> Any | None:
    """Parse JSON-like LLM output into Python objects.

    Handles:
    - strict JSON
    - fenced JSON blocks
    - Python literal dict/list strings with single quotes / True / False / None
    - prose-wrapped object/array outputs by extracting likely bracket blocks
    """
    text = _strip_markdown_fences(raw or "")
    if not text:
        return None

    candidates: list[str] = [text]
    for opener, closer in (("{", "}"), ("[", "]")):
        block = _extract_bracket_block(text, opener, closer)
        if block and block not in candidates:
            candidates.append(block)

    for candidate in candidates:
        parsed = _try_parse_candidate(candidate)
        if parsed is not None:
            return parsed
    return None
