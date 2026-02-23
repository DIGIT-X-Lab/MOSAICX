"""MOSAICX-specific tools available to the RLM sandbox during query sessions."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_token(token: str) -> str:
    """Lightweight token normalization for keyword matching."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _extract_terms(text: str) -> list[str]:
    return [_normalize_token(t) for t in _TOKEN_RE.findall(text.lower())]


def search_documents(
    query: str,
    *,
    documents: dict[str, str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search loaded documents by keyword.

    Parameters
    ----------
    query:
        Search query (keywords).
    documents:
        Dict mapping source name to text content.
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[dict]
        Matching results with keys: source, snippet, score.
    """
    raw_terms = _extract_terms(query)
    stopwords = {
        "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with",
        "is", "are", "was", "were", "be", "been", "being", "what", "which",
        "who", "whom", "when", "where", "why", "how", "does", "do", "did",
        "please", "show", "tell", "about", "can", "you", "your", "my", "me",
        "from", "that", "this", "those", "these", "there", "their", "them",
        "no", "patient", "study", "scan", "ct", "mri",
        "findings", "impression", "compared", "comparison", "today",
    }
    query_terms = [t for t in raw_terms if len(t) >= 2 and t not in stopwords]
    if not query_terms:
        query_terms = raw_terms
    results: list[dict[str, Any]] = []

    for name, text in documents.items():
        if not query_terms:
            continue

        tokens = _extract_terms(text)
        counts = Counter(tokens)
        term_counts = {term: counts.get(term, 0) for term in query_terms}
        score = sum(term_counts.values())
        if score > 0:
            snippet = ""
            best_line_score = 0
            for raw_line in text.splitlines():
                line = " ".join(raw_line.split()).strip()
                if not line:
                    continue
                line_tokens = Counter(_extract_terms(line))
                line_score = sum(line_tokens.get(term, 0) for term in query_terms)
                if line_score > best_line_score:
                    best_line_score = line_score
                    snippet = line

            if not snippet:
                # Anchor snippet around the strongest non-stopword match term.
                anchor = max(term_counts, key=term_counts.get)
                line_match = re.compile(rf"\b{re.escape(anchor)}s?\b", flags=re.IGNORECASE)
                match = line_match.search(text)
                idx = match.start() if match else 0
                start = max(0, idx - 120)
                end = min(len(text), idx + 320)
                snippet = " ".join(text[start:end].split())

            results.append({"source": name, "snippet": snippet, "score": score})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def get_document(name: str, *, documents: dict[str, str]) -> str:
    """Retrieve a document by name.

    Parameters
    ----------
    name:
        Source name (filename).
    documents:
        Dict mapping source name to text content.

    Returns
    -------
    str
        Full document text.

    Raises
    ------
    KeyError
        If the document is not found.
    """
    if name not in documents:
        raise KeyError(
            f"Document '{name}' not found. Available: {list(documents.keys())}"
        )
    return documents[name]


def save_artifact(
    data: list[dict[str, Any]] | dict[str, Any],
    path: Path | str,
    *,
    format: str = "csv",
) -> str:
    """Save query results as an artifact file.

    Parameters
    ----------
    data:
        Data to save. For CSV: list of dicts. For JSON: any JSON-serializable
        object.
    path:
        Output file path.
    format:
        Output format: "csv" or "json".

    Returns
    -------
    str
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        if not isinstance(data, list) or not data:
            raise ValueError("CSV format requires a non-empty list of dicts")
        fieldnames = list(data[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        path.write_text(buf.getvalue(), encoding="utf-8")
    elif format == "json":
        path.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        raise ValueError(f"Unsupported format: {format}")

    return str(path)
