"""MOSAICX-specific tools available to the RLM sandbox during query sessions."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


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
    query_lower = query.lower()
    query_terms = query_lower.split()
    results: list[dict[str, Any]] = []

    for name, text in documents.items():
        text_lower = text.lower()
        # Score by term frequency
        score = sum(text_lower.count(term) for term in query_terms)
        if score > 0:
            # Find best snippet
            idx = text_lower.find(query_terms[0])
            start = max(0, idx - 50)
            end = min(len(text), idx + 200)
            snippet = text[start:end]
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
