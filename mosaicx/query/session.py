"""Query session management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .loaders import SourceMeta, load_source


class QuerySession:
    """Stateful session for conversational Q&A over documents and data.

    Parameters
    ----------
    sources:
        List of file paths to load as data sources.
    """

    def __init__(self, sources: list[Path | str] | None = None) -> None:
        self._catalog: list[SourceMeta] = []
        self._data: dict[str, Any] = {}  # name -> loaded data
        self._conversation: list[dict[str, str]] = []
        self._closed: bool = False

        if sources:
            for src in sources:
                meta, data = load_source(src)
                self._catalog.append(meta)
                self._data[meta.name] = data

    @property
    def catalog(self) -> list[SourceMeta]:
        """Metadata for all loaded sources."""
        return list(self._catalog)

    @property
    def conversation(self) -> list[dict[str, str]]:
        """Conversation history as list of {role, content} dicts."""
        return list(self._conversation)

    @property
    def closed(self) -> bool:
        """Whether the session has been closed."""
        return self._closed

    @property
    def data(self) -> dict[str, Any]:
        """Access loaded data by source name."""
        return dict(self._data)

    def add_turn(self, role: str, content: str) -> None:
        """Append a conversation turn.

        Parameters
        ----------
        role:
            Speaker role (e.g. "user", "assistant").
        content:
            Message content.
        """
        self._conversation.append({"role": role, "content": content})

    def close(self) -> None:
        """Close the session and release resources."""
        self._closed = True
        self._data.clear()
