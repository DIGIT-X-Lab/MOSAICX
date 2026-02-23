"""Query session management."""

from __future__ import annotations

import glob
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

    def __init__(
        self,
        sources: list[Path | str] | None = None,
        template: str | None = None,
        sub_lm: str | None = None,
    ) -> None:
        self._catalog: list[SourceMeta] = []
        self._data: dict[str, Any] = {}  # name -> loaded data
        self._conversation: list[dict[str, str]] = []
        self._state: dict[str, Any] = {}
        self._closed: bool = False
        self._template = template
        self._sub_lm = sub_lm

        if sources:
            for src in sources:
                for resolved in self._expand_source(src):
                    meta, data = load_source(resolved)
                    meta = meta.model_copy(update={"name": self._unique_name(meta.name)})
                    self._catalog.append(meta)
                    self._data[meta.name] = data

    def _unique_name(self, base_name: str) -> str:
        if base_name not in self._data:
            return base_name
        i = 2
        while True:
            candidate = f"{base_name} ({i})"
            if candidate not in self._data:
                return candidate
            i += 1

    def _expand_source(self, src: Path | str) -> list[Path]:
        raw = str(src)
        # Glob patterns
        if any(ch in raw for ch in ("*", "?", "[")):
            matches = [Path(p) for p in sorted(glob.glob(raw))]
            if not matches:
                raise FileNotFoundError(f"No files matched source pattern: {raw}")
            return [m for m in matches if m.is_file()]

        p = Path(src)
        if p.is_dir():
            files = sorted(x for x in p.iterdir() if x.is_file())
            if not files:
                raise ValueError(f"No files found in source directory: {p}")
            return files
        if not p.exists():
            raise FileNotFoundError(f"Source not found: {p}")
        return [p]

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
    def template(self) -> str | None:
        """Optional extraction template hint for query workflows."""
        return self._template

    @property
    def sub_lm(self) -> str | None:
        """Optional lightweight model override for sub-queries."""
        return self._sub_lm

    @property
    def data(self) -> dict[str, Any]:
        """Access loaded data by source name."""
        return dict(self._data)

    @property
    def state(self) -> dict[str, Any]:
        """Structured session state for robust multi-turn reasoning."""
        return dict(self._state)

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value by key."""
        return self._state.get(key, default)

    def set_state(self, **kwargs: Any) -> None:
        """Update one or more state values."""
        if self._closed:
            raise ValueError("Cannot update state on a closed session.")
        self._state.update(kwargs)

    def add_text_source(self, name: str, text: str) -> SourceMeta:
        """Add an in-memory text source to the session.

        Parameters
        ----------
        name:
            Source name (e.g. filename).
        text:
            Full text content.

        Returns
        -------
        SourceMeta
            Metadata for the added source.
        """
        meta = SourceMeta(
            name=name,
            format="txt",
            source_type="document",
            size=len(text.encode("utf-8")),
            preview=text[:200],
        )
        self._catalog.append(meta)
        self._data[name] = text
        return meta

    def add_turn(self, role: str, content: str) -> None:
        """Append a conversation turn.

        Parameters
        ----------
        role:
            Speaker role (e.g. "user", "assistant").
        content:
            Message content.

        Raises
        ------
        ValueError
            If the session is closed.
        """
        if self._closed:
            raise ValueError("Cannot add turns to a closed session.")
        self._conversation.append({"role": role, "content": content})

    def close(self) -> None:
        """Close the session and release resources."""
        self._closed = True
        self._data.clear()
        self._state.clear()

    def ask(
        self,
        question: str,
        *,
        max_iterations: int = 20,
        verbose: bool = False,
    ) -> str:
        """Ask a question directly on the session."""
        try:
            import dspy
        except ImportError as exc:
            raise RuntimeError(
                "DSPy is required for query.ask(). Install with: pip install dspy"
            ) from exc

        if getattr(dspy.settings, "lm", None) is None:
            # Lazy SDK-level configuration so `session.ask(...)` works out-of-the-box.
            from ..sdk import _ensure_configured

            _ensure_configured()

        from .engine import QueryEngine

        engine = QueryEngine(
            session=self,
            max_iterations=max_iterations,
            verbose=verbose,
            sub_lm=self._sub_lm,
        )
        return engine.ask(question)

    def ask_structured(
        self,
        question: str,
        *,
        max_iterations: int = 20,
        verbose: bool = False,
        top_k_citations: int = 3,
    ) -> dict[str, Any]:
        """Ask a question and return structured answer metadata.

        Returns answer text plus citations and confidence.
        """
        try:
            import dspy
        except ImportError as exc:
            raise RuntimeError(
                "DSPy is required for query.ask_structured(). Install with: pip install dspy"
            ) from exc

        if getattr(dspy.settings, "lm", None) is None:
            from ..sdk import _ensure_configured

            _ensure_configured()

        from .engine import QueryEngine

        engine = QueryEngine(
            session=self,
            max_iterations=max_iterations,
            verbose=verbose,
            sub_lm=self._sub_lm,
        )
        return engine.ask_structured(question, top_k_citations=top_k_citations)
