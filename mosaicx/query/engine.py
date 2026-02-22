# mosaicx/query/engine.py
"""RLM-powered query engine for conversational Q&A over documents and data.

Uses ``dspy.RLM`` to let a language model programmatically explore loaded
documents via MOSAICX tools (search, retrieve, save).  The engine wraps
session management, document preparation, and conversation tracking.

DSPy is imported lazily inside ``ask()`` so that the module can be imported
even when dspy is not fully configured.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mosaicx.query.session import QuerySession


def _text_for_data(value: Any) -> str:
    """Convert a loaded data value to a text representation for tools.

    Parameters
    ----------
    value:
        Loaded data -- could be str, dict, list, or a pandas DataFrame.

    Returns
    -------
    str
        Text representation suitable for keyword search.
    """
    if isinstance(value, str):
        return value

    # pandas DataFrame
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return str(value.to_string(max_rows=200))
    except ImportError:
        pass

    # dict / list -- serialize to JSON
    try:
        return json.dumps(value, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


class QueryEngine:
    """Conversational query engine backed by ``dspy.RLM``.

    Wraps a :class:`~mosaicx.query.session.QuerySession` and uses the
    Recursive Language Model to answer questions about loaded documents.

    Parameters
    ----------
    session:
        An active (not closed) :class:`~mosaicx.query.session.QuerySession`.
    max_iterations:
        Maximum REPL iterations for the RLM. Default ``20``.
    verbose:
        Whether to log detailed RLM execution info. Default ``False``.
    """

    def __init__(
        self,
        *,
        session: QuerySession,
        max_iterations: int = 20,
        verbose: bool = False,
    ) -> None:
        from mosaicx.query.session import QuerySession

        if not isinstance(session, QuerySession):
            raise TypeError(
                f"session must be a QuerySession, got {type(session).__name__}"
            )
        if session.closed:
            raise ValueError("Cannot create QueryEngine with a closed session.")

        self._session = session
        self._max_iterations = max_iterations
        self._verbose = verbose

        # Pre-compute text representations of all documents for tool use
        self._documents: dict[str, str] = {
            name: _text_for_data(data)
            for name, data in self._session.data.items()
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session(self) -> QuerySession:
        """The underlying :class:`~mosaicx.query.session.QuerySession`."""
        return self._session

    @property
    def documents(self) -> dict[str, str]:
        """Text representations of loaded documents, keyed by source name."""
        return dict(self._documents)

    @property
    def max_iterations(self) -> int:
        """Maximum REPL iterations for the RLM."""
        return self._max_iterations

    @property
    def verbose(self) -> bool:
        """Whether verbose logging is enabled."""
        return self._verbose

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        """Ask a question about the loaded documents.

        Parameters
        ----------
        question:
            Natural language question.

        Returns
        -------
        str
            The RLM-generated answer.

        Raises
        ------
        ValueError
            If the session is closed or the question is empty.
        """
        if self._session.closed:
            raise ValueError("Cannot ask questions on a closed session.")
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        import dspy

        from mosaicx.query.tools import get_document, save_artifact, search_documents

        # Build catalog summary for the RLM context
        catalog_lines = []
        for meta in self._session.catalog:
            catalog_lines.append(
                f"- {meta.name} ({meta.format}, {meta.source_type}, "
                f"{meta.size} bytes)"
            )
        catalog_text = "\n".join(catalog_lines) if catalog_lines else "(no sources)"

        # Build conversation context
        history_lines = []
        for turn in self._session.conversation:
            role = turn["role"]
            content = turn["content"]
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "(new session)"

        # Bind documents into tool closures so RLM tools have access
        docs = self._documents

        def _search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
            """Search loaded documents by keyword. Returns matching snippets."""
            return search_documents(query, documents=docs, top_k=top_k)

        def _get(name: str) -> str:
            """Retrieve a full document by name."""
            return get_document(name, documents=docs)

        def _save(data: list[dict[str, Any]] | dict[str, Any], path: str, format: str = "csv") -> str:
            """Save query results as a CSV or JSON artifact file."""
            return save_artifact(data, path, format=format)

        tools = [
            dspy.Tool(_search, name="search_documents", desc="Search loaded documents by keyword."),
            dspy.Tool(_get, name="get_document", desc="Retrieve a full document by name."),
            dspy.Tool(_save, name="save_artifact", desc="Save query results as a CSV or JSON file."),
        ]

        rlm = dspy.RLM(
            "catalog, history, question -> answer",
            max_iterations=self._max_iterations,
            verbose=self._verbose,
            tools=tools,
        )

        prediction = rlm(
            catalog=catalog_text,
            history=history_text,
            question=question.strip(),
        )

        answer = str(prediction.answer)

        # Track conversation
        self._session.add_turn("user", question.strip())
        self._session.add_turn("assistant", answer)

        return answer
