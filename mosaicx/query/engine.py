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
import re
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
        sub_lm: str | None = None,
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
        self._sub_lm = sub_lm

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

    @property
    def sub_lm(self) -> str | None:
        """Optional lightweight model used for RLM sub-queries."""
        return self._sub_lm

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _run_query_once(self, question: str) -> tuple[str, list[dict[str, Any]]]:
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

        # Seed retrieval to help the RLM start from concrete evidence.
        seed_hits = search_documents(question.strip(), documents=docs, top_k=6)
        retrieval_lines = []
        for h in seed_hits:
            retrieval_lines.append(
                f"- {h.get('source', 'unknown')} (score={h.get('score', 0)}): "
                f"{str(h.get('snippet', '')).strip()[:220]}"
            )
        retrieval_text = "\n".join(retrieval_lines) if retrieval_lines else "(no keyword hits)"

        rlm_kwargs: dict[str, Any] = {
            "max_iterations": self._max_iterations,
            "verbose": self._verbose,
            "tools": tools,
        }
        if self._sub_lm:
            try:
                from mosaicx.config import get_config
                from mosaicx.metrics import make_harmony_lm

                cfg = get_config()
                rlm_kwargs["sub_lm"] = make_harmony_lm(
                    self._sub_lm,
                    api_key=cfg.api_key,
                    api_base=cfg.api_base,
                    temperature=cfg.lm_temperature,
                )
            except Exception:
                # Keep query available even if sub-model init fails.
                pass

        rlm = dspy.RLM(
            "catalog, history, retrieval, question -> answer",
            **rlm_kwargs,
        )

        prediction = rlm(
            catalog=catalog_text,
            history=history_text,
            retrieval=retrieval_text,
            question=question.strip(),
        )

        answer = str(prediction.answer).strip()
        return answer, seed_hits

    def _build_citations(
        self,
        *,
        question: str,
        answer: str,
        seed_hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        from mosaicx.query.tools import search_documents

        docs = self._documents
        query = " ".join([question.strip(), answer.strip()]).strip()
        hits = search_documents(query, documents=docs, top_k=max(top_k, 6))
        combined = []
        seen = set()
        for h in [*seed_hits, *hits]:
            source = str(h.get("source") or "")
            snippet = str(h.get("snippet") or "")
            score = int(h.get("score") or 0)
            key = (source, snippet[:120])
            if not source or not snippet or key in seen:
                continue
            seen.add(key)
            combined.append({
                "source": source,
                "snippet": snippet.strip(),
                "score": score,
            })

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

    def _citation_confidence(self, question: str, citations: list[dict[str, Any]]) -> float:
        if not citations:
            return 0.0
        terms = {t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) >= 3}
        denom = max(len(terms), 1)
        max_score = max(int(c.get("score", 0)) for c in citations)
        return max(0.0, min(1.0, max_score / denom))

    def ask_structured(
        self,
        question: str,
        *,
        top_k_citations: int = 3,
    ) -> dict[str, Any]:
        """Ask a question and return an answer with grounding metadata."""
        if self._session.closed:
            raise ValueError("Cannot ask questions on a closed session.")
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        fallback_used = False
        fallback_reason: str | None = None

        try:
            answer, seed_hits = self._run_query_once(question.strip())
        except Exception as exc:
            from mosaicx.query.tools import search_documents

            fallback_used = True
            fallback_reason = f"{type(exc).__name__}: {exc}"
            seed_hits = search_documents(
                question.strip(),
                documents=self._documents,
                top_k=max(3, top_k_citations),
            )
            if seed_hits:
                top = seed_hits[0]
                snippet = " ".join(str(top.get("snippet") or "").split())
                answer = (
                    f"LLM unavailable. Best matching evidence is from "
                    f"{top.get('source', 'unknown')}: {snippet[:220]}"
                )
            else:
                answer = (
                    "LLM unavailable and no matching evidence was found in the loaded sources."
                )

        citations = self._build_citations(
            question=question.strip(),
            answer=answer,
            seed_hits=seed_hits,
            top_k=max(1, top_k_citations),
        )
        confidence = self._citation_confidence(question.strip(), citations)

        self._session.add_turn("user", question.strip())
        self._session.add_turn("assistant", answer)

        return {
            "question": question.strip(),
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "sources_consulted": sorted({c["source"] for c in citations}),
            "turn_index": len(self._session.conversation) // 2,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        }

    def ask(self, question: str) -> str:
        """Ask a question about the loaded documents.

        Returns only the answer text for backwards compatibility.
        """
        payload = self.ask_structured(question)
        return str(payload["answer"])
