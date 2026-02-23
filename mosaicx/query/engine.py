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


_NON_ANSWER_MARKERS = (
    "unable to retrieve",
    "unable to provide",
    "unable to access",
    "cannot retrieve",
    "cannot provide",
    "cannot access",
    "can't retrieve",
    "can't provide",
    "do not have access",
    "not available",
    "insufficient information",
    "no further details are available",
)
_DELTA_QUESTION_MARKERS = (
    "how much",
    "change",
    "difference",
    "delta",
    "increase",
    "decrease",
    "grew",
    "shrank",
    "went up",
    "went down",
)
_QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "what", "which",
    "who", "whom", "when", "where", "why", "how", "does", "do", "did",
    "please", "show", "tell", "about", "can", "you", "your", "my", "me",
    "from", "that", "this", "those", "these", "there", "their", "them",
    "report", "reports", "patient", "study", "scan", "ct", "mri",
}


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

        guidance = (
            "Use only grounded evidence from retrieval/tool outputs. "
            "If evidence exists, answer directly and do not claim sources are unavailable. "
            "For timeline/date questions, return concise chronological bullets. "
            "Avoid repeating headings."
        )

        rlm = dspy.RLM(
            "guidance, catalog, history, retrieval, question -> answer",
            **rlm_kwargs,
        )

        prediction = rlm(
            guidance=guidance,
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
        question_terms = {
            t
            for t in re.findall(r"[a-z0-9]+", question.lower())
            if len(t) >= 3 and t not in _QUERY_STOPWORDS
        }
        is_delta_question = self._is_delta_question(question)

        def _citation_rank(snippet: str, base_score: int) -> int:
            snip = " ".join(str(snippet).split())
            if not snip:
                return -1000
            has_mm = bool(re.search(r"\b\d+(?:\.\d+)?\s*mm\b", snip.lower()))
            has_date = bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b", snip))
            snip_terms = {
                t
                for t in re.findall(r"[a-z0-9]+", snip.lower())
                if len(t) >= 3 and t not in _QUERY_STOPWORDS
            }
            overlap = len(question_terms & snip_terms)
            rank = overlap * 20 + min(base_score, 12)
            if has_mm:
                rank += 12
            if has_date:
                rank += 5
            if len(snip) < 26 and not (has_mm or has_date):
                rank -= 8
            if snip.lower() in {
                "radiology report",
                "follow-up radiology report",
                "ct chest/abdomen/pelvis with contrast",
            }:
                rank -= 20
            return rank

        query = " ".join([question.strip(), answer.strip()]).strip()
        hits = search_documents(query, documents=docs, top_k=max(top_k, 6))
        combined = []
        seen = set()
        for h in [*seed_hits, *hits]:
            source = str(h.get("source") or "")
            snippet = " ".join(str(h.get("snippet") or "").split())
            score = int(h.get("score") or 0)
            key = (source, snippet[:120])
            if not source or not snippet or key in seen:
                continue
            rank = _citation_rank(snippet, score)
            if rank < 0:
                continue
            if is_delta_question and not re.search(r"\b\d+(?:\.\d+)?\s*mm\b", snippet.lower()):
                # For size-change questions, prefer explicit measurements.
                rank -= 25
                if rank < 0:
                    continue
            seen.add(key)
            combined.append({
                "source": source,
                "snippet": snippet,
                "score": score,
                "rank": rank,
            })

        combined.sort(key=lambda x: (x.get("rank", 0), x.get("score", 0)), reverse=True)

        # First pass: maximize source diversity.
        selected: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        for item in combined:
            src = item["source"]
            if src in seen_sources:
                continue
            selected.append(item)
            seen_sources.add(src)
            if len(selected) >= top_k:
                return selected

        # Second pass: fill remaining slots.
        for item in combined:
            if len(selected) >= top_k:
                break
            if item in selected:
                continue
            selected.append(item)
        return selected

    def _citation_confidence(self, question: str, citations: list[dict[str, Any]]) -> float:
        if not citations:
            return 0.0
        terms = {t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) >= 3}
        denom = max(len(terms), 1)
        max_score = max(int(c.get("score", 0)) for c in citations)
        return max(0.0, min(1.0, max_score / denom))

    def _is_delta_question(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        return any(marker in q for marker in _DELTA_QUESTION_MARKERS)

    def _extract_mm_values(self, text: str) -> list[float]:
        vals: list[float] = []
        for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*mm\b", text.lower()):
            try:
                vals.append(float(m.group(1)))
            except (TypeError, ValueError):
                continue
        return vals

    def _answer_conflicts_with_measurements(
        self,
        *,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> bool:
        citation_vals: list[float] = []
        for c in citations:
            citation_vals.extend(self._extract_mm_values(str(c.get("snippet") or "")))
        unique_vals = sorted(set(citation_vals))
        if len(unique_vals) < 2:
            return False

        answer_text = " ".join(answer.lower().split())
        answer_vals = set(self._extract_mm_values(answer_text))

        # Strong contradiction patterns.
        if ("no change" in answer_text or "0 mm" in answer_text or "zero mm" in answer_text) and unique_vals[0] != unique_vals[-1]:
            return True
        if answer_vals and answer_vals.isdisjoint(set(unique_vals)):
            return True
        return False

    def _deterministic_delta_answer(
        self,
        *,
        citations: list[dict[str, Any]],
    ) -> str | None:
        entries: list[tuple[str, int, str, float]] = []
        for idx, c in enumerate(citations):
            source = str(c.get("source") or "unknown")
            snippet = " ".join(str(c.get("snippet") or "").split())
            vals = self._extract_mm_values(snippet)
            if not vals:
                continue
            date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", source) or re.search(r"\b\d{4}-\d{2}-\d{2}\b", snippet)
            date_key = date_match.group(0) if date_match else f"z{idx:03d}"
            entries.append((date_key, idx, source, vals[0]))

        # Keep first measurement per source/date key.
        if len(entries) < 2:
            return None
        entries.sort(key=lambda x: (x[0], x[1]))
        start = entries[0]
        end = entries[-1]
        start_val = start[3]
        end_val = end[3]
        delta = round(end_val - start_val, 3)
        if abs(delta) < 1e-9:
            direction = "no change"
            sign_delta = "0"
        elif delta > 0:
            direction = "increase"
            sign_delta = f"+{delta:g}"
        else:
            direction = "decrease"
            sign_delta = f"{delta:g}"

        return (
            f"Grounded size change: {start_val:g} mm -> {end_val:g} mm "
            f"({sign_delta} mm, {direction}). "
            f"Sources: {start[2]} and {end[2]}."
        )

    def _looks_like_non_answer(self, answer: str) -> bool:
        text = " ".join(answer.lower().split())
        if not text:
            return True
        return any(marker in text for marker in _NON_ANSWER_MARKERS)

    def _fallback_answer_from_citations(
        self,
        *,
        question: str,
        citations: list[dict[str, Any]],
    ) -> str | None:
        if not citations:
            return None

        q = question.lower()
        is_timeline = "timeline" in q or "chronolog" in q or "over time" in q
        lines: list[str] = []

        if is_timeline:
            dated: list[tuple[str, str, str]] = []
            undated: list[tuple[str, str]] = []
            for c in citations[:6]:
                source = str(c.get("source") or "unknown")
                snippet = " ".join(str(c.get("snippet") or "").split())
                date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", snippet)
                if date_match:
                    dated.append((date_match.group(0), source, snippet))
                else:
                    undated.append((source, snippet))
            dated.sort(key=lambda x: x[0])
            if dated:
                lines.append("Timeline from grounded evidence:")
                for date_str, source, snippet in dated[:4]:
                    lines.append(f"- {date_str} | {source}: {snippet[:220]}")
            for source, snippet in undated[:2]:
                lines.append(f"- {source}: {snippet[:220]}")
        else:
            lines.append("Grounded summary from loaded sources:")
            for c in citations[:4]:
                source = str(c.get("source") or "unknown")
                snippet = " ".join(str(c.get("snippet") or "").split())
                lines.append(f"- {source}: {snippet[:240]}")

        text = "\n".join(lines).strip()
        return text or None

    def _rescue_answer_with_evidence(
        self,
        *,
        question: str,
        citations: list[dict[str, Any]],
    ) -> str | None:
        if not citations:
            return None

        try:
            import dspy
        except Exception:
            return self._fallback_answer_from_citations(question=question, citations=citations)

        evidence_blocks: list[str] = []
        for idx, c in enumerate(citations[:6], start=1):
            source = str(c.get("source") or "unknown")
            snippet = " ".join(str(c.get("snippet") or "").split())
            evidence_blocks.append(f"[{idx}] {source}: {snippet[:360]}")

        guidance = (
            "Answer the question using only the evidence snippets. "
            "Do not say the files are unavailable when evidence is present. "
            "If the question asks for a timeline, return chronological bullets with dates. "
            "Be concise and concrete."
        )
        predictor = dspy.Predict("guidance, question, evidence -> answer")
        try:
            pred = predictor(
                guidance=guidance,
                question=question.strip(),
                evidence="\n".join(evidence_blocks),
            )
            rescued = str(getattr(pred, "answer", "")).strip()
            if rescued and not self._looks_like_non_answer(rescued):
                return rescued
        except Exception:
            pass

        return self._fallback_answer_from_citations(question=question, citations=citations)

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
        rescue_used = False
        rescue_reason: str | None = None

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
                    "LLM unavailable. Could not compute a full answer; use the evidence table below."
                )

        citations = self._build_citations(
            question=question.strip(),
            answer=answer,
            seed_hits=seed_hits,
            top_k=max(1, top_k_citations),
        )
        needs_rescue = bool(citations and self._looks_like_non_answer(answer))
        if citations and self._is_delta_question(question) and self._answer_conflicts_with_measurements(
            answer=answer,
            citations=citations,
        ):
            deterministic = self._deterministic_delta_answer(citations=citations)
            if deterministic:
                answer = deterministic
                rescue_used = True
                rescue_reason = "numeric_delta_from_evidence"
                needs_rescue = False

        if needs_rescue:
            rescued = self._rescue_answer_with_evidence(
                question=question.strip(),
                citations=citations,
            )
            if rescued:
                answer = rescued
                rescue_used = True
                rescue_reason = "non_answer_with_evidence"
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
            "rescue_used": rescue_used,
            "rescue_reason": rescue_reason,
        }

    def ask(self, question: str) -> str:
        """Ask a question about the loaded documents.

        Returns only the answer text for backwards compatibility.
        """
        payload = self.ask_structured(question)
        return str(payload["answer"])
