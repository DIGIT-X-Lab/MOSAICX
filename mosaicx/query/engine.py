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
    "not specified",
    "not provided",
    "not mentioned",
    "unknown",
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
    "kind", "used", "use", "throughout", "across", "overall", "imaging",
    "image", "images", "imaged",
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
            rows = int(len(value))
            cols = int(len(value.columns))
            col_preview = ", ".join([str(c) for c in value.columns[:30]])
            if cols > 30:
                col_preview += ", ..."

            # Keep prompt context compact: schema + small head/tail sample.
            head_n = min(8, rows)
            tail_n = min(4, max(0, rows - head_n))
            frames = [value.head(head_n)]
            if tail_n > 0:
                frames.append(value.tail(tail_n))
            sample = pd.concat(frames, axis=0).drop_duplicates()
            sample_text = sample.to_string(max_rows=12, max_cols=16)
            return (
                f"DataFrame {rows} rows x {cols} cols\n"
                f"Columns: {col_preview}\n"
                f"Sample rows:\n{sample_text}"
            )
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

    def _merge_hits(
        self,
        hits: list[dict[str, Any]],
        *,
        max_hits: int,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            source = str(hit.get("source") or "")
            snippet = " ".join(str(hit.get("snippet") or "").split())
            if not source or not snippet:
                continue
            key = (source, snippet[:180])
            if key in seen:
                continue
            seen.add(key)
            item = dict(hit)
            item["snippet"] = snippet
            merged.append(item)
            if len(merged) >= max_hits:
                break
        return merged

    def _run_query_once(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        import dspy

        from mosaicx.query.tools import (
            analyze_table_question,
            compute_table_stat,
            get_document,
            get_table_schema,
            list_tables,
            run_table_sql,
            sample_table_rows,
            save_artifact,
            search_documents,
            search_tables,
        )

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
        data = self._session.data

        def _search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
            """Search loaded documents by keyword. Returns matching snippets."""
            return search_documents(query, documents=docs, top_k=top_k)

        def _search_tables(query: str, top_k: int = 5) -> list[dict[str, Any]]:
            """Search tabular sources and return row-level evidence snippets."""
            return search_tables(query, data=data, top_k=top_k)

        def _list_tables() -> list[dict[str, Any]]:
            """List loaded CSV/Parquet/Excel sources with shape and columns."""
            return list_tables(data=data)

        def _table_schema(name: str, max_columns: int = 120) -> dict[str, Any]:
            """Inspect table schema: dtypes, missingness, and unique counts."""
            return get_table_schema(name, data=data, max_columns=max_columns)

        def _table_rows(name: str, columns_csv: str = "", limit: int = 5, strategy: str = "head") -> list[dict[str, Any]]:
            """Sample rows from a tabular source."""
            return sample_table_rows(
                name,
                data=data,
                columns_csv=columns_csv,
                limit=limit,
                strategy=strategy,
            )

        def _table_stat(
            name: str,
            column: str,
            operation: str = "mean",
            group_by: str = "",
            where: str = "",
            top_n: int = 10,
        ) -> list[dict[str, Any]]:
            """Compute stats from tabular data (mean/median/min/max/count/etc.)."""
            return compute_table_stat(
                name,
                data=data,
                column=column,
                operation=operation,
                group_by=group_by,
                where=where,
                top_n=top_n,
            )

        def _table_sql(name: str, sql: str, limit: int = 50) -> list[dict[str, Any]]:
            """Run a read-only SQL query over a tabular source (DuckDB)."""
            try:
                return run_table_sql(
                    name,
                    data=data,
                    sql=sql,
                    limit=limit,
                )
            except ImportError:
                return [
                    {
                        "error": "duckdb_not_installed",
                        "hint": "Install mosaicx[query] to enable run_table_sql.",
                    }
                ]

        def _analyze_table_question(question: str, top_k: int = 5) -> list[dict[str, Any]]:
            """Compute deterministic evidence snippets for common cohort-stat questions."""
            return analyze_table_question(question, data=data, top_k=top_k)

        def _get(name: str) -> str:
            """Retrieve a full document by name."""
            return get_document(name, documents=docs)

        def _save(data: list[dict[str, Any]] | dict[str, Any], path: str, format: str = "csv") -> str:
            """Save query results as a CSV or JSON artifact file."""
            return save_artifact(data, path, format=format)

        tools = [
            dspy.Tool(_search, name="search_documents", desc="Search loaded documents by keyword."),
            dspy.Tool(_search_tables, name="search_tables", desc="Search tabular sources and return row-level evidence."),
            dspy.Tool(_list_tables, name="list_tables", desc="List loaded tabular sources with row/column counts."),
            dspy.Tool(_table_schema, name="get_table_schema", desc="Get detailed schema for a table source."),
            dspy.Tool(_table_rows, name="sample_table_rows", desc="Sample rows from a table source."),
            dspy.Tool(_table_stat, name="compute_table_stat", desc="Compute statistics (mean/median/min/max/count/etc.) for a column."),
            dspy.Tool(_table_sql, name="run_table_sql", desc="Run read-only SQL over a tabular source for complex analytics."),
            dspy.Tool(_analyze_table_question, name="analyze_table_question", desc="Derive computed evidence for cohort/statistics questions."),
            dspy.Tool(_get, name="get_document", desc="Retrieve a full document by name."),
            dspy.Tool(_save, name="save_artifact", desc="Save query results as a CSV or JSON file."),
        ]

        # Seed retrieval to help the RLM start from concrete evidence.
        text_hits = search_documents(question.strip(), documents=docs, top_k=6)
        table_hits = search_tables(question.strip(), data=data, top_k=6)
        computed_hits = analyze_table_question(question.strip(), data=data, top_k=4)
        seed_hits = self._merge_hits(
            [*computed_hits, *table_hits, *text_hits],
            max_hits=12,
        )
        retrieval_lines = []
        for h in seed_hits:
            evidence_type = str(h.get("evidence_type") or "text")
            retrieval_lines.append(
                f"- [{evidence_type}] {h.get('source', 'unknown')} (score={h.get('score', 0)}): "
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
            "For CSV/Parquet/Excel questions, inspect schema and compute stats with table tools before answering. "
            "For complex cohort questions, use run_table_sql and cite computed outputs. "
            "Never estimate numeric cohort statistics from memory; compute them. "
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
        from mosaicx.query.tools import analyze_table_question, search_documents, search_tables

        docs = self._documents
        data = self._session.data
        question_terms = {
            t
            for t in re.findall(r"[a-z0-9]+", question.lower())
            if len(t) >= 3 and t not in _QUERY_STOPWORDS
        }
        is_delta_question = self._is_delta_question(question)

        def _citation_rank(snippet: str, base_score: int, evidence_type: str) -> int:
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
            if evidence_type == "table_row":
                rank += 14
            elif evidence_type == "table_stat":
                rank += 30
            return rank

        query = " ".join([question.strip(), answer.strip()]).strip()
        text_hits = search_documents(query, documents=docs, top_k=max(top_k, 6))
        table_hits = search_tables(query, data=data, top_k=max(top_k, 8))
        computed_hits = analyze_table_question(question.strip(), data=data, top_k=max(top_k, 5))
        hits = self._merge_hits([*computed_hits, *table_hits, *text_hits], max_hits=max(top_k, 12))
        combined = []
        seen = set()
        for h in [*seed_hits, *hits]:
            source = str(h.get("source") or "")
            snippet = " ".join(str(h.get("snippet") or "").split())
            score = int(h.get("score") or 0)
            evidence_type = str(h.get("evidence_type") or "text")
            key = (source, snippet[:120])
            if not source or not snippet or key in seen:
                continue
            rank = _citation_rank(snippet, score, evidence_type)
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
                "evidence_type": evidence_type,
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

    def _normalize_term(self, token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    def _salient_terms(self, text: str, *, min_len: int = 3) -> set[str]:
        return {
            self._normalize_term(t)
            for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) >= min_len and self._normalize_term(t) not in _QUERY_STOPWORDS
        }

    def _extract_answer_values(self, answer: str) -> set[str]:
        values: set[str] = set()
        lowered = answer.lower()

        # Clinical/date literals where exact match should strongly impact grounding.
        for m in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", answer):
            values.add(m.lower())
        for m in re.findall(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|%)\b", lowered):
            values.add(" ".join(m.split()))

        # Common modality literals, including short tokens that normal term logic drops.
        for modality in ("ct", "mri", "pet", "xray", "x-ray", "ultrasound", "us"):
            if re.search(rf"\b{re.escape(modality)}\b", lowered):
                values.add(modality)

        return values

    def _citation_confidence(
        self,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> float:
        if not citations:
            return 0.0

        question_terms = self._salient_terms(question, min_len=3)
        citation_blob = " ".join(
            " ".join(str(c.get("snippet") or "").split()).lower()
            for c in citations
        )
        citation_terms = self._salient_terms(citation_blob, min_len=3)

        if question_terms:
            question_coverage = len(question_terms & citation_terms) / len(question_terms)
        else:
            question_coverage = 1.0

        answer_values = self._extract_answer_values(answer)
        if answer_values:
            answer_value_support = sum(1 for v in answer_values if v in citation_blob) / len(answer_values)
        else:
            answer_value_support = None

        answer_terms = self._salient_terms(answer, min_len=2)
        if answer_terms:
            answer_term_support = len(answer_terms & citation_terms) / len(answer_terms)
        else:
            answer_term_support = None

        if answer_value_support is None and answer_term_support is None:
            answer_support = 0.5
        elif answer_value_support is None:
            answer_support = answer_term_support or 0.0
        elif answer_term_support is None:
            answer_support = answer_value_support
        else:
            answer_support = max(answer_value_support, answer_term_support)

        max_rank = max(int(c.get("rank", c.get("score", 0)) or 0) for c in citations)
        rank_strength = max(0.0, min(1.0, max_rank / 30.0))

        unique_sources = len({str(c.get("source") or "") for c in citations if c.get("source")})
        source_diversity = max(0.0, min(1.0, unique_sources / 2.0))

        confidence = (
            0.25 * rank_strength
            + 0.25 * question_coverage
            + 0.35 * answer_support
            + 0.15 * source_diversity
        )

        # Penalize literal mismatches (e.g., answer says MRI but evidence says CT).
        if answer_values and answer_value_support == 0.0:
            confidence *= 0.6
        elif answer_value_support is not None and answer_value_support < 0.5:
            confidence *= 0.8
        if answer_term_support is not None and answer_term_support < 0.2:
            confidence *= 0.85

        return max(0.0, min(1.0, confidence))

    def _is_delta_question(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        return any(marker in q for marker in _DELTA_QUESTION_MARKERS)

    def _reconcile_answer_with_evidence(
        self,
        *,
        question: str,
        draft_answer: str,
        citations: list[dict[str, Any]],
    ) -> str | None:
        """Use an LLM adjudication pass to correct unsupported draft answers."""
        if not citations:
            return None

        try:
            import dspy
        except Exception:
            return None

        evidence_blocks: list[str] = []
        for idx, c in enumerate(citations[:8], start=1):
            source = str(c.get("source") or "unknown")
            snippet = " ".join(str(c.get("snippet") or "").split())
            evidence_blocks.append(f"[{idx}] {source}: {snippet[:360]}")

        guidance = (
            "Decide whether the draft answer is fully supported by the evidence. "
            "If unsupported or contradicted, provide a corrected grounded answer using only evidence. "
            "If evidence is insufficient, say so briefly. "
            "status must be one of: supported, corrected, insufficient."
        )
        predictor = dspy.Predict("guidance, question, draft_answer, evidence -> status, revised_answer")
        try:
            pred = predictor(
                guidance=guidance,
                question=question.strip(),
                draft_answer=draft_answer.strip(),
                evidence="\n".join(evidence_blocks),
            )
        except Exception:
            return None

        status = " ".join(str(getattr(pred, "status", "")).lower().split())
        revised = str(getattr(pred, "revised_answer", "")).strip()
        if not revised:
            return None

        # Only replace when model explicitly flags mismatch or when draft was a non-answer.
        if "correct" in status or "contrad" in status or "unsupported" in status:
            return revised
        if self._looks_like_non_answer(draft_answer) and revised != draft_answer.strip():
            return revised
        return None

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
        if citations:
            reconciled = self._reconcile_answer_with_evidence(
                question=question.strip(),
                draft_answer=answer,
                citations=citations,
            )
            if reconciled and reconciled.strip() != answer.strip():
                answer = reconciled.strip()
                rescue_used = True
                rescue_reason = "evidence_reconciler"

        needs_rescue = bool(citations and self._looks_like_non_answer(answer))

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
        confidence = self._citation_confidence(question.strip(), answer, citations)

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
