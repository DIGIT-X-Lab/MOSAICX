# mosaicx/query/engine.py
"""RLM-powered query engine for conversational Q&A over documents and data.

Uses ``dspy.RLM`` to let a language model programmatically explore loaded
documents via MOSAICX tools (search, retrieve, save).  The engine wraps
session management, document preparation, and conversation tracking.

DSPy is imported lazily inside ``ask()`` so that the module can be imported
even when dspy is not fully configured.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import TYPE_CHECKING, Any

from mosaicx.runtime_env import import_dspy
from mosaicx.query.control_plane import (
    EvidenceVerifier,
    IntentDecision,
    IntentRouter,
    ProgrammaticTableAnalyst,
    ReActTabularPlanner,
)

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
_COREFERENCE_MARKERS = {
    "they",
    "them",
    "those",
    "that",
    "it",
    "this",
    "these",
    "ones",
}
_VALUE_LIST_MARKERS = (
    "what are they",
    "which ones",
    "list them",
    "what are those",
    "what are the values",
    "what are the categories",
)
_SCHEMA_QUESTION_MARKERS = (
    "column names",
    "column name",
    "all columns",
    "all of the column names",
    "field names",
    "column headers",
    "table headers",
    "list columns",
    "what are the columns",
)
_SCHEMA_ENTITY_HINTS = {
    "column",
    "columns",
    "colum",
    "colums",
    "field",
    "fields",
    "header",
    "headers",
    "feature",
    "features",
    "variable",
    "variables",
}
_NUMERIC_STAT_MARKERS = (
    "average",
    "mean",
    "median",
    "min",
    "minimum",
    "max",
    "maximum",
    "sum",
    "std",
    "standard deviation",
    "percent",
    "ratio",
    "correlation",
    "change",
    "difference",
)
_DISTRIBUTION_MARKERS = (
    "distribution",
    "breakdown",
    "split by",
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


def _is_adapter_parse_exception(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    return (
        "adapterparseerror" in lowered
        or "jsonadapter" in lowered
        or "cannot be serialized to a json object" in lowered
    )


def _build_twostep_adapter(dspy: Any) -> Any | None:
    """Construct TwoStepAdapter across DSPy versions."""
    if not hasattr(dspy, "TwoStepAdapter"):
        return None
    lm = getattr(getattr(dspy, "settings", None), "lm", None)
    try:
        return dspy.TwoStepAdapter(extraction_model=lm)
    except TypeError:
        try:
            return dspy.TwoStepAdapter()
        except Exception:
            return None
    except Exception:
        return None


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
        self._document_chunk_index: dict[str, list[dict[str, Any]]] | None = None
        self._intent_router = IntentRouter()
        self._react_planner = ReActTabularPlanner()
        self._programmatic_analyst = ProgrammaticTableAnalyst()
        self._evidence_verifier = EvidenceVerifier()

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

    def _ensure_document_chunk_index(self) -> dict[str, list[dict[str, Any]]]:
        if self._document_chunk_index is not None:
            return self._document_chunk_index
        try:
            from mosaicx.query.tools import build_document_chunks
        except Exception:
            self._document_chunk_index = {}
            return self._document_chunk_index

        long_docs = {
            name: text
            for name, text in self._documents.items()
            if len(str(text or "")) >= 1200
        }
        if not long_docs:
            self._document_chunk_index = {}
            return self._document_chunk_index

        try:
            self._document_chunk_index = build_document_chunks(
                long_docs,
                chunk_chars=1800,
                overlap_chars=220,
                max_chunks_per_document=500,
            )
        except Exception:
            self._document_chunk_index = {}
        return self._document_chunk_index

    def _run_query_once(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        dspy = import_dspy()

        from mosaicx.query.tools import (
            analyze_table_question,
            get_document_chunk,
            compute_table_stat,
            get_document,
            get_table_schema,
            list_document_chunks,
            list_distinct_values,
            list_tables,
            profile_table,
            run_table_sql,
            sample_table_rows,
            save_artifact,
            search_document_chunks,
            search_documents,
            search_tables,
            suggest_table_columns,
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
        history_text = self._recent_history_text(turns=8, max_chars=360)

        # Bind documents into tool closures so RLM tools have access
        docs = self._documents
        data = self._session.data
        chunk_index = self._ensure_document_chunk_index()

        def _search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
            """Search loaded documents by keyword. Returns matching snippets."""
            return search_documents(query, documents=docs, top_k=top_k)

        def _search_chunks(query: str, top_k: int = 8) -> list[dict[str, Any]]:
            """Search long documents at chunk granularity for deep evidence."""
            return search_document_chunks(
                query,
                documents=docs,
                top_k=top_k,
                chunk_index=chunk_index,
            )

        def _list_chunks(source: str = "", limit: int = 200) -> list[dict[str, Any]]:
            """List available long-document chunks with chunk ids and ranges."""
            return list_document_chunks(
                chunk_index=chunk_index,
                source=source,
                limit=limit,
            )

        def _get_chunk(source: str, chunk_id: int) -> dict[str, Any]:
            """Retrieve one long-document chunk by source name and chunk id."""
            return get_document_chunk(
                source,
                chunk_id,
                chunk_index=chunk_index,
            )

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

        def _table_profile(name: str, max_columns: int = 80) -> dict[str, Any]:
            """Create schema-agnostic profile with roles, missingness, and summary stats."""
            return profile_table(
                name,
                data=data,
                max_columns=max_columns,
            )

        def _list_distinct_values(name: str, column: str, where: str = "", limit: int = 25) -> list[dict[str, Any]]:
            """List distinct values for a column with counts."""
            return list_distinct_values(
                name,
                data=data,
                column=column,
                where=where,
                limit=limit,
            )

        def _suggest_table_columns(question: str, top_k: int = 8) -> list[dict[str, Any]]:
            """Suggest likely columns for a question across loaded tables."""
            return suggest_table_columns(
                question,
                data=data,
                top_k=top_k,
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
            dspy.Tool(_search_chunks, name="search_document_chunks", desc="Search long documents at chunk-level for deeper evidence."),
            dspy.Tool(_list_chunks, name="list_document_chunks", desc="List long-document chunks with chunk ids and ranges."),
            dspy.Tool(_get_chunk, name="get_document_chunk", desc="Retrieve one long-document chunk by source and chunk id."),
            dspy.Tool(_search_tables, name="search_tables", desc="Search tabular sources and return row-level evidence."),
            dspy.Tool(_list_tables, name="list_tables", desc="List loaded tabular sources with row/column counts."),
            dspy.Tool(_table_schema, name="get_table_schema", desc="Get detailed schema for a table source."),
            dspy.Tool(_table_profile, name="profile_table", desc="Create schema-agnostic profile with role inference and summary stats."),
            dspy.Tool(_list_distinct_values, name="list_distinct_values", desc="List distinct values in a column with counts."),
            dspy.Tool(_suggest_table_columns, name="suggest_table_columns", desc="Suggest likely columns for the current question."),
            dspy.Tool(_table_rows, name="sample_table_rows", desc="Sample rows from a table source."),
            dspy.Tool(_table_stat, name="compute_table_stat", desc="Compute statistics (mean/median/min/max/count/etc.) for a column."),
            dspy.Tool(_table_sql, name="run_table_sql", desc="Run read-only SQL over a tabular source for complex analytics."),
            dspy.Tool(_analyze_table_question, name="analyze_table_question", desc="Derive computed evidence for cohort/statistics questions."),
            dspy.Tool(_get, name="get_document", desc="Retrieve a full document by name."),
            dspy.Tool(_save, name="save_artifact", desc="Save query results as a CSV or JSON file."),
        ]

        # Seed retrieval to help the RLM start from concrete evidence.
        text_hits = search_documents(question.strip(), documents=docs, top_k=6)
        chunk_hits = search_document_chunks(
            question.strip(),
            documents=docs,
            top_k=8,
            chunk_index=chunk_index,
        )
        table_hits = search_tables(question.strip(), data=data, top_k=6)
        computed_hits = analyze_table_question(question.strip(), data=data, top_k=4)
        column_hits = suggest_table_columns(question.strip(), data=data, top_k=6)
        seed_hits = self._merge_hits(
            [*computed_hits, *column_hits, *table_hits, *chunk_hits, *text_hits],
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
            "For CSV/Parquet/Excel questions, inspect schema, profile tables, and compute stats with table tools before answering. "
            "For category listing questions, use list_distinct_values and include concrete values. "
            "Resolve follow-up pronouns (e.g., 'what are they') using recent user turns. "
            "Use suggest_table_columns when column names are unfamiliar. "
            "For complex cohort questions, use run_table_sql and cite computed outputs. "
            "For subject/patient counts, use deterministic distinct counts (nunique), not sampled rows. "
            "Never estimate numeric cohort statistics from memory; compute them. "
            "For long documents, use search_document_chunks then get_document_chunk for precise evidence. "
            "For timeline/date questions, return concise chronological bullets. "
            "Avoid repeating headings."
        )

        rlm = dspy.RLM(
            "guidance, catalog, history, retrieval, question -> answer",
            **rlm_kwargs,
        )

        def _invoke_rlm():
            return rlm(
                guidance=guidance,
                catalog=catalog_text,
                history=history_text,
                retrieval=retrieval_text,
                question=question.strip(),
            )

        try:
            prediction = _invoke_rlm()
        except Exception as exc:
            adapter = _build_twostep_adapter(dspy)
            if not _is_adapter_parse_exception(exc) or adapter is None:
                raise
            with dspy.context(adapter=adapter):
                prediction = _invoke_rlm()

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
        from mosaicx.query.tools import (
            analyze_table_question,
            search_document_chunks,
            search_documents,
            search_tables,
            suggest_table_columns,
        )

        docs = self._documents
        data = self._session.data
        question_terms = {
            t
            for t in re.findall(r"[a-z0-9]+", question.lower())
            if len(t) >= 3 and t not in _QUERY_STOPWORDS
        }
        is_delta_question = self._is_delta_question(question)
        is_count_plus_values = self._is_count_plus_values_request(question)
        needs_count_evidence = (
            ("how many" in question.lower())
            or ("number of" in question.lower())
            or ("count" in question.lower())
        )
        is_schema_question = self._is_schema_question(question)
        is_schema_count = (
            self._is_schema_count_question(question)
            or self._looks_like_schema_count_answer(answer)
        )
        tabular_evidence_types = {
            "table_row",
            "table_stat",
            "table_sql",
            "table_value",
            "table_schema",
            "table_column",
        }
        if is_schema_question:
            allowed = {"table_schema", "table_column", "text_chunk"}
            if is_schema_count:
                allowed = {"table_schema"}
            schema_hits = [
                h
                for h in seed_hits
                if str(h.get("evidence_type") or "").strip().lower() in allowed
            ]
            if schema_hits:
                return self._merge_hits(schema_hits, max_hits=max(top_k, 6))[: max(1, top_k)]

        focus_source = ""
        focus_column = ""
        for hit in seed_hits:
            source = str(hit.get("source") or "").strip()
            column = str(hit.get("column") or "").strip()
            evidence_type = str(hit.get("evidence_type") or "").strip().lower()
            if source and evidence_type in tabular_evidence_types and not focus_source:
                focus_source = source
            if column and evidence_type in {"table_stat", "table_value", "table_column"}:
                focus_column = column
            if focus_source and focus_column:
                break
        if not focus_source:
            focus_source = str(self._session.get_state("last_tabular_source", "")).strip()
        if not focus_column:
            focus_column = str(self._session.get_state("last_tabular_column", "")).strip()
        is_category_breakdown = self._is_category_count_breakdown_request(
            question,
            column=focus_column,
        )

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
            elif evidence_type == "table_sql":
                rank += 28
            elif evidence_type == "table_value":
                rank += 26
            elif evidence_type == "table_schema":
                rank += 22
            elif evidence_type == "table_column":
                rank += 18
            elif evidence_type == "text_chunk":
                rank += 16
            return rank

        query = " ".join([question.strip(), answer.strip()]).strip()
        text_hits = search_documents(query, documents=docs, top_k=max(top_k, 6))
        chunk_hits = search_document_chunks(
            query,
            documents=docs,
            top_k=max(top_k, 8),
            chunk_index=self._ensure_document_chunk_index(),
        )
        table_hits = search_tables(query, data=data, top_k=max(top_k, 8))
        computed_hits = analyze_table_question(question.strip(), data=data, top_k=max(top_k, 5))
        column_hits = suggest_table_columns(question.strip(), data=data, top_k=max(top_k, 6))
        hits = self._merge_hits(
            [*computed_hits, *column_hits, *table_hits, *chunk_hits, *text_hits],
            max_hits=max(top_k, 14),
        )
        combined = []
        seen = set()
        for h in [*seed_hits, *hits]:
            source = str(h.get("source") or "")
            snippet = " ".join(str(h.get("snippet") or "").split())
            score = int(h.get("score") or 0)
            evidence_type = str(h.get("evidence_type") or "text")
            evidence_type_lc = evidence_type.strip().lower()
            key = (source, snippet[:120])
            if not source or not snippet or key in seen:
                continue
            rank = _citation_rank(snippet, score, evidence_type)
            if rank < 0:
                continue
            column = str(h.get("column") or "").strip()
            if focus_source and evidence_type_lc in tabular_evidence_types:
                if source == focus_source:
                    rank += 10
                else:
                    rank -= 18
            if focus_column and evidence_type_lc in {"table_stat", "table_value", "table_column"}:
                if column == focus_column:
                    rank += 22
                elif column:
                    rank -= 35
            if is_delta_question and not re.search(r"\b\d+(?:\.\d+)?\s*mm\b", snippet.lower()):
                # For size-change questions, prefer explicit measurements.
                rank -= 25
                if rank < 0:
                    continue
            seen.add(key)
            item: dict[str, Any] = {
                "source": source,
                "snippet": snippet,
                "score": score,
                "rank": rank,
                "evidence_type": evidence_type,
            }
            for extra_key in (
                "chunk_id",
                "start",
                "end",
                "column",
                "value",
                "count",
                "operation",
                "backend",
                "sql",
                "row_count",
            ):
                if extra_key in h:
                    item[extra_key] = h.get(extra_key)
            combined.append(item)

        combined.sort(key=lambda x: (x.get("rank", 0), x.get("score", 0)), reverse=True)

        # Prefer evidence-type diversity for count+enumeration questions.
        selected: list[dict[str, Any]] = []
        selected_keys: set[tuple[str, str]] = set()

        def _add_selected(item: dict[str, Any]) -> None:
            key = (str(item.get("source") or ""), str(item.get("snippet") or "")[:120])
            if key in selected_keys:
                return
            selected.append(item)
            selected_keys.add(key)

        if is_count_plus_values or is_category_breakdown:
            focused_value_items = [
                item
                for item in combined
                if (
                    str(item.get("evidence_type") or "") == "table_value"
                    and (not focus_source or str(item.get("source") or "") == focus_source)
                    and (not focus_column or str(item.get("column") or "") == focus_column)
                )
            ]
            fallback_value_items = [
                item
                for item in combined
                if str(item.get("evidence_type") or "") == "table_value"
            ]
            for value_item in (focused_value_items or fallback_value_items)[:4]:
                _add_selected(value_item)
            if needs_count_evidence:
                count_item = next(
                    (
                        item
                        for item in combined
                        if (
                            str(item.get("evidence_type") or "") in {"table_stat", "table_sql"}
                            and (not focus_source or str(item.get("source") or "") == focus_source)
                            and (
                                str(item.get("evidence_type") or "") == "table_sql"
                                or not focus_column
                                or str(item.get("column") or "") == focus_column
                            )
                        )
                    ),
                    None,
                )
                if count_item is None:
                    count_item = next(
                        (
                            item
                            for item in combined
                            if str(item.get("evidence_type") or "") in {"table_stat", "table_sql"}
                        ),
                        None,
                    )
                if count_item is not None:
                    _add_selected(count_item)

        # Keep at least one chunk-level citation when available for long-document grounding.
        chunk_item = next(
            (item for item in combined if str(item.get("evidence_type") or "") == "text_chunk"),
            None,
        )
        if chunk_item is not None and len(selected) < top_k:
            _add_selected(chunk_item)

        # First pass: maximize source diversity.
        seen_sources: set[str] = {str(item.get("source") or "") for item in selected}
        for item in combined:
            src = item["source"]
            if src in seen_sources:
                continue
            _add_selected(item)
            seen_sources.add(src)
            if len(selected) >= top_k:
                return selected

        # Second pass: fill remaining slots.
        for item in combined:
            if len(selected) >= top_k:
                break
            _add_selected(item)
        return selected

    def _normalize_term(self, token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    def _compact_history_content(self, text: str, *, max_chars: int = 360) -> str:
        value = " ".join(str(text).split())
        if len(value) <= max_chars:
            return value
        clipped = value[: max_chars - 1].rstrip()
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        return clipped + "…"

    def _recent_history_text(self, *, turns: int = 8, max_chars: int = 360) -> str:
        lines: list[str] = []
        for turn in self._session.conversation[-max(1, turns):]:
            role = str(turn.get("role") or "")
            content = self._compact_history_content(
                str(turn.get("content") or ""),
                max_chars=max_chars,
            )
            lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "(new session)"

    def _build_table_profiles(
        self,
        *,
        max_columns: int = 120,
        top_values: int = 4,
    ) -> dict[str, dict[str, Any]]:
        from mosaicx.query.tools import profile_table

        tabular_sources = self._tabular_source_names()
        if not tabular_sources:
            return {}

        cached = self._session.get_state("_table_profiles")
        if isinstance(cached, dict):
            if set(cached.keys()) == set(tabular_sources):
                return cached

        profiles: dict[str, dict[str, Any]] = {}
        for source_name in tabular_sources:
            try:
                profiles[source_name] = profile_table(
                    source_name,
                    data=self._session.data,
                    max_columns=max_columns,
                    top_values=top_values,
                )
            except Exception:
                continue

        if profiles:
            self._session.set_state(_table_profiles=profiles)
        return profiles

    def _is_coreference_followup(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        q_terms = {
            self._normalize_term(t)
            for t in re.findall(r"[a-z0-9]+", q)
            if len(t) >= 2
        }
        if any(marker in q for marker in _VALUE_LIST_MARKERS):
            topical_terms = {
                t
                for t in q_terms
                if t not in _COREFERENCE_MARKERS and t not in _QUERY_STOPWORDS
            }
            # Treat as follow-up only when the user did not provide a concrete topic.
            if not topical_terms:
                return True
            return False
        return bool(q_terms & _COREFERENCE_MARKERS) and len(q_terms) <= 10

    def _latest_user_question(self) -> str | None:
        for turn in reversed(self._session.conversation):
            if str(turn.get("role") or "").strip().lower() != "user":
                continue
            content = str(turn.get("content") or "").strip()
            if content:
                return content
        return None

    def _latest_assistant_answer(self) -> str | None:
        for turn in reversed(self._session.conversation):
            if str(turn.get("role") or "").strip().lower() != "assistant":
                continue
            content = str(turn.get("content") or "").strip()
            if content:
                return content
        return None

    def _coerce_string_list(self, value: Any, *, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if not text or text in out:
                continue
            out.append(text)
            if len(out) >= max(1, limit):
                break
        return out

    def _rewrite_followup_with_state(
        self,
        *,
        question: str,
        prior_question: str,
        prior_answer: str,
        query_state: dict[str, Any],
    ) -> str | None:
        active_columns = self._coerce_string_list(query_state.get("active_columns"), limit=6)
        active_sources = self._coerce_string_list(query_state.get("active_sources"), limit=4)
        entities = self._coerce_string_list(query_state.get("entities"), limit=6)
        metrics = self._coerce_string_list(query_state.get("metrics"), limit=6)
        timeframe = str(query_state.get("timeframe") or "").strip()

        if not any((active_columns, active_sources, entities, metrics, timeframe)):
            return None

        state_payload = {
            "active_columns": active_columns,
            "active_sources": active_sources,
            "entities": entities,
            "metrics": metrics,
            "timeframe": timeframe,
        }
        state_json = json.dumps(state_payload, ensure_ascii=False)

        try:
            dspy = import_dspy()

            if getattr(dspy.settings, "lm", None) is not None:
                predictor = dspy.Predict(
                    "guidance, question, prior_question, prior_answer, state_json -> standalone_question"
                )
                guidance = (
                    "Rewrite follow-up into a standalone question. "
                    "Use only details from prior turns/state_json. "
                    "Prefer explicit table/column mentions when available. "
                    "Return only the rewritten question text."
                )
                pred = predictor(
                    guidance=guidance,
                    question=question.strip(),
                    prior_question=self._compact_history_content(prior_question or "", max_chars=220),
                    prior_answer=self._compact_history_content(prior_answer or "", max_chars=220),
                    state_json=state_json,
                )
                rewritten = " ".join(str(getattr(pred, "standalone_question", "")).split())
                if rewritten and rewritten != question.strip():
                    return rewritten
        except Exception:
            pass

        first_col = active_columns[0] if active_columns else ""
        first_source = active_sources[0] if active_sources else ""
        q = " ".join(question.strip().split())
        if self._has_value_list_marker(q) and first_col:
            if first_source:
                return f"What are the distinct values of {first_col} in {first_source}?"
            return f"What are the distinct values of {first_col}?"
        if first_col:
            if first_source:
                return f"{q} about {first_col} in {first_source}"
            return f"{q} about {first_col}"
        if entities:
            return f"{q} about {entities[0]}"
        return None

    def _resolve_followup_question(self, question: str) -> str:
        q = question.strip()
        last_intent = str(self._session.get_state("last_intent", "")).strip().lower()
        state_source = str(self._session.get_state("last_tabular_source", "")).strip()
        if last_intent in {"schema", "schema_count"}:
            if self._is_ambiguous_count_question(q):
                if state_source:
                    return f"How many columns are there in {state_source}?"
                return "How many columns are there?"
            if self._has_value_list_marker(q) and self._is_coreference_followup(q):
                if state_source:
                    return f"What are the column names in {state_source}?"
                return "What are the column names?"

        if not self._is_coreference_followup(q):
            return q

        query_state = self._session.get_state("query_state", {})
        if not isinstance(query_state, dict):
            query_state = {}

        state_column = str(self._session.get_state("last_tabular_column", "")).strip()
        if state_source:
            query_state.setdefault("active_sources", [])
            if isinstance(query_state.get("active_sources"), list):
                query_state["active_sources"] = [state_source, *query_state["active_sources"]]
        if state_column:
            query_state.setdefault("active_columns", [])
            if isinstance(query_state.get("active_columns"), list):
                query_state["active_columns"] = [state_column, *query_state["active_columns"]]

        rewritten = self._rewrite_followup_with_state(
            question=q,
            prior_question=self._latest_user_question() or "",
            prior_answer=self._latest_assistant_answer() or "",
            query_state=query_state,
        )
        if rewritten:
            return rewritten
        return q

    def _parse_json_object(self, raw: str) -> dict[str, Any] | None:
        text = str(raw or "").strip()
        if not text:
            return None
        candidates = [text]
        if "```" in text:
            fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
            candidates.extend(fenced)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    def _update_structured_query_state(
        self,
        *,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
        route: IntentDecision,
    ) -> None:
        prev = self._session.get_state("query_state", {})
        state: dict[str, Any] = dict(prev) if isinstance(prev, dict) else {}

        sources = sorted({str(c.get("source") or "").strip() for c in citations if c.get("source")})
        columns = sorted({str(c.get("column") or "").strip() for c in citations if c.get("column")})
        if not columns and route.intent != "schema":
            state_col = str(self._session.get_state("last_tabular_column", "")).strip()
            if state_col:
                columns = [state_col]
        values = sorted({str(c.get("value") or "").strip() for c in citations if c.get("value") not in {None, ""}})
        if sources:
            state["active_sources"] = sources[:6]
        if route.intent == "schema":
            state["active_columns"] = []
            state["schema_focus"] = True
        elif columns:
            state["active_columns"] = columns[:10]
        state["last_route_intent"] = route.intent
        if values:
            state["recent_values"] = values[:10]

        # Optional DSPy pass to maintain compact semantic memory across turns.
        try:
            dspy = import_dspy()

            if getattr(dspy.settings, "lm", None) is not None:
                citation_summary = "\n".join(
                    f"- {str(c.get('source') or 'unknown')} | "
                    f"column={str(c.get('column') or '')} | "
                    f"value={str(c.get('value') or '')} | "
                    f"{' '.join(str(c.get('snippet') or '').split())[:180]}"
                    for c in citations[:8]
                )
                predictor = dspy.Predict(
                    "guidance, prior_state_json, question, answer, citation_summary "
                    "-> state_json"
                )
                guidance = (
                    "Update memory state for future follow-up questions. "
                    "Return strict JSON object with keys: "
                    "entities (list[str]), metrics (list[str]), active_columns (list[str]), "
                    "active_sources (list[str]), timeframe (str), unresolved_reference (bool). "
                    "Use empty lists/empty string when unknown."
                )
                pred = predictor(
                    guidance=guidance,
                    prior_state_json=json.dumps(state, ensure_ascii=False),
                    question=question.strip(),
                    answer=answer.strip(),
                    citation_summary=citation_summary or "(none)",
                )
                parsed = self._parse_json_object(str(getattr(pred, "state_json", "")))
                if isinstance(parsed, dict):
                    for key in (
                        "entities",
                        "metrics",
                        "active_columns",
                        "active_sources",
                        "timeframe",
                        "unresolved_reference",
                    ):
                        if key in parsed:
                            state[key] = parsed[key]
        except Exception:
            pass

        self._session.set_state(query_state=state)

    def _is_count_plus_values_request(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        has_count = ("how many" in q) or ("number of" in q) or ("count" in q)
        if not has_count:
            return False
        return bool(
            re.search(r"\band\b[^?]{0,160}\bwhat\s+are\b", q)
            or re.search(r"\band\b[^?]{0,160}\bwhich\b", q)
            or re.search(r"\band\b[^?]{0,160}\blist\b", q)
        )

    def _looks_like_schema_token(self, token: str) -> bool:
        value = self._normalize_term(str(token).strip().lower())
        if not value:
            return False
        if value in _SCHEMA_ENTITY_HINTS:
            return True
        column_like = {"column", "columns", "colum", "colums"}
        close = difflib.get_close_matches(
            value,
            list(column_like),
            n=1,
            cutoff=0.8,
        )
        return bool(close)

    def _is_ambiguous_count_question(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        if not (("how many" in q) or ("number of" in q) or ("count" in q)):
            return False
        tokens = [self._normalize_term(t) for t in re.findall(r"[a-z0-9]+", q)]
        informative = [
            t
            for t in tokens
            if t
            and t not in _QUERY_STOPWORDS
            and t not in {"how", "many", "number", "count", "total", "there"}
        ]
        return len(informative) == 0

    def _has_value_list_marker(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        return any(marker in q for marker in _VALUE_LIST_MARKERS) or self._is_count_plus_values_request(q)

    def _is_schema_count_question(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        if not self._is_schema_question(q):
            return False
        return ("how many" in q) or ("number of" in q) or ("count" in q) or ("total" in q)

    def _is_schema_question(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        if any(marker in q for marker in _SCHEMA_QUESTION_MARKERS):
            return True

        tokens = [self._normalize_term(t) for t in re.findall(r"[a-z0-9]+", q)]
        has_schema_entity = any(self._looks_like_schema_token(t) for t in tokens)
        if not has_schema_entity:
            return False

        schema_context_terms = {
            "name",
            "names",
            "list",
            "all",
            "different",
            "header",
            "headers",
            "field",
            "fields",
            "feature",
            "features",
            "variable",
            "variables",
            "count",
            "number",
            "many",
            "total",
            "what",
            "which",
            "how",
        }
        return bool(set(tokens) & schema_context_terms)

    def _detect_aggregate_operation(self, question: str) -> str | None:
        q = " ".join(question.lower().split())
        q = re.sub(r"\bi\s+mean\b", " ", q)
        if "average" in q or "mean" in q or "avg" in q:
            return "mean"
        if "median" in q:
            return "median"
        if "minimum" in q or "lowest" in q or re.search(r"\bmin\b", q):
            return "min"
        if "maximum" in q or "highest" in q or re.search(r"\bmax\b", q):
            return "max"
        if "sum" in q or "total" in q:
            return "sum"
        if "standard deviation" in q or re.search(r"\bstd\b", q):
            return "std"
        return None

    def _is_distribution_request(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        return any(marker in q for marker in _DISTRIBUTION_MARKERS)

    def _is_category_count_breakdown_request(
        self,
        question: str,
        *,
        column: str = "",
        distinct_rows: list[dict[str, Any]] | None = None,
    ) -> bool:
        q = " ".join(question.lower().split())
        has_count = ("how many" in q) or ("number of" in q) or ("count" in q)
        if not has_count:
            return False
        if self._is_distribution_request(question):
            return True
        if any(
            marker in q
            for marker in (
                "male",
                "female",
                "sex",
                "gender",
                "ethnicity",
                "race",
                "group",
                "groups",
                "category",
                "categories",
            )
        ):
            return True
        if column:
            col_text = re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip()
            if col_text and col_text in q and " and " in q:
                return True
        values = distinct_rows or []
        if values and " and " in q:
            matched = 0
            for row in values[:10]:
                value = str(row.get("value") or "").strip().lower()
                if not value:
                    continue
                if re.search(rf"\b{re.escape(value)}\b", q):
                    matched += 1
                if matched >= 2:
                    return True
        return False

    def _tabular_source_names(self) -> list[str]:
        out: list[str] = []
        for meta in self._session.catalog:
            if getattr(meta, "source_type", "") != "dataframe":
                continue
            name = str(getattr(meta, "name", "") or "").strip()
            if name:
                out.append(name)
        return out

    def _resolve_source_from_question(
        self,
        question: str,
        sources: list[str],
    ) -> str | None:
        q = " ".join(question.lower().split())
        for source in sources:
            if source.lower() in q:
                return source
        return None

    def _try_deterministic_schema_answer(
        self,
        question: str,
    ) -> tuple[str, list[dict[str, Any]], str] | None:
        if not self._has_tabular_sources() or not self._is_schema_question(question):
            return None

        try:
            import pandas as pd
        except Exception:
            return None

        sources = self._tabular_source_names()
        if not sources:
            return None

        explicit = self._resolve_source_from_question(question, sources)
        state_source = str(self._session.get_state("last_tabular_source", "")).strip()
        if explicit:
            selected = [explicit]
        elif state_source and state_source in sources:
            selected = [state_source]
        elif len(sources) == 1:
            selected = [sources[0]]
        else:
            selected = sources

        wants_schema_count = self._is_schema_count_question(question)
        seed_hits: list[dict[str, Any]] = []
        lines: list[str] = []
        for source in selected:
            value = self._session.data.get(source)
            if not isinstance(value, pd.DataFrame):
                continue
            cols = [str(c) for c in value.columns]
            preview = ", ".join(cols[:40]) + (", ..." if len(cols) > 40 else "")
            seed_hits.append(
                {
                    "source": source,
                    "snippet": f"Schema columns ({len(cols)}): {preview}",
                    "score": 90,
                    "evidence_type": "table_schema",
                    "column_count": len(cols),
                }
            )
            if wants_schema_count:
                lines.append(f"There are {len(cols)} columns in {source}.")
            elif len(selected) == 1:
                lines.append(f"Columns in {source} ({len(cols)}): {', '.join(cols)}")
            else:
                lines.append(f"{source} ({len(cols)}): {', '.join(cols)}")
            self._session.set_state(
                last_tabular_source=source,
                last_intent="schema_count" if wants_schema_count else "schema",
            )

        if not lines:
            return None
        return (
            "\n".join(lines),
            self._merge_hits(seed_hits, max_hits=12),
            "schema_count" if wants_schema_count else "schema",
        )

    def _normalize_planned_column_name(self, df: Any, column: str) -> str:
        """Normalize planner-returned column names to exact dataframe headers."""
        value = str(column or "").strip()
        if not value:
            return ""
        col_map = {str(c): str(c) for c in df.columns}
        if value in col_map:
            return col_map[value]
        col_norm_map = {
            re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_"): str(c)
            for c in df.columns
        }
        value_norm = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return col_norm_map.get(value_norm, "")

    def _resolve_planned_column_with_llm(
        self,
        *,
        question: str,
        source: str,
        intent: str,
        df: Any,
        table_profiles: dict[str, dict[str, Any]],
    ) -> str:
        """Recover missing planner column using LM + session state."""
        available_columns = [str(c) for c in df.columns]
        if not available_columns:
            return ""

        state_column = str(self._session.get_state("last_tabular_column", "")).strip()
        if state_column in available_columns and (
            self._is_coreference_followup(question)
            or self._has_value_list_marker(question)
            or self._is_ambiguous_count_question(question)
        ):
            return state_column

        try:
            dspy = import_dspy()
        except Exception:
            return ""
        if getattr(dspy.settings, "lm", None) is None:
            return ""

        profile = table_profiles.get(source, {})
        col_lines: list[str] = []
        for col in profile.get("columns", [])[:140]:
            if not isinstance(col, dict):
                continue
            name = str(col.get("name") or "").strip()
            if not name:
                continue
            role = str(col.get("role") or "unknown")
            preview_values: list[str] = []
            top_values = col.get("top_values")
            if isinstance(top_values, list):
                for item in top_values[:3]:
                    if isinstance(item, dict) and item.get("value") is not None:
                        preview_values.append(str(item.get("value")))
            if preview_values:
                col_lines.append(f"- {name} [{role}] values={','.join(preview_values)}")
            else:
                col_lines.append(f"- {name} [{role}]")

        if not col_lines:
            col_lines = [f"- {name}" for name in available_columns[:120]]

        guidance = (
            "Select exactly one existing column name from the provided list. "
            "For count_distinct/list_values/mixed intents, prefer categorical/boolean fields over identifier keys. "
            "For aggregate intents, prefer numeric fields. "
            "Return only the column name."
        )
        try:
            pred = dspy.Predict(
                "guidance, question, intent, source, columns, last_column -> column"
            )(
                guidance=guidance,
                question=question.strip(),
                intent=str(intent).strip(),
                source=source,
                columns="\n".join(col_lines)[:16000],
                last_column=state_column,
            )
        except Exception:
            return ""

        candidate = self._normalize_planned_column_name(
            df,
            str(getattr(pred, "column", "")).strip(),
        )
        return candidate

    def _plan_tabular_question_with_llm(self, question: str) -> dict[str, Any] | None:
        """Use DSPy ReAct planning to route tabular questions."""
        if not self._has_tabular_sources():
            return None

        table_profiles = self._build_table_profiles(max_columns=120, top_values=4)
        if not table_profiles:
            return None

        history_text = self._recent_history_text(turns=4, max_chars=220)
        plan = self._react_planner.plan(
            question=question.strip(),
            history=history_text,
            table_profiles=table_profiles,
        )
        if plan is None:
            return None

        sources = self._tabular_source_names()
        source = str(plan.source or "").strip()
        if not source and len(sources) == 1:
            source = sources[0]
        if not source and sources:
            state_source = str(self._session.get_state("last_tabular_source", "")).strip()
            if state_source in sources:
                source = state_source
        if not source or source not in self._session.data:
            return None

        try:
            import pandas as pd
        except Exception:
            return None
        df = self._session.data.get(source)
        if not isinstance(df, pd.DataFrame):
            return None

        column = self._normalize_planned_column_name(df, str(plan.column or "").strip())

        intent = str(plan.intent or "").strip().lower()
        if intent not in {"schema", "count_rows", "count_distinct", "list_values", "aggregate", "mixed"}:
            return None
        if intent not in {"schema", "count_rows", "mixed"} and not column:
            column = self._resolve_planned_column_with_llm(
                question=question.strip(),
                source=source,
                intent=intent,
                df=df,
                table_profiles=table_profiles,
            )
        if intent not in {"schema", "count_rows", "mixed"} and not column:
            return None

        operation = str(plan.operation or "").strip().lower() or None
        if operation not in {None, "mean", "median", "min", "max", "sum", "std"}:
            operation = None

        return {
            "intent": intent,
            "source": source,
            "column": column,
            "operation": operation,
            "include_values": bool(plan.include_values),
        }

    def _execute_planned_tabular_answer(
        self,
        *,
        question: str,
        plan: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]], str] | None:
        if not plan:
            return None

        from mosaicx.query.tools import (
            compute_table_stat,
            list_distinct_values,
        )

        source = str(plan.get("source") or "").strip()
        if not source or source not in self._session.data:
            return None
        intent = str(plan.get("intent") or "").strip().lower()
        column = str(plan.get("column") or "").strip()
        operation = str(plan.get("operation") or "").strip().lower() or None
        include_values = bool(plan.get("include_values"))

        if intent == "schema":
            return self._try_deterministic_schema_answer(question)

        q = " ".join(question.lower().split())
        wants_count = intent in {"count_rows", "count_distinct", "mixed"} or (
            ("how many" in q) or ("number of" in q) or ("count" in q)
        )
        wants_distribution = self._is_distribution_request(question)
        wants_values = (
            include_values
            or intent in {"list_values", "mixed"}
            or wants_distribution
            or self._has_value_list_marker(question)
            or self._is_count_plus_values_request(question)
        )

        seed_hits: list[dict[str, Any]] = []
        if column:
            seed_hits.append(
                {
                    "source": source,
                    "column": column,
                    "role": "llm_planned",
                    "score": 96,
                    "snippet": f"LLM planner selected column: {column}",
                    "evidence_type": "table_column",
                }
            )

        if intent == "count_rows":
            row_count = int(len(self._session.data[source]))
            seed_hits.append(
                {
                    "source": source,
                    "snippet": f"Computed row_count from {row_count} rows: {row_count} (engine=pandas)",
                    "score": 88,
                    "evidence_type": "table_stat",
                    "operation": "row_count",
                    "column": "__rows__",
                    "value": row_count,
                    "backend": "pandas",
                }
            )
            self._session.set_state(
                last_tabular_source=source,
                last_tabular_column="__rows__",
                last_intent="count_rows",
            )
            return f"There are {row_count} rows in {source}.", self._merge_hits(seed_hits, max_hits=16), "count_rows"

        if intent == "aggregate":
            if operation not in {"mean", "median", "min", "max", "sum", "std"}:
                operation = self._detect_aggregate_operation(question)
            if operation not in {"mean", "median", "min", "max", "sum", "std"} or not column:
                return None
            try:
                agg_rows = compute_table_stat(
                    source,
                    data=self._session.data,
                    column=column,
                    operation=operation,
                )
            except Exception:
                return None
            if not agg_rows:
                return None
            row = agg_rows[0]
            value = str(row.get("value"))
            backend = str(row.get("backend") or "table_engine")
            row_count = row.get("row_count")
            non_null = row.get("non_null")
            seed_hits.append(
                {
                    "source": source,
                    "snippet": (
                        f"Computed {operation} of {column} from {row_count} rows "
                        f"(non-null {non_null}): {value} (engine={backend})"
                    ),
                    "score": 90,
                    "evidence_type": "table_stat",
                    "operation": operation,
                    "column": column,
                    "value": row.get("value"),
                    "backend": backend,
                }
            )
            self._session.set_state(
                last_tabular_source=source,
                last_tabular_column=column,
                last_intent="aggregate",
            )
            return f"{operation} of {column}: {value}.", self._merge_hits(seed_hits, max_hits=16), "aggregate"

        if intent not in {"count_distinct", "list_values", "mixed"} or not column:
            return None

        try:
            distinct_rows = list_distinct_values(
                source,
                data=self._session.data,
                column=column,
                limit=12,
            )
        except Exception:
            distinct_rows = []

        count_value: str | None = None
        try:
            count_rows = compute_table_stat(
                source,
                data=self._session.data,
                column=column,
                operation="nunique",
            )
        except Exception:
            count_rows = []
        if count_rows:
            row = count_rows[0]
            count_value = str(row.get("value"))
            backend = str(row.get("backend") or "table_engine")
            seed_hits.append(
                {
                    "source": source,
                    "snippet": (
                        f"Computed unique_count of {column} from {row.get('row_count')} rows: "
                        f"{count_value} (engine={backend})"
                    ),
                    "score": 90,
                    "evidence_type": "table_stat",
                    "operation": "nunique",
                    "column": column,
                    "value": row.get("value"),
                    "backend": backend,
                }
            )
        if distinct_rows:
            for row in distinct_rows:
                backend = str(row.get("backend") or "table_engine")
                seed_hits.append(
                    {
                        "source": source,
                        "snippet": (
                            f"Distinct {column}: {row.get('value')} "
                            f"(count={row.get('count')}, engine={backend})"
                        ),
                        "score": 88,
                        "evidence_type": "table_value",
                        "operation": "distinct_values",
                        "column": column,
                        "value": row.get("value"),
                        "count": int(row.get("count") or 0),
                        "backend": backend,
                    }
                )

        if count_value is None and distinct_rows:
            count_value = str(len(distinct_rows))
        if count_value is None and not distinct_rows:
            return None

        wants_breakdown = self._is_category_count_breakdown_request(
            question,
            column=column,
            distinct_rows=distinct_rows,
        )
        if wants_distribution and distinct_rows:
            distribution_preview = ", ".join(
                f"{r.get('value')}={int(r.get('count') or 0)}"
                for r in distinct_rows[:10]
            )
            answer = f"{column} distribution ({count_value} groups): {distribution_preview}."
            deterministic_intent = "count_values"
        elif wants_count and wants_breakdown and distinct_rows:
            distribution_preview = ", ".join(
                f"{r.get('value')}={int(r.get('count') or 0)}"
                for r in distinct_rows[:10]
            )
            answer = f"{column} distribution ({count_value} groups): {distribution_preview}."
            deterministic_intent = "count_values"
        elif wants_count and wants_values and distinct_rows:
            values_preview = ", ".join(str(r.get("value")) for r in distinct_rows[:8])
            answer = f"There are {count_value} distinct {column} values: {values_preview}."
            deterministic_intent = "count_values"
        elif wants_values and distinct_rows:
            values_preview = ", ".join(str(r.get("value")) for r in distinct_rows[:12])
            answer = f"Distinct {column} values: {values_preview}."
            deterministic_intent = "list_values"
        else:
            answer = f"There are {count_value} distinct {column} values."
            deterministic_intent = "count_distinct"

        self._session.set_state(
            last_tabular_source=source,
            last_tabular_column=column,
            last_intent=deterministic_intent,
        )
        return answer, self._merge_hits(seed_hits, max_hits=16), deterministic_intent

    def _resolve_tabular_target_with_llm(self, question: str) -> tuple[str, str] | None:
        """Use configured LM to resolve the best table/column for a tabular question."""
        plan = self._plan_tabular_question_with_llm(question)
        if not plan:
            return None
        source = str(plan.get("source") or "")
        column = str(plan.get("column") or "")
        if source and column:
            return source, column
        return None

    def _try_deterministic_tabular_answer(
        self,
        question: str,
    ) -> tuple[str, list[dict[str, Any]], str] | None:
        if not self._has_tabular_sources():
            return None

        schema = self._try_deterministic_schema_answer(question)
        if schema is not None:
            return schema

        from mosaicx.query.tools import (
            compute_table_stat,
            list_distinct_values,
            suggest_table_columns,
        )

        q_raw = question.strip()
        q = " ".join(q_raw.lower().split())
        resolved_q = self._resolve_followup_question(q_raw)
        route = self._intent_router.route(
            question=resolved_q,
            history=self._recent_history_text(turns=6, max_chars=220),
            has_tabular_sources=self._has_tabular_sources(),
        )

        llm_plan = self._plan_tabular_question_with_llm(resolved_q)
        if llm_plan is not None:
            planned = self._execute_planned_tabular_answer(
                question=resolved_q,
                plan=llm_plan,
            )
            if planned is not None:
                return planned
        wants_count = route.wants_count or ("how many" in q) or ("number of" in q) or ("count" in q)
        wants_values = (
            route.wants_values
            or self._has_value_list_marker(q_raw)
            or self._is_count_plus_values_request(resolved_q)
        )
        wants_distribution = self._is_distribution_request(resolved_q)
        if wants_distribution:
            wants_count = True
            wants_values = True
        aggregate_op = route.operation or self._detect_aggregate_operation(q_raw)
        wants_row_count = bool(
            {
                self._normalize_term(t)
                for t in re.findall(r"[a-z0-9]+", q)
            }
            & {"row", "record", "entry", "sample", "observation"}
        )
        if route.intent == "count" and wants_count and wants_row_count:
            wants_values = False
        if route.intent == "count_values":
            wants_count = True
            wants_values = True

        if llm_plan:
            intent = str(llm_plan.get("intent") or "")
            if intent == "count_rows":
                wants_count = True
                wants_row_count = True
            elif intent == "count_distinct":
                wants_count = True
            elif intent == "list_values":
                wants_values = True
            elif intent == "aggregate":
                aggregate_op = str(llm_plan.get("operation") or aggregate_op or "").strip().lower() or aggregate_op
            if bool(llm_plan.get("include_values")):
                wants_values = True
        if aggregate_op not in {None, "mean", "median", "min", "max", "sum", "std"}:
            aggregate_op = None

        if not wants_count and not wants_values and not aggregate_op:
            return None

        if llm_plan and str(llm_plan.get("source") or "") and str(llm_plan.get("column") or ""):
            source = str(llm_plan.get("source"))
            column = str(llm_plan.get("column"))
            top = {
                "source": source,
                "column": column,
                "role": "llm_planned",
                "score": 96,
                "snippet": f"LLM planner selected column: {column}",
                "evidence_type": "table_column",
            }
        else:
            llm_target = self._resolve_tabular_target_with_llm(resolved_q)
            if llm_target is not None:
                source, column = llm_target
                top = {
                    "source": source,
                    "column": column,
                    "role": "llm_resolved",
                    "score": 94,
                    "snippet": f"LLM-resolved column: {column}",
                    "evidence_type": "table_column",
                }
            else:
                column_hits = suggest_table_columns(
                    resolved_q,
                    data=self._session.data,
                    top_k=6,
                )
                if not column_hits:
                    return None
                top = column_hits[0]
                source = str(top.get("source") or "")
                column = str(top.get("column") or "")
                if not source or not column:
                    return None

        seed_hits: list[dict[str, Any]] = [top]
        distinct_rows: list[dict[str, Any]] = []
        count_value: str | None = None
        deterministic_intent = "count_distinct"

        if aggregate_op:
            deterministic_intent = "aggregate"
            try:
                agg_rows = compute_table_stat(
                    source,
                    data=self._session.data,
                    column=column,
                    operation=aggregate_op,
                )
            except Exception:
                return None
            if not agg_rows:
                return None
            row = agg_rows[0]
            value = str(row.get("value"))
            backend = str(row.get("backend") or "table_engine")
            row_count = row.get("row_count")
            non_null = row.get("non_null")
            snippet = (
                f"Computed {aggregate_op} of {column} from {row_count} rows "
                f"(non-null {non_null}): {value} (engine={backend})"
            )
            seed_hits.append(
                {
                    "source": source,
                    "snippet": snippet,
                    "score": 88,
                    "evidence_type": "table_stat",
                    "operation": aggregate_op,
                    "column": column,
                    "value": row.get("value"),
                    "backend": backend,
                }
            )
            self._session.set_state(
                last_tabular_source=source,
                last_tabular_column=column,
                last_intent=deterministic_intent,
            )
            answer = f"{aggregate_op} of {column}: {value}."
            return answer, self._merge_hits(seed_hits, max_hits=16), deterministic_intent

        if wants_count:
            if wants_row_count and source in self._session.data:
                deterministic_intent = "count_rows"
                row_count = int(len(self._session.data[source]))
                count_value = str(row_count)
                seed_hits.append(
                    {
                        "source": source,
                        "snippet": f"Computed row_count from {row_count} rows: {row_count} (engine=pandas)",
                        "score": 82,
                        "evidence_type": "table_stat",
                        "operation": "row_count",
                        "column": "__rows__",
                        "value": row_count,
                        "backend": "pandas",
                    }
                )
            else:
                count_rows = compute_table_stat(
                    source,
                    data=self._session.data,
                    column=column,
                    operation="nunique",
                )
                if count_rows:
                    row = count_rows[0]
                    backend = str(row.get("backend") or "table_engine")
                    count_value = str(row.get("value"))
                    seed_hits.append(
                        {
                            "source": source,
                            "snippet": (
                                f"Computed unique_count of {column} from {row.get('row_count')} rows: "
                                f"{count_value} (engine={backend})"
                            ),
                            "score": 84,
                            "evidence_type": "table_stat",
                            "operation": "nunique",
                            "column": column,
                            "value": row.get("value"),
                            "backend": backend,
                        }
                    )

        if wants_values or (wants_count and ("male" in q or "female" in q)):
            deterministic_intent = "list_values" if not wants_count else deterministic_intent
            distinct_rows = list_distinct_values(
                source,
                data=self._session.data,
                column=column,
                limit=12,
            )
            for row in distinct_rows:
                backend = str(row.get("backend") or "table_engine")
                seed_hits.append(
                    {
                        "source": source,
                        "snippet": (
                            f"Distinct {column}: {row.get('value')} "
                            f"(count={row.get('count')}, engine={backend})"
                        ),
                        "score": 84,
                        "evidence_type": "table_value",
                        "operation": "distinct_values",
                        "column": column,
                        "value": row.get("value"),
                        "count": int(row.get("count") or 0),
                        "backend": backend,
                    }
                )

        if not seed_hits:
            return None

        wants_breakdown = self._is_category_count_breakdown_request(
            q_raw,
            column=column,
            distinct_rows=distinct_rows,
        )
        if wants_count and distinct_rows:
            if count_value is None:
                count_value = str(len(distinct_rows))
            if wants_distribution or wants_breakdown:
                distribution_preview = ", ".join(
                    f"{r.get('value')}={int(r.get('count') or 0)}"
                    for r in distinct_rows[:10]
                )
                answer = f"{column} distribution ({count_value} groups): {distribution_preview}."
            else:
                values_preview = ", ".join(str(r.get("value")) for r in distinct_rows[:8])
                answer = (
                    f"There are {count_value} distinct {column} values: {values_preview}."
                )
        elif wants_values and distinct_rows:
            values_preview = ", ".join(str(r.get("value")) for r in distinct_rows[:12])
            answer = f"Distinct {column} values: {values_preview}."
        elif wants_count and count_value is not None:
            answer = f"There are {count_value} distinct {column} values."
        else:
            return None

        self._session.set_state(
            last_tabular_source=source,
            last_tabular_column=column,
            last_intent=deterministic_intent,
        )

        return answer, self._merge_hits(seed_hits, max_hits=16), deterministic_intent

    def _attempt_programmatic_sql_answer(
        self,
        question: str,
    ) -> tuple[str, list[dict[str, Any]], str] | None:
        if not self._has_tabular_sources():
            return None

        from mosaicx.query.tools import run_table_sql

        profiles = self._build_table_profiles(max_columns=120, top_values=4)
        if not profiles:
            return None

        plan = self._programmatic_analyst.propose_sql(
            question=question.strip(),
            history=self._recent_history_text(turns=8, max_chars=240),
            table_profiles=profiles,
        )
        if not plan:
            return None

        source = str(plan.get("source") or "").strip()
        sql = str(plan.get("sql") or "").strip()
        rationale = str(plan.get("rationale") or "").strip()
        if not source or not sql:
            return None

        try:
            rows = run_table_sql(
                source,
                data=self._session.data,
                sql=sql,
                limit=25,
            )
        except Exception:
            return None
        if not rows:
            return None
        if isinstance(rows[0], dict) and rows[0].get("error"):
            return None

        q = " ".join(question.lower().split())
        wants_count = ("how many" in q) or ("number of" in q) or ("count" in q)
        wants_values = self._has_value_list_marker(question) or self._is_count_plus_values_request(question)
        wants_distribution = self._is_distribution_request(question)

        preview_rows = rows[:6]
        serialized_rows = json.dumps(preview_rows, default=str, ensure_ascii=False)

        def _is_int_like(value: Any) -> bool:
            text = str(value).strip()
            if not text:
                return False
            try:
                parsed = float(text)
            except ValueError:
                return False
            return parsed.is_integer()

        value_column = ""
        count_column = ""
        if preview_rows and all(isinstance(r, dict) for r in preview_rows):
            columns = list(preview_rows[0].keys())
            for candidate in columns:
                c_norm = re.sub(r"[^a-z0-9]+", "_", str(candidate).lower()).strip("_")
                if c_norm in {"count", "cnt", "n", "freq", "frequency"}:
                    if all(_is_int_like(r.get(candidate)) for r in preview_rows):
                        count_column = str(candidate)
                        break
            if count_column:
                for candidate in columns:
                    if str(candidate) == count_column:
                        continue
                    value_column = str(candidate)
                    break

        answer: str
        if value_column and count_column and (wants_distribution or wants_values):
            pairs = [
                f"{str(r.get(value_column))}={int(float(str(r.get(count_column) or 0)))}"
                for r in preview_rows[:10]
            ]
            values = [str(r.get(value_column)) for r in preview_rows[:10]]
            if wants_distribution:
                answer = f"{value_column} distribution ({len(rows)} groups): {', '.join(pairs)}."
            elif wants_count:
                answer = f"There are {len(rows)} distinct {value_column} values: {', '.join(values)}."
            else:
                answer = f"Distinct {value_column} values: {', '.join(values)}."
        elif len(preview_rows) == 1 and isinstance(preview_rows[0], dict) and len(preview_rows[0]) == 1:
            key, value = next(iter(preview_rows[0].items()))
            key_norm = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if wants_count and any(token in key_norm for token in ("count", "n", "total", "rows")):
                answer = f"There are {value}."
            else:
                answer = f"{key}: {value}."
        elif len(preview_rows) == 1:
            answer = f"Computed result: {serialized_rows}"
        else:
            answer = f"Computed result rows ({len(preview_rows)} shown): {serialized_rows}"

        snippet = (
            f"Computed SQL on {source}: {sql} "
            f"-> {serialized_rows[:320]}"
        )
        if rationale:
            snippet += f" (rationale: {rationale[:120]})"
        seed_hits = [
            {
                "source": source,
                "snippet": snippet,
                "score": 92,
                "evidence_type": "table_sql",
                "sql": sql,
                "row_count": len(rows),
            }
        ]
        if value_column and count_column:
            for row in preview_rows[:10]:
                try:
                    count_value = int(float(str(row.get(count_column) or 0)))
                except ValueError:
                    count_value = 0
                seed_hits.append(
                    {
                        "source": source,
                        "snippet": (
                            f"Distinct {value_column}: {row.get(value_column)} "
                            f"(count={count_value}, engine=duckdb-sql)"
                        ),
                        "score": 90,
                        "evidence_type": "table_value",
                        "operation": "distinct_values",
                        "column": value_column,
                        "value": row.get(value_column),
                        "count": count_value,
                        "backend": "duckdb-sql",
                    }
                )
            seed_hits.append(
                {
                    "source": source,
                    "snippet": (
                        f"Computed unique_count of {value_column} from SQL result: "
                        f"{len(rows)} groups (engine=duckdb-sql)"
                    ),
                    "score": 88,
                    "evidence_type": "table_stat",
                    "operation": "nunique",
                    "column": value_column,
                    "value": len(rows),
                    "backend": "duckdb-sql",
                }
            )
        self._session.set_state(
            last_tabular_source=source,
            last_tabular_sql=sql,
            last_tabular_column=value_column or self._session.get_state("last_tabular_column", ""),
            last_intent="sql_analytic",
        )
        return answer, self._merge_hits(seed_hits, max_hits=8), "sql_analytic"

    def _question_mentions_explicit_column(self, question: str) -> bool:
        if not self._has_tabular_sources():
            return False
        q = " ".join(question.lower().split())
        q_terms = {
            self._normalize_term(t)
            for t in re.findall(r"[a-z0-9]+", q)
            if len(t) >= 2
        }
        if not q_terms:
            return False
        profiles = self._build_table_profiles(max_columns=200, top_values=0)
        for profile in profiles.values():
            columns = profile.get("columns", [])
            if not isinstance(columns, list):
                continue
            for col in columns:
                if not isinstance(col, dict):
                    continue
                name = str(col.get("name") or "").strip()
                if not name:
                    continue
                name_lc = name.lower()
                if name_lc in q:
                    return True
                col_terms = {
                    self._normalize_term(t)
                    for t in re.findall(r"[a-z0-9]+", name_lc)
                    if len(t) >= 2
                }
                if col_terms and len(col_terms) <= 4 and col_terms.issubset(q_terms):
                    return True
        return False

    def _should_try_programmatic_first(
        self,
        *,
        question: str,
        route: IntentDecision,
    ) -> bool:
        if not self._has_tabular_sources():
            return False
        if self._is_schema_question(question):
            return False
        if route.intent == "schema":
            return False
        q = " ".join(question.lower().split())
        row_count_markers = {"row", "rows", "record", "records", "entry", "entries", "observation", "observations", "sample", "samples"}
        q_terms = {
            self._normalize_term(t)
            for t in re.findall(r"[a-z0-9]+", q)
            if len(t) >= 2
        }
        is_row_count_only = (
            route.intent == "count"
            and bool(q_terms & row_count_markers)
            and not self._has_value_list_marker(question)
            and not self._is_distribution_request(question)
        )
        if is_row_count_only:
            return False

        analytic_like = (
            route.intent in {"aggregate", "count_values", "mixed"}
            or (route.intent == "count" and not is_row_count_only)
            or self._requires_computed_evidence(question)
        )
        if not analytic_like:
            return False

        explicit_column = self._question_mentions_explicit_column(question)
        if (
            route.intent == "count"
            and not explicit_column
            and not self._is_distribution_request(question)
            and not self._has_value_list_marker(question)
            and not self._is_count_plus_values_request(question)
        ):
            return False
        explicit_simple_aggregate = (
            explicit_column
            and route.intent in {"count", "aggregate"}
            and not self._is_distribution_request(question)
            and not self._has_value_list_marker(question)
        )
        if explicit_simple_aggregate:
            return False
        return True

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

        # Reward deterministic tabular answers when computed evidence explicitly
        # supports both counts and category values.
        computed_citations = [
            c
            for c in citations
            if str(c.get("evidence_type") or "").strip().lower() in {"table_stat", "table_value", "table_sql"}
        ]
        if computed_citations:
            computed_blob = " ".join(
                " ".join(str(c.get("snippet") or "").split()).lower()
                for c in computed_citations
            )
            answer_numbers = re.findall(r"\d+(?:\.\d+)?", answer)
            numeric_support = None
            if answer_numbers:
                numeric_support = sum(1 for n in answer_numbers if n in computed_blob) / len(answer_numbers)

            def _norm_value(value: str) -> str:
                return " ".join(str(value or "").lower().split())

            table_value_citations = [
                c for c in computed_citations
                if str(c.get("evidence_type") or "").strip().lower() == "table_value"
            ]
            citation_values: list[str] = []
            for c in table_value_citations:
                value = str(c.get("value") or "").strip()
                if not value:
                    snippet = str(c.get("snippet") or "")
                    match = re.search(r"Distinct\s+.+?:\s+(.+?)\s+\(count=", snippet)
                    value = match.group(1).strip() if match else ""
                if value:
                    citation_values.append(value)

            answer_lc = answer.lower()
            matched_values = 0
            if citation_values:
                for value in citation_values:
                    value_lc = value.lower()
                    if value_lc and value_lc in answer_lc:
                        matched_values += 1
            table_value_coverage = (
                matched_values / len(citation_values)
                if citation_values
                else None
            )

            # When answer provides explicit category labels, suppress boost
            # if labels are not grounded in value-level computed evidence.
            answer_labels: set[str] = set()
            for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_/-]{0,24})\s*=", answer):
                answer_labels.add(_norm_value(match.group(1)))
            list_match = re.search(r"(?i)distinct\s+[^:]{1,80}\s+values:\s*(.+)", answer)
            if list_match:
                raw_items = re.split(r",|\band\b", list_match.group(1))
                for item in raw_items:
                    cleaned = re.sub(r"\(.*?\)", "", item).strip(" .;:-")
                    if cleaned:
                        answer_labels.add(_norm_value(cleaned))

            citation_value_norms = {_norm_value(v) for v in citation_values if _norm_value(v)}
            if answer_labels and citation_value_norms:
                unsupported = {v for v in answer_labels if v and v not in citation_value_norms}
                if unsupported and table_value_coverage is not None:
                    table_value_coverage = min(table_value_coverage, 0.25)

            if table_value_coverage is not None:
                if table_value_coverage >= 0.8:
                    confidence = max(confidence, 0.82)
                elif table_value_coverage >= 0.5:
                    confidence = max(confidence, 0.74)
            elif numeric_support is not None:
                if numeric_support >= 0.8:
                    confidence = max(confidence, 0.78)
                elif numeric_support >= 0.5:
                    confidence = max(confidence, 0.68)

        return max(0.0, min(1.0, confidence))

    def _chunk_literal_support(
        self,
        *,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> float | None:
        literals = self._extract_answer_values(answer)
        if not literals:
            return None
        chunk_blob = " ".join(
            " ".join(str(c.get("snippet") or "").split()).lower()
            for c in citations
            if str(c.get("evidence_type") or "").strip().lower() == "text_chunk"
        )
        if not chunk_blob:
            return None
        supported = sum(1 for lit in literals if lit in chunk_blob)
        return supported / len(literals)

    def _is_temporal_comparison_question(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        if self._is_delta_question(q):
            return True
        if ("when was" in q or "which date" in q or "what date" in q) and (
            any(m in q for m in ("small", "smallest", "lowest", "minimum", "min"))
            or any(m in q for m in ("large", "largest", "highest", "maximum", "max"))
        ):
            return True
        return any(m in q for m in ("increase", "decrease", "grew", "shrank", "went up", "went down"))

    def _format_mm_value(self, mm_value: float) -> str:
        if abs(mm_value - round(mm_value)) < 1e-6:
            return f"{int(round(mm_value))} mm"
        return f"{mm_value:.1f} mm"

    def _extract_temporal_observations_from_citations(
        self,
        citations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        seen: set[tuple[str, float, str]] = set()
        for citation in citations[:24]:
            snippet = " ".join(str(citation.get("snippet") or "").split())
            if not snippet:
                continue
            source = str(citation.get("source") or "").strip()

            date = None
            date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", snippet)
            if date_match:
                date = date_match.group(0)
            if not date and source:
                src_date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", source)
                if src_date_match:
                    date = src_date_match.group(0)

            for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(mm|cm)\b", snippet.lower()):
                numeric = float(match.group(1))
                unit = str(match.group(2)).lower()
                mm_value = numeric * (10.0 if unit == "cm" else 1.0)
                value_display = f"{match.group(1)} {unit}"
                key = (date or "", round(mm_value, 4), source)
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    {
                        "date": date,
                        "source": source,
                        "value_mm": mm_value,
                        "value_display": value_display,
                    }
                )
        return observations

    def _infer_temporal_answer_from_citations(
        self,
        *,
        question: str,
        citations: list[dict[str, Any]],
    ) -> str | None:
        if not citations or not self._is_temporal_comparison_question(question):
            return None

        observations = self._extract_temporal_observations_from_citations(citations)
        if len(observations) < 2:
            return None

        q = " ".join(question.lower().split())
        with_dates = [obs for obs in observations if obs.get("date")]
        if len(with_dates) >= 2:
            ordered = sorted(with_dates, key=lambda obs: str(obs.get("date") or ""))
        else:
            ordered = sorted(observations, key=lambda obs: float(obs.get("value_mm") or 0.0))
        earliest = ordered[0]
        latest = ordered[-1]

        if ("when was" in q or "which date" in q or "what date" in q) and any(
            m in q for m in ("small", "smallest", "lowest", "minimum", "min")
        ):
            smallest = min(observations, key=lambda obs: float(obs.get("value_mm") or 0.0))
            when = str(smallest.get("date") or smallest.get("source") or "the available evidence")
            return (
                f"The smallest recorded measurement was {smallest['value_display']} on {when}."
            )

        if ("when was" in q or "which date" in q or "what date" in q) and any(
            m in q for m in ("large", "largest", "highest", "maximum", "max")
        ):
            largest = max(observations, key=lambda obs: float(obs.get("value_mm") or 0.0))
            when = str(largest.get("date") or largest.get("source") or "the available evidence")
            return (
                f"The largest recorded measurement was {largest['value_display']} on {when}."
            )

        if not self._is_delta_question(q) and not any(
            m in q for m in ("increase", "decrease", "difference", "change", "delta", "grew", "shrank")
        ):
            return None

        earliest_when = str(earliest.get("date") or earliest.get("source") or "earlier report")
        latest_when = str(latest.get("date") or latest.get("source") or "later report")
        delta_mm = float(latest.get("value_mm") or 0.0) - float(earliest.get("value_mm") or 0.0)
        if abs(delta_mm) < 0.1:
            return (
                "No measurable size change was found: "
                f"{earliest['value_display']} ({earliest_when}) and "
                f"{latest['value_display']} ({latest_when})."
            )

        direction = "increased" if delta_mm > 0 else "decreased"
        delta_display = self._format_mm_value(abs(delta_mm))
        return (
            f"Lesion size {direction} from {earliest['value_display']} ({earliest_when}) "
            f"to {latest['value_display']} ({latest_when}), a change of {delta_display}."
        )

    def _looks_like_schema_answer(self, answer: str) -> bool:
        a = " ".join(str(answer).lower().split())
        return any(marker in a for marker in ("column", "columns", "field", "fields", "header", "headers"))

    def _looks_like_schema_count_answer(self, answer: str) -> bool:
        a = " ".join(str(answer).lower().split())
        has_schema_term = any(
            marker in a for marker in ("column", "columns", "field", "fields", "header", "headers")
        )
        has_count_term = any(
            marker in a for marker in ("there are", "number of", "count", "total")
        )
        return has_schema_term and has_count_term

    def _is_delta_question(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        return any(marker in q for marker in _DELTA_QUESTION_MARKERS)

    def _requires_computed_evidence(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        analytic_markers = (
            "how many",
            "number of",
            "count",
            "average",
            "mean",
            "median",
            "min",
            "max",
            "sum",
            "std",
            "percent",
            "ratio",
            "prevalence",
            "incidence",
            "distribution",
            "outlier",
            "correlation",
            "change",
            "difference",
        )
        return any(marker in q for marker in analytic_markers)

    def _requires_numeric_stat_evidence(self, question: str) -> bool:
        q = " ".join(question.lower().split())
        return any(marker in q for marker in _NUMERIC_STAT_MARKERS)

    def _has_tabular_sources(self) -> bool:
        return any(getattr(meta, "source_type", "") == "dataframe" for meta in self._session.catalog)

    def _has_computed_citations(
        self,
        citations: list[dict[str, Any]],
        *,
        require_numeric_stat: bool = False,
    ) -> bool:
        allowed = {"table_stat", "table_sql"}
        if not require_numeric_stat:
            allowed.add("table_value")
        return any(
            str(c.get("evidence_type") or "").strip().lower() in allowed
            for c in citations
        )

    def _has_value_citations(self, citations: list[dict[str, Any]]) -> bool:
        return any(
            str(c.get("evidence_type") or "").strip().lower() == "table_value"
            for c in citations
        )

    def _has_chunk_citations(self, citations: list[dict[str, Any]]) -> bool:
        return any(
            str(c.get("evidence_type") or "").strip().lower() == "text_chunk"
            for c in citations
        )

    def _has_long_documents(self) -> bool:
        for meta in self._session.catalog:
            if str(getattr(meta, "source_type", "")).strip().lower() != "document":
                continue
            name = str(getattr(meta, "name", "") or "").strip()
            if not name:
                continue
            text = str(self._documents.get(name, "") or "")
            if len(text) >= 1200:
                return True
        return False

    def _needs_longdoc_chunk_grounding(
        self,
        *,
        question: str,
        route: IntentDecision,
        deterministic_intent: str | None,
    ) -> bool:
        if not self._has_long_documents():
            return False
        if self._is_schema_question(question):
            return False
        if self._has_tabular_sources() and (
            route.intent in {"count", "count_values", "aggregate"}
            or self._requires_computed_evidence(question)
            or deterministic_intent in {
                "count_rows",
                "count_distinct",
                "count_values",
                "list_values",
                "aggregate",
                "sql_analytic",
            }
        ):
            return False
        return True

    def _is_reconciler_sensitive_deterministic_intent(self, intent: str | None) -> bool:
        return str(intent or "").strip().lower() in {
            "count_rows",
            "count_distinct",
            "count_values",
            "list_values",
            "aggregate",
            "sql_analytic",
        }

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
            revised = self._evidence_verifier.revise(
                question=question,
                draft_answer=draft_answer,
                citations=citations,
            )
            if revised:
                return revised
        except Exception:
            pass

        try:
            dspy = import_dspy()
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
            dspy = import_dspy()
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
        fallback_code: str | None = None
        rescue_used = False
        rescue_reason: str | None = None
        deterministic_used = False
        deterministic_intent: str | None = None
        longdoc_literal_support: float | None = None

        resolved_question = self._resolve_followup_question(question.strip())
        route = self._intent_router.route(
            question=resolved_question,
            history=self._recent_history_text(turns=8, max_chars=240),
            has_tabular_sources=self._has_tabular_sources(),
        )
        requires_value_listing = (
            self._has_tabular_sources()
            and (
                route.intent == "count_values"
                or self._is_count_plus_values_request(resolved_question)
            )
        )
        effective_top_k = max(1, int(top_k_citations))
        if requires_value_listing:
            # Keep enough room for count evidence + category values.
            effective_top_k = max(effective_top_k, 6)

        answer = ""
        seed_hits: list[dict[str, Any]] = []
        if self._should_try_programmatic_first(question=resolved_question, route=route):
            sql_primary = self._attempt_programmatic_sql_answer(resolved_question)
            if sql_primary is not None:
                answer, seed_hits, deterministic_intent = sql_primary
                deterministic_used = True

        if not answer:
            deterministic = self._try_deterministic_tabular_answer(question.strip())
            if deterministic is not None:
                answer, seed_hits, deterministic_intent = deterministic
                deterministic_used = True
            else:
                run_error: Exception | None = None
                attempt_questions = [
                    resolved_question,
                    (
                        f"{resolved_question}\n"
                        "Return a concise plain-text answer grounded only in tool evidence. "
                        "Avoid JSON wrappers."
                    ),
                ]
                for idx, attempt_question in enumerate(attempt_questions):
                    try:
                        answer, seed_hits = self._run_query_once(attempt_question)
                        run_error = None
                        break
                    except Exception as exc:
                        run_error = exc
                        exc_text = f"{type(exc).__name__}: {exc}"
                        is_adapter_parse = (
                            "AdapterParseError" in exc_text
                            or "JSONAdapter" in exc_text
                            or "cannot be serialized to a JSON object" in exc_text.lower()
                        )
                        if not is_adapter_parse:
                            break
                        if idx + 1 >= len(attempt_questions):
                            break

                if run_error is not None:
                    from mosaicx.query.tools import (
                        analyze_table_question,
                        search_document_chunks,
                        search_documents,
                        search_tables,
                        suggest_table_columns,
                    )

                    fallback_used = True
                    fallback_reason = f"{type(run_error).__name__}: {run_error}"
                    fallback_code = "adapter_parse_error" if "AdapterParseError" in fallback_reason else "lm_runtime_error"
                    text_hits = search_documents(
                        resolved_question,
                        documents=self._documents,
                        top_k=max(3, effective_top_k),
                    )
                    chunk_hits = search_document_chunks(
                        resolved_question,
                        documents=self._documents,
                        top_k=max(4, effective_top_k + 1),
                        chunk_index=self._ensure_document_chunk_index(),
                    )
                    table_hits = search_tables(
                        resolved_question,
                        data=self._session.data,
                        top_k=max(3, effective_top_k),
                    )
                    computed_hits = analyze_table_question(
                        resolved_question,
                        data=self._session.data,
                        top_k=max(3, effective_top_k),
                    )
                    column_hits = suggest_table_columns(
                        resolved_question,
                        data=self._session.data,
                        top_k=max(3, effective_top_k),
                    )
                    seed_hits = self._merge_hits(
                        [*computed_hits, *column_hits, *table_hits, *chunk_hits, *text_hits],
                        max_hits=max(4, effective_top_k * 2),
                    )
                    if seed_hits:
                        rescued = self._rescue_answer_with_evidence(
                            question=resolved_question,
                            citations=seed_hits,
                        )
                        if rescued:
                            answer = rescued
                            rescue_used = True
                            rescue_reason = "fallback_evidence_recovery"
                        else:
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
            question=resolved_question,
            answer=answer,
            seed_hits=seed_hits,
            top_k=effective_top_k,
        )
        skip_reconcile_for_deterministic = bool(
            deterministic_used
            and self._is_reconciler_sensitive_deterministic_intent(deterministic_intent)
            and self._has_computed_citations(citations)
            and not self._looks_like_non_answer(answer)
        )
        if citations and deterministic_intent not in {"schema", "schema_count"} and not skip_reconcile_for_deterministic:
            reconciled = self._reconcile_answer_with_evidence(
                question=resolved_question,
                draft_answer=answer,
                citations=citations,
            )
            if reconciled and reconciled.strip() != answer.strip():
                answer = reconciled.strip()
                rescue_used = True
                rescue_reason = "evidence_reconciler"

        temporal_override = self._infer_temporal_answer_from_citations(
            question=resolved_question,
            citations=citations,
        )
        if temporal_override and temporal_override.strip() != answer.strip():
            apply_temporal_override = False
            if self._looks_like_non_answer(answer):
                apply_temporal_override = True
            else:
                current_conf = self._citation_confidence(resolved_question, answer, citations)
                temporal_conf = self._citation_confidence(resolved_question, temporal_override, citations)
                # Apply only when inferred temporal answer is materially better grounded.
                if temporal_conf >= current_conf + 0.08:
                    apply_temporal_override = True
                elif current_conf < 0.45 and temporal_conf >= 0.55:
                    apply_temporal_override = True

            if apply_temporal_override:
                answer = temporal_override.strip()
                rescue_used = True
                rescue_reason = "temporal_comparison_guard"
                citations = self._build_citations(
                    question=resolved_question,
                    answer=answer,
                    seed_hits=seed_hits,
                    top_k=max(effective_top_k, 4),
                )
                longdoc_literal_support = self._chunk_literal_support(
                    answer=answer,
                    citations=citations,
                )

        needs_rescue = bool(
            deterministic_intent not in {"schema", "schema_count"}
            and citations
            and self._looks_like_non_answer(answer)
        )

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
                    question=resolved_question,
                    answer=answer,
                    seed_hits=seed_hits,
                    top_k=effective_top_k,
                )

        requires_computed = self._has_tabular_sources() and not self._is_schema_question(resolved_question) and (
            route.intent in {"count", "count_values", "aggregate", "mixed"}
            or self._requires_computed_evidence(resolved_question)
        )
        requires_numeric_stat = route.needs_numeric_stat or self._requires_numeric_stat_evidence(resolved_question)
        if requires_computed and not self._has_computed_citations(
            citations,
            require_numeric_stat=requires_numeric_stat,
        ):
            from mosaicx.query.tools import analyze_table_question

            computed_hits = analyze_table_question(
                resolved_question,
                data=self._session.data,
                top_k=max(3, effective_top_k),
            )
            if computed_hits:
                citations = self._build_citations(
                    question=resolved_question,
                    answer=answer,
                    seed_hits=[*computed_hits, *seed_hits],
                    top_k=effective_top_k,
                )
                skip_reconcile_for_deterministic = bool(
                    deterministic_used
                    and self._is_reconciler_sensitive_deterministic_intent(deterministic_intent)
                    and self._has_computed_citations(citations)
                    and not self._looks_like_non_answer(answer)
                )
                if not skip_reconcile_for_deterministic:
                    reconciled = self._reconcile_answer_with_evidence(
                        question=resolved_question,
                        draft_answer=answer,
                        citations=citations,
                    )
                    if reconciled and reconciled.strip() != answer.strip():
                        answer = reconciled.strip()
                        rescue_used = True
                        rescue_reason = "analytic_reconciler"

            if not self._has_computed_citations(
                citations,
                require_numeric_stat=requires_numeric_stat,
            ):
                sql_attempt = self._attempt_programmatic_sql_answer(resolved_question)
                if sql_attempt is not None:
                    sql_answer, sql_hits, sql_intent = sql_attempt
                    answer = sql_answer
                    deterministic_used = True
                    deterministic_intent = sql_intent
                    citations = self._build_citations(
                        question=resolved_question,
                        answer=answer,
                        seed_hits=[*sql_hits, *seed_hits],
                        top_k=effective_top_k,
                    )
                    rescue_used = True
                    rescue_reason = "programmatic_sql"

            if not self._has_computed_citations(
                citations,
                require_numeric_stat=requires_numeric_stat,
            ):
                answer = (
                    "I cannot provide a reliable numeric answer yet because no computed "
                    "table evidence was produced. Ask for schema/profile or specify a column."
                )
                rescue_used = True
                rescue_reason = "missing_computed_evidence"

        if requires_value_listing and not self._has_value_citations(citations):
            from mosaicx.query.tools import analyze_table_question

            value_hits = [
                h
                for h in analyze_table_question(
                    resolved_question,
                    data=self._session.data,
                    top_k=max(6, effective_top_k * 2),
                )
                if str(h.get("evidence_type") or "").strip().lower() == "table_value"
            ]
            if value_hits:
                citations = self._build_citations(
                    question=resolved_question,
                    answer=answer,
                    seed_hits=[*value_hits, *seed_hits],
                    top_k=max(effective_top_k, 6),
                )
                skip_reconcile_for_deterministic = bool(
                    deterministic_used
                    and self._is_reconciler_sensitive_deterministic_intent(deterministic_intent)
                    and self._has_computed_citations(citations)
                    and not self._looks_like_non_answer(answer)
                )
                if not skip_reconcile_for_deterministic:
                    reconciled = self._reconcile_answer_with_evidence(
                        question=resolved_question,
                        draft_answer=answer,
                        citations=citations,
                    )
                    if reconciled and reconciled.strip() != answer.strip():
                        answer = reconciled.strip()
                        rescue_used = True
                        rescue_reason = "value_evidence_reconciler"
            if not self._has_value_citations(citations):
                answer = (
                    "I can confirm the distinct count, but I cannot reliably list the "
                    "actual category values because no value-level evidence was produced."
                )
                rescue_used = True
                rescue_reason = "missing_value_evidence"

        needs_longdoc_chunk_grounding = self._needs_longdoc_chunk_grounding(
            question=resolved_question,
            route=route,
            deterministic_intent=deterministic_intent,
        )
        if needs_longdoc_chunk_grounding and not self._has_chunk_citations(citations):
            from mosaicx.query.tools import search_document_chunks

            chunk_query = " ".join([resolved_question, answer]).strip()
            chunk_hits = search_document_chunks(
                chunk_query,
                documents=self._documents,
                top_k=max(4, effective_top_k + 1),
                chunk_index=self._ensure_document_chunk_index(),
            )
            if chunk_hits:
                citations = self._build_citations(
                    question=resolved_question,
                    answer=answer,
                    seed_hits=[*chunk_hits, *seed_hits],
                    top_k=max(effective_top_k, 4),
                )
                rescue_used = True
                rescue_reason = rescue_reason or "longdoc_chunk_grounding"
                if self._looks_like_non_answer(answer):
                    rescued = self._rescue_answer_with_evidence(
                        question=question.strip(),
                        citations=citations,
                    )
                    if rescued:
                        answer = rescued
                        rescue_reason = "longdoc_chunk_recovery"

        if needs_longdoc_chunk_grounding and self._has_chunk_citations(citations):
            longdoc_literal_support = self._chunk_literal_support(
                answer=answer,
                citations=citations,
            )
            if (
                longdoc_literal_support is not None
                and longdoc_literal_support < 0.5
                and not self._looks_like_non_answer(answer)
            ):
                rescued = self._rescue_answer_with_evidence(
                    question=question.strip(),
                    citations=citations,
                )
                if rescued and rescued.strip() != answer.strip():
                    answer = rescued.strip()
                    rescue_used = True
                    rescue_reason = "longdoc_literal_guard"
                    citations = self._build_citations(
                        question=resolved_question,
                        answer=answer,
                        seed_hits=seed_hits,
                        top_k=max(effective_top_k, 4),
                    )
                    longdoc_literal_support = self._chunk_literal_support(
                        answer=answer,
                        citations=citations,
                    )

        if self._has_tabular_sources() and self._is_schema_question(resolved_question):
            schema_answer = self._try_deterministic_schema_answer(resolved_question)
            should_override_schema = (
                deterministic_intent not in {"schema", "schema_count"}
                or not self._looks_like_schema_answer(answer)
            )
            if schema_answer is not None and should_override_schema:
                schema_text, schema_hits, schema_intent = schema_answer
                answer = schema_text
                deterministic_used = True
                deterministic_intent = schema_intent
                rescue_used = True
                rescue_reason = "schema_truth_guard"
                seed_hits = self._merge_hits([*schema_hits, *seed_hits], max_hits=16)
                citations = self._build_citations(
                    question=resolved_question,
                    answer=answer,
                    seed_hits=seed_hits,
                    top_k=max(effective_top_k, 4),
                )

        confidence = self._citation_confidence(resolved_question, answer, citations)

        self._session.add_turn("user", question.strip())
        self._session.add_turn("assistant", answer)
        self._session.set_state(
            last_user_question=question.strip(),
            last_resolved_question=resolved_question,
            last_answer=answer,
            last_confidence=confidence,
            last_route_intent=route.intent,
        )
        state_route = route
        if deterministic_intent in {"schema", "schema_count"}:
            state_route = IntentDecision(
                intent="schema",
                wants_schema=True,
                wants_count=False,
                wants_values=False,
                operation=None,
                needs_numeric_stat=False,
            )
        self._update_structured_query_state(
            question=resolved_question,
            answer=answer,
            citations=citations,
            route=state_route,
        )

        effective_intent = deterministic_intent or route.intent

        return {
            "question": question.strip(),
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "sources_consulted": sorted({c["source"] for c in citations}),
            "turn_index": len(self._session.conversation) // 2,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "fallback_code": fallback_code,
            "rescue_used": rescue_used,
            "rescue_reason": rescue_reason,
            "deterministic_used": deterministic_used,
            "deterministic_intent": deterministic_intent,
            "intent": effective_intent,
            "route_intent": route.intent,
            "longdoc_literal_support": longdoc_literal_support,
        }

    def ask(self, question: str) -> str:
        """Ask a question about the loaded documents.

        Returns only the answer text for backwards compatibility.
        """
        payload = self.ask_structured(question)
        return str(payload["answer"])
