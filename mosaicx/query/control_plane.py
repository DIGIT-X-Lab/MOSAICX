"""DSPy control-plane modules for robust query orchestration.

This module separates planning/verification narration logic from the
deterministic data-plane execution in ``mosaicx.query.tools``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_VALUE_MARKERS = (
    "what are they",
    "which ones",
    "list them",
    "what are those",
    "what are the values",
    "what are the categories",
)
_SCHEMA_MARKERS = (
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
_COUNT_MARKERS = ("how many", "number of", "count")
_DISTRIBUTION_MARKERS = ("distribution", "breakdown", "split by")
_AGG_OPERATION_MARKERS: tuple[tuple[str, str], ...] = (
    ("mean", "mean"),
    ("average", "mean"),
    ("avg", "mean"),
    ("median", "median"),
    ("minimum", "min"),
    ("lowest", "min"),
    ("min", "min"),
    ("maximum", "max"),
    ("highest", "max"),
    ("max", "max"),
    ("sum", "sum"),
    ("total", "sum"),
    ("standard deviation", "std"),
    ("std", "std"),
)


@dataclass(slots=True)
class IntentDecision:
    """Structured intent routing output."""

    intent: str
    wants_count: bool = False
    wants_values: bool = False
    wants_schema: bool = False
    operation: str | None = None
    needs_numeric_stat: bool = False


@dataclass(slots=True)
class TabularPlan:
    """Planner output for deterministic tabular executor."""

    intent: str
    source: str | None = None
    column: str | None = None
    operation: str | None = None
    include_values: bool = False


def _compact(text: str, limit: int = 220) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    clipped = value[: limit - 1].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped + "…"


class IntentRouter:
    """Intent router using DSPy + deterministic fallback."""

    def route(
        self,
        *,
        question: str,
        history: str,
        has_tabular_sources: bool,
    ) -> IntentDecision:
        q = " ".join(question.lower().split())

        wants_schema = any(marker in q for marker in _SCHEMA_MARKERS)
        wants_count = self._wants_count(q)
        wants_values = self._wants_values(q, wants_count=wants_count)
        wants_distribution = self._wants_distribution(q)
        operation = self._detect_operation(q)
        needs_numeric_stat = operation is not None

        # Deterministic route first for stability.
        if wants_schema:
            return IntentDecision(
                intent="schema",
                wants_schema=True,
                wants_values=False,
                wants_count=False,
                operation=None,
                needs_numeric_stat=False,
            )
        if wants_distribution:
            return IntentDecision(
                intent="count_values",
                wants_schema=False,
                wants_values=True,
                wants_count=True,
                operation=None,
                needs_numeric_stat=False,
            )
        if wants_count and wants_values:
            return IntentDecision(
                intent="count_values",
                wants_schema=False,
                wants_values=True,
                wants_count=True,
                operation=None,
                needs_numeric_stat=False,
            )
        if wants_count:
            return IntentDecision(
                intent="count",
                wants_schema=False,
                wants_values=False,
                wants_count=True,
                operation=None,
                needs_numeric_stat=False,
            )
        if operation is not None:
            return IntentDecision(
                intent="aggregate",
                wants_schema=False,
                wants_values=False,
                wants_count=False,
                operation=operation,
                needs_numeric_stat=True,
            )

        # Optional DSPy route for ambiguous cases.
        if has_tabular_sources:
            llm = self._route_with_dspy(question=question, history=history)
            if llm is not None:
                return llm

        return IntentDecision(intent="text_qa")

    def _wants_count(self, q: str) -> bool:
        return any(marker in q for marker in _COUNT_MARKERS)

    def _wants_values(self, q: str, *, wants_count: bool) -> bool:
        if any(marker in q for marker in _VALUE_MARKERS):
            return True
        if not wants_count:
            return False
        return bool(
            re.search(r"\band\b[^?]{0,120}\bwhat\s+are\b", q)
            or re.search(r"\band\b[^?]{0,120}\bwhich\b", q)
            or re.search(r"\band\b[^?]{0,120}\blist\b", q)
        )

    def _wants_distribution(self, q: str) -> bool:
        return any(marker in q for marker in _DISTRIBUTION_MARKERS)

    def _detect_operation(self, q: str) -> str | None:
        for marker, op in _AGG_OPERATION_MARKERS:
            if marker == "min":
                if re.search(r"\bmin\b", q):
                    return op
                continue
            if marker == "max":
                if re.search(r"\bmax\b", q):
                    return op
                continue
            if marker == "std":
                if re.search(r"\bstd\b", q):
                    return op
                continue
            if marker in q:
                return op
        return None

    def _route_with_dspy(self, *, question: str, history: str) -> IntentDecision | None:
        try:
            import dspy
        except Exception:
            return None
        if getattr(dspy.settings, "lm", None) is None:
            return None

        guidance = (
            "Classify user intent for query routing. "
            "Use one of intents: text_qa, schema, count, count_values, aggregate, mixed. "
            "Return aggregate_operation only when intent=aggregate. "
            "Return needs_numeric_stat true only for numeric aggregate stats."
        )
        try:
            pred = dspy.Predict(
                "guidance, question, history -> intent, aggregate_operation, needs_numeric_stat"
            )(
                guidance=guidance,
                question=question.strip(),
                history=_compact(history, limit=1200),
            )
        except Exception:
            return None

        intent = " ".join(str(getattr(pred, "intent", "")).lower().split())
        op = " ".join(str(getattr(pred, "aggregate_operation", "")).lower().split()) or None
        if op not in {None, "mean", "median", "min", "max", "sum", "std"}:
            op = None
        needs_numeric = " ".join(str(getattr(pred, "needs_numeric_stat", "")).lower().split()) in {
            "true",
            "yes",
            "1",
        }
        if intent not in {"text_qa", "schema", "count", "count_values", "aggregate", "mixed"}:
            return None
        return IntentDecision(
            intent=intent,
            wants_schema=intent == "schema",
            wants_count=intent in {"count", "count_values"},
            wants_values=intent == "count_values",
            operation=op,
            needs_numeric_stat=needs_numeric,
        )


class ReActTabularPlanner:
    """ReAct-based tabular planner with safe fallback."""

    def plan(
        self,
        *,
        question: str,
        history: str,
        table_profiles: dict[str, dict[str, Any]],
    ) -> TabularPlan | None:
        if not table_profiles:
            return None
        plan = self._plan_with_react(
            question=question,
            history=history,
            table_profiles=table_profiles,
        )
        if plan is not None:
            return plan
        return self._plan_with_predict(
            question=question,
            history=history,
            table_profiles=table_profiles,
        )

    def _plan_with_react(
        self,
        *,
        question: str,
        history: str,
        table_profiles: dict[str, dict[str, Any]],
    ) -> TabularPlan | None:
        try:
            import dspy
        except Exception:
            return None
        if getattr(dspy.settings, "lm", None) is None:
            return None

        tables = sorted(table_profiles.keys())

        def list_tables() -> list[str]:
            """List available tabular source names."""
            return tables

        def list_columns(source: str) -> list[str]:
            """List columns for a specific source."""
            profile = table_profiles.get(source, {})
            cols = profile.get("columns", [])
            return [str(c.get("name")) for c in cols if isinstance(c, dict) and c.get("name")]

        def describe_column(source: str, column: str) -> dict[str, Any]:
            """Describe column role and top values for planning."""
            profile = table_profiles.get(source, {})
            for col in profile.get("columns", []):
                if not isinstance(col, dict):
                    continue
                if str(col.get("name")) == str(column):
                    return {
                        "name": str(col.get("name")),
                        "role": str(col.get("role") or "unknown"),
                        "top_values": col.get("top_values") or [],
                    }
            return {"name": str(column), "role": "unknown", "top_values": []}

        class _PlanSig(dspy.Signature):
            question: str = dspy.InputField(desc="User query")
            history: str = dspy.InputField(desc="Recent user/assistant context")
            intent: str = dspy.OutputField(desc="schema|count_rows|count_distinct|list_values|aggregate|mixed")
            source: str = dspy.OutputField(desc="Source filename from available tables")
            column: str = dspy.OutputField(desc="Column name if applicable")
            operation: str = dspy.OutputField(desc="mean|median|min|max|sum|std|none")
            include_values: bool = dspy.OutputField(desc="Whether distinct category values are needed")

        tools = [
            dspy.Tool(list_tables, name="list_tables", desc="List loaded table names."),
            dspy.Tool(list_columns, name="list_columns", desc="List columns for a table source."),
            dspy.Tool(describe_column, name="describe_column", desc="Inspect role/top values for a column."),
        ]

        try:
            react = dspy.ReAct(_PlanSig, tools=tools, max_iters=6)
            pred = react(question=question.strip(), history=_compact(history, limit=1000))
        except Exception:
            return None

        return self._normalize_plan(
            intent=getattr(pred, "intent", ""),
            source=getattr(pred, "source", ""),
            column=getattr(pred, "column", ""),
            operation=getattr(pred, "operation", ""),
            include_values=getattr(pred, "include_values", False),
            table_profiles=table_profiles,
        )

    def _plan_with_predict(
        self,
        *,
        question: str,
        history: str,
        table_profiles: dict[str, dict[str, Any]],
    ) -> TabularPlan | None:
        try:
            import dspy
        except Exception:
            return None
        if getattr(dspy.settings, "lm", None) is None:
            return None

        blocks: list[str] = []
        for source, profile in table_profiles.items():
            lines = [f"Table: {source}"]
            for col in profile.get("columns", [])[:80]:
                if not isinstance(col, dict):
                    continue
                col_name = str(col.get("name") or "")
                if not col_name:
                    continue
                role = str(col.get("role") or "unknown")
                top_vals = col.get("top_values") or []
                preview_vals = []
                if isinstance(top_vals, list):
                    for v in top_vals[:3]:
                        if isinstance(v, dict) and v.get("value") is not None:
                            preview_vals.append(str(v.get("value")))
                preview = f" values={','.join(preview_vals)}" if preview_vals else ""
                lines.append(f"- {col_name} [{role}]{preview}")
            blocks.append("\n".join(lines))

        guidance = (
            "Plan deterministic tabular execution. "
            "intent must be one of schema, count_rows, count_distinct, list_values, aggregate, mixed. "
            "operation must be one of mean, median, min, max, sum, std, none."
        )
        try:
            pred = dspy.Predict(
                "guidance, question, history, tables -> intent, source, column, operation, include_values"
            )(
                guidance=guidance,
                question=question.strip(),
                history=_compact(history, limit=1000),
                tables="\n\n".join(blocks)[:22000],
            )
        except Exception:
            return None

        return self._normalize_plan(
            intent=getattr(pred, "intent", ""),
            source=getattr(pred, "source", ""),
            column=getattr(pred, "column", ""),
            operation=getattr(pred, "operation", ""),
            include_values=getattr(pred, "include_values", False),
            table_profiles=table_profiles,
        )

    def _normalize_plan(
        self,
        *,
        intent: Any,
        source: Any,
        column: Any,
        operation: Any,
        include_values: Any,
        table_profiles: dict[str, dict[str, Any]],
    ) -> TabularPlan | None:
        intent_norm = " ".join(str(intent).lower().split())
        intent_map = {
            "schema": "schema",
            "schema_lookup": "schema",
            "count_rows": "count_rows",
            "row_count": "count_rows",
            "count_distinct": "count_distinct",
            "distinct_count": "count_distinct",
            "list_values": "list_values",
            "aggregate": "aggregate",
            "mixed": "mixed",
        }
        intent_final = intent_map.get(intent_norm, "")
        if not intent_final:
            return None

        source_norm = " ".join(str(source).split())
        if not source_norm and len(table_profiles) == 1:
            source_norm = next(iter(table_profiles.keys()))
        if source_norm and source_norm not in table_profiles:
            source_norm = ""

        col_norm = " ".join(str(column).split())
        if source_norm and col_norm:
            cols = [
                str(c.get("name"))
                for c in table_profiles.get(source_norm, {}).get("columns", [])
                if isinstance(c, dict) and c.get("name")
            ]
            if col_norm not in cols:
                col_map = {re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_"): c for c in cols}
                key = re.sub(r"[^a-z0-9]+", "_", col_norm.lower()).strip("_")
                col_norm = col_map.get(key, "")

        op_norm = " ".join(str(operation).lower().split())
        if op_norm in {"none", "n/a", ""}:
            op_norm = None
        if op_norm not in {None, "mean", "median", "min", "max", "sum", "std"}:
            op_norm = None

        include_values_bool = " ".join(str(include_values).lower().split()) in {
            "true",
            "1",
            "yes",
            "y",
        }
        if isinstance(include_values, bool):
            include_values_bool = include_values

        return TabularPlan(
            intent=intent_final,
            source=source_norm or None,
            column=col_norm or None,
            operation=op_norm,
            include_values=include_values_bool,
        )


class ProgrammaticTableAnalyst:
    """ProgramOfThought-based SQL planner for complex tabular analytics."""

    _DENY_SQL_TERMS = {
        "insert", "update", "delete", "drop", "alter", "create", "truncate", "attach",
        "detach", "copy", "export", "import", "vacuum", "pragma", "call",
    }

    def propose_sql(
        self,
        *,
        question: str,
        history: str,
        table_profiles: dict[str, dict[str, Any]],
    ) -> dict[str, str] | None:
        if not table_profiles:
            return None

        try:
            import dspy
        except Exception:
            return None
        if getattr(dspy.settings, "lm", None) is None:
            return None
        schema_text = self._schema_text(table_profiles)
        pot_plan = self._propose_sql_with_program_of_thought(
            dspy=dspy,
            question=question,
            history=history,
            schema=schema_text,
            table_profiles=table_profiles,
        )
        if pot_plan is not None:
            return pot_plan

        return self._propose_sql_with_codeact(
            dspy=dspy,
            question=question,
            history=history,
            schema=schema_text,
            table_profiles=table_profiles,
        )

    def _schema_text(self, table_profiles: dict[str, dict[str, Any]]) -> str:
        schema_blocks: list[str] = []
        for source, profile in table_profiles.items():
            lines = [f"Table: {source}", f"Rows: {profile.get('rows', 'unknown')}"]
            for col in profile.get("columns", [])[:80]:
                if not isinstance(col, dict) or not col.get("name"):
                    continue
                col_name = str(col.get("name"))
                role = str(col.get("role") or "unknown")
                values = col.get("top_values") if isinstance(col.get("top_values"), list) else []
                preview_values: list[str] = []
                for item in values[:4]:
                    if isinstance(item, dict) and item.get("value") is not None:
                        preview_values.append(str(item.get("value")))
                preview = f" top_values={','.join(preview_values)}" if preview_values else ""
                lines.append(f"- {col_name} [{role}]{preview}")
            schema_blocks.append("\n".join(lines))
        return "\n\n".join(schema_blocks)[:18000]

    def _normalize_sql_plan(
        self,
        *,
        source: Any,
        sql: Any,
        rationale: Any,
        table_profiles: dict[str, dict[str, Any]],
    ) -> dict[str, str] | None:
        source_text = " ".join(str(source or "").split())
        sql_text = " ".join(str(sql or "").split())
        rationale_text = _compact(str(rationale or ""), limit=260)
        if not source_text or source_text not in table_profiles:
            return None
        if not self._is_safe_sql(sql_text):
            return None
        if "_mosaicx_table" not in sql_text:
            return None
        return {"source": source_text, "sql": sql_text, "rationale": rationale_text}

    def _propose_sql_with_program_of_thought(
        self,
        *,
        dspy: Any,
        question: str,
        history: str,
        schema: str,
        table_profiles: dict[str, dict[str, Any]],
    ) -> dict[str, str] | None:
        if not hasattr(dspy, "ProgramOfThought"):
            return None

        guidance = (
            "Generate a safe, read-only SQL query for a single selected source table. "
            "Use only SELECT queries and reference table alias `_mosaicx_table`. "
            "For distribution/category questions, return rows with aliases `value` and `count` "
            "(GROUP BY category, COUNT(*) AS count). "
            "For count-distinct questions, use COUNT(DISTINCT ...) with alias `count`. "
            "Return source as one table name from schema and sql as executable query."
        )
        try:
            pot = dspy.ProgramOfThought(
                "guidance, question, history, schema -> source, sql, rationale"
            )
            pred = pot(
                guidance=guidance,
                question=question.strip(),
                history=_compact(history, limit=900),
                schema=schema,
            )
        except Exception:
            return None

        return self._normalize_sql_plan(
            source=getattr(pred, "source", ""),
            sql=getattr(pred, "sql", ""),
            rationale=getattr(pred, "rationale", ""),
            table_profiles=table_profiles,
        )

    def _propose_sql_with_codeact(
        self,
        *,
        dspy: Any,
        question: str,
        history: str,
        schema: str,
        table_profiles: dict[str, dict[str, Any]],
    ) -> dict[str, str] | None:
        if not hasattr(dspy, "CodeAct"):
            return None

        tables = sorted(table_profiles.keys())

        def list_tables() -> list[str]:
            """List available table names."""
            return tables

        def list_columns(source: str) -> list[str]:
            """List columns for a table source."""
            profile = table_profiles.get(str(source), {})
            cols = profile.get("columns", [])
            return [str(c.get("name")) for c in cols if isinstance(c, dict) and c.get("name")]

        def describe_column(source: str, column: str) -> dict[str, Any]:
            """Describe one column role/top values for planning."""
            profile = table_profiles.get(str(source), {})
            for col in profile.get("columns", []):
                if not isinstance(col, dict):
                    continue
                if str(col.get("name")) == str(column):
                    return {
                        "name": str(col.get("name")),
                        "role": str(col.get("role") or "unknown"),
                        "top_values": col.get("top_values") or [],
                    }
            return {"name": str(column), "role": "unknown", "top_values": []}

        try:
            codeact = dspy.CodeAct(
                (
                    "question, history, schema -> source, sql, rationale"
                ),
                tools=[list_tables, list_columns, describe_column],
                max_iters=5,
            )
            pred = codeact(
                question=question.strip(),
                history=_compact(history, limit=700),
                schema=schema,
            )
        except Exception:
            return None

        return self._normalize_sql_plan(
            source=getattr(pred, "source", ""),
            sql=getattr(pred, "sql", ""),
            rationale=getattr(pred, "rationale", ""),
            table_profiles=table_profiles,
        )

    def _is_safe_sql(self, sql: str) -> bool:
        if not sql:
            return False
        text = " ".join(sql.lower().split())
        if not text.startswith("select"):
            return False
        if ";" in text:
            return False
        if any(term in text for term in self._DENY_SQL_TERMS):
            return False
        return True


class EvidenceVerifier:
    """Evidence-grounding verifier with BestOfN and comparison fallback."""

    NON_ANSWER_MARKERS = (
        "unable to retrieve",
        "unable to provide",
        "cannot provide",
        "cannot retrieve",
        "not available",
        "insufficient information",
    )

    def revise(
        self,
        *,
        question: str,
        draft_answer: str,
        citations: list[dict[str, Any]],
    ) -> str | None:
        if not citations:
            return None

        try:
            import dspy
        except Exception:
            return None
        if getattr(dspy.settings, "lm", None) is None:
            return None

        evidence = "\n".join(
            f"[{i}] {str(c.get('source') or 'unknown')}: "
            f"{_compact(str(c.get('snippet') or ''), limit=320)}"
            for i, c in enumerate(citations[:8], start=1)
        )
        guidance = (
            "Decide if draft answer is grounded in the evidence. "
            "Return status in supported|corrected|insufficient and provide revised_answer."
        )
        base = dspy.ChainOfThought(
            "guidance, question, draft_answer, evidence -> status, revised_answer"
        )

        def reward_fn(args: dict[str, Any], pred: Any) -> float:
            revised = " ".join(str(getattr(pred, "revised_answer", "")).lower().split())
            status = " ".join(str(getattr(pred, "status", "")).lower().split())
            if not revised:
                return 0.0
            if any(marker in revised for marker in self.NON_ANSWER_MARKERS):
                return 0.05
            blob = " ".join(str(c.get("snippet") or "").lower() for c in citations)
            terms = {
                t
                for t in re.findall(r"[a-z0-9]+", revised)
                if len(t) >= 3
            }
            overlap = sum(1 for t in terms if t in blob)
            score = overlap / max(len(terms), 1)
            if "correct" in status or "support" in status:
                score += 0.1
            return max(0.0, min(1.0, score))

        attempts: list[dict[str, Any]] = []
        for idx in range(3):
            try:
                lm = dspy.settings.lm
                if lm is not None:
                    with dspy.context(lm=lm.copy(rollout_id=idx + 1, temperature=1.0)):
                        p = base(
                            guidance=guidance,
                            question=question.strip(),
                            draft_answer=draft_answer.strip(),
                            evidence=evidence,
                        )
                else:
                    p = base(
                        guidance=guidance,
                        question=question.strip(),
                        draft_answer=draft_answer.strip(),
                        evidence=evidence,
                    )
                attempts.append(dict(p))
            except Exception:
                continue

        mcc_candidate = ""
        if len(attempts) == 3 and hasattr(dspy, "MultiChainComparison"):
            try:
                mcc = dspy.MultiChainComparison(
                    "question, evidence -> revised_answer",
                    M=3,
                    temperature=0.2,
                )
                mcc_pred = mcc(
                    completions=attempts,
                    question=question.strip(),
                    evidence=evidence,
                )
                mcc_candidate = str(getattr(mcc_pred, "revised_answer", "")).strip()
            except Exception:
                mcc_candidate = ""

        best_candidate = ""
        if hasattr(dspy, "BestOfN"):
            try:
                best_mod = dspy.BestOfN(
                    module=base,
                    N=3,
                    reward_fn=reward_fn,
                    threshold=0.85,
                )
                best_pred = best_mod(
                    guidance=guidance,
                    question=question.strip(),
                    draft_answer=draft_answer.strip(),
                    evidence=evidence,
                )
                best_candidate = str(getattr(best_pred, "revised_answer", "")).strip()
            except Exception:
                best_candidate = ""

        candidates = [c for c in [mcc_candidate, best_candidate] if c]
        if not candidates:
            return None

        # Select candidate with strongest lexical grounding.
        blob = " ".join(str(c.get("snippet") or "").lower() for c in citations)
        best_text = ""
        best_score = -1.0
        for candidate in candidates:
            terms = {t for t in re.findall(r"[a-z0-9]+", candidate.lower()) if len(t) >= 3}
            if not terms:
                continue
            score = sum(1 for t in terms if t in blob) / max(len(terms), 1)
            if score > best_score:
                best_score = score
                best_text = candidate
        if not best_text:
            return None
        if best_text.strip() == draft_answer.strip():
            return None
        return best_text.strip()
