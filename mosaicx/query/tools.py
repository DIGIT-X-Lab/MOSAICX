"""MOSAICX-specific tools available to the RLM sandbox during query sessions."""

from __future__ import annotations

import csv
import io
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DEFAULT_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "what", "which",
    "who", "whom", "when", "where", "why", "how", "does", "do", "did",
    "please", "show", "tell", "about", "can", "you", "your", "my", "me",
    "from", "that", "this", "those", "these", "there", "their", "them",
    "no", "patient", "study", "scan", "ct", "mri", "table", "tables",
    "dataset", "data", "cohort", "findings", "impression", "compared",
    "comparison", "today",
}
_TERM_SYNONYMS: dict[str, set[str]] = {
    "cancer": {
        "carcinoma",
        "malignancy",
        "malignant",
        "neoplasm",
        "tumor",
        "oncology",
        "adenocarcinoma",
        "metastasis",
        "metastatic",
    },
    "pathology": {
        "diagnosis",
        "disease",
        "carcinoma",
        "malignancy",
        "tumor",
    },
    "lesion": {
        "nodule",
        "mass",
        "node",
        "lymph",
    },
}
_OPERATION_ALIASES = {
    "mean": {"mean", "average", "avg"},
    "median": {"median"},
    "min": {"min", "minimum", "lowest", "smallest"},
    "max": {"max", "maximum", "highest", "largest"},
    "sum": {"sum", "total"},
    "std": {"std", "stdev", "standard", "deviation"},
    "count": {"count", "rows", "row", "many", "number", "total"},
    "nunique": {"distinct", "unique"},
    "missing_count": {"missing", "null", "na", "nan"},
}
_ENTITY_COUNT_TERMS = {
    "subject",
    "patient",
    "participant",
    "case",
    "id",
    "identifier",
    "individual",
    "person",
}
_ROW_COUNT_TERMS = {"row", "rows", "record", "records", "entry", "entries", "observation", "observations", "sample", "samples"}


def _normalize_token(token: str) -> str:
    """Lightweight token normalization for keyword matching."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _extract_terms(text: str) -> list[str]:
    return [_normalize_token(t) for t in _TOKEN_RE.findall(text.lower())]


def _expand_terms(terms: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term and term not in seen:
            expanded.append(term)
            seen.add(term)
        for syn in _TERM_SYNONYMS.get(term, ()):
            syn_n = _normalize_token(syn)
            if syn_n and syn_n not in seen:
                expanded.append(syn_n)
                seen.add(syn_n)
    return expanded


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    try:
        import pandas as pd

        if pd.isna(value):
            return "null"
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _iter_tables(data: dict[str, Any]):
    try:
        import pandas as pd
    except ImportError:
        return
    for name, value in data.items():
        if isinstance(value, pd.DataFrame):
            yield name, value


def _resolve_table(name: str, *, data: dict[str, Any]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for table tools") from exc

    if name not in data:
        raise KeyError(f"Table '{name}' not found. Available: {list(data.keys())}")
    value = data[name]
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Source '{name}' is not tabular (got {type(value).__name__})")
    return value


def _resolve_column(df, column: str) -> str:
    columns = [str(c) for c in df.columns]
    col_map = {_normalize_column_name(c): c for c in columns}
    if column in columns:
        return column
    n = _normalize_column_name(column)
    if n in col_map:
        return col_map[n]
    raise KeyError(f"Column '{column}' not found. Available: {columns}")


def _choose_column(question: str, columns: list[str]) -> str | None:
    q_terms = [t for t in _extract_terms(question) if t not in _DEFAULT_STOPWORDS]
    if not q_terms:
        return None
    q_compact = "_".join(q_terms)

    best: tuple[int, str] | None = None
    for col in columns:
        col_norm = _normalize_column_name(col)
        col_terms = [t for t in col_norm.split("_") if t]
        score = 0
        if col_norm and col_norm in q_compact:
            score += 120
        overlap = len(set(col_terms) & set(q_terms))
        score += overlap * 15
        if col_norm in q_terms:
            score += 60
        if score > 0 and (best is None or score > best[0]):
            best = (score, col)
    if best is None:
        return None
    return best[1]


def _detect_operation(question: str) -> str | None:
    q_terms = set(_extract_terms(question))
    for op, aliases in _OPERATION_ALIASES.items():
        if q_terms & aliases:
            return op
    q_norm = " ".join(question.lower().split())
    if "how many" in q_norm or "number of" in q_norm:
        return "count"
    if "average" in question.lower():
        return "mean"
    return None


def _should_use_distinct_count(question: str, selected_col: str) -> bool:
    col_terms = set(_extract_terms(selected_col))
    asks_subject_count = _is_entity_count_question(question)
    col_looks_identifier = bool(col_terms & _ENTITY_COUNT_TERMS)
    return asks_subject_count and col_looks_identifier


def _normalize_operation(operation: str) -> str:
    op = operation.strip().lower()
    if op in {"avg", "average"}:
        return "mean"
    if op in {"unique", "distinct"}:
        return "nunique"
    if op in {"missing", "null", "na"}:
        return "missing_count"
    return op


def _sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _question_terms(question: str) -> set[str]:
    return set(_extract_terms(question))


def _is_entity_count_question(question: str) -> bool:
    q_terms = _question_terms(question)
    q_norm = " ".join(question.lower().split())
    if bool(q_terms & _ENTITY_COUNT_TERMS):
        return True
    return (
        ("how many" in q_norm or "number of" in q_norm)
        and "cohort" in q_terms
    )


def _is_row_count_question(question: str) -> bool:
    q_terms = _question_terms(question)
    q_norm = " ".join(question.lower().split())
    asks_rows = bool(q_terms & _ROW_COUNT_TERMS)
    asks_how_many = "how many" in q_norm or "number of" in q_norm
    asks_entities = bool(q_terms & _ENTITY_COUNT_TERMS)
    return asks_rows or (asks_how_many and not asks_entities)


def infer_table_roles(
    name: str,
    *,
    data: dict[str, Any],
    max_columns: int = 300,
) -> dict[str, Any]:
    """Infer coarse semantic roles for columns in a tabular source."""
    import pandas as pd
    from pandas.api import types as ptypes

    df = _resolve_table(name, data=data)
    rows = int(len(df))
    columns = [str(c) for c in df.columns[: max(1, max_columns)]]

    roles: dict[str, list[str]] = {
        "id": [],
        "numeric": [],
        "categorical": [],
        "boolean": [],
        "datetime": [],
        "text": [],
    }
    id_candidates: list[dict[str, Any]] = []
    column_roles: dict[str, str] = {}

    for col in columns:
        s = df[col]
        dtype = str(s.dtype)
        non_null = int(s.notna().sum())
        unique = int(s.nunique(dropna=True))
        unique_ratio = (unique / non_null) if non_null else 0.0
        col_terms = set(_extract_terms(col))

        role = "categorical"
        if ptypes.is_bool_dtype(s):
            role = "boolean"
        elif ptypes.is_numeric_dtype(s):
            role = "numeric"
        elif ptypes.is_datetime64_any_dtype(s):
            role = "datetime"
        else:
            sample = s.dropna().astype(str).head(80)
            avg_len = float(sample.str.len().mean()) if not sample.empty else 0.0
            date_ratio = 0.0
            if not sample.empty:
                col_lc = str(col).lower()
                has_datetime_name_hint = any(
                    hint in col_lc for hint in ("date", "time", "timestamp", "dob")
                )
                has_digit = sample.str.contains(r"\d", regex=True)
                has_separator = sample.str.contains(r"[-/:]", regex=True)
                parse_candidate_ratio = float((has_digit & has_separator).mean())
                should_try_datetime_parse = (
                    has_datetime_name_hint or parse_candidate_ratio >= 0.45
                )
                if should_try_datetime_parse:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        parsed_dates = pd.to_datetime(sample, errors="coerce")
                    date_ratio = float(parsed_dates.notna().mean())
            if date_ratio >= 0.7:
                role = "datetime"
            elif avg_len >= 48 and unique >= min(max(15, non_null // 3), non_null):
                role = "text"
            else:
                role = "categorical"

        roles[role].append(col)
        column_roles[col] = role

        id_hint = bool(col_terms & _ENTITY_COUNT_TERMS)
        eligible = (
            non_null >= max(3, int(rows * 0.5))
            and unique_ratio >= 0.95
            and role in {"categorical", "numeric", "datetime"}
        )
        if eligible:
            score = int(round(unique_ratio * 100)) + (20 if id_hint else 0)
            id_candidates.append(
                {
                    "column": col,
                    "score": score,
                    "non_null": non_null,
                    "unique": unique,
                    "unique_ratio": round(unique_ratio, 6),
                    "dtype": dtype,
                    "role": role,
                }
            )

    id_candidates.sort(
        key=lambda x: (int(x.get("score", 0)), float(x.get("unique_ratio", 0.0))),
        reverse=True,
    )
    roles["id"] = [str(x["column"]) for x in id_candidates[:10]]

    return {
        "source": name,
        "rows": rows,
        "column_count": int(len(df.columns)),
        "roles": roles,
        "id_candidates": id_candidates[:20],
        "column_roles": column_roles,
    }


def suggest_table_columns(
    question: str,
    *,
    data: dict[str, Any],
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Suggest relevant columns for a question across all loaded tables."""
    q_terms = [t for t in _extract_terms(question) if t not in _DEFAULT_STOPWORDS]
    if not q_terms:
        q_terms = _extract_terms(question)
    if not q_terms:
        return []

    op = _detect_operation(question)
    q_term_set = set(q_terms)
    out: list[dict[str, Any]] = []
    for name, df in _iter_tables(data) or ():
        role_info = infer_table_roles(name, data=data, max_columns=300)
        column_roles: dict[str, str] = role_info.get("column_roles", {})
        id_cols = set(role_info.get("roles", {}).get("id", []))
        for col in [str(c) for c in df.columns]:
            col_norm = _normalize_column_name(col)
            col_terms = [t for t in col_norm.split("_") if t]
            overlap = len(set(col_terms) & q_term_set)
            substring_hits = sum(1 for t in q_terms if t in col_norm)
            score = overlap * 25 + substring_hits * 12

            role = column_roles.get(col, "categorical")
            if op in {"mean", "median", "min", "max", "sum", "std"} and role == "numeric":
                score += 20
            if op in {"nunique", "count"} and col in id_cols:
                score += 25
            if _is_entity_count_question(question) and col in id_cols:
                score += 35

            if score <= 0:
                continue
            out.append(
                {
                    "source": name,
                    "column": col,
                    "role": role,
                    "score": int(score),
                    "snippet": f"Column match: {col} (role={role}, score={int(score)})",
                    "evidence_type": "table_column",
                }
            )

    out.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
    return out[: max(1, top_k)]


def profile_table(
    name: str,
    *,
    data: dict[str, Any],
    max_columns: int = 80,
    top_values: int = 6,
) -> dict[str, Any]:
    """Build a schema-agnostic EDA profile for a table source."""
    import pandas as pd

    df = _resolve_table(name, data=data)
    rows = int(len(df))
    role_info = infer_table_roles(name, data=data, max_columns=max_columns)
    column_roles = role_info.get("column_roles", {})

    columns_out: list[dict[str, Any]] = []
    max_columns = max(1, min(int(max_columns), len(df.columns)))
    for col in [str(c) for c in df.columns[:max_columns]]:
        s = df[col]
        non_null = int(s.notna().sum())
        missing = int(rows - non_null)
        unique = int(s.nunique(dropna=True))
        unique_ratio = (unique / non_null) if non_null else 0.0
        role = str(column_roles.get(col, "categorical"))

        col_item: dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "role": role,
            "non_null": non_null,
            "missing": missing,
            "missing_pct": round((missing / rows) if rows else 0.0, 6),
            "unique": unique,
            "unique_ratio": round(unique_ratio, 6),
        }

        if role == "numeric":
            nums = pd.to_numeric(s, errors="coerce").dropna()
            if not nums.empty:
                col_item["summary"] = {
                    "mean": _format_scalar(float(nums.mean())),
                    "std": _format_scalar(float(nums.std())) if nums.shape[0] > 1 else "0",
                    "min": _format_scalar(float(nums.min())),
                    "p25": _format_scalar(float(nums.quantile(0.25))),
                    "median": _format_scalar(float(nums.median())),
                    "p75": _format_scalar(float(nums.quantile(0.75))),
                    "max": _format_scalar(float(nums.max())),
                }
        elif role in {"categorical", "boolean"}:
            vc = s.dropna().astype(str).value_counts().head(max(1, min(int(top_values), 20)))
            if not vc.empty:
                col_item["top_values"] = [
                    {"value": str(idx), "count": int(cnt)}
                    for idx, cnt in vc.items()
                ]

        columns_out.append(col_item)

    return {
        "source": name,
        "rows": rows,
        "column_count": int(len(df.columns)),
        "roles": role_info.get("roles", {}),
        "id_candidates": role_info.get("id_candidates", [])[:8],
        "columns": columns_out,
    }


def list_tables(
    *,
    data: dict[str, Any],
    max_columns: int = 30,
) -> list[dict[str, Any]]:
    """List tabular sources with shape and column preview."""
    out: list[dict[str, Any]] = []
    for name, df in _iter_tables(data) or ():
        columns = [str(c) for c in df.columns]
        shown = columns[: max(1, max_columns)]
        out.append(
            {
                "source": name,
                "rows": int(len(df)),
                "columns": shown,
                "column_count": int(len(columns)),
                "has_more_columns": len(columns) > len(shown),
            }
        )
    return out


def get_table_schema(
    name: str,
    *,
    data: dict[str, Any],
    max_columns: int = 200,
) -> dict[str, Any]:
    """Return schema summary for a table source."""
    df = _resolve_table(name, data=data)
    rows = int(len(df))
    columns: list[dict[str, Any]] = []
    for c in [str(x) for x in df.columns[: max(1, max_columns)]]:
        s = df[c]
        non_null = int(s.notna().sum())
        missing = int(rows - non_null)
        uniq = int(s.nunique(dropna=True))
        columns.append(
            {
                "name": c,
                "dtype": str(s.dtype),
                "non_null": non_null,
                "missing": missing,
                "missing_pct": (missing / rows) if rows else 0.0,
                "unique": uniq,
            }
        )
    return {
        "source": name,
        "rows": rows,
        "column_count": int(len(df.columns)),
        "columns": columns,
    }


def sample_table_rows(
    name: str,
    *,
    data: dict[str, Any],
    columns_csv: str = "",
    limit: int = 5,
    strategy: str = "head",
) -> list[dict[str, Any]]:
    """Sample rows from a table source.

    Parameters
    ----------
    name:
        Table source name.
    data:
        Session data map.
    columns_csv:
        Optional comma-separated subset of columns.
    limit:
        Number of rows to return.
    strategy:
        head, tail, or random.
    """
    df = _resolve_table(name, data=data)
    limit = max(1, min(int(limit), 100))

    selected_df = df
    if columns_csv.strip():
        selected_cols: list[str] = []
        for raw in columns_csv.split(","):
            raw = raw.strip()
            if not raw:
                continue
            selected_cols.append(_resolve_column(df, raw))
        if selected_cols:
            selected_df = selected_df[selected_cols]

    strategy_norm = strategy.strip().lower()
    if strategy_norm == "tail":
        sampled = selected_df.tail(limit)
    elif strategy_norm == "random":
        sampled = selected_df.sample(min(limit, len(selected_df)), random_state=0)
    else:
        sampled = selected_df.head(limit)

    out: list[dict[str, Any]] = []
    for idx, row in sampled.iterrows():
        row_obj = {"__row_index": int(idx) if isinstance(idx, (int,)) else str(idx)}
        for col, val in row.items():
            row_obj[str(col)] = _format_scalar(val)
        out.append(row_obj)
    return out


def _compute_table_stat_duckdb(
    name: str,
    *,
    df: Any,
    column: str,
    operation: str,
    group_by: str,
    where: str,
    top_n: int,
) -> list[dict[str, Any]]:
    import duckdb

    col = _resolve_column(df, column)
    op = _normalize_operation(operation)
    if op not in {
        "mean",
        "median",
        "min",
        "max",
        "sum",
        "std",
        "count",
        "nunique",
        "missing_count",
    }:
        raise ValueError(f"Unsupported operation: {op}")

    group_col = _resolve_column(df, group_by) if group_by.strip() else ""
    table_name = "_mosaicx_table"
    where_sql = f" WHERE {where.strip()}" if where.strip() else ""
    top_n = max(1, min(int(top_n), 50))

    numeric_expr = f"TRY_CAST({_sql_ident(col)} AS DOUBLE)"
    agg_expr = {
        "mean": f"AVG({numeric_expr})",
        "median": f"MEDIAN({numeric_expr})",
        "min": f"MIN({numeric_expr})",
        "max": f"MAX({numeric_expr})",
        "sum": f"SUM({numeric_expr})",
        "std": f"STDDEV_SAMP({numeric_expr})",
        "count": f"COUNT({_sql_ident(col)})",
        "nunique": f"COUNT(DISTINCT {_sql_ident(col)})",
        "missing_count": f"SUM(CASE WHEN {_sql_ident(col)} IS NULL THEN 1 ELSE 0 END)",
    }[op]

    with duckdb.connect(database=":memory:") as conn:
        conn.register(table_name, df)

        try:
            row_count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {table_name}{where_sql}"
                ).fetchone()[0]
            )
        except Exception as exc:
            raise ValueError(f"Invalid where expression: {exc}") from exc

        if group_col:
            query = (
                f"SELECT {_sql_ident(group_col)} AS grp, {agg_expr} AS val "
                f"FROM {table_name}{where_sql} "
                f"GROUP BY {_sql_ident(group_col)} "
                f"ORDER BY val DESC NULLS LAST "
                f"LIMIT {top_n}"
            )
            try:
                rows = conn.execute(query).fetchall()
            except Exception as exc:
                raise ValueError(f"Failed to compute grouped stat: {exc}") from exc

            out: list[dict[str, Any]] = []
            for grp, val in rows:
                out.append(
                    {
                        "source": name,
                        "group": _format_scalar(grp),
                        "column": col,
                        "operation": op,
                        "value": _format_scalar(val),
                        "row_count": row_count,
                        "backend": "duckdb",
                    }
                )
            return out

        if op in {"mean", "median", "min", "max", "sum", "std"}:
            non_null_expr = f"COUNT({numeric_expr})"
        else:
            non_null_expr = f"COUNT({_sql_ident(col)})"
        query = (
            f"SELECT {agg_expr} AS val, {non_null_expr} AS non_null "
            f"FROM {table_name}{where_sql}"
        )
        try:
            value, non_null = conn.execute(query).fetchone()
        except Exception as exc:
            raise ValueError(f"Failed to compute stat: {exc}") from exc

    if op in {"mean", "median", "min", "max", "sum", "std"} and int(non_null or 0) == 0:
        raise ValueError(f"Column '{col}' has no numeric values for operation '{op}'")

    return [
        {
            "source": name,
            "column": col,
            "operation": op,
            "value": _format_scalar(value) if op in {"mean", "median", "min", "max", "sum", "std"} else int(value or 0),
            "row_count": int(row_count),
            "non_null": int(non_null or 0),
            "backend": "duckdb",
        }
    ]


def _compute_table_stat_pandas(
    name: str,
    *,
    df: Any,
    column: str,
    operation: str,
    group_by: str,
    where: str,
    top_n: int,
) -> list[dict[str, Any]]:
    import pandas as pd

    working = df
    if where.strip():
        try:
            working = working.query(where, engine="python")
        except Exception as exc:
            raise ValueError(f"Invalid where expression: {exc}") from exc

    col = _resolve_column(working, column)
    op = _normalize_operation(operation)

    group_col = ""
    if group_by.strip():
        group_col = _resolve_column(working, group_by)

    if group_col:
        if op in {"count", "nunique", "missing_count"}:
            grouped = working.groupby(group_col, dropna=False)[col]
            if op == "count":
                series = grouped.count()
            elif op == "nunique":
                series = grouped.nunique(dropna=True)
            elif op == "missing_count":
                series = grouped.apply(lambda s: int(s.isna().sum()))
            else:
                raise ValueError(f"Unsupported operation with group_by: {op}")
        else:
            num = pd.to_numeric(working[col], errors="coerce")
            grouped = pd.DataFrame(
                {"__group": working[group_col], "__value": num}
            ).groupby("__group", dropna=False)["__value"]
            if op == "mean":
                series = grouped.mean()
            elif op == "median":
                series = grouped.median()
            elif op == "min":
                series = grouped.min()
            elif op == "max":
                series = grouped.max()
            elif op == "sum":
                series = grouped.sum()
            elif op == "std":
                series = grouped.std()
            else:
                raise ValueError(f"Unsupported operation with group_by: {op}")

        rows: list[dict[str, Any]] = []
        for idx, value in series.sort_values(ascending=False).head(max(1, min(int(top_n), 50))).items():
            rows.append(
                {
                    "source": name,
                    "group": _format_scalar(idx),
                    "column": col,
                    "operation": op,
                    "value": _format_scalar(value),
                    "row_count": int(len(working)),
                    "backend": "pandas",
                }
            )
        return rows

    series = working[col]
    if op in {"mean", "median", "min", "max", "sum", "std"}:
        nums = pd.to_numeric(series, errors="coerce")
        valid = nums.dropna()
        if valid.empty:
            raise ValueError(
                f"Column '{col}' has no numeric values for operation '{op}'"
            )
        if op == "mean":
            value = float(valid.mean())
        elif op == "median":
            value = float(valid.median())
        elif op == "min":
            value = float(valid.min())
        elif op == "max":
            value = float(valid.max())
        elif op == "sum":
            value = float(valid.sum())
        elif op == "std":
            value = float(valid.std())
        else:
            raise ValueError(f"Unsupported operation: {op}")
        return [
            {
                "source": name,
                "column": col,
                "operation": op,
                "value": _format_scalar(value),
                "row_count": int(len(working)),
                "non_null": int(valid.shape[0]),
                "backend": "pandas",
            }
        ]

    if op == "count":
        value = int(series.notna().sum())
    elif op == "nunique":
        value = int(series.nunique(dropna=True))
    elif op == "missing_count":
        value = int(series.isna().sum())
    else:
        raise ValueError(f"Unsupported operation: {op}")
    return [
        {
            "source": name,
            "column": col,
            "operation": op,
            "value": value,
            "row_count": int(len(working)),
            "non_null": int(series.notna().sum()),
            "backend": "pandas",
        }
    ]


def compute_table_stat(
    name: str,
    *,
    data: dict[str, Any],
    column: str,
    operation: str = "mean",
    group_by: str = "",
    where: str = "",
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Compute common statistics from a tabular source.

    Prefers DuckDB (fast, deterministic SQL execution) when available and
    falls back to pandas to keep query usable in minimal environments.
    """
    df = _resolve_table(name, data=data)
    op = _normalize_operation(operation)

    try:
        return _compute_table_stat_duckdb(
            name,
            df=df,
            column=column,
            operation=op,
            group_by=group_by,
            where=where,
            top_n=top_n,
        )
    except ImportError:
        pass
    except Exception as duckdb_exc:
        # Fall back to pandas for broader expression compatibility.
        try:
            return _compute_table_stat_pandas(
                name,
                df=df,
                column=column,
                operation=op,
                group_by=group_by,
                where=where,
                top_n=top_n,
            )
        except Exception:
            raise duckdb_exc

    return _compute_table_stat_pandas(
        name,
        df=df,
        column=column,
        operation=op,
        group_by=group_by,
        where=where,
        top_n=top_n,
    )


def run_table_sql(
    name: str,
    *,
    data: dict[str, Any],
    sql: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Run a read-only SQL query against a tabular source via DuckDB."""
    import duckdb

    df = _resolve_table(name, data=data)
    query = sql.strip()
    if not query:
        raise ValueError("sql must be non-empty")
    query_lc = query.lower().lstrip()
    if not query_lc.startswith("select"):
        raise ValueError("Only SELECT SQL statements are allowed.")
    if ";" in query.rstrip(";"):
        raise ValueError("Only a single SQL statement is allowed.")

    safe_limit = max(1, min(int(limit), 500))
    with duckdb.connect(database=":memory:") as conn:
        conn.register("_mosaicx_table", df)
        try:
            result = conn.execute(query).fetchdf()
        except Exception as exc:
            raise ValueError(f"SQL execution failed: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for _, row in result.head(safe_limit).iterrows():
        obj: dict[str, Any] = {}
        for col, val in row.items():
            obj[str(col)] = _format_scalar(val)
        rows.append(obj)
    return rows


def analyze_table_question(
    question: str,
    *,
    data: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Derive deterministic evidence snippets for common cohort-stat questions."""
    op = _detect_operation(question)
    out: list[dict[str, Any]] = []

    for name, df in _iter_tables(data) or ():
        columns = [str(c) for c in df.columns]
        if not op:
            continue
        role_info = infer_table_roles(name, data=data, max_columns=300)
        id_candidates = [str(c) for c in role_info.get("roles", {}).get("id", [])]
        numeric_candidates = [str(c) for c in role_info.get("roles", {}).get("numeric", [])]
        selected_col = _choose_column(question, columns)
        if selected_col is None:
            suggested = suggest_table_columns(question, data={name: df}, top_k=6)
            if suggested:
                selected_col = str(suggested[0].get("column") or "")
        if selected_col is None:
            if _is_entity_count_question(question) and id_candidates:
                selected_col = id_candidates[0]
            elif op in {"mean", "median", "min", "max", "sum", "std"} and numeric_candidates:
                selected_col = numeric_candidates[0]
            elif columns:
                selected_col = columns[0]
        if not selected_col:
            continue

        # Generic "how many rows/records" should compute table row count.
        if op == "count" and _is_row_count_question(question):
            row_count = int(len(df))
            out.append(
                {
                    "source": name,
                    "snippet": f"Computed row_count from {row_count} rows: {row_count} (engine=pandas)",
                    "score": 82,
                    "evidence_type": "table_stat",
                    "operation": "row_count",
                    "column": "__rows__",
                    "value": row_count,
                    "backend": "pandas",
                }
            )
            continue

        op_to_run = op
        if op == "count" and (
            _should_use_distinct_count(question, selected_col)
            or (_is_entity_count_question(question) and selected_col in id_candidates)
        ):
            op_to_run = "nunique"
        try:
            computed = compute_table_stat(
                name,
                data=data,
                column=selected_col,
                operation=op_to_run,
            )
        except Exception:
            continue
        for row in computed:
            value = row.get("value")
            row_count = row.get("row_count")
            non_null = row.get("non_null")
            backend = str(row.get("backend") or "table_engine")
            snippet_op = str(row.get("operation") or op_to_run)
            if snippet_op == "nunique" and _is_entity_count_question(question):
                snippet_op = "unique_count"
            if non_null is not None:
                snippet = (
                    f"Computed {snippet_op} of {selected_col} from {row_count} rows "
                    f"(non-null {non_null}): {value} (engine={backend})"
                )
            else:
                snippet = (
                    f"Computed {snippet_op} of {selected_col} from {row_count} rows: "
                    f"{value} (engine={backend})"
                )
            out.append(
                {
                    "source": name,
                    "snippet": snippet,
                    "score": 80,
                    "evidence_type": "table_stat",
                    "operation": str(row.get("operation") or op_to_run),
                    "column": selected_col,
                    "value": value,
                    "backend": backend,
                }
            )

    out.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
    return out[: max(1, top_k)]


def search_tables(
    query: str,
    *,
    data: dict[str, Any],
    top_k: int = 5,
    max_rows: int = 6000,
) -> list[dict[str, Any]]:
    """Search tabular sources and return row-level snippets.

    This complements ``search_documents`` for CSV/Parquet/Excel sources.
    """
    query_terms = [t for t in _extract_terms(query) if len(t) >= 2 and t not in _DEFAULT_STOPWORDS]
    if not query_terms:
        query_terms = _extract_terms(query)
    query_terms = _expand_terms(query_terms)
    if not query_terms:
        return []

    out: list[dict[str, Any]] = []
    for name, df in _iter_tables(data) or ():
        columns = [str(c) for c in df.columns]
        col_norm_map = {c: _normalize_column_name(c) for c in columns}
        candidate_cols = [
            c for c in columns
            if any(term in col_norm_map[c] for term in query_terms)
        ]
        if not candidate_cols:
            candidate_cols = columns[: min(12, len(columns))]

        # Prefer most informative rows while staying bounded for large tables.
        view = df.head(max(1, min(int(max_rows), len(df))))
        for idx, row in view.iterrows():
            matched_pairs: list[tuple[str, str]] = []
            row_score = 0
            for c in candidate_cols:
                val_text = _format_scalar(row[c]).lower()
                col_text = col_norm_map[c]
                hits = sum(1 for t in query_terms if t in col_text or t in val_text)
                if hits:
                    row_score += hits
                    matched_pairs.append((c, _format_scalar(row[c])))
            if row_score <= 0:
                continue

            preview_pairs = matched_pairs[:4]
            preview = ", ".join([f"{k}={v}" for k, v in preview_pairs])
            row_label = int(idx) if isinstance(idx, int) else str(idx)
            snippet = f"row {row_label}: {preview}"
            out.append(
                {
                    "source": name,
                    "snippet": snippet,
                    "score": int(row_score),
                    "evidence_type": "table_row",
                    "row_index": row_label,
                }
            )

    out.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
    return out[: max(1, top_k)]


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
    query_terms = [t for t in raw_terms if len(t) >= 2 and t not in _DEFAULT_STOPWORDS]
    if not query_terms:
        query_terms = raw_terms
    query_terms = _expand_terms(query_terms)
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

            results.append(
                {
                    "source": name,
                    "snippet": snippet,
                    "score": score,
                    "evidence_type": "text",
                }
            )

    results.sort(key=lambda r: int(r["score"]), reverse=True)
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
