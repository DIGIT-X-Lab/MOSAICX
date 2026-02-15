"""Tabular export — CSV, Parquet, JSONL.

Provides three entry-points for writing structured extraction results
to flat-file formats suitable for analytics and downstream pipelines.

Functions
---------
export_csv
    Write results to a CSV file with configurable row strategies.
export_jsonl
    Write results as newline-delimited JSON (one object per line).
export_parquet
    Write results to an Apache Parquet file via *pandas* + *pyarrow*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_one_row(results: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per result; findings are serialised as a JSON string column."""
    rows: list[dict[str, Any]] = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "findings"}
        row["findings"] = json.dumps(r.get("findings", []))
        rows.append(row)
    return pd.DataFrame(rows)


def _flatten_findings_rows(results: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per finding; report-level metadata is repeated on every row."""
    rows: list[dict[str, Any]] = []
    for r in results:
        meta = {k: v for k, v in r.items() if k != "findings"}
        findings = r.get("findings", [])
        for finding in findings:
            row = {**meta, **finding}
            rows.append(row)
    return pd.DataFrame(rows)


_STRATEGIES = {
    "one_row": _flatten_one_row,
    "findings_rows": _flatten_findings_rows,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_csv(
    results: list[dict[str, Any]],
    path: Path | str,
    *,
    strategy: str = "one_row",
) -> None:
    """Write *results* to a CSV file.

    Parameters
    ----------
    results:
        List of extraction-result dicts.
    path:
        Destination file path.
    strategy:
        ``"one_row"`` (default) — one CSV row per result, findings
        serialised as a JSON string column.
        ``"findings_rows"`` — one CSV row per individual finding,
        with report-level metadata repeated.
    """
    path = Path(path)
    flatten = _STRATEGIES[strategy]
    df = flatten(results)
    df.to_csv(path, index=False)


def export_jsonl(
    results: list[dict[str, Any]],
    path: Path | str,
) -> None:
    """Write *results* as newline-delimited JSON (JSONL).

    Each result dict is written as a single JSON line.

    Parameters
    ----------
    results:
        List of extraction-result dicts.
    path:
        Destination file path.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, default=str) + "\n")


def export_parquet(
    results: list[dict[str, Any]],
    path: Path | str,
    *,
    strategy: str = "one_row",
) -> None:
    """Write *results* to an Apache Parquet file.

    Uses the same flattening strategies as :func:`export_csv`.

    Parameters
    ----------
    results:
        List of extraction-result dicts.
    path:
        Destination file path.
    strategy:
        ``"one_row"`` or ``"findings_rows"`` — see :func:`export_csv`.
    """
    path = Path(path)
    flatten = _STRATEGIES[strategy]
    df = flatten(results)
    df.to_parquet(path, index=False)
