"""Source loaders for the query engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel


class SourceMeta(BaseModel):
    """Metadata about a loaded source."""

    name: str
    format: str
    source_type: str  # "json", "dataframe", "document"
    size: int  # bytes
    preview: str | None = None  # first N chars or summary


def load_source(path: Path | str) -> tuple[SourceMeta, Any]:
    """Load a source file and return (metadata, data).

    Supported formats:
    - .json -> (meta, dict/list)
    - .csv -> (meta, pandas.DataFrame)
    - .parquet -> (meta, pandas.DataFrame)
    - .xlsx/.xls -> (meta, pandas.DataFrame)
    - .txt/.md -> (meta, str)
    - .pdf -> (meta, str)  # via document loader
    """
    path = Path(path)
    suffix = path.suffix.lower()
    size = path.stat().st_size

    if suffix == ".json":
        return _load_json(path, size)
    elif suffix == ".csv":
        return _load_csv(path, size)
    elif suffix == ".parquet":
        return _load_parquet(path, size)
    elif suffix in (".xlsx", ".xls"):
        return _load_excel(path, size)
    elif suffix in (".txt", ".md"):
        return _load_text(path, size)
    elif suffix == ".pdf":
        return _load_pdf(path, size)
    else:
        # Fall back to text
        return _load_text(path, size)


def _load_json(path: Path, size: int) -> tuple[SourceMeta, Any]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    preview = json.dumps(data, default=str)[:200]
    return (
        SourceMeta(
            name=path.name,
            format="json",
            source_type="json",
            size=size,
            preview=preview,
        ),
        data,
    )


def _load_csv(path: Path, size: int) -> tuple[SourceMeta, Any]:
    import pandas as pd

    df = pd.read_csv(path)
    preview = f"{len(df)} rows x {len(df.columns)} cols: {list(df.columns)}"
    return (
        SourceMeta(
            name=path.name,
            format="csv",
            source_type="dataframe",
            size=size,
            preview=preview,
        ),
        df,
    )


def _load_parquet(path: Path, size: int) -> tuple[SourceMeta, Any]:
    import pandas as pd

    df = pd.read_parquet(path)
    preview = f"{len(df)} rows x {len(df.columns)} cols: {list(df.columns)}"
    return (
        SourceMeta(
            name=path.name,
            format="parquet",
            source_type="dataframe",
            size=size,
            preview=preview,
        ),
        df,
    )


def _load_excel(path: Path, size: int) -> tuple[SourceMeta, Any]:
    import pandas as pd

    df = pd.read_excel(path)
    preview = f"{len(df)} rows x {len(df.columns)} cols: {list(df.columns)}"
    return (
        SourceMeta(
            name=path.name,
            format="excel",
            source_type="dataframe",
            size=size,
            preview=preview,
        ),
        df,
    )


def _load_text(path: Path, size: int) -> tuple[SourceMeta, str]:
    text = path.read_text(encoding="utf-8")
    preview = text[:200]
    return (
        SourceMeta(
            name=path.name,
            format=path.suffix.lstrip(".") or "txt",
            source_type="document",
            size=size,
            preview=preview,
        ),
        text,
    )


def _load_pdf(path: Path, size: int) -> tuple[SourceMeta, str]:
    from mosaicx.documents.loader import load_document

    doc = load_document(path)
    preview = doc.text[:200]
    return (
        SourceMeta(
            name=path.name,
            format="pdf",
            source_type="document",
            size=size,
            preview=preview,
        ),
        doc.text,
    )
