"""Report summarisation helpers for MOSAICX API."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from ..constants import DEFAULT_LLM_MODEL
from ..summarizer import PatientSummary, load_reports, summarize_with_llm


def summarize_reports(
    paths: Union[Sequence[Union[Path, str]], Path, str],
    *,
    patient_id: Optional[str],
    model: str = DEFAULT_LLM_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> PatientSummary:
    """Summarize one or many reports into a :class:`PatientSummary`.

    ``paths`` accepts a single ``Path``/``str`` or any sequence of them. Directories
    are scanned recursively for ``.pdf`` and ``.txt`` files before summarisation.
    ``Path`` and string inputs are both accepted for convenience.
    """

    collected_paths: List[Path] = []
    raw_sources: Iterable[Union[Path, str]]
    if isinstance(paths, (str, Path)):
        raw_sources = [paths]
    else:
        raw_sources = paths

    for src in raw_sources:
        path_obj = Path(src)
        if not path_obj.exists():
            raise FileNotFoundError(f"Report source not found: {path_obj}")
        if path_obj.is_dir():
            for candidate in path_obj.rglob("*"):
                if candidate.suffix.lower() in {".pdf", ".txt"}:
                    collected_paths.append(candidate)
        else:
            collected_paths.append(path_obj)

    docs = load_reports(collected_paths)
    if not docs:
        raise ValueError("No textual content found in the provided inputs (.pdf/.txt).")

    return summarize_with_llm(
        docs,
        patient_id=patient_id,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )


__all__ = ["summarize_reports"]
