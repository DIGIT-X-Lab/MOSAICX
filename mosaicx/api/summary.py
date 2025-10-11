"""
MOSAICX API - Report Summarisation Helpers

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Structure first. Insight follows.

Author: Lalith Kumar Shiyam Sundar, PhD
Lab: DIGIT-X Lab
Department: Department of Radiology
University: LMU University Hospital | LMU Munich

Overview:
---------
Provide a thin veneer over the summariser pipeline so applications and tests
can generate longitudinal patient narratives without invoking the CLI.  The
API accepts loose collections of paths, resolves report content, and returns a
validated ``PatientSummary`` instance.

Key Behaviours:
--------------
- Accepts single files, directories, or mixed iterables of paths, scanning
  recursively for supported formats (PDF, DOCX, PPTX, TXT, ...).
- Shares LLM configuration defaults with other modules to maintain consistent
  behaviour across extraction and summarisation features.
- Raises explicit errors for missing inputs to aid early validation in calling
  services.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from ..constants import DEFAULT_LLM_MODEL
from ..document_loader import DOC_SUFFIXES
from ..summarizer import (
    PatientSummary,
    load_reports,
    save_summary_artifacts,
    summarize_with_llm,
)


def summarize_reports(
    paths: Union[Sequence[Union[Path, str]], Path, str],
    *,
    patient_id: Optional[str],
    model: str = DEFAULT_LLM_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    artifacts: Optional[Sequence[str]] = None,
    json_path: Optional[Path] = None,
    pdf_path: Optional[Path] = None,
) -> PatientSummary:
    """Summarize one or many reports into a :class:`PatientSummary`.

    ``paths`` accepts a single ``Path``/``str`` or any sequence of them. Directories
    are scanned recursively for supported formats before summarisation. ``Path`` and
    string inputs are both accepted for convenience.

    Optional ``artifacts`` allows writing JSON and/or PDF outputs. Pass values from
    ``{"json", "pdf"}``. When omitted, no files are written.
    """

    collected_paths: List[Path] = []
    raw_sources: Iterable[Union[Path, str]]
    if isinstance(paths, (str, Path)):
        raw_sources = [paths]
    else:
        raw_sources = paths

    allowed_suffixes = set(DOC_SUFFIXES.keys())

    for src in raw_sources:
        path_obj = Path(src)
        if not path_obj.exists():
            raise FileNotFoundError(f"Report source not found: {path_obj}")
        if path_obj.is_dir():
            for candidate in path_obj.rglob("*"):
                if candidate.suffix.lower() in allowed_suffixes:
                    collected_paths.append(candidate)
        else:
            if path_obj.suffix.lower() not in allowed_suffixes:
                raise ValueError(
                    f"Unsupported report format: {path_obj.suffix or '<none>'}. "
                    f"Supported extensions: {', '.join(sorted(allowed_suffixes))}."
                )
            collected_paths.append(path_obj)

    docs = load_reports(collected_paths)
    if not docs:
        raise ValueError(
            f"No textual content found in the provided inputs (supported: {', '.join(sorted(allowed_suffixes))})."
        )

    summary = summarize_with_llm(
        docs,
        patient_id=patient_id,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )

    if artifacts:
        save_summary_artifacts(
            summary,
            artifacts=artifacts,
            json_path=json_path,
            pdf_path=pdf_path,
            patient_id=patient_id,
            emit_messages=False,
        )

    return summary


__all__ = ["summarize_reports"]
