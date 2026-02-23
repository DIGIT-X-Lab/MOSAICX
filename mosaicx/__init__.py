# mosaicx/__init__.py
"""
MOSAICX -- Medical Document Structuring Platform.

Three core SDK functions:
    - mosaicx.extract(text=..., documents=..., template=..., mode=..., score=..., workers=...)
    - mosaicx.deidentify(text=..., documents=..., mode=..., workers=...)
    - mosaicx.summarize(reports=..., documents=..., patient_id=..., optimized=...)

Utilities:
    - mosaicx.generate_schema(description, name=None, example_text=None, save=False)
    - mosaicx.list_schemas()
    - mosaicx.list_modes()
    - mosaicx.list_templates()
    - mosaicx.evaluate(pipeline, testset_path, optimized=None)
    - mosaicx.health()

Configuration:
    - mosaicx.config.MosaicxConfig / get_config()
    - mosaicx.documents.load_document()
    - mosaicx.cli.cli (Click entry point)

Legacy file-based API (still works, prefer SDK functions):
    - mosaicx.extract_file(document_path, schema=None, mode=None, template=None)
    - mosaicx.summarize_files(document_paths)
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Any, Union

try:
    __version__ = _pkg_version("mosaicx")
except PackageNotFoundError:
    __version__ = "2.0.0a1"


# ---------------------------------------------------------------------------
# DSPy configuration helper (shared by LLM-dependent wrappers)
# ---------------------------------------------------------------------------

def _configure_dspy() -> None:
    """Configure DSPy with the LM from MosaicxConfig.

    Raises ``RuntimeError`` if DSPy cannot be imported or the API key
    is missing.  Imported lazily so this module stays importable without
    DSPy installed.
    """
    from .runtime_env import ensure_runtime_env

    ensure_runtime_env()

    from .config import get_config  # noqa: delay until called

    cfg = get_config()
    if not cfg.api_key:
        raise RuntimeError(
            "No API key configured. Set MOSAICX_API_KEY or add api_key "
            "to your config."
        )
    try:
        import dspy
    except ImportError:
        raise RuntimeError(
            "DSPy is required for this function. Install with: pip install dspy"
        )
    from .metrics import TokenTracker, make_harmony_lm, set_tracker

    lm = make_harmony_lm(cfg.lm, api_key=cfg.api_key, api_base=cfg.api_base, temperature=cfg.lm_temperature)
    adapter = None
    try:
        adapter = dspy.JSONAdapter()
    except Exception:
        adapter = None
    try:
        if adapter is not None:
            dspy.configure(lm=lm, adapter=adapter)
        else:
            dspy.configure(lm=lm)
    except TypeError:
        dspy.configure(lm=lm)
        if adapter is not None:
            try:
                dspy.settings.adapter = adapter
            except Exception:
                pass

    # Install token usage tracker

    tracker = TokenTracker()
    set_tracker(tracker)
    dspy.settings.usage_tracker = tracker
    dspy.settings.track_usage = True


# ---------------------------------------------------------------------------
# Document loading helper (OCR config wiring)
# ---------------------------------------------------------------------------


def _load_doc_with_config(path: Path) -> "LoadedDocument":
    """Load a document using OCR settings from config."""
    from .documents.loader import load_document
    from .config import get_config

    cfg = get_config()
    return load_document(
        path,
        ocr_engine=cfg.ocr_engine,
        force_ocr=cfg.force_ocr,
        ocr_langs=cfg.ocr_langs,
        chandra_backend=cfg.chandra_backend if cfg.chandra_backend != "auto" else None,
        quality_threshold=cfg.quality_threshold,
        page_timeout=cfg.ocr_page_timeout,
    )


# ---------------------------------------------------------------------------
# Legacy file-based API wrappers (load document from disk, then process)
# ---------------------------------------------------------------------------


def extract_file(
    document_path: Union[str, Path],
    schema: str | None = None,
    mode: str | None = None,
    template: str | None = None,
) -> dict[str, Any]:
    """Extract structured data from a clinical document file.

    .. note::
        Legacy wrapper. Prefer ``mosaicx.extract(documents=path)`` instead.

    Parameters
    ----------
    document_path:
        Path to the document file (PDF, DOCX, TXT, etc.).
    schema:
        Name of a saved schema from ~/.mosaicx/schemas/.
        Extracts into the specified schema shape.
    mode:
        Extraction mode name (e.g., "radiology", "pathology").
        Runs a specialized multi-step pipeline.
    template:
        Path to a YAML template file. Compiles to a Pydantic model
        and extracts into that shape.

    Returns
    -------
    dict
        Extraction results. Key is typically ``extracted`` containing
        the structured data matching the schema/template.

    Notes
    -----
    If none of schema/mode/template are provided, the LLM auto-infers
    a schema from the document content.
    schema, mode, and template are mutually exclusive.
    """
    # Validate mutual exclusivity
    provided = sum(x is not None for x in (schema, mode, template))
    if provided > 1:
        raise ValueError(
            "schema, mode, and template are mutually exclusive. "
            "Provide at most one."
        )

    from .config import get_config

    cfg = get_config()
    doc = _load_doc_with_config(Path(document_path))
    _configure_dspy()

    if schema is not None:
        # Schema mode: load saved schema and extract into it
        from .pipelines.extraction import extract_with_schema

        extracted = extract_with_schema(doc.text, schema, cfg.schema_dir)
        return {"extracted": extracted}

    if mode is not None:
        # Mode: run a specialized multi-step pipeline
        from .pipelines.extraction import extract_with_mode

        output_data, _metrics = extract_with_mode(doc.text, mode)
        return output_data

    if template is not None:
        # YAML template: compile and extract
        from .schemas.template_compiler import compile_template_file
        from .pipelines.extraction import DocumentExtractor

        tpl_path = Path(template)
        output_schema = compile_template_file(tpl_path)
        extractor = DocumentExtractor(output_schema=output_schema)
        result = extractor(document_text=doc.text)
        output: dict[str, Any] = {}
        if hasattr(result, "extracted"):
            val = result.extracted
            output["extracted"] = (
                val.model_dump() if hasattr(val, "model_dump") else val
            )
        return output

    # Auto mode: LLM infers schema from document content
    from .pipelines.extraction import DocumentExtractor

    extractor = DocumentExtractor()
    result = extractor(document_text=doc.text)
    output: dict[str, Any] = {}
    if hasattr(result, "extracted"):
        val = result.extracted
        output["extracted"] = (
            val.model_dump() if hasattr(val, "model_dump") else val
        )
    if hasattr(result, "inferred_schema"):
        output["inferred_schema"] = result.inferred_schema.model_dump()
    return output


def summarize_files(
    document_paths: list[Union[str, Path]],
) -> dict[str, Any]:
    """Summarize a collection of clinical report files into a patient timeline.

    .. note::
        Legacy wrapper. Prefer ``mosaicx.summarize(documents=[...])`` instead.

    Parameters
    ----------
    document_paths:
        List of paths to the report files.

    Returns
    -------
    dict
        Keys: ``events`` (list of timeline event dicts) and
        ``narrative`` (str).
    """
    from .pipelines.summarizer import ReportSummarizer  # lazy (triggers dspy)

    report_texts: list[str] = []
    for p in document_paths:
        doc = _load_doc_with_config(Path(p))
        report_texts.append(doc.text)

    if not report_texts:
        raise ValueError("No reports provided to summarize.")

    _configure_dspy()
    summarizer_mod = ReportSummarizer()
    result = summarizer_mod(reports=report_texts)

    return {
        "events": [e.model_dump() for e in result.events],
        "narrative": result.narrative,
    }


# ---------------------------------------------------------------------------
# SDK convenience imports (the primary public interface)
# ---------------------------------------------------------------------------
# These provide the unified SDK API so that:
#     from mosaicx import extract, deidentify, summarize, generate_schema
# uses the SDK functions directly.

from mosaicx.sdk import (  # noqa: E402
    deidentify,
    evaluate,
    extract,
    generate_schema,
    health,
    list_modes,
    list_schemas,
    list_templates,
    query,
    summarize,
    verify,
    verify_batch,
)


__all__ = [
    "__version__",
    # SDK core functions
    "extract",
    "deidentify",
    "summarize",
    # SDK utilities
    "generate_schema",
    "list_schemas",
    "list_modes",
    "list_templates",
    "evaluate",
    "health",
    "verify",
    "verify_batch",
    "query",
    # Legacy file-based API
    "extract_file",
    "summarize_files",
]
