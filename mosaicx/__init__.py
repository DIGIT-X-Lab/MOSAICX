# mosaicx/__init__.py
"""
MOSAICX — Medical Document Structuring Platform.

Public API:
    - mosaicx.extract(document_path, schema=None, mode=None, template=None)
    - mosaicx.summarize(document_paths)
    - mosaicx.generate_schema(description, example_text=None)
    - mosaicx.deidentify(text, mode="remove")
    - mosaicx.config.MosaicxConfig / get_config()
    - mosaicx.documents.load_document()
    - mosaicx.cli.cli (Click entry point)
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
    dspy.configure(lm=dspy.LM(cfg.lm, api_key=cfg.api_key))


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
# Public API wrappers
# ---------------------------------------------------------------------------


def extract(
    document_path: Union[str, Path],
    schema: str | None = None,
    mode: str | None = None,
    template: str | None = None,
) -> dict[str, Any]:
    """Extract structured data from a clinical document.

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

        return extract_with_mode(doc.text, mode)

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


def summarize(
    document_paths: list[Union[str, Path]],
) -> dict[str, Any]:
    """Summarize a collection of clinical reports into a patient timeline.

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


def generate_schema(
    description: str,
    example_text: str | None = None,
) -> dict[str, Any]:
    """Generate a Pydantic schema from a natural-language description.

    Parameters
    ----------
    description:
        Natural-language description of the document type to structure.
    example_text:
        Optional example document text for grounding.

    Returns
    -------
    dict
        Keys: ``schema_spec`` (dict) and ``compiled_model`` (the
        generated Pydantic BaseModel class).
    """
    from .pipelines.schema_gen import SchemaGenerator  # lazy (triggers dspy)

    _configure_dspy()
    generator = SchemaGenerator()
    result = generator(
        description=description,
        example_text=example_text or "",
    )
    return {
        "schema_spec": result.schema_spec.model_dump(),
        "compiled_model": result.compiled_model,
    }


def deidentify(
    text: str,
    mode: str = "remove",
) -> str:
    """De-identify clinical text by removing or replacing PHI.

    Parameters
    ----------
    text:
        The clinical text to de-identify.
    mode:
        De-identification strategy:
        - ``"remove"``       -- LLM + regex, replace PHI with [REDACTED].
        - ``"pseudonymize"`` -- LLM + regex, replace with fake values.
        - ``"dateshift"``    -- LLM + regex, shift dates consistently.
        - ``"regex"``        -- Regex-only scrubbing (no LLM needed).

    Returns
    -------
    str
        The de-identified text.
    """
    from .pipelines.deidentifier import regex_scrub_phi  # lazy, no dspy dep

    if mode == "regex":
        return regex_scrub_phi(text)

    # Full LLM + regex pipeline
    from .pipelines.deidentifier import Deidentifier  # lazy (triggers dspy)

    _configure_dspy()
    deid = Deidentifier()
    result = deid(document_text=text, mode=mode)
    return result.redacted_text


__all__ = [
    "__version__",
    "extract",
    "summarize",
    "generate_schema",
    "deidentify",
]
