# mosaicx/__init__.py
"""
MOSAICX — Medical Document Structuring Platform.

Public API:
    - mosaicx.extract(document_path, template="auto")
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
    template: str = "auto",
) -> dict[str, Any]:
    """Extract structured data from a clinical document.

    Parameters
    ----------
    document_path:
        Path to the document file (PDF, DOCX, TXT, etc.).
    template:
        Template name, YAML file path, or ``"auto"`` for the default
        3-step demographics/findings/diagnoses extraction.

    Returns
    -------
    dict
        Extraction results.  In default (auto) mode the keys are
        ``demographics``, ``findings``, and ``diagnoses``.  In custom
        template mode the key is ``extracted``.
    """
    from .pipelines.extraction import DocumentExtractor  # lazy (triggers dspy)

    doc = _load_doc_with_config(Path(document_path))

    # Resolve template to an output schema
    output_schema = None
    if template != "auto":
        tpl_path = Path(template)
        if tpl_path.exists() and tpl_path.suffix in (".yaml", ".yml"):
            from .schemas.template_compiler import compile_template_file

            output_schema = compile_template_file(tpl_path)
        else:
            # Registry lookup is informational only; extraction still uses
            # the default pipeline (registry templates don't produce a
            # Pydantic model directly).
            from .schemas.radreport.registry import get_template

            tpl_info = get_template(template)
            if tpl_info is None:
                raise ValueError(
                    f"Template not found: {template!r}. "
                    "Provide a .yaml file path or a registered template name."
                )

    _configure_dspy()
    extractor = DocumentExtractor(output_schema=output_schema)
    result = extractor(document_text=doc.text)

    # Convert DSPy Prediction to a plain dict
    output: dict[str, Any] = {}
    if hasattr(result, "extracted"):
        val = result.extracted
        output["extracted"] = (
            val.model_dump() if hasattr(val, "model_dump") else val
        )
    else:
        if hasattr(result, "demographics"):
            output["demographics"] = result.demographics.model_dump()
        if hasattr(result, "findings"):
            output["findings"] = [f.model_dump() for f in result.findings]
        if hasattr(result, "diagnoses"):
            output["diagnoses"] = [d.model_dump() for d in result.diagnoses]
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
