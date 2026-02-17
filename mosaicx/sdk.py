# mosaicx/sdk.py
"""
MOSAICX Python SDK -- programmatic access without the CLI.

This module provides high-level functions that wrap the internal DSPy
pipelines. Every function:
    - Accepts plain Python types (str, dict, Path).
    - Returns plain Python dicts (not DSPy Predictions or Pydantic models).
    - Configures DSPy automatically on first use.
    - Supports loading optimized DSPy programs via the ``optimized`` parameter.

Quick start::

    from mosaicx.sdk import extract, deidentify, summarize, generate_schema

    result  = extract("Patient presents with chest pain...", mode="radiology")
    clean   = deidentify("John Doe, SSN 123-45-6789")
    summary = summarize(["Report 1 text...", "Report 2 text..."])
    schema  = generate_schema("echo report with LVEF and valve grades")

All heavy dependencies (DSPy, pipeline modules) are imported lazily so
this module stays importable even in environments where DSPy is not
installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_configured: bool = False


# ---------------------------------------------------------------------------
# DSPy configuration
# ---------------------------------------------------------------------------


def _ensure_configured() -> None:
    """Configure DSPy if not already done. Called automatically by all SDK functions.

    Uses :func:`mosaicx.config.get_config` to read settings and configures
    the DSPy LM exactly the same way the CLI does.

    Raises
    ------
    RuntimeError
        If DSPy is not installed or the API key is missing.
    """
    global _configured
    if _configured:
        return

    from .config import get_config

    cfg = get_config()
    if not cfg.api_key:
        raise RuntimeError(
            "No API key configured. Set MOSAICX_API_KEY or add api_key "
            "to your config."
        )

    try:
        import dspy  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "DSPy is required for SDK functions. Install with: pip install dspy"
        )

    from .metrics import TokenTracker, make_harmony_lm, set_tracker

    lm = make_harmony_lm(cfg.lm, api_key=cfg.api_key, api_base=cfg.api_base, temperature=cfg.lm_temperature)
    dspy.configure(lm=lm)

    tracker = TokenTracker()
    set_tracker(tracker)
    dspy.settings.usage_tracker = tracker
    dspy.settings.track_usage = True

    _configured = True
    logger.info("DSPy configured with model %s", cfg.lm)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prediction_to_dict(prediction: Any) -> dict[str, Any]:
    """Convert a DSPy Prediction (or similar) to a plain dict.

    Pydantic models nested inside the prediction are serialised via
    ``model_dump()``.  Lists of Pydantic models are handled recursively.
    """
    output: dict[str, Any] = {}
    for key in prediction.keys():
        val = getattr(prediction, key)
        if hasattr(val, "model_dump"):
            output[key] = val.model_dump()
        elif isinstance(val, list):
            output[key] = [
                v.model_dump() if hasattr(v, "model_dump") else v for v in val
            ]
        else:
            output[key] = val
    return output


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def extract(
    text: str,
    *,
    mode: str = "auto",
    schema_name: str | None = None,
    optimized: str | Path | None = None,
) -> dict[str, Any]:
    """Extract structured data from document text.

    Parameters
    ----------
    text:
        Document text to extract from.
    mode:
        Extraction mode. ``"auto"`` lets the LLM infer the schema.
        ``"radiology"`` and ``"pathology"`` run specialised multi-step
        pipelines.
    schema_name:
        Name of a saved schema (from ``~/.mosaicx/schemas/``) to extract
        into.  Mutually exclusive with ``mode`` other than ``"auto"``.
    optimized:
        Path to an optimized DSPy program to load. Only applicable for
        ``mode="auto"`` or schema-based extraction.

    Returns
    -------
    dict
        Extracted data.  Structure depends on mode / schema.

    Raises
    ------
    ValueError
        If ``mode`` and ``schema_name`` are both specified (and mode is
        not ``"auto"``), or if the mode is unknown.
    FileNotFoundError
        If ``schema_name`` refers to a schema that does not exist.
    """
    if schema_name is not None and mode not in ("auto",):
        raise ValueError(
            "schema_name and mode are mutually exclusive. "
            "Use schema_name with mode='auto' (the default), or use mode alone."
        )

    _ensure_configured()

    # --- Schema-based extraction ---
    if schema_name is not None:
        from .config import get_config
        from .pipelines.extraction import extract_with_schema

        cfg = get_config()
        extracted = extract_with_schema(text, schema_name, cfg.schema_dir)
        return {"extracted": extracted}

    # --- Mode-based extraction (radiology, pathology, ...) ---
    if mode not in ("auto",):
        # Trigger lazy loading of mode pipeline modules
        import mosaicx.pipelines.radiology  # noqa: F401
        import mosaicx.pipelines.pathology  # noqa: F401
        from .pipelines.extraction import extract_with_mode

        output_data, metrics = extract_with_mode(text, mode)
        if metrics is not None:
            output_data["_metrics"] = {
                "total_duration_s": metrics.total_duration_s,
                "total_tokens": metrics.total_tokens,
                "steps": [
                    {
                        "name": s.name,
                        "duration_s": s.duration_s,
                        "input_tokens": s.input_tokens,
                        "output_tokens": s.output_tokens,
                    }
                    for s in metrics.steps
                ],
            }
        return output_data

    # --- Auto extraction (LLM infers schema) ---
    from .pipelines.extraction import DocumentExtractor

    extractor = DocumentExtractor()

    if optimized is not None:
        from .evaluation.optimize import load_optimized

        extractor = load_optimized(DocumentExtractor, Path(optimized))

    result = extractor(document_text=text)
    output: dict[str, Any] = {}

    if hasattr(result, "extracted"):
        val = result.extracted
        output["extracted"] = val.model_dump() if hasattr(val, "model_dump") else val
    if hasattr(result, "inferred_schema"):
        output["inferred_schema"] = result.inferred_schema.model_dump()

    return output


# ---------------------------------------------------------------------------
# deidentify
# ---------------------------------------------------------------------------


def deidentify(
    text: str,
    *,
    mode: str = "remove",
) -> dict[str, Any]:
    """Remove PHI from text.

    Parameters
    ----------
    text:
        Text containing Protected Health Information.
    mode:
        De-identification strategy:
        - ``"remove"``       -- Replace PHI with ``[REDACTED]``.
        - ``"pseudonymize"`` -- Replace PHI with realistic fake values.
        - ``"dateshift"``    -- Shift dates by a consistent random offset.
        - ``"regex"``        -- Regex-only scrubbing (no LLM needed).

    Returns
    -------
    dict
        Keys: ``"redacted_text"`` (str).

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported values.
    """
    valid_modes = {"remove", "pseudonymize", "dateshift", "regex"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown deidentify mode: {mode!r}. "
            f"Choose from: {sorted(valid_modes)}"
        )

    from .pipelines.deidentifier import regex_scrub_phi

    if mode == "regex":
        return {"redacted_text": regex_scrub_phi(text)}

    _ensure_configured()

    from .pipelines.deidentifier import Deidentifier

    deid = Deidentifier()
    result = deid(document_text=text, mode=mode)
    return {"redacted_text": result.redacted_text}


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def summarize(
    reports: list[str],
    *,
    patient_id: str = "unknown",
    optimized: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize multiple reports into a patient timeline.

    Parameters
    ----------
    reports:
        List of report texts.
    patient_id:
        Patient identifier for the summary.
    optimized:
        Path to an optimized DSPy program to load.

    Returns
    -------
    dict
        Keys: ``"narrative"`` (str), ``"events"`` (list of event dicts).

    Raises
    ------
    ValueError
        If *reports* is empty.
    """
    if not reports:
        raise ValueError("No reports provided to summarize.")

    _ensure_configured()

    from .pipelines.summarizer import ReportSummarizer

    summarizer = ReportSummarizer()

    if optimized is not None:
        from .evaluation.optimize import load_optimized

        summarizer = load_optimized(ReportSummarizer, Path(optimized))

    result = summarizer(reports=reports, patient_id=patient_id)

    return {
        "events": [e.model_dump() for e in result.events],
        "narrative": result.narrative,
    }


# ---------------------------------------------------------------------------
# generate_schema
# ---------------------------------------------------------------------------


def generate_schema(
    description: str,
    *,
    name: str | None = None,
    example_text: str | None = None,
    save: bool = False,
) -> dict[str, Any]:
    """Generate a Pydantic schema from a plain-English description.

    Parameters
    ----------
    description:
        Natural language description of desired fields.
    name:
        Optional schema name.  If omitted the LLM will choose one.
    example_text:
        Optional example document text to guide schema generation.
    save:
        If ``True``, persist the schema to ``~/.mosaicx/schemas/``.

    Returns
    -------
    dict
        Keys: ``"name"`` (str), ``"fields"`` (list of field dicts),
        ``"json_schema"`` (dict -- the JSON Schema representation).

    Raises
    ------
    RuntimeError
        If DSPy is not configured or not installed.
    """
    _ensure_configured()

    from .pipelines.schema_gen import SchemaGenerator, save_schema

    generator = SchemaGenerator()
    result = generator(
        description=description,
        example_text=example_text or "",
    )

    spec = result.schema_spec

    # If the caller provided a name, override the LLM-chosen class_name
    if name is not None:
        spec = spec.model_copy(update={"class_name": name})

    compiled_model = result.compiled_model

    output: dict[str, Any] = {
        "name": spec.class_name,
        "fields": [f.model_dump() for f in spec.fields],
        "json_schema": compiled_model.model_json_schema(),
    }

    if save:
        from .config import get_config

        cfg = get_config()
        saved_path = save_schema(spec, schema_dir=cfg.schema_dir)
        output["saved_to"] = str(saved_path)
        logger.info("Schema saved to %s", saved_path)

    return output


# ---------------------------------------------------------------------------
# list_schemas
# ---------------------------------------------------------------------------


def list_schemas() -> list[str]:
    """List names of all saved schemas.

    Returns
    -------
    list[str]
        Schema names (alphabetically sorted).  Empty list if the schema
        directory does not exist or contains no schemas.
    """
    from .config import get_config
    from .pipelines.schema_gen import list_schemas as _list_schemas

    cfg = get_config()
    specs = _list_schemas(cfg.schema_dir)
    return [s.class_name for s in specs]


# ---------------------------------------------------------------------------
# list_modes
# ---------------------------------------------------------------------------


def list_modes() -> list[dict[str, str]]:
    """List available extraction modes with descriptions.

    Returns
    -------
    list[dict]
        Each dict has keys ``"name"`` and ``"description"``.
    """
    from .pipelines.modes import list_modes as _list_modes

    return [
        {"name": name, "description": desc}
        for name, desc in _list_modes()
    ]


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def evaluate(
    pipeline: str,
    testset_path: str | Path,
    *,
    optimized: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a pipeline against a labeled test set.

    Parameters
    ----------
    pipeline:
        Pipeline name (e.g., ``"radiology"``, ``"pathology"``,
        ``"extract"``, ``"summarize"``, ``"deidentify"``, ``"schema"``).
    testset_path:
        Path to a ``.jsonl`` file with labeled examples.
    optimized:
        Path to an optimized DSPy program.  If ``None``, the baseline
        (unoptimized) program is evaluated.

    Returns
    -------
    dict
        Keys: ``"mean"``, ``"median"``, ``"std"`` (or ``None`` if fewer
        than 2 examples), ``"min"``, ``"max"``, ``"count"``,
        ``"scores"`` (list[float]).

    Raises
    ------
    ValueError
        If ``pipeline`` is not recognised.
    FileNotFoundError
        If ``testset_path`` does not exist.
    """
    import statistics

    _ensure_configured()

    from .evaluation.dataset import load_jsonl
    from .evaluation.metrics import get_metric
    from .evaluation.optimize import get_pipeline_class, load_optimized

    testset_path = Path(testset_path)

    pipeline_cls = get_pipeline_class(pipeline)
    metric = get_metric(pipeline)
    test_examples = load_jsonl(testset_path, pipeline)

    if optimized is not None:
        module = load_optimized(pipeline_cls, Path(optimized))
    else:
        module = pipeline_cls()

    scores: list[float] = []
    for example in test_examples:
        try:
            prediction = module(**dict(example.inputs()))
            score = metric(example, prediction)
        except Exception as exc:
            logger.warning("Example failed: %s", exc)
            score = 0.0
        scores.append(score)

    result: dict[str, Any] = {
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "std": statistics.stdev(scores) if len(scores) >= 2 else None,
        "min": min(scores),
        "max": max(scores),
        "count": len(scores),
        "scores": scores,
    }

    return result


# ---------------------------------------------------------------------------
# batch_extract
# ---------------------------------------------------------------------------


def batch_extract(
    texts: list[str],
    *,
    mode: str = "auto",
    schema_name: str | None = None,
) -> list[dict[str, Any]]:
    """Extract structured data from multiple documents.

    A convenience wrapper that calls :func:`extract` for each text in
    *texts*.  All documents use the same mode / schema.

    Parameters
    ----------
    texts:
        List of document texts.
    mode:
        Extraction mode (see :func:`extract`).
    schema_name:
        Schema name (see :func:`extract`).

    Returns
    -------
    list[dict]
        One result dict per input text.  Failed extractions produce
        a dict with an ``"error"`` key.
    """
    results: list[dict[str, Any]] = []
    for text in texts:
        try:
            result = extract(text, mode=mode, schema_name=schema_name)
        except Exception as exc:
            logger.warning("batch_extract failed for one document: %s", exc)
            result = {"error": str(exc)}
        results.append(result)
    return results
