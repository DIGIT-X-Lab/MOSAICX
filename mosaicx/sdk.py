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

    result   = extract("Patient presents with chest pain...", template="chest_ct")
    clean    = deidentify("John Doe, SSN 123-45-6789")
    summary  = summarize(["Report 1 text...", "Report 2 text..."])
    template = generate_schema("echo report with LVEF and valve grades")

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


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    """Convert PipelineMetrics to a plain dict."""
    return {
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


def _compute_completeness_dict(model_instance: Any, text: str) -> dict[str, Any]:
    """Compute completeness scoring and return as a plain dict."""
    from dataclasses import asdict

    from .evaluation.completeness import compute_report_completeness

    comp = compute_report_completeness(
        model_instance, text, type(model_instance)
    )
    return asdict(comp)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def extract(
    text: str,
    *,
    template: str | Path | None = None,
    mode: str = "auto",
    score: bool = False,
    optimized: str | Path | None = None,
) -> dict[str, Any]:
    """Extract structured data from document text.

    Parameters
    ----------
    text:
        Document text to extract from.
    template:
        Template name (built-in or user-created), or path to a YAML
        template file.  Resolved via :func:`mosaicx.report.resolve_template`.
    mode:
        Extraction mode. ``"auto"`` lets the LLM infer the structure.
        ``"radiology"`` and ``"pathology"`` run specialised multi-step
        pipelines.  Ignored when *template* is provided.
    score:
        If ``True``, compute completeness scoring against the template
        and include it in the output under ``"completeness"``.
    optimized:
        Path to an optimized DSPy program to load. Only applicable for
        ``mode="auto"`` or template-based extraction.

    Returns
    -------
    dict
        Extracted data.  Structure depends on mode / template.  When
        *score* is ``True``, includes a ``"completeness"`` key.

    Raises
    ------
    ValueError
        If conflicting parameters are provided, or if the template/mode
        is unknown.
    """
    _ensure_configured()

    # --- Template-based extraction ---
    if template is not None:
        from .report import detect_mode, resolve_template

        template_str = str(template)
        template_model, tpl_name = resolve_template(template=template_str)

        effective_mode = detect_mode(tpl_name)

        if effective_mode is not None and template_model is None:
            # Built-in template with mode pipeline
            import mosaicx.pipelines.radiology  # noqa: F401
            import mosaicx.pipelines.pathology  # noqa: F401

            if score:
                from .pipelines.extraction import extract_with_mode_raw
                from .report import _find_primary_model

                output_data, metrics, raw_pred = extract_with_mode_raw(
                    text, effective_mode
                )
                model_instance = _find_primary_model(raw_pred)
                if model_instance is not None:
                    output_data["completeness"] = _compute_completeness_dict(
                        model_instance, text
                    )
            else:
                from .pipelines.extraction import extract_with_mode

                output_data, metrics = extract_with_mode(text, effective_mode)
            if metrics is not None:
                output_data["_metrics"] = _metrics_to_dict(metrics)
            return output_data
        elif template_model is not None:
            from .pipelines.extraction import DocumentExtractor

            extractor = DocumentExtractor(output_schema=template_model)
            if optimized is not None:
                from .evaluation.optimize import load_optimized

                extractor = load_optimized(DocumentExtractor, Path(optimized))
            result = extractor(document_text=text)
            output: dict[str, Any] = {}
            if hasattr(result, "extracted"):
                val = result.extracted
                if hasattr(val, "model_dump"):
                    output["extracted"] = val.model_dump()
                    if score:
                        output["completeness"] = _compute_completeness_dict(
                            val, text
                        )
                else:
                    output["extracted"] = val
            return output
        else:
            raise ValueError(
                f"Template {template!r} resolved but produced no extraction template."
            )

    # --- Mode-based extraction (radiology, pathology, ...) ---
    if mode not in ("auto",):
        # Trigger lazy loading of mode pipeline modules
        import mosaicx.pipelines.radiology  # noqa: F401
        import mosaicx.pipelines.pathology  # noqa: F401

        if score:
            from .pipelines.extraction import extract_with_mode_raw
            from .report import _find_primary_model

            output_data, metrics, raw_pred = extract_with_mode_raw(text, mode)
            model_instance = _find_primary_model(raw_pred)
            if model_instance is not None:
                output_data["completeness"] = _compute_completeness_dict(
                    model_instance, text
                )
        else:
            from .pipelines.extraction import extract_with_mode

            output_data, metrics = extract_with_mode(text, mode)
        if metrics is not None:
            output_data["_metrics"] = _metrics_to_dict(metrics)
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
# report
# ---------------------------------------------------------------------------


def report(
    text: str,
    *,
    template: str | Path | None = None,
    schema_name: str | None = None,
    describe: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Extract structured data and score completeness against a template.

    .. deprecated::
        Use :func:`extract` with ``score=True`` instead.

    Parameters
    ----------
    text:
        Document text to extract from.
    template:
        Built-in RDES template name (e.g. ``"chest_ct"``) or path to
        a YAML template file, or a saved schema name.
    schema_name:
        *Deprecated* -- use *template* instead.
    describe:
        Plain-English description to AI-generate a template on the fly.
    mode:
        Explicit pipeline mode override.  If ``None``, auto-detected
        from the template.

    Returns
    -------
    dict
        Keys: ``"extracted"`` (dict), ``"completeness"`` (dict with
        ``overall``, ``required_coverage``, ``optional_coverage``,
        ``missing_required``, etc.), ``"template_name"`` (str | None),
        ``"mode_used"`` (str | None), ``"metrics"`` (dict | None).

    Raises
    ------
    ValueError
        If more than one of *template*, *schema_name*, *describe* is
        provided, or if the template/schema cannot be resolved.
    """
    import warnings

    warnings.warn(
        "sdk.report() is deprecated. Use sdk.extract(score=True) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Map deprecated params to template
    effective_template = template
    if schema_name is not None:
        if effective_template is not None:
            raise ValueError(
                "Provide at most one of template or schema_name."
            )
        effective_template = schema_name
    if describe is not None:
        raise ValueError(
            "The 'describe' parameter is no longer supported in sdk.report(). "
            "Use 'mosaicx template create --describe' to create a template first, "
            "then pass the template name."
        )

    _ensure_configured()

    from .report import resolve_template, run_report

    template_str = str(effective_template) if effective_template is not None else None
    template_model, tpl_name = resolve_template(template=template_str)

    result = run_report(
        document_text=text,
        template_model=template_model,
        template_name=tpl_name,
        mode=mode,
    )

    from dataclasses import asdict

    return asdict(result)


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
    template: str | None = None,
) -> list[dict[str, Any]]:
    """Extract structured data from multiple documents.

    A convenience wrapper that calls :func:`extract` for each text in
    *texts*.  All documents use the same mode / template.

    Parameters
    ----------
    texts:
        List of document texts.
    mode:
        Extraction mode (see :func:`extract`).
    template:
        Template name or YAML file path (see :func:`extract`).

    Returns
    -------
    list[dict]
        One result dict per input text.  Failed extractions produce
        a dict with an ``"error"`` key.
    """
    results: list[dict[str, Any]] = []
    for text in texts:
        try:
            result = extract(text, mode=mode, template=template)
        except Exception as exc:
            logger.warning("batch_extract failed for one document: %s", exc)
            result = {"error": str(exc)}
        results.append(result)
    return results
