# mosaicx/report.py
"""Structured reporting orchestrator.

Wires together template resolution, extraction pipelines, and completeness
scoring into a single ``run_report()`` call.  Used by both the CLI
``mosaicx report`` command and the SDK ``mosaicx.sdk.report()`` function.

Key functions:
    - resolve_template: Unified template/schema/description resolution.
    - detect_mode: Auto-detect pipeline mode from a template name.
    - run_report: End-to-end structured report extraction + scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _find_primary_model(prediction: Any) -> BaseModel | None:
    """Find the first Pydantic BaseModel instance in a dspy.Prediction.

    Mode pipelines return predictions containing multiple values — some are
    plain strings, some are Pydantic models, some are lists of models.
    This helper returns the first *single* ``BaseModel`` instance, which is
    the natural "completeness target" (e.g. ``ReportSections`` for
    radiology).
    """
    for key in prediction.keys():
        val = getattr(prediction, key, None)
        if isinstance(val, BaseModel):
            return val
    return None


@dataclass
class ReportResult:
    """Container for structured report output."""

    extracted: dict[str, Any]
    completeness: dict[str, Any]
    template_name: str | None = None
    mode_used: str | None = None
    metrics: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


def resolve_template(
    template: str | None = None,
    schema_dir: Path | None = None,
) -> tuple[type[BaseModel] | None, str | None]:
    """Resolve a template specification into a Pydantic model class.

    Resolution order for *template*:

    1. File path (``.yaml`` / ``.yml`` suffix + exists on disk)
    2. User templates dir (``~/.mosaicx/templates/<name>.yaml``)
    3. Built-in YAML template (package ``templates/`` dir)
    4. Legacy saved schema (``~/.mosaicx/schemas/<name>.json``)
    5. Error with suggestions

    Parameters
    ----------
    template:
        A built-in template name (e.g. ``"chest_ct"``), a filesystem
        path to a YAML file, or the name of a saved schema.
    schema_dir:
        Directory to search for saved schemas.  Defaults to config.

    Returns
    -------
    tuple[type[BaseModel] | None, str | None]
        ``(model_class, template_name)``.  ``model_class`` is ``None``
        only when no template source is provided.
    """
    if template is None:
        return None, None

    # 1. File path (.yaml/.yml suffix + exists on disk)
    path = Path(template)
    if path.suffix.lower() in (".yaml", ".yml") and path.exists():
        from .schemas.template_compiler import compile_template_file

        model = compile_template_file(path)
        return model, path.stem

    # 2. User templates dir (~/.mosaicx/templates/<name>.yaml)
    user_tpl_path = _find_user_template_yaml(template)
    if user_tpl_path is not None:
        from .schemas.template_compiler import compile_template_file

        model = compile_template_file(user_tpl_path)
        return model, template

    # 3. Built-in YAML template
    builtin_path = _find_builtin_template_yaml(template)
    if builtin_path is not None:
        from .schemas.template_compiler import compile_template_file

        return compile_template_file(builtin_path), template

    # 4. Legacy saved schema (~/.mosaicx/schemas/<name>.json)
    saved_model = _try_load_saved_schema(template, schema_dir)
    if saved_model is not None:
        return saved_model, template

    raise ValueError(
        f"Template {template!r} not found. It is not a YAML file, "
        f"a built-in template, or a saved schema. "
        f"Use 'mosaicx template list' to see available templates."
    )


def _try_load_saved_schema(
    name: str, schema_dir: Path | None = None
) -> type[BaseModel] | None:
    """Attempt to load a saved schema by name. Returns None if not found."""
    try:
        from .pipelines.schema_gen import compile_schema, load_schema

        if schema_dir is None:
            from .config import get_config

            schema_dir = get_config().schema_dir
        spec = load_schema(name, schema_dir)
        return compile_schema(spec)
    except (FileNotFoundError, Exception):
        return None


def _find_builtin_template_yaml(name: str) -> Path | None:
    """Look for a built-in YAML template file in the templates directory."""
    templates_dir = Path(__file__).parent / "schemas" / "radreport" / "templates"
    if not templates_dir.is_dir():
        return None
    for suffix in (".yaml", ".yml"):
        path = templates_dir / f"{name}{suffix}"
        if path.exists():
            return path
    return None


def _find_user_template_yaml(name: str) -> Path | None:
    """Look for a user-created YAML template in ~/.mosaicx/templates/."""
    from .config import get_config

    templates_dir = get_config().templates_dir
    if not templates_dir.is_dir():
        return None
    for suffix in (".yaml", ".yml"):
        path = templates_dir / f"{name}{suffix}"
        if path.exists():
            return path
    return None


def _read_mode_from_yaml(path: Path) -> str | None:
    """Read the ``mode`` field from a YAML template file without full compilation."""
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("mode")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


def detect_mode(template_name: str | None) -> str | None:
    """Auto-detect pipeline mode from a template name or YAML file.

    Checks in order:

    1. User template YAML (``~/.mosaicx/templates/<name>.yaml``) — reads ``mode`` field
    2. Built-in YAML template (package ``templates/`` dir) — reads ``mode`` field

    Returns ``None`` if the template is not found or has no mode.
    """
    if template_name is None:
        return None

    # 1. User template YAML
    user_path = _find_user_template_yaml(template_name)
    if user_path is not None:
        mode = _read_mode_from_yaml(user_path)
        if mode is not None:
            return mode

    # 2. Built-in YAML template
    builtin_path = _find_builtin_template_yaml(template_name)
    if builtin_path is not None:
        mode = _read_mode_from_yaml(builtin_path)
        if mode is not None:
            return mode

    return None


# ---------------------------------------------------------------------------
# Report orchestrator
# ---------------------------------------------------------------------------


def run_report(
    document_text: str,
    template_model: type[BaseModel] | None = None,
    template_name: str | None = None,
    mode: str | None = None,
) -> ReportResult:
    """Run structured report extraction with completeness scoring.

    Parameters
    ----------
    document_text:
        Full text of the clinical document.
    template_model:
        Optional Pydantic model class for schema-based extraction.
    template_name:
        Template name (used for mode auto-detection and metadata).
    mode:
        Explicit pipeline mode override.  If ``None``, auto-detected
        from *template_name*.

    Returns
    -------
    ReportResult
        Extracted data, completeness scores, and metadata.
    """
    from .evaluation.completeness import compute_report_completeness

    # Determine mode
    effective_mode = mode
    if effective_mode is None:
        effective_mode = detect_mode(template_name)

    output_data: dict[str, Any] = {}
    metrics_data: dict[str, Any] | None = None
    result_model_instance: BaseModel | None = None
    result_model_class: type[BaseModel] | None = template_model

    if effective_mode is not None:
        # Use a registered pipeline mode
        import mosaicx.pipelines.radiology  # noqa: F401
        import mosaicx.pipelines.pathology  # noqa: F401
        from .pipelines.extraction import extract_with_mode_raw

        output_data, metrics, raw_prediction = extract_with_mode_raw(
            document_text, effective_mode
        )
        if metrics is not None:
            metrics_data = {
                "total_duration_s": metrics.total_duration_s,
                "total_tokens": metrics.total_tokens,
                "steps": len(metrics.steps),
            }
        # Find the primary Pydantic model for completeness scoring
        result_model_instance = _find_primary_model(raw_prediction)
        if result_model_instance is not None:
            result_model_class = type(result_model_instance)
    elif template_model is not None:
        # Use DocumentExtractor with schema
        from .pipelines.extraction import DocumentExtractor

        extractor = DocumentExtractor(output_schema=template_model)
        result = extractor(document_text=document_text)
        if hasattr(result, "extracted"):
            val = result.extracted
            if hasattr(val, "model_dump"):
                result_model_instance = val
                output_data["extracted"] = val.model_dump()
            else:
                output_data["extracted"] = val
        pipe_metrics = getattr(extractor, "_last_metrics", None)
        if pipe_metrics is not None:
            metrics_data = {
                "total_duration_s": pipe_metrics.total_duration_s,
                "total_tokens": pipe_metrics.total_tokens,
                "steps": len(pipe_metrics.steps),
            }
    else:
        # No mode, no template -- use auto extraction
        from .pipelines.extraction import DocumentExtractor

        extractor = DocumentExtractor()
        result = extractor(document_text=document_text)
        if hasattr(result, "extracted"):
            val = result.extracted
            if hasattr(val, "model_dump"):
                result_model_instance = val
                result_model_class = type(val)
                output_data["extracted"] = val.model_dump()
            else:
                output_data["extracted"] = val
        pipe_metrics = getattr(extractor, "_last_metrics", None)
        if pipe_metrics is not None:
            metrics_data = {
                "total_duration_s": pipe_metrics.total_duration_s,
                "total_tokens": pipe_metrics.total_tokens,
                "steps": len(pipe_metrics.steps),
            }

    # Compute completeness if we have a model instance to score
    completeness_dict: dict[str, Any] = {}
    if result_model_instance is not None and result_model_class is not None:
        from dataclasses import asdict

        comp = compute_report_completeness(
            result_model_instance, document_text, result_model_class
        )
        completeness_dict = asdict(comp)

    return ReportResult(
        extracted=output_data,
        completeness=completeness_dict,
        template_name=template_name,
        mode_used=effective_mode,
        metrics=metrics_data,
    )
