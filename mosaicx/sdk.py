# mosaicx/sdk.py
"""
MOSAICX Python SDK -- programmatic access without the CLI.

This module provides three core functions that wrap the internal DSPy
pipelines, plus utility functions for schema management, evaluation, and
health checks.

Core functions::

    from mosaicx.sdk import extract, deidentify, summarize

    result   = extract("Patient presents with chest pain...", template="chest_ct")
    result   = extract(documents="scan.pdf", mode="radiology")
    results  = extract(documents=["a.pdf", "b.pdf"], workers=4)
    clean    = deidentify("John Doe, SSN 123-45-6789")
    clean    = deidentify(documents="record.pdf")
    summary  = summarize(["Report 1 text...", "Report 2 text..."])
    summary  = summarize(documents=["r1.pdf", "r2.pdf"], patient_id="P001")

Utilities::

    from mosaicx.sdk import generate_schema, health, list_modes, list_templates

    template = generate_schema("echo report with LVEF and valve grades")
    status   = health()

All heavy dependencies (DSPy, pipeline modules) are imported lazily so
this module stays importable even in environments where DSPy is not
installed.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mosaicx.query.session import QuerySession

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

    from .runtime_env import ensure_runtime_env

    ensure_runtime_env()

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


def _attach_envelope(
    output: dict[str, Any],
    *,
    pipeline: str,
    template: str | None = None,
    metrics: Any = None,
    provenance: bool = False,
    verification: dict[str, Any] | None = None,
    document: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> None:
    """Attach a ``_mosaicx`` metadata envelope to *output* in-place.

    Parameters
    ----------
    output:
        The result dict to modify.
    pipeline:
        Pipeline name (e.g. ``"radiology"``, ``"deidentify"``).
    template:
        Template name, or ``None`` if no template was used.
    metrics:
        A ``PipelineMetrics`` instance, or ``None``.
    """
    from .envelope import build_envelope

    duration: float | None = None
    tokens: dict[str, int] | None = None
    if metrics is not None:
        duration = metrics.total_duration_s
        tokens = {
            "input": metrics.total_input_tokens,
            "output": metrics.total_output_tokens,
        }

    output["_mosaicx"] = build_envelope(
        pipeline=pipeline,
        template=template,
        duration_s=duration,
        tokens=tokens,
        provenance=provenance,
        verification=verification,
        document=document,
    )


def _set_envelope_fields(
    output: dict[str, Any],
    *,
    document: dict[str, Any] | list[dict[str, Any]] | None = None,
    provenance: bool | None = None,
    verification: dict[str, Any] | None = None,
) -> None:
    """Patch selected ``_mosaicx`` subfields after the envelope is attached."""
    env = output.get("_mosaicx")
    if not isinstance(env, dict):
        return
    if document is not None:
        env["document"] = document
    if provenance is not None:
        env["provenance"] = provenance
    if verification is not None:
        env["verification"] = verification


# ---------------------------------------------------------------------------
# Document resolution helpers
# ---------------------------------------------------------------------------


def _load_doc_with_config(path: Path) -> Any:
    """Load a document using OCR settings from config.

    Returns a ``LoadedDocument`` instance from :mod:`mosaicx.documents.loader`.
    """
    from .config import get_config
    from .documents.loader import load_document

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


def _build_document_meta(doc: Any, filepath: str | Path | None = None) -> dict[str, Any]:
    """Build the ``_document`` metadata dict from a loaded document.

    Parameters
    ----------
    doc:
        A ``LoadedDocument`` instance.
    filepath:
        Original file path (used for the ``"file"`` key). If ``None``,
        the ``source_path`` from the document is used.

    Returns
    -------
    dict
        Keys: ``"file"``, ``"format"``, ``"page_count"``,
        ``"ocr_engine_used"``, ``"quality_warning"``.
    """
    name = Path(filepath).name if filepath is not None else doc.source_path.name
    return {
        "file": name,
        "format": doc.format,
        "page_count": doc.page_count,
        "ocr_engine_used": doc.ocr_engine_used,
        "quality_warning": doc.quality_warning if doc.quality_warning else None,
    }


def _resolve_documents(
    documents: str | Path | bytes | list[str | Path],
    filename: str | None = None,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Resolve the ``documents`` parameter into loaded document texts.

    Handles four input types:

    1. ``bytes`` -- write to temp file (extension from *filename*), load,
       return text, cleanup temp file.
    2. ``str`` or ``Path`` pointing to a **file** -- load directly.
    3. ``str`` or ``Path`` pointing to a **directory** -- discover all
       supported files and load each.
    4. ``list[str | Path]`` -- load each path in the list.

    Parameters
    ----------
    documents:
        The documents parameter from the public API.
    filename:
        Original filename, required when *documents* is ``bytes``.

    Returns
    -------
    list[tuple[str, str, dict]]
        List of ``(filepath_str, loaded_text, document_metadata)`` tuples.
        ``filepath_str`` is the display name for progress callbacks.

    Raises
    ------
    ValueError
        If *documents* is ``bytes`` and *filename* is not provided.
    FileNotFoundError
        If a file path does not exist.
    """
    from .documents.engines.base import SUPPORTED_FORMATS

    results: list[tuple[str, str, dict[str, Any]]] = []

    if isinstance(documents, bytes):
        # bytes -> write to temp file, load, cleanup
        if not filename:
            raise ValueError(
                "filename is required when documents is bytes "
                "(needed for format detection from extension)."
            )
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(documents)
            tmp_path = Path(tmp.name)
        try:
            doc = _load_doc_with_config(tmp_path)
            meta = _build_document_meta(doc, filepath=filename)
            results.append((filename, doc.text, meta))
        finally:
            tmp_path.unlink(missing_ok=True)
        return results

    if isinstance(documents, (str, Path)):
        doc_path = Path(documents)
        if doc_path.is_dir():
            # Directory -> discover all supported files
            file_list = sorted(
                p for p in doc_path.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
            )
            if not file_list:
                raise ValueError(
                    f"No supported documents found in directory: {doc_path}"
                )
            for fp in file_list:
                doc = _load_doc_with_config(fp)
                meta = _build_document_meta(doc, filepath=fp)
                results.append((fp.name, doc.text, meta))
            return results
        else:
            # Single file
            if not doc_path.exists():
                raise FileNotFoundError(f"Document not found: {doc_path}")
            doc = _load_doc_with_config(doc_path)
            meta = _build_document_meta(doc, filepath=doc_path)
            results.append((doc_path.name, doc.text, meta))
            return results

    if isinstance(documents, list):
        for item in documents:
            fp = Path(item)
            if not fp.exists():
                raise FileNotFoundError(f"Document not found: {fp}")
            doc = _load_doc_with_config(fp)
            meta = _build_document_meta(doc, filepath=fp)
            results.append((fp.name, doc.text, meta))
        return results

    raise TypeError(
        f"Unsupported documents type: {type(documents).__name__}. "
        "Expected str, Path, bytes, or list[str | Path]."
    )


def _resolve_verification_sources(
    *,
    source_text: str | None,
    sources: list[str | Path] | None,
    document: str | Path | None,
) -> tuple[str, list[str]]:
    """Resolve verification sources into one combined source text."""
    chunks: list[tuple[str, str]] = []

    if source_text:
        chunks.append(("source_text", source_text))

    if sources:
        for item in sources:
            p = Path(item)
            if p.exists():
                if p.is_dir():
                    for name, text, _meta in _resolve_documents(p):
                        chunks.append((name, text))
                else:
                    doc = _load_doc_with_config(p)
                    chunks.append((p.name, doc.text))
            else:
                # Allow direct raw-text sources in the list.
                raw = str(item).strip()
                if raw:
                    chunks.append(("source_text", raw))

    if document is not None:
        p = Path(document)
        doc = _load_doc_with_config(p)
        chunks.append((p.name, doc.text))

    if not chunks:
        raise ValueError("Must provide either source_text or document (or sources)")

    source_names = [name for name, _ in chunks]
    if len(chunks) == 1:
        return chunks[0][1], source_names

    combined = "\n\n".join(
        f"[SOURCE: {name}]\n{text}" for name, text in chunks
    )
    return combined, source_names


def _compact_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _source_snippet_for(source_text: str, needle: str) -> tuple[str, str] | None:
    if not source_text.strip() or not needle.strip():
        return None

    needle_norm = " ".join(needle.split())
    pattern = re.escape(needle_norm).replace(r"\ ", r"\s+")
    if "/" in needle_norm:
        pattern = pattern.replace("/", r"\s*/\s*")

    match = re.search(pattern, source_text, flags=re.IGNORECASE)
    if match is None:
        return None

    radius = 80
    start = max(0, match.start() - radius)
    end = min(len(source_text), match.end() + radius)
    matched = " ".join(source_text[match.start():match.end()].split())
    snippet = " ".join(source_text[start:end].split())
    return matched, snippet


def _claim_comparison_from_report(
    *,
    claim: str,
    source_text: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Build a normalized claim grounding payload from verify report output."""
    field_verdicts = report.get("field_verdicts", [])
    issues = report.get("issues", [])
    evidence_items = report.get("evidence", [])

    claimed_val: str | None = None
    source_val: str | None = None
    evidence_val: str | None = None

    claim_rows: list[dict[str, Any]] = []
    for fv in field_verdicts:
        if not isinstance(fv, dict):
            continue
        fp = str(fv.get("field_path") or "").lower().strip()
        if fp in ("", "claim"):
            claim_rows.append(fv)

    if not claim_rows and len(field_verdicts) == 1 and isinstance(field_verdicts[0], dict):
        claim_rows = [field_verdicts[0]]

    preferred_rows = (
        [fv for fv in claim_rows if fv.get("status") in ("mismatch", "unsupported")]
        + [fv for fv in claim_rows if fv.get("status") == "verified"]
        + [fv for fv in claim_rows if fv.get("status") not in ("mismatch", "unsupported", "verified")]
    )
    for fv in preferred_rows:
        if claimed_val is None:
            claimed_val = _compact_text(fv.get("claimed_value"))
        if source_val is None:
            source_val = _compact_text(fv.get("source_value"))
        if evidence_val is None:
            evidence_val = _compact_text(fv.get("evidence_excerpt"))

    # Parse natural-language issue descriptions when model emits prose only.
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        detail = str(issue.get("detail") or "")
        if not detail:
            continue
        if evidence_val is None:
            evidence_val = _compact_text(detail)

        m = re.search(
            r"(?i)(?:claim(?:ed| states?)\s+)(.+?)(?:,?\s+but\s+source(?:\s+\w+)?\s+|"
            r"\s+does not match source\s+)(.+?)(?:[.]|$)",
            detail,
        )
        if m:
            if claimed_val is None:
                claimed_val = _compact_text(m.group(1))
            if source_val is None:
                source_val = _compact_text(m.group(2))
            break

    if (source_val is None or evidence_val is None) and isinstance(evidence_items, list):
        for ev in evidence_items:
            if not isinstance(ev, dict):
                continue
            if source_val is None:
                source_val = _compact_text(ev.get("excerpt"))
            if evidence_val is None:
                evidence_val = _compact_text(
                    ev.get("supports")
                    or ev.get("contradicts")
                    or ev.get("excerpt")
                )
            if source_val is not None and evidence_val is not None:
                break

    if source_val is None and evidence_val:
        bp = re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", evidence_val)
        if bp:
            source_val = bp.group(0)

    if source_val is None:
        seed = claimed_val or claim
        candidates: list[str] = []
        bp = re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", seed)
        if bp:
            candidates.append(bp.group(0))
        candidates.extend(re.findall(r"\d+(?:\.\d+)?", seed))
        terms = [
            t for t in re.findall(r"[A-Za-z]{4,}", seed)
            if t.lower() not in {"with", "from", "that", "this", "patient", "claim", "states"}
        ]
        candidates.extend(terms[:3])

        for cand in candidates:
            result = _source_snippet_for(source_text, cand)
            if result is not None:
                matched, snippet = result
                source_val = matched
                if evidence_val is None:
                    evidence_val = snippet
                break

    if evidence_val is None and source_val is not None:
        result = _source_snippet_for(source_text, source_val)
        if result is not None:
            _matched, snippet = result
            evidence_val = snippet

    if claimed_val is None:
        claimed_val = claim

    grounded = bool((source_val and source_val.strip()) or (evidence_val and evidence_val.strip()))
    return {
        "claimed": claimed_val,
        "source": source_val,
        "evidence": evidence_val,
        "grounded": grounded,
    }


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def _extract_single_text(
    text: str,
    *,
    template: str | Path | None,
    mode: str,
    score: bool,
    optimized: str | Path | None,
    verify: bool,
    verify_level: str,
    provenance: bool,
) -> dict[str, Any]:
    """Core extraction logic for a single text input.

    This is the internal workhorse called by :func:`extract` for each
    document.  It handles template resolution, mode selection, and
    completeness scoring.
    """
    # Validate mode early, before configuring DSPy
    if mode not in ("auto",) and template is None:
        import mosaicx.pipelines.pathology  # noqa: F401
        import mosaicx.pipelines.radiology  # noqa: F401
        from .pipelines.modes import list_modes

        available = list_modes()
        if mode not in available:
            raise ValueError(
                f"Unknown mode {mode!r}. Available: {', '.join(sorted(available))}"
            )

    _ensure_configured()

    def _finalize_output(
        output: dict[str, Any],
        *,
        pipeline: str,
        template_name: str | None,
        metrics: Any = None,
    ) -> dict[str, Any]:
        verification_summary: dict[str, Any] | None = None

        if provenance:
            from .provenance.resolve import build_provenance

            output["_provenance"] = build_provenance(output, text)

        if verify:
            from .verify.engine import verify as _verify

            report = _verify(
                extraction=output,
                source_text=text,
                level=verify_level,
            ).to_dict()
            output["_verification"] = report
            verification_summary = {
                "verdict": report.get("verdict"),
                "confidence": report.get("confidence"),
                "level": report.get("level"),
                "issues": len(report.get("issues", [])),
            }

        _attach_envelope(
            output,
            pipeline=pipeline,
            template=template_name,
            metrics=metrics,
            provenance=provenance,
            verification=verification_summary,
        )
        return output

    # --- Template-based extraction ---
    if template is not None:
        from .report import detect_mode, resolve_template

        template_str = str(template)
        template_model, tpl_name = resolve_template(template=template_str)

        effective_mode = detect_mode(tpl_name)

        if effective_mode is not None and template_model is None:
            # Built-in template with mode pipeline
            import mosaicx.pipelines.pathology  # noqa: F401
            import mosaicx.pipelines.radiology  # noqa: F401

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
            return _finalize_output(
                output_data,
                pipeline=effective_mode,
                template_name=tpl_name,
                metrics=metrics,
            )
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
            return _finalize_output(
                output,
                pipeline="extraction",
                template_name=tpl_name,
                metrics=getattr(extractor, "_last_metrics", None),
            )
        else:
            raise ValueError(
                f"Template {template!r} resolved but produced no extraction template."
            )

    # --- Mode-based extraction (radiology, pathology, ...) ---
    if mode not in ("auto",):
        # Trigger lazy loading of mode pipeline modules
        import mosaicx.pipelines.pathology  # noqa: F401
        import mosaicx.pipelines.radiology  # noqa: F401

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
        return _finalize_output(
            output_data,
            pipeline=mode,
            template_name=None,
            metrics=metrics,
        )

    # --- Auto extraction (LLM infers schema) ---
    if score:
        logger.warning(
            "score=True has no effect in auto mode (no template to score against). "
            "Provide a template to enable completeness scoring."
        )

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

    return _finalize_output(
        output,
        pipeline="extraction",
        template_name=None,
        metrics=getattr(extractor, "_last_metrics", None),
    )


def extract(
    text: str | None = None,
    *,
    documents: str | Path | bytes | list[str | Path] | None = None,
    filename: str | None = None,
    template: str | Path | None = None,
    mode: str = "auto",
    score: bool = False,
    optimized: str | Path | None = None,
    verify: bool = False,
    verify_level: str = "quick",
    provenance: bool = False,
    workers: int = 1,
    on_progress: Callable[[str, bool, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Extract structured data from text or document files.

    Parameters
    ----------
    text:
        Document text to extract from.  Mutually exclusive with
        *documents*.
    documents:
        Document source(s). Accepts:

        - ``bytes`` -- raw file content (requires *filename*).
        - ``str`` or ``Path`` to a **file** -- loaded directly.
        - ``str`` or ``Path`` to a **directory** -- discovers all
          supported files.
        - ``list[str | Path]`` -- processes each path.

        Mutually exclusive with *text*.
    filename:
        Original filename. Required when *documents* is ``bytes`` (for
        format detection from the extension).
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
    verify:
        If ``True``, run post-extraction verification and include
        ``"_verification"`` in each result.
    verify_level:
        Verification depth used when ``verify=True``. One of
        ``"quick"``, ``"standard"``, ``"thorough"``.
    provenance:
        If ``True``, attach deterministic field-level provenance under
        ``"_provenance"``.
    workers:
        Number of parallel extraction workers for multi-file processing.
        Document loading is always sequential (pypdfium2 is not
        thread-safe), but extraction is parallelised.
    on_progress:
        Optional callback ``(filename, success, result_or_none)`` called
        after each file completes (multi-file mode only).

    Returns
    -------
    dict | list[dict]
        **Smart return**: single input returns ``dict``, multiple inputs
        returns ``list[dict]``.  Each result dict includes a
        ``"_document"`` key with loading metadata when loaded from a file.

    Raises
    ------
    ValueError
        If both *text* and *documents* are provided, if neither is
        provided, or if the template/mode is unknown.
    FileNotFoundError
        If a document path does not exist.
    """
    # --- Input validation ---
    if text is not None and documents is not None:
        raise ValueError("Provide either text or documents, not both.")
    if text is None and documents is None:
        raise ValueError("Provide text or documents.")

    # --- Text-only path (single result) ---
    if text is not None:
        return _extract_single_text(
            text,
            template=template,
            mode=mode,
            score=score,
            optimized=optimized,
            verify=verify,
            verify_level=verify_level,
            provenance=provenance,
        )

    # --- Document-based path ---
    assert documents is not None

    # Resolve documents: load all files sequentially (OCR not thread-safe)
    loaded = _resolve_documents(documents, filename=filename)

    # Determine if this is a single-input call (smart return)
    is_single = (
        isinstance(documents, bytes)
        or (isinstance(documents, (str, Path)) and Path(documents).is_file())
    )

    # Extract from each loaded document
    def _do_extract(
        name: str, doc_text: str, doc_meta: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            result = _extract_single_text(
                doc_text,
                template=template,
                mode=mode,
                score=score,
                optimized=optimized,
                verify=verify,
                verify_level=verify_level,
                provenance=provenance,
            )
            result["_document"] = doc_meta
            _set_envelope_fields(result, document=doc_meta, provenance=provenance)
            return name, result, None
        except Exception as exc:
            return name, None, f"{type(exc).__name__}: {exc}"

    if len(loaded) == 1:
        # Single document -- no threading needed
        name, doc_text, doc_meta = loaded[0]
        name, result, error = _do_extract(name, doc_text, doc_meta)
        if error:
            result_dict: dict[str, Any] = {
                "error": error, "_document": doc_meta,
            }
            if on_progress:
                on_progress(name, False, None)
            if is_single:
                return result_dict
            return [result_dict]
        if on_progress:
            on_progress(name, True, result)
        if is_single:
            return result  # type: ignore[return-value]
        return [result]  # type: ignore[list-item]

    # Multiple documents -- parallel extraction
    results: list[dict[str, Any]] = [{}] * len(loaded)  # preserve order
    index_map = {name: i for i, (name, _, _) in enumerate(loaded)}

    max_w = min(max(1, workers), 32)
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        futures = {
            pool.submit(_do_extract, name, doc_text, doc_meta): (name, doc_meta)
            for name, doc_text, doc_meta in loaded
        }
        for future in as_completed(futures):
            name, doc_meta = futures[future]
            fname, result, error = future.result()
            idx = index_map[fname]
            if error:
                results[idx] = {"error": error, "_document": doc_meta}
                if on_progress:
                    on_progress(fname, False, None)
            else:
                assert result is not None
                results[idx] = result
                if on_progress:
                    on_progress(fname, True, result)

    return results


# ---------------------------------------------------------------------------
# deidentify
# ---------------------------------------------------------------------------


def _deidentify_single_text(
    text: str,
    *,
    mode: str,
) -> dict[str, Any]:
    """Core de-identification logic for a single text input."""
    valid_modes = {"remove", "pseudonymize", "dateshift", "regex"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown deidentify mode: {mode!r}. "
            f"Choose from: {sorted(valid_modes)}"
        )

    from .pipelines.deidentifier import regex_scrub_phi

    if mode == "regex":
        output: dict[str, Any] = {"redacted_text": regex_scrub_phi(text)}
        _attach_envelope(output, pipeline="deidentify")
        return output

    _ensure_configured()

    from .pipelines.deidentifier import Deidentifier

    deid = Deidentifier()
    result = deid(document_text=text, mode=mode)
    output = {"redacted_text": result.redacted_text}
    _attach_envelope(
        output,
        pipeline="deidentify",
        metrics=getattr(deid, "_last_metrics", None),
    )
    return output


def deidentify(
    text: str | None = None,
    *,
    documents: str | Path | bytes | list[str | Path] | None = None,
    filename: str | None = None,
    mode: str = "remove",
    workers: int = 1,
    on_progress: Callable[[str, bool, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Remove PHI from text or document files.

    Parameters
    ----------
    text:
        Text containing Protected Health Information.  Mutually
        exclusive with *documents*.
    documents:
        Document source(s). Accepts the same types as
        :func:`extract` -- ``bytes``, file path, directory, or list of
        paths.  Mutually exclusive with *text*.
    filename:
        Original filename. Required when *documents* is ``bytes``.
    mode:
        De-identification strategy:

        - ``"remove"``       -- Replace PHI with ``[REDACTED]``.
        - ``"pseudonymize"`` -- Replace PHI with realistic fake values.
        - ``"dateshift"``    -- Shift dates by a consistent random offset.
        - ``"regex"``        -- Regex-only scrubbing (no LLM needed).
    workers:
        Number of parallel workers for multi-file processing.
    on_progress:
        Optional callback ``(filename, success, result_or_none)`` called
        after each file completes (multi-file mode only).

    Returns
    -------
    dict | list[dict]
        **Smart return**: single input returns ``dict``, multiple inputs
        returns ``list[dict]``.  Keys: ``"redacted_text"`` (str).  When
        loaded from a file, includes a ``"_document"`` key with metadata.

    Raises
    ------
    ValueError
        If both *text* and *documents* are provided, if neither is
        provided, or if ``mode`` is not one of the supported values.
    FileNotFoundError
        If a document path does not exist.
    """
    # --- Input validation ---
    if text is not None and documents is not None:
        raise ValueError("Provide either text or documents, not both.")
    if text is None and documents is None:
        raise ValueError("Provide text or documents.")

    # --- Text-only path (single result) ---
    if text is not None:
        return _deidentify_single_text(text, mode=mode)

    # --- Document-based path ---
    assert documents is not None

    loaded = _resolve_documents(documents, filename=filename)

    is_single = (
        isinstance(documents, bytes)
        or (isinstance(documents, (str, Path)) and Path(documents).is_file())
    )

    def _do_deid(
        name: str, doc_text: str, doc_meta: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            result = _deidentify_single_text(doc_text, mode=mode)
            result["_document"] = doc_meta
            return name, result, None
        except Exception as exc:
            return name, None, f"{type(exc).__name__}: {exc}"

    if len(loaded) == 1:
        name, doc_text, doc_meta = loaded[0]
        name, result, error = _do_deid(name, doc_text, doc_meta)
        if error:
            result_dict: dict[str, Any] = {
                "error": error, "_document": doc_meta,
            }
            if on_progress:
                on_progress(name, False, None)
            if is_single:
                return result_dict
            return [result_dict]
        if on_progress:
            on_progress(name, True, result)
        if is_single:
            return result  # type: ignore[return-value]
        return [result]  # type: ignore[list-item]

    # Multiple documents -- parallel
    results: list[dict[str, Any]] = [{}] * len(loaded)
    index_map = {name: i for i, (name, _, _) in enumerate(loaded)}

    max_w = min(max(1, workers), 32)
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        futures = {
            pool.submit(_do_deid, name, doc_text, doc_meta): (name, doc_meta)
            for name, doc_text, doc_meta in loaded
        }
        for future in as_completed(futures):
            name, doc_meta = futures[future]
            fname, result, error = future.result()
            idx = index_map[fname]
            if error:
                results[idx] = {"error": error, "_document": doc_meta}
                if on_progress:
                    on_progress(fname, False, None)
            else:
                assert result is not None
                results[idx] = result
                if on_progress:
                    on_progress(fname, True, result)

    return results


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def summarize(
    reports: list[str] | None = None,
    *,
    documents: str | Path | list[str | Path] | None = None,
    patient_id: str = "unknown",
    optimized: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize multiple reports into a patient timeline.

    Parameters
    ----------
    reports:
        List of report texts.  Mutually exclusive with *documents*.
    documents:
        File paths to load reports from.  Accepts a list of paths,
        a single path, or a directory (discovers all supported files).
        Mutually exclusive with *reports*.  No ``bytes`` support
        (summarize merges all reports into one timeline).
    patient_id:
        Patient identifier for the summary.
    optimized:
        Path to an optimized DSPy program to load.

    Returns
    -------
    dict
        Keys: ``"narrative"`` (str), ``"events"`` (list of event dicts).
        When *documents* is used, includes ``"_document"`` with a list
        of loading metadata dicts.

    Raises
    ------
    ValueError
        If both *reports* and *documents* are provided, or if neither
        is provided, or if the resulting report list is empty.
    FileNotFoundError
        If any document path does not exist.
    """
    if reports is not None and documents is not None:
        raise ValueError("Provide either reports or documents, not both.")
    if reports is None and documents is None:
        raise ValueError("Provide reports or documents.")

    doc_metadata_list: list[dict[str, Any]] | None = None

    if documents is not None:
        loaded = _resolve_documents(documents)
        reports = []
        doc_metadata_list = []
        for _name, doc_text, doc_meta in loaded:
            if doc_text:
                reports.append(doc_text)
                doc_metadata_list.append(doc_meta)

    assert reports is not None  # guaranteed by validation above

    if not reports:
        raise ValueError("No reports provided to summarize.")

    _ensure_configured()

    from .pipelines.summarizer import ReportSummarizer

    summarizer = ReportSummarizer()

    if optimized is not None:
        from .evaluation.optimize import load_optimized

        summarizer = load_optimized(ReportSummarizer, Path(optimized))

    result = summarizer(reports=reports, patient_id=patient_id)

    output: dict[str, Any] = {
        "events": [e.model_dump() for e in result.events],
        "narrative": result.narrative,
    }
    if doc_metadata_list is not None:
        output["_document"] = doc_metadata_list
    _attach_envelope(
        output,
        pipeline="summarize",
        metrics=getattr(summarizer, "_last_metrics", None),
    )
    return output


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
# list_templates
# ---------------------------------------------------------------------------


def list_templates() -> list[dict[str, Any]]:
    """List available extraction templates (built-in and user-created).

    Returns
    -------
    list[dict]
        Each dict has keys ``"name"``, ``"description"``, ``"mode"``,
        and ``"source"`` (``"built-in"`` or ``"user"``).
    """
    from .config import get_config
    from .schemas.radreport.registry import list_templates as _list_builtin

    cfg = get_config()
    templates: list[dict[str, Any]] = []

    for tpl in _list_builtin():
        templates.append({
            "name": tpl.name,
            "description": tpl.description,
            "mode": tpl.mode,
            "source": "built-in",
        })

    if cfg.templates_dir.is_dir():
        from .schemas.template_compiler import parse_template

        for f in sorted(cfg.templates_dir.glob("*.yaml")) + sorted(
            cfg.templates_dir.glob("*.yml")
        ):
            try:
                meta = parse_template(f.read_text(encoding="utf-8"))
                templates.append({
                    "name": f.stem,
                    "description": meta.description or "",
                    "mode": meta.mode,
                    "source": "user",
                })
            except Exception:
                templates.append({
                    "name": f.stem,
                    "description": "(invalid YAML)",
                    "mode": None,
                    "source": "user",
                })

    return templates


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
# Verify
# ---------------------------------------------------------------------------


def verify(
    *,
    extraction: dict[str, Any] | None = None,
    claim: str | None = None,
    source_text: str | None = None,
    sources: list[str | Path] | None = None,
    document: str | Path | None = None,
    level: str = "quick",
) -> dict[str, Any]:
    """Verify an extraction or claim against source text.

    Parameters
    ----------
    extraction:
        Structured extraction dict to verify.
    claim:
        A single claim string to verify.
    source_text:
        Source document text to verify against. If *document* is provided
        instead, text is loaded from the file.
    sources:
        One or more source files or text strings to verify against.
        When multiple are provided, they are combined into a single source
        context.
    document:
        Path to source document file. Text is loaded automatically.
    level:
        Verification level: "quick", "standard", or "thorough".

    Returns
    -------
    dict
        Verification report with keys: verdict, confidence, level, issues.
    """
    combined_source_text, _source_names = _resolve_verification_sources(
        source_text=source_text,
        sources=sources,
        document=document,
    )

    if level in ("standard", "thorough"):
        _ensure_configured()

    from .verify.engine import verify as _verify

    report = _verify(
        extraction=extraction,
        claim=claim,
        source_text=combined_source_text,
        level=level,
    )
    out = report.to_dict()

    expected_effective = {
        "quick": "deterministic",
        "standard": "spot_check",
        "thorough": "audit",
    }.get(level)
    effective_level = out.get("level")
    fallback_used = bool(expected_effective and effective_level != expected_effective)

    out["requested_level"] = level
    out["effective_level"] = effective_level
    out["fallback_used"] = fallback_used
    verification_mode = "claim" if claim is not None and extraction is None else "extraction"
    out["verification_mode"] = verification_mode
    out["confidence_score"] = out.get("confidence")

    if fallback_used:
        fallback_issue = next(
            (
                i for i in out.get("issues", [])
                if i.get("type") in {"llm_unavailable", "rlm_unavailable"}
            ),
            None,
        )
        out["fallback_reason"] = (
            fallback_issue.get("detail")
                if fallback_issue is not None
                else f"Requested {level} but executed {effective_level}"
        )

    decision = out.get("verdict")
    out["claim_truth"] = None
    if verification_mode == "claim" and claim is not None:
        claim_comparison = _claim_comparison_from_report(
            claim=claim,
            source_text=combined_source_text,
            report=out,
        )
        out["claim_comparison"] = claim_comparison
        grounded = bool(claim_comparison.get("grounded"))
        out["grounded"] = grounded

        # Avoid hard pass/fail labels when we have no source-grounding payload.
        if decision in {"verified", "contradicted"} and not grounded:
            decision = "insufficient_evidence"

        support_map = {
            "verified": 1.0,
            "partially_supported": 0.5,
            "insufficient_evidence": 0.25,
            "contradicted": 0.0,
        }
        out["support_score"] = support_map.get(decision, out.get("confidence", 0.0))
        if decision == "verified":
            out["claim_truth"] = True
        elif decision == "contradicted":
            out["claim_truth"] = False
    else:
        out["support_score"] = out.get("confidence")

    out["decision"] = decision

    return out


def verify_batch(
    *,
    claims: list[str] | None = None,
    extractions: list[dict[str, Any]] | None = None,
    source_text: str | None = None,
    sources: list[str | Path] | None = None,
    document: str | Path | None = None,
    level: str = "quick",
) -> list[dict[str, Any]]:
    """Batch verify multiple claims or extractions against source(s)."""
    if not claims and not extractions:
        raise ValueError("Provide claims or extractions for batch verification.")

    if claims and extractions:
        raise ValueError("Provide claims or extractions, not both.")

    results: list[dict[str, Any]] = []
    if claims:
        for claim in claims:
            results.append(
                verify(
                    claim=claim,
                    source_text=source_text,
                    sources=sources,
                    document=document,
                    level=level,
                )
            )
    else:
        assert extractions is not None
        for extraction in extractions:
            results.append(
                verify(
                    extraction=extraction,
                    source_text=source_text,
                    sources=sources,
                    document=document,
                    level=level,
                )
            )
    return results


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


def query(
    sources: list[str | Path] | None = None,
    template: str | None = None,
    sub_lm: str | None = None,
) -> QuerySession:
    """Create a query session for conversational Q&A over documents and data.

    Parameters
    ----------
    sources:
        List of file paths to load as data sources. Supports CSV, JSON,
        Parquet, Excel, PDF, and plain text files.
    template:
        Optional extraction template hint for future query steps.
    sub_lm:
        Optional lightweight model override for sub-queries.

    Returns
    -------
    QuerySession
        A stateful session that holds loaded data and conversation
        history. The caller can then create a
        :class:`~mosaicx.query.engine.QueryEngine` from the session
        for LLM-powered Q&A.

    Examples
    --------
    ::

        from mosaicx.sdk import query

        session = query(sources=["data.csv", "notes.txt"])
        print(session.catalog)   # inspect loaded sources
        session.close()          # release resources
    """
    if not sources:
        raise ValueError("At least one source path is required.")

    from .query.session import QuerySession as _QuerySession

    return _QuerySession(sources=sources, template=template, sub_lm=sub_lm)


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def health() -> dict[str, Any]:
    """Check MOSAICX configuration status and available capabilities.

    Does NOT make an LLM call. Reads configuration and scans available
    modes/templates to report what the system can do.

    Returns
    -------
    dict
        Keys: ``"version"``, ``"configured"``, ``"lm_model"``,
        ``"api_base"``, ``"available_modes"``, ``"available_templates"``,
        ``"ocr_engine"``.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    from .config import get_config

    try:
        version = _pkg_version("mosaicx")
    except PackageNotFoundError:
        version = "2.0.0a1"

    cfg = get_config()

    # Modes (no DSPy needed -- just registry scan)
    import mosaicx.pipelines.pathology  # noqa: F401
    import mosaicx.pipelines.radiology  # noqa: F401

    from .pipelines.modes import list_modes as _list_modes

    modes = [name for name, _desc in _list_modes()]

    # Templates (delegate to list_templates to avoid duplication)
    templates = [t["name"] for t in list_templates()]

    return {
        "version": version,
        "configured": bool(cfg.api_key),
        "lm_model": cfg.lm,
        "api_base": cfg.api_base,
        "available_modes": modes,
        "available_templates": templates,
        "ocr_engine": cfg.ocr_engine,
    }
