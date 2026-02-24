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

    from .runtime_env import configure_dspy_lm

    from .config import get_config

    cfg = get_config()
    if not cfg.api_key:
        raise RuntimeError(
            "No API key configured. Set MOSAICX_API_KEY or add api_key "
            "to your config."
        )

    from .metrics import TokenTracker, make_harmony_lm, set_tracker

    lm = make_harmony_lm(cfg.lm, api_key=cfg.api_key, api_base=cfg.api_base, temperature=cfg.lm_temperature)
    try:
        dspy, _adapter_name = configure_dspy_lm(
            lm,
            preferred_cache_dir=cfg.home_dir / ".dspy_cache",
        )
    except ImportError as exc:
        raise RuntimeError(
            "DSPy is required for SDK functions. Install with: pip install dspy"
        ) from exc

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


def _best_bp_value_from_text(text: str, *, claim_bp: str | None = None) -> str | None:
    """Return the most relevant BP-style value from text.

    Uses light context scoring so we prefer actual measured values over
    nearby reference ranges when both appear together.
    """
    text_value = str(text or "")
    if not text_value.strip():
        return None

    matches = list(re.finditer(r"\b\d{2,3}\s*/\s*\d{2,3}\b", text_value))
    if not matches:
        return None

    claim_norm = _normalize_numeric_text(claim_bp or "")
    claim_parts = claim_norm.split("/") if claim_norm else []
    best_score: int | None = None
    best_value: str | None = None

    for match in matches:
        candidate = " ".join(match.group(0).split())
        cand_norm = _normalize_numeric_text(candidate)
        cand_parts = cand_norm.split("/")
        ctx_start = max(0, match.start() - 72)
        ctx_end = min(len(text_value), match.end() + 72)
        context = text_value[ctx_start:ctx_end].lower()

        score = 0
        if ("blood pressure" in context) or re.search(r"\bbp\b", context):
            score += 5
        if any(marker in context for marker in ("vital", "reading", "measured", "measurement", "status")):
            score += 2
        if any(marker in context for marker in ("normal range", "reference", "target", "goal")):
            score -= 2
        if claim_parts and len(claim_parts) == 2 and len(cand_parts) == 2:
            if cand_parts[1] == claim_parts[1]:
                score += 2
            if cand_parts[0] == claim_parts[0]:
                score += 1
            if cand_norm == claim_norm:
                score += 3

        if best_score is None or score > best_score:
            best_score = score
            best_value = candidate

    return best_value


def _normalize_numeric_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _looks_like_absence_statement(text: str | None) -> bool:
    value = _compact_text(text)
    if not value:
        return False
    lowered = value.lower()
    cues = (
        "does not contain",
        "not contain",
        "not found",
        "no evidence",
        "not present",
        "unable to find",
        "cannot find",
        "missing",
        "no information found",
    )
    if any(cue in lowered for cue in cues):
        return True
    # Covers variants like "No blood pressure information found in source."
    return bool(re.search(r"\bno\b[^.]{0,100}\bfound\b", lowered))


def _looks_like_runtime_failure(text: str | None) -> bool:
    """Return True when text is transport/adapter failure prose, not evidence."""
    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return False
    cues = (
        "unavailable",
        "connection error",
        "adapter jsonadapter failed",
        "failed to parse",
        "lm response cannot be serialized",
        "internalservererror",
    )
    return any(cue in lowered for cue in cues)


def _claim_values_clearly_conflict(claimed: Any, source: Any) -> bool:
    claimed_text = _compact_text(claimed) or ""
    source_text = _compact_text(source) or ""
    if not claimed_text or not source_text:
        return False

    c_bp = re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", claimed_text)
    s_bp = re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", source_text)
    if c_bp and s_bp:
        return _normalize_numeric_text(c_bp.group(0)) != _normalize_numeric_text(s_bp.group(0))

    c_nums = re.findall(r"\d+(?:\.\d+)?", claimed_text)
    s_nums = re.findall(r"\d+(?:\.\d+)?", source_text)
    if c_nums and s_nums and c_nums != s_nums:
        return True

    return False


def _claim_values_clearly_match(claimed: Any, source: Any) -> bool:
    """Return True when claim/source carry the same concrete value signal."""
    claimed_text = _compact_text(claimed) or ""
    source_text = _compact_text(source) or ""
    if not claimed_text or not source_text:
        return False

    c_bp = re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", claimed_text)
    s_bp = re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", source_text)
    if c_bp and s_bp:
        return _normalize_numeric_text(c_bp.group(0)) == _normalize_numeric_text(s_bp.group(0))

    c_nums = re.findall(r"\d+(?:\.\d+)?", claimed_text)
    s_nums = re.findall(r"\d+(?:\.\d+)?", source_text)
    if c_nums and s_nums:
        return c_nums == s_nums

    return False


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

    claim_bp_match = re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", str(claim or ""))
    claim_bp = claim_bp_match.group(0) if claim_bp_match else None

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

    # If claim is BP-like, prefer a full BP pair from evidence/source text
    # over partial numeric tokens (e.g., "120").
    if claim_bp:
        preferred_bp = None
        if source_val:
            preferred_bp = _best_bp_value_from_text(source_val, claim_bp=claim_bp)
        if preferred_bp is None and evidence_val:
            preferred_bp = _best_bp_value_from_text(evidence_val, claim_bp=claim_bp)
        if preferred_bp is None:
            preferred_bp = _best_bp_value_from_text(source_text, claim_bp=claim_bp)
        if preferred_bp is not None:
            source_val = preferred_bp
            if evidence_val is None or _looks_like_absence_statement(evidence_val):
                result = _source_snippet_for(source_text, preferred_bp)
                if result is not None:
                    _matched, snippet = result
                    evidence_val = snippet

    if evidence_val is None and source_val is not None:
        result = _source_snippet_for(source_text, source_val)
        if result is not None:
            _matched, snippet = result
            evidence_val = snippet
    elif source_val is not None and _looks_like_absence_statement(evidence_val):
        # If grounded source text exists but prose says "not found", prefer
        # an actual source snippet to keep claim comparison coherent.
        result = _source_snippet_for(source_text, source_val)
        if result is not None:
            _matched, snippet = result
            evidence_val = snippet

    if claimed_val is None:
        claimed_val = claim

    grounded_source = False
    if source_val:
        source_match = _source_snippet_for(source_text, source_val)
        if source_match is not None:
            grounded_source = True
            if evidence_val is None or _looks_like_absence_statement(evidence_val) or _looks_like_runtime_failure(evidence_val):
                _matched, snippet = source_match
                evidence_val = snippet
        else:
            # If model-emitted source value is not in source text, re-ground from source.
            if claim_bp:
                recovered_bp = _best_bp_value_from_text(source_text, claim_bp=claim_bp)
                if recovered_bp is not None:
                    recovered_match = _source_snippet_for(source_text, recovered_bp)
                    if recovered_match is not None:
                        grounded_source = True
                        source_val = recovered_bp
                        _matched, snippet = recovered_match
                        evidence_val = snippet
            if not grounded_source:
                source_val = None

    grounded = bool(
        grounded_source
        or (
            evidence_val
            and evidence_val.strip()
            and not _looks_like_absence_statement(evidence_val)
            and not _looks_like_runtime_failure(evidence_val)
        )
    )
    return {
        "claimed": claimed_val,
        "source": source_val,
        "evidence": evidence_val,
        "grounded": grounded,
    }


def _normalize_claim_decision(value: Any) -> str | None:
    text = " ".join(str(value or "").lower().split())
    if text in {"verified", "supported", "support", "match", "true"}:
        return "verified"
    if text in {"contradicted", "conflict", "mismatch", "false"}:
        return "contradicted"
    if text in {"insufficient_evidence", "insufficient", "unknown", "inconclusive", "uncertain"}:
        return "insufficient_evidence"
    return None


def _adjudicate_claim_decision_with_dspy(
    *,
    claim: str,
    claim_comparison: dict[str, Any],
    current_decision: str,
    citations: list[dict[str, Any]],
) -> str | None:
    """Adjudicate ambiguous claim outcomes with DSPy comparison modules.

    This stage is only used when claim values are grounded but neither clearly
    matching nor clearly conflicting. It uses ``MultiChainComparison`` and
    ``BestOfN`` when available, with deterministic constraints on output labels.
    """
    try:
        import dspy
    except Exception:
        return None

    if getattr(dspy.settings, "lm", None) is None:
        return None

    claimed = _compact_text(claim_comparison.get("claimed")) or claim
    source = _compact_text(claim_comparison.get("source"))
    evidence = _compact_text(claim_comparison.get("evidence"))
    if not source and not evidence:
        return None

    supporting = []
    for c in citations[:4]:
        if not isinstance(c, dict):
            continue
        snippet = _compact_text(c.get("snippet"))
        if not snippet:
            continue
        supporting.append(f"{c.get('source', 'source_document')}: {snippet}")
    evidence_blob = "\n".join(
        p for p in [
            f"claimed={claimed}",
            f"source={source}" if source else "",
            f"evidence={evidence}" if evidence else "",
            *supporting,
        ]
        if p
    )

    guidance = (
        "Adjudicate claim truth from grounded evidence. "
        "Return final_decision as one of: verified, contradicted, insufficient_evidence."
    )
    base = dspy.ChainOfThought(
        "guidance, claim, evidence, current_decision -> final_decision, rationale"
    )

    attempts: list[dict[str, Any]] = []
    for idx in range(3):
        try:
            lm = dspy.settings.lm
            if lm is not None:
                with dspy.context(lm=lm.copy(rollout_id=idx + 1, temperature=1.0)):
                    pred = base(
                        guidance=guidance,
                        claim=claim.strip(),
                        evidence=evidence_blob,
                        current_decision=current_decision,
                    )
            else:
                pred = base(
                    guidance=guidance,
                    claim=claim.strip(),
                    evidence=evidence_blob,
                    current_decision=current_decision,
                )
            try:
                attempts.append(_prediction_to_dict(pred))
            except Exception:
                attempts.append(
                    {
                        "final_decision": getattr(pred, "final_decision", ""),
                        "rationale": getattr(pred, "rationale", ""),
                    }
                )
        except Exception:
            continue

    mcc_decision: str | None = None
    if len(attempts) == 3 and hasattr(dspy, "MultiChainComparison"):
        try:
            mcc = dspy.MultiChainComparison(
                "claim, evidence -> final_decision",
                M=3,
                temperature=0.2,
            )
            mcc_pred = mcc(
                completions=attempts,
                claim=claim.strip(),
                evidence=evidence_blob,
            )
            mcc_decision = _normalize_claim_decision(getattr(mcc_pred, "final_decision", ""))
        except Exception:
            mcc_decision = None

    best_decision: str | None = None
    if hasattr(dspy, "BestOfN"):
        def reward_fn(_args: dict[str, Any], pred: Any) -> float:
            decision = _normalize_claim_decision(getattr(pred, "final_decision", ""))
            if decision is None:
                return 0.0
            score = 0.0
            if source and _claim_values_clearly_match(claimed, source):
                score = 1.0 if decision == "verified" else 0.0
            elif source and _claim_values_clearly_conflict(claimed, source):
                score = 1.0 if decision == "contradicted" else 0.0
            else:
                # For ambiguous grounded cases, prefer stable non-hallucinated labels.
                if decision == current_decision:
                    score = 0.9
                elif decision == "insufficient_evidence":
                    score = 0.8
                else:
                    score = 0.5
            return score

        try:
            best_mod = dspy.BestOfN(
                module=base,
                N=3,
                reward_fn=reward_fn,
                threshold=0.8,
            )
            best_pred = best_mod(
                guidance=guidance,
                claim=claim.strip(),
                evidence=evidence_blob,
                current_decision=current_decision,
            )
            best_decision = _normalize_claim_decision(getattr(best_pred, "final_decision", ""))
        except Exception:
            best_decision = None

    for candidate in (mcc_decision, best_decision):
        if candidate in {"verified", "contradicted", "insufficient_evidence"}:
            return candidate
    return None


def _verification_citations_from_report(
    *,
    report: dict[str, Any],
    verification_mode: str,
    max_citations: int = 8,
) -> list[dict[str, Any]]:
    """Build a normalized evidence list for verify outputs.

    Aligns verify evidence semantics with query citations so downstream tools can
    reason over a consistent structure.
    """
    citations: list[dict[str, Any]] = []
    field_verdicts = report.get("field_verdicts", [])
    evidence_items = report.get("evidence", [])

    def _to_int(value: Any) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    def _status_rank(status: str, *, claim_row: bool) -> int:
        status_lc = status.strip().lower()
        if status_lc == "mismatch":
            return 96 if claim_row else 92
        if status_lc == "verified":
            return 92 if claim_row else 86
        if status_lc == "unsupported":
            return 74
        return 60

    for fv in field_verdicts:
        if not isinstance(fv, dict):
            continue
        snippet = _compact_text(fv.get("evidence_excerpt") or fv.get("source_value"))
        if not snippet:
            continue
        field_path = str(fv.get("field_path") or "").strip()
        status = str(fv.get("status") or "not_checked").strip().lower()
        is_claim_row = verification_mode == "claim" and field_path in {"", "claim"}
        evidence_type = str(
            fv.get("evidence_type")
            or ("claim_evidence" if is_claim_row else "field_evidence")
        ).strip() or "field_evidence"
        rank = _status_rank(status, claim_row=is_claim_row)

        evidence_score = fv.get("evidence_score")
        if isinstance(evidence_score, (int, float)):
            if evidence_score <= 1.0:
                rank += int(round(max(0.0, evidence_score) * 10))
            else:
                rank += min(12, int(round(max(0.0, evidence_score))))

        citation: dict[str, Any] = {
            "source": str(fv.get("evidence_source") or "source_document"),
            "snippet": snippet,
            "evidence_type": evidence_type,
            "score": rank,
            "rank": rank,
            "status": status,
            "field_path": field_path or None,
            "claimed_value": _compact_text(fv.get("claimed_value")),
            "source_value": _compact_text(fv.get("source_value")),
        }
        if status in {"verified", "mismatch"}:
            citation["relevance"] = "high"
        elif status == "unsupported":
            citation["relevance"] = "medium"
        else:
            citation["relevance"] = "low"

        chunk_id = _to_int(fv.get("evidence_chunk_id"))
        start = _to_int(fv.get("evidence_start"))
        end = _to_int(fv.get("evidence_end"))
        if chunk_id is not None:
            citation["chunk_id"] = chunk_id
        if start is not None:
            citation["start"] = start
        if end is not None:
            citation["end"] = end

        citations.append(citation)

    for ev in evidence_items:
        if not isinstance(ev, dict):
            continue
        snippet = _compact_text(ev.get("contradicts") or ev.get("supports") or ev.get("excerpt"))
        if not snippet:
            continue
        rank = 78 if ev.get("supports") or ev.get("contradicts") else 70
        citations.append(
            {
                "source": str(ev.get("source") or "source_document"),
                "snippet": snippet,
                "evidence_type": "supporting_text",
                "score": rank,
                "rank": rank,
                "relevance": "medium",
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for citation in citations:
        key = (
            str(citation.get("source") or ""),
            str(citation.get("evidence_type") or ""),
            str(citation.get("snippet") or "")[:180],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)

    deduped.sort(
        key=lambda item: (
            int(item.get("rank", item.get("score", 0)) or 0),
            int(item.get("score", 0) or 0),
        ),
        reverse=True,
    )
    return deduped[: max(1, int(max_citations))]


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
    include_debug: bool = False,
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
    include_debug:
        Include verbose diagnostics (`issues`, `citations`, routing metadata).

    Returns
    -------
    dict
        Canonical verification payload with self-explanatory fields.
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

    verification_mode = "claim" if claim is not None and extraction is None else "extraction"
    out["citations"] = _verification_citations_from_report(
        report=out,
        verification_mode=verification_mode,
    )

    fallback_reason: str | None = None
    if fallback_used:
        fallback_issue = next(
            (
                i for i in out.get("issues", [])
                if i.get("type") in {"llm_unavailable", "rlm_unavailable"}
            ),
            None,
        )
        fallback_reason = (
            fallback_issue.get("detail")
                if fallback_issue is not None
                else f"Requested {level} but executed {effective_level}"
        )

    decision = out.get("verdict")
    claim_truth: bool | None = None
    grounded = bool(out.get("citations"))
    claim_comparison: dict[str, Any] | None = None
    adjudication_method: str | None = None
    if verification_mode == "claim" and claim is not None:
        claim_comparison = _claim_comparison_from_report(
            claim=claim,
            source_text=combined_source_text,
            report=out,
        )
        grounded = bool(claim_comparison.get("grounded"))

        claim_conflict = grounded and _claim_values_clearly_conflict(
            claim_comparison.get("claimed"),
            claim_comparison.get("source"),
        )
        claim_match = grounded and _claim_values_clearly_match(
            claim_comparison.get("claimed"),
            claim_comparison.get("source"),
        )

        if claim_conflict:
            decision = "contradicted"
            has_conflict_issue = any(
                isinstance(i, dict) and str(i.get("type") or "") in {"claim_value_conflict", "audit_mismatch"}
                for i in out.get("issues", [])
            )
            if not has_conflict_issue:
                out.setdefault("issues", []).append(
                    {
                        "type": "claim_value_conflict",
                        "field": "claim",
                        "detail": (
                            f"Claimed value ({_compact_text(claim_comparison.get('claimed'))}) "
                            f"conflicts with grounded source value ({_compact_text(claim_comparison.get('source'))})."
                        ),
                        "severity": "critical",
                    }
                )
        else:
            has_critical_issue = any(
                isinstance(i, dict) and str(i.get("severity") or "").lower() == "critical"
                for i in out.get("issues", [])
            )
            if claim_match and not has_critical_issue and str(decision or "") in {
                "partially_supported",
                "insufficient_evidence",
                "inconclusive",
            }:
                decision = "verified"
                out["match_rescued"] = True
                out["confidence"] = max(float(out.get("confidence") or 0.0), 0.85)
                filtered_issues: list[dict[str, Any]] = []
                for issue in out.get("issues", []):
                    if not isinstance(issue, dict):
                        filtered_issues.append(issue)
                        continue
                    if str(issue.get("severity") or "").lower() == "critical":
                        filtered_issues.append(issue)
                        continue
                    field = str(issue.get("field") or "").strip().lower()
                    detail = str(issue.get("detail") or "")
                    if field in {"", "claim"} and _looks_like_absence_statement(detail):
                        continue
                    filtered_issues.append(issue)
                out["issues"] = filtered_issues
                filtered_citations: list[dict[str, Any]] = []
                for citation in out.get("citations", []):
                    if (
                        isinstance(citation, dict)
                        and _looks_like_absence_statement(str(citation.get("snippet") or ""))
                    ):
                        continue
                    filtered_citations.append(citation)
                out["citations"] = filtered_citations
                grounded_snippet = _compact_text(
                    claim_comparison.get("evidence")
                    or claim_comparison.get("source")
                )
                if grounded_snippet:
                    out.setdefault("citations", [])
                    out["citations"] = [
                        {
                            "source": "source_document",
                            "snippet": grounded_snippet,
                            "evidence_type": "claim_evidence",
                            "score": 92,
                            "rank": 92,
                            "relevance": "high",
                            "claimed_value": _compact_text(claim_comparison.get("claimed")),
                            "source_value": _compact_text(claim_comparison.get("source")),
                        }
                    ] + list(out["citations"])

        # DSPy adjudication is only allowed for ambiguous grounded claims.
        ambiguous_grounded = (
            grounded
            and not claim_conflict
            and not claim_match
            and str(decision or "") in {"partially_supported", "insufficient_evidence", "inconclusive"}
        )
        if ambiguous_grounded:
            adjudicated = _adjudicate_claim_decision_with_dspy(
                claim=claim,
                claim_comparison=claim_comparison,
                current_decision=str(decision or "insufficient_evidence"),
                citations=list(out.get("citations") or []),
            )
            if adjudicated in {"verified", "contradicted", "insufficient_evidence"} and adjudicated != decision:
                decision = adjudicated
                adjudication_method = "dspy_mcc_bestofn"

        if not out.get("citations"):
            fallback_snippet = _compact_text(
                claim_comparison.get("evidence")
                or claim_comparison.get("source")
            )
            if fallback_snippet:
                out["citations"] = [
                    {
                        "source": "source_document",
                        "snippet": fallback_snippet,
                        "evidence_type": "claim_evidence",
                        "score": 70,
                        "rank": 70,
                        "relevance": "medium",
                        "claimed_value": _compact_text(claim_comparison.get("claimed")),
                        "source_value": _compact_text(claim_comparison.get("source")),
                    }
                ]

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
            claim_truth = True
        elif decision == "contradicted":
            claim_truth = False
    else:
        out["support_score"] = out.get("confidence")
    sources_consulted = sorted(
        {
            str(c.get("source") or "")
            for c in out.get("citations", [])
            if str(c.get("source") or "").strip()
        }
    )

    compact_issues: list[dict[str, Any]] = []
    for issue in out.get("issues", []):
        if not isinstance(issue, dict):
            continue
        compact_issues.append(
            {
                "type": str(issue.get("type") or "issue"),
                "severity": str(issue.get("severity") or "warning"),
                "field": str(issue.get("field") or ""),
                "message": str(issue.get("detail") or "").strip(),
            }
        )

    top_support_score = out.get("support_score")
    if top_support_score is None:
        top_support_score = out.get("confidence")

    result: dict[str, Any] = {
        "result": str(decision or "insufficient_evidence"),
        "confidence": float(out.get("confidence") or 0.0),
        "verify_type": verification_mode,
        "requested_mode": level,
        "executed_mode": str(effective_level or out.get("level") or ""),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "based_on_source": bool(grounded),
        "support_score": float(top_support_score or 0.0),
    }
    if verification_mode == "claim":
        result["claim_is_true"] = claim_truth
        result["claim"] = str(claim or "").strip()
        source_value = _compact_text((claim_comparison or {}).get("source"))
        evidence_value = _compact_text((claim_comparison or {}).get("evidence"))
        if source_value and _looks_like_runtime_failure(evidence_value):
            snippet_match = _source_snippet_for(combined_source_text, source_value)
            if snippet_match is not None:
                _matched, snippet = snippet_match
                evidence_value = snippet
            else:
                evidence_value = f"Source contains {source_value}."
        result["source_value"] = source_value
        result["evidence"] = evidence_value
    else:
        field_verdicts = [
            fv for fv in out.get("field_verdicts", [])
            if isinstance(fv, dict) and str(fv.get("status") or "").strip()
        ]
        result["field_checks"] = {
            "verified": sum(1 for fv in field_verdicts if str(fv.get("status")) == "verified"),
            "total": len(field_verdicts),
        }

    if include_debug:
        support_score = out.get("support_score")
        if support_score is None:
            support_score = out.get("confidence")
        debug_payload: dict[str, Any] = {
            "support_score": float(support_score or 0.0),
            "based_on_source": bool(grounded),
            "verify_type": verification_mode,
            "requested_mode": level,
            "executed_mode": str(effective_level or out.get("level") or ""),
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
            "issues": compact_issues,
            "citations": list(out.get("citations") or []),
            "sources_consulted": sources_consulted,
        }
        if adjudication_method:
            debug_payload["adjudication"] = adjudication_method
        result["debug"] = debug_payload

    return result


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
