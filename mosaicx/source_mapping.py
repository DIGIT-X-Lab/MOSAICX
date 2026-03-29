# mosaicx/source_mapping.py
"""Unified ``_source`` block builder for extraction and deidentification.

Produces a self-documenting JSON structure that maps every extracted field
or redacted PHI item back to its exact location in the source document
(page number + bounding box).  The ``_guide`` section explains how to
transform coordinates for PDF overlay or image rendering.

Usage (both pipelines)::

    from mosaicx.source_mapping import build_source_block

    output["_source"] = build_source_block(doc, fields={
        "patient_name": "Sarah Johnson",
        "findings[0].location": "right upper lobe",
    })
"""

from __future__ import annotations

import re
from typing import Any

from .documents.models import LoadedDocument


def _date_alternatives(value: str) -> list[str]:
    """Generate alternative date formats for an ISO or common date string."""
    import datetime

    alternatives: list[str] = []
    # Try parsing ISO format (YYYY-MM-DD)
    for fmt_in in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%m/%d/%Y"):
        try:
            dt = datetime.datetime.strptime(value.strip(), fmt_in)
            # Generate common output formats
            alternatives.extend([
                dt.strftime("%B %d, %Y"),        # March 15, 1985
                dt.strftime("%B %d,%Y"),         # March 15,1985
                dt.strftime("%b %d, %Y"),        # Mar 15, 1985
                dt.strftime("%d %B %Y"),         # 15 March 1985
                dt.strftime("%d.%m.%Y"),         # 15.03.1985
                dt.strftime("%m/%d/%Y"),         # 03/15/1985
                dt.strftime("%d/%m/%Y"),         # 15/03/1985
                dt.strftime("%Y-%m-%d"),         # 1985-03-15
            ])
            break
        except ValueError:
            continue
    return alternatives


def _find_tight_excerpt(
    source_text: str,
    value: str,
) -> tuple[str | None, int | None, int | None]:
    """Find a tight excerpt for *value* in *source_text*.

    Returns ``(excerpt, start, end)`` where excerpt is the full line
    containing the value, trimmed to be readable but short.
    Returns ``(None, None, None)`` if not found.

    Handles date reformatting: if the value is ``"1985-03-15"`` but the
    document says ``"March 15, 1985"``, it tries alternative date formats.
    """
    if not value or not source_text:
        return None, None, None

    needle = str(value).strip()
    if not needle:
        return None, None, None

    # Normalize whitespace for matching
    text_norm = " ".join(source_text.split())
    needle_norm = " ".join(needle.split())

    idx = text_norm.find(needle_norm)
    if idx < 0:
        # Case-insensitive fallback
        idx = text_norm.lower().find(needle_norm.lower())

    # If not found, try alternative date formats
    if idx < 0:
        for alt in _date_alternatives(needle):
            alt_norm = " ".join(alt.split())
            idx = text_norm.find(alt_norm)
            if idx < 0:
                idx = text_norm.lower().find(alt_norm.lower())
            if idx >= 0:
                needle = alt
                needle_norm = alt_norm
                break

    if idx < 0:
        return None, None, None

    # Now find the match in the original (non-normalized) text.
    # Map normalized index back to original by counting chars.
    orig_idx = source_text.find(needle)
    if orig_idx < 0:
        # Try case-insensitive on original
        orig_idx = source_text.lower().find(needle.lower())
    if orig_idx < 0:
        # Use normalized position as approximation
        orig_idx = idx

    start = orig_idx
    end = orig_idx + len(needle)

    # Build tight excerpt: extend to nearest line boundaries
    line_start = source_text.rfind("\n", 0, start)
    line_start = 0 if line_start < 0 else line_start + 1
    line_end = source_text.find("\n", end)
    line_end = len(source_text) if line_end < 0 else line_end

    excerpt = source_text[line_start:line_end].strip()
    # Cap at 120 chars
    if len(excerpt) > 120:
        # Center on the value
        val_offset = start - line_start
        trim_start = max(0, val_offset - 30)
        trim_end = min(len(excerpt), val_offset + len(needle) + 30)
        excerpt = excerpt[trim_start:trim_end].strip()

    return excerpt, start, end


def _flatten_dict(
    d: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten a nested dict/list to dot-path keys.

    ``{"a": {"b": 1}, "items": [{"x": 2}]}``
    becomes ``{"a.b": 1, "items[0].x": 2}``
    """
    out: dict[str, Any] = {}
    for key, val in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            out.update(_flatten_dict(val, path))
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    out.update(_flatten_dict(item, f"{path}[{i}]"))
                else:
                    out[f"{path}[{i}]"] = item
        else:
            out[path] = val
    return out


def build_source_block(
    doc: LoadedDocument,
    fields: dict[str, Any] | None = None,
    redaction_map: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the unified ``_source`` block for JSON output.

    Call with *fields* for extraction output, or *redaction_map* for
    deidentification output.  Both produce the same structure.

    Parameters
    ----------
    doc:
        The loaded document with text_blocks and page_dimensions.
    fields:
        For extraction: dict of ``{field_key: value}`` (flat or nested —
        nested dicts/lists are auto-flattened to dot-path keys).
    redaction_map:
        For deidentification: list of redaction entry dicts with
        ``original``, ``replacement``, ``phi_type``, and optionally
        ``spans``.

    Returns
    -------
    dict
        The ``_source`` block ready to insert into the output JSON.
    """
    from .pipelines.provenance import locate_in_document

    # Build the guide
    page_dims = doc.page_dimensions or []
    is_image = doc.format in ("jpg", "jpeg", "png", "tiff", "tif", "bmp")
    is_text = doc.format in ("txt", "md", "markdown")

    guide: dict[str, Any] = {
        "version": "1.0",
        "bbox_format": "[x0, y0, x1, y1]",
    }

    if is_text:
        guide["coordinate_space"] = "none"
        guide["how_to_use"] = (
            "Text files have no spatial coordinates. "
            "Fields have excerpts and grounded status but spans will be empty."
        )
    elif is_image:
        guide["coordinate_space"] = "image_pixels"
        guide["origin"] = "top-left"
        guide["page_dimensions"] = [list(d) for d in page_dims]
        guide["how_to_use"] = (
            "Bbox coordinates are image pixels (origin top-left). "
            "Use spans[].bbox directly for overlay rectangles."
        )
    else:
        guide["coordinate_space"] = "pdf_points"
        guide["origin"] = "bottom-left"
        guide["render_dpi"] = 200
        guide["page_dimensions"] = [list(d) for d in page_dims]
        guide["to_fitz_rect"] = "fitz.Rect(x0, page_h - y1, x1, page_h - y0)"
        guide["to_image_px"] = (
            "scale = render_dpi / 72; "
            "(x0 * scale, (page_h - y1) * scale, x1 * scale, (page_h - y0) * scale)"
        )
        guide["how_to_use"] = (
            "Each field has 'spans' with page number and bbox in PDF points. "
            "Use page_dimensions[page-1] for (width, height). "
            "Transform bbox with to_fitz_rect for PyMuPDF overlay, "
            "or to_image_px for rendered image overlay at render_dpi."
        )

    # Build fields
    source_fields: dict[str, dict[str, Any]] = {}

    if fields is not None:
        # Extraction mode: flatten nested dicts to dot-path keys
        if any(isinstance(v, (dict, list)) for v in fields.values()):
            flat = _flatten_dict(fields)
        else:
            flat = dict(fields)

        for field_key, value in flat.items():
            if field_key.startswith("_"):
                continue

            val_str = str(value) if value is not None else ""
            excerpt, start, end = _find_tight_excerpt(doc.text, val_str)

            spans: list[dict[str, Any]] = []
            if start is not None and end is not None and doc.text_blocks:
                raw_spans = locate_in_document(doc, start, end)
                if raw_spans:
                    spans = raw_spans

            entry: dict[str, Any] = {"value": value}
            if excerpt:
                entry["excerpt"] = excerpt
            entry["grounded"] = excerpt is not None
            entry["spans"] = spans
            source_fields[field_key] = entry

    elif redaction_map is not None:
        # Deidentification mode
        for i, item in enumerate(redaction_map):
            original = item.get("original", "")
            phi_type = item.get("phi_type", "UNKNOWN")
            replacement = item.get("replacement", "[REDACTED]")

            # Use a readable key
            field_key = f"{phi_type.lower()}_{i}"

            # Get or build spans
            existing_spans = item.get("spans", [])
            if existing_spans:
                spans = existing_spans
            elif original and doc.text_blocks:
                excerpt, start, end = _find_tight_excerpt(doc.text, original)
                if start is not None and end is not None:
                    raw_spans = locate_in_document(doc, start, end)
                    spans = raw_spans or []
                else:
                    spans = []
            else:
                spans = []

            # Tight excerpt
            excerpt, _, _ = _find_tight_excerpt(doc.text, original)

            entry = {
                "value": original,
                "replacement": replacement,
                "phi_type": phi_type,
            }
            if excerpt:
                entry["excerpt"] = excerpt
            # Pass through LLM reasoning/excerpt if present
            if item.get("reasoning"):
                entry["reasoning"] = item["reasoning"]
            if item.get("excerpt") and not excerpt:
                entry["excerpt"] = item["excerpt"]
            entry["grounded"] = bool(spans)
            entry["spans"] = spans
            source_fields[field_key] = entry

    result: dict[str, Any] = {
        "_guide": guide,
        "fields": source_fields,
    }
    return result


def strip_for_open_source(output_data: dict[str, Any]) -> dict[str, Any]:
    """Strip a full extraction/deidentification JSON to open-source tier.

    Keeps ``extracted`` + ``_evidence`` (reasoning + excerpt per field).
    Removes ``_source`` (coordinates), ``_extraction_contract``,
    ``_planner``, and other internal diagnostics.

    The ``_evidence`` block is built from ``_source.fields`` — each field
    gets its ``reasoning`` and ``excerpt`` but no coordinates or spans.
    """
    result: dict[str, Any] = {}

    # Keep the primary output
    if "extracted" in output_data:
        result["extracted"] = output_data["extracted"]
    if "redacted_text" in output_data:
        result["redacted_text"] = output_data["redacted_text"]
    if "mode" in output_data:
        result["mode"] = output_data["mode"]

    # Build _evidence from _source.fields (reasoning + excerpt only)
    source = output_data.get("_source", {})
    fields = source.get("fields", {})
    if fields:
        evidence: dict[str, dict[str, str]] = {}
        for key, info in fields.items():
            ev: dict[str, str] = {}
            if info.get("excerpt"):
                ev["excerpt"] = info["excerpt"]
            if info.get("reasoning"):
                ev["reasoning"] = info["reasoning"]
            if info.get("llm_excerpt"):
                ev["excerpt"] = info["llm_excerpt"]
            if ev:
                evidence[key] = ev
        if evidence:
            result["_evidence"] = evidence

    return result
