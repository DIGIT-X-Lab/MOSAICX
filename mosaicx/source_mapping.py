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

import difflib
import re
from typing import Any

from .documents.models import LoadedDocument


def _normalize_textish(value: Any) -> str:
    """Normalize a scalar-ish value for loose equality checks."""
    text = str(value or "").strip()
    return " ".join(text.split()).lower()


def _normalize_compact_text(value: Any) -> str:
    """Normalize text for OCR block matching across spacing / punctuation drift."""
    text = str(value or "").strip().lower()
    return re.sub(r"[^0-9a-z]+", "", text)


_HEADER_DEMOGRAPHIC_FIELDS = {
    "patient_name",
    "name",
    "patient_age",
    "age",
    "patient_sex",
    "sex",
    "gender",
    "date_of_birth",
    "dob",
    "patient_id",
    "mrn",
    "uhid",
    "ip_no",
}


def _field_candidate_variants(field_key: str, value: Any) -> list[str]:
    """Expand short demographic fields into OCR-friendly match variants."""
    text = str(value or "").strip()
    if not text:
        return []

    variants = [text]
    key = field_key.lower()

    if key in {"patient_age", "age"} and text.isdigit():
        variants.extend([
            text,
            f"{text}y",
            f"{text} y",
            f"{text}yr",
            f"{text} yr",
            f"{text}yrs",
            f"{text} yrs",
            f"{text}year",
            f"{text} years",
        ])
    elif key in {"patient_sex", "sex", "gender"}:
        lowered = text.lower()
        if lowered.startswith("f"):
            variants.extend(["female", "f"])
        elif lowered.startswith("m"):
            variants.extend(["male", "m"])
    elif key in {"patient_name", "name"}:
        stripped = re.sub(r"^(mr|mrs|ms|miss|dr)\.?\s+", "", text, flags=re.IGNORECASE).strip()
        if stripped and stripped.lower() != text.lower():
            variants.append(stripped)

    seen: set[str] = set()
    out: list[str] = []
    for variant in variants:
        cleaned = str(variant or "").strip()
        if not cleaned:
            continue
        token = cleaned.lower()
        if token in seen:
            continue
        seen.add(token)
        out.append(cleaned)
    return out


def _block_priority_bonus(doc: LoadedDocument, block: Any, field_key: str, block_text: str) -> float:
    """Bias header demographic fields toward top-of-page header blocks."""
    key = field_key.lower()
    if key not in _HEADER_DEMOGRAPHIC_FIELDS:
        return 0.0

    bonus = 0.0
    page = int(getattr(block, "page", 1) or 1)
    if page == 1:
        bonus += 0.08

    bbox = getattr(block, "bbox", None) or [0, 0, 0, 0]
    page_dims = doc.page_dimensions or []
    if 1 <= page <= len(page_dims) and len(bbox) == 4:
        _page_w, page_h = page_dims[page - 1]
        if page_h:
            top_from_top = max(0.0, page_h - float(bbox[3])) / float(page_h)
            if top_from_top <= 0.25:
                bonus += 0.10
            elif top_from_top <= 0.40:
                bonus += 0.05

    compact = _normalize_compact_text(block_text)
    if key in {"patient_age", "age", "patient_sex", "sex", "gender"}:
        if re.search(r"\d", block_text) and ("female" in compact or "male" in compact or compact.endswith("f") or compact.endswith("m")):
            bonus += 0.10
    if key in {"patient_id", "mrn", "uhid", "ip_no"}:
        if re.search(r"\d{6,}", block_text):
            bonus += 0.06
        if any(token in compact for token in ("uhid", "ipno", "patientid", "mrn", "ipnumber")):
            bonus += 0.06

    return bonus


def _blocks_share_line(left: Any, right: Any) -> bool:
    """Return True when two OCR blocks are plausibly on the same text line."""
    if getattr(left, "page", None) != getattr(right, "page", None):
        return False

    lx0, ly0, lx1, ly1 = list(getattr(left, "bbox", None) or [0, 0, 0, 0])
    rx0, ry0, rx1, ry1 = list(getattr(right, "bbox", None) or [0, 0, 0, 0])
    lheight = max(1.0, ly1 - ly0)
    rheight = max(1.0, ry1 - ry0)
    lmid = (ly0 + ly1) / 2.0
    rmid = (ry0 + ry1) / 2.0
    vertical_gap = abs(lmid - rmid)
    overlap = min(ly1, ry1) - max(ly0, ry0)

    return (
        overlap >= 0.35 * min(lheight, rheight)
        or vertical_gap <= 0.65 * max(lheight, rheight)
    )


def _merge_spans(blocks: list[Any]) -> tuple[list[dict[str, Any]], str]:
    """Create one merged span across contiguous OCR blocks and combined text."""
    sorted_blocks = sorted(
        blocks,
        key=lambda block: (
            int(getattr(block, "page", 1) or 1),
            float((getattr(block, "bbox", [0, 0, 0, 0])[0])),
            int(getattr(block, "start", 0) or 0),
        ),
    )
    page = int(getattr(sorted_blocks[0], "page", 1) or 1)
    x0 = min(float(block.bbox[0]) for block in sorted_blocks)
    y0 = min(float(block.bbox[1]) for block in sorted_blocks)
    x1 = max(float(block.bbox[2]) for block in sorted_blocks)
    y1 = max(float(block.bbox[3]) for block in sorted_blocks)
    start = min(int(getattr(block, "start", 0) or 0) for block in sorted_blocks)
    end = max(int(getattr(block, "end", 0) or 0) for block in sorted_blocks)
    text = " ".join(str(getattr(block, "text", "") or "").strip() for block in sorted_blocks).strip()
    return [{
        "page": page,
        "bbox": [x0, y0, x1, y1],
        "start": start,
        "end": end,
    }], text


def _union_anchor(anchor: dict[str, Any] | None, spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Expand a demographic anchor with newly accepted spans on the same page."""
    if not spans:
        return anchor

    page = spans[0]["page"]
    x0 = min(float(span["bbox"][0]) for span in spans)
    y0 = min(float(span["bbox"][1]) for span in spans)
    x1 = max(float(span["bbox"][2]) for span in spans)
    y1 = max(float(span["bbox"][3]) for span in spans)
    if anchor is None:
        return {"page": page, "bbox": [x0, y0, x1, y1]}

    if int(anchor.get("page", page)) != int(page):
        return anchor

    ax0, ay0, ax1, ay1 = list(anchor.get("bbox", [x0, y0, x1, y1]))
    return {
        "page": page,
        "bbox": [
            min(float(ax0), x0),
            min(float(ay0), y0),
            max(float(ax1), x1),
            max(float(ay1), y1),
        ],
    }


def _anchor_priority_bonus(anchor: dict[str, Any] | None, spans: list[dict[str, Any]]) -> float:
    """Prefer demographic spans that stay inside the same header cluster."""
    if anchor is None or not spans:
        return 0.0

    page = int(spans[0]["page"])
    if int(anchor.get("page", page)) != page:
        return 0.0

    ax0, ay0, ax1, ay1 = list(anchor.get("bbox", [0, 0, 0, 0]))
    sx0 = min(float(span["bbox"][0]) for span in spans)
    sy0 = min(float(span["bbox"][1]) for span in spans)
    sx1 = max(float(span["bbox"][2]) for span in spans)
    sy1 = max(float(span["bbox"][3]) for span in spans)
    smid_y = (sy0 + sy1) / 2.0
    amid_y = (float(ay0) + float(ay1)) / 2.0
    vertical_gap = abs(smid_y - amid_y)
    overlap_y = min(float(ay1), sy1) - max(float(ay0), sy0)
    overlap_x = min(float(ax1), sx1) - max(float(ax0), sx0)

    bonus = 0.22
    if overlap_y >= 0:
        bonus += 0.12
    elif vertical_gap <= 24.0:
        bonus += 0.08

    if overlap_x >= -18.0:
        bonus += 0.06

    return bonus


def _candidate_units(doc: LoadedDocument, field_key: str) -> list[tuple[list[dict[str, Any]], str]]:
    """Return OCR units to rank against: single blocks plus header line windows."""
    units: list[tuple[list[dict[str, Any]], str]] = []
    if not doc.text_blocks:
        return units

    # Always keep single-block candidates.
    for block in doc.text_blocks:
        text = str(getattr(block, "text", "") or "").strip()
        if not text:
            continue
        units.append((
            [{
                "page": block.page,
                "bbox": list(block.bbox),
                "start": block.start,
                "end": block.end,
            }],
            text,
        ))

    key = field_key.lower()
    if key not in _HEADER_DEMOGRAPHIC_FIELDS:
        return units

    by_page: dict[int, list[Any]] = {}
    for block in doc.text_blocks:
        text = str(getattr(block, "text", "") or "").strip()
        if not text:
            continue
        by_page.setdefault(int(getattr(block, "page", 1) or 1), []).append(block)

    for page, page_blocks in by_page.items():
        sorted_blocks = sorted(
            page_blocks,
            key=lambda block: (
                -float((getattr(block, "bbox", [0, 0, 0, 0])[3])),
                float((getattr(block, "bbox", [0, 0, 0, 0])[0])),
                int(getattr(block, "start", 0) or 0),
            ),
        )

        rows: list[list[Any]] = []
        for block in sorted_blocks:
            placed = False
            for row in rows:
                if _blocks_share_line(row[-1], block):
                    row.append(block)
                    placed = True
                    break
            if not placed:
                rows.append([block])

        for row in rows:
            row.sort(key=lambda block: float((getattr(block, "bbox", [0, 0, 0, 0])[0])))
            n = len(row)
            for start_idx in range(n):
                window: list[Any] = [row[start_idx]]
                for end_idx in range(start_idx, n):
                    if end_idx > start_idx:
                        prev = row[end_idx - 1]
                        curr = row[end_idx]
                        gap = float(curr.bbox[0]) - float(prev.bbox[2])
                        row_height = max(
                            1.0,
                            max(float(item.bbox[3]) - float(item.bbox[1]) for item in window + [curr]),
                        )
                        if gap > max(18.0, row_height * 2.5):
                            break
                        window.append(curr)

                    if len(window) < 2:
                        continue

                    spans, text = _merge_spans(window)
                    if text:
                        units.append((spans, text))

    return units


def _find_best_text_block_span(
    doc: LoadedDocument,
    *,
    candidates: list[str],
    field_key: str = "",
    preferred_anchor: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str | None] | None:
    """Find the best matching OCR text block for a field.

    PPStructureV3 OCR often preserves the right block geometry while the
    reconstructed document text adds Markdown headings, collapses spaces, or
    duplicates repeated headers across pages.  For source highlighting we
    prefer matching directly against ``doc.text_blocks`` before falling back
    to global-text offset lookup.
    """
    if not doc.text_blocks:
        return None

    short_variant_fields = {"patient_age", "age", "patient_sex", "sex", "gender"}
    min_candidate_len = 2 if field_key.lower() in short_variant_fields else 4

    normalized_candidates: list[tuple[str, str, str]] = []
    for raw in candidates:
        text = str(raw or "").strip()
        compact = _normalize_compact_text(text)
        if len(compact) < min_candidate_len:
            continue
        normalized_candidates.append((text, compact, _normalize_textish(text)))

    if not normalized_candidates:
        return None

    best: tuple[float, int, int, list[dict[str, Any]], str] | None = None

    for spans, block_text in _candidate_units(doc, field_key):
        if not block_text or not spans:
            continue

        block_compact = _normalize_compact_text(block_text)
        if len(block_compact) < 4:
            continue

        best_score = 0.0
        best_excerpt = None
        for original, cand_compact, _cand_loose in normalized_candidates:
            score = 0.0
            if cand_compact in block_compact:
                # Strong exact-ish match after stripping punctuation / spaces.
                # Prefer tighter OCR regions over merged row windows that only
                # contain the value as a substring. Earlier versions capped
                # both the tight block and the larger row at the same score,
                # which let a larger merged row win on tie-break.
                tightness = min(1.0, len(cand_compact) / max(len(block_compact), 1))
                score = 1.0 + 0.5 * tightness
            elif block_compact in cand_compact and len(block_compact) >= max(4, len(cand_compact) // 2):
                tightness = min(1.0, len(block_compact) / max(len(cand_compact), 1))
                score = 0.88 + 0.08 * tightness
            else:
                ratio = difflib.SequenceMatcher(
                    None,
                    cand_compact,
                    block_compact,
                    autojunk=False,
                ).ratio()
                if ratio >= 0.9:
                    score = ratio

            if score > best_score:
                best_score = score
                best_excerpt = original

        if best_score <= 0 or best_excerpt is None:
            continue

        unit = type("SourceUnit", (), {
            "page": spans[0]["page"],
            "bbox": spans[0]["bbox"],
            "start": spans[0]["start"],
        })()
        best_score += _block_priority_bonus(doc, unit, field_key, block_text)
        best_score += _anchor_priority_bonus(preferred_anchor, spans)

        candidate = (best_score, spans[0]["page"], spans[0]["start"], spans, best_excerpt)
        if best is None:
            best = candidate
            continue

        prev_score, prev_page, prev_start, *_ = best
        # Prefer higher score, then earlier page, then earlier block position.
        if (
            best_score > prev_score
            or (
                abs(best_score - prev_score) < 1e-6
                and (
                    spans[0]["page"] < prev_page
                    or (spans[0]["page"] == prev_page and spans[0]["start"] < prev_start)
                )
            )
        ):
            best = candidate

    if best is None:
        return None

    _score, _page, _start, spans, excerpt = best
    return spans, excerpt


def _derive_source_value_metadata(
    *,
    value: Any,
    llm_excerpt: str,
    field_evidence: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Derive raw source value and canonicalization metadata.

    v1 intentionally keeps this simple:
    - if no LLM excerpt exists, we do not infer a distinct source value
    - if the normalized extracted value already appears in the LLM excerpt,
      we treat the source value as identical to the extracted value
    - otherwise the LLM excerpt becomes the raw source value and we record
      that the final value came from LLM-side canonicalization
    """
    if value is None:
        return None, None

    value_text = str(value).strip()
    if not value_text:
        return None, None

    raw_excerpt = str(llm_excerpt or "").strip()
    if not raw_excerpt:
        return value, None

    if isinstance(field_evidence, dict):
        explicit_source_value = field_evidence.get("source_value")
        explicit_canonicalization = field_evidence.get("canonicalization")
        if explicit_source_value not in (None, ""):
            raw_excerpt = str(explicit_source_value).strip() or raw_excerpt
        if isinstance(explicit_canonicalization, dict):
            return raw_excerpt, dict(explicit_canonicalization)

    value_norm = _normalize_textish(value_text)
    excerpt_norm = _normalize_textish(raw_excerpt)

    if value_norm and value_norm in excerpt_norm:
        return value, None

    return raw_excerpt, {
        "applied": True,
        "method": "llm_extraction",
        "from": raw_excerpt,
        "to": value,
    }


def _date_alternatives(value: str) -> list[str]:
    """Generate alternative date formats for an ISO or common date string."""
    import datetime

    alternatives: list[str] = []
    # Try parsing ISO format (YYYY-MM-DD)
    for fmt_in in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            dt = datetime.datetime.strptime(value.strip(), fmt_in)
            # Generate common output formats
            alternatives.extend([
                dt.strftime("%B %d, %Y"),        # March 15, 1985
                dt.strftime("%B %d,%Y"),         # March 15,1985
                dt.strftime("%b %d, %Y"),        # Mar 15, 1985
                dt.strftime("%d %B %Y"),         # 15 March 1985
                dt.strftime("%d.%m.%Y"),         # 15.03.1985
                dt.strftime("%d-%m-%Y"),         # 15-03-1985
                dt.strftime("%m/%d/%Y"),         # 03/15/1985
                dt.strftime("%d/%m/%Y"),         # 15/03/1985
                dt.strftime("%Y-%m-%d"),         # 1985-03-15
                dt.strftime("%d %b %Y"),         # 15 Mar 1985
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

    # Fuzzy fallback: use SequenceMatcher for near-matches
    if idx < 0 and len(needle_norm) >= 4:
        from .pipelines.provenance import _fuzzy_find
        fuzzy_result = _fuzzy_find(text_norm, needle_norm, [])
        if fuzzy_result:
            idx = fuzzy_result[0]
            # Use the matched text as the needle for original-text lookup
            needle_norm = text_norm[fuzzy_result[0]:fuzzy_result[1]]
            needle = needle_norm

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
    field_evidence: dict[str, dict[str, Any]] | None = None,
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
    field_evidence:
        Optional dict of ``{field_key: {"excerpt": ..., "reasoning": ...}}``
        from the structured extractor. When provided, the LLM's verbatim
        excerpt is used for grounding before falling back to the normalized
        extracted value. Additional metadata such as ``evidence_selection``
        is preserved on the resulting field entry.

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
        guide["matcher"] = "ocr_block_v3"
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

        ev = field_evidence or {}
        demographic_anchor: dict[str, Any] | None = None

        for field_key, value in flat.items():
            if field_key.startswith("_"):
                continue

            val_str = str(value) if value is not None else ""

            # Try grounding with LLM's verbatim excerpt first (more
            # reliable than the reformatted extracted value), then fall
            # back to the extracted value itself.
            llm_excerpt = ""
            field_ev = ev.get(field_key, {}) if isinstance(ev, dict) else {}
            if isinstance(field_ev, dict):
                llm_excerpt = field_ev.get("excerpt", "") or ""

            excerpt, start, end = None, None, None
            spans: list[dict[str, Any]] = []
            explicit_source_value = ""
            if isinstance(field_ev, dict):
                explicit_source_value = str(field_ev.get("source_value") or "").strip()

            candidates = [llm_excerpt, explicit_source_value, val_str]
            for expanded in _field_candidate_variants(field_key, value):
                candidates.append(expanded)

            block_match = _find_best_text_block_span(
                doc,
                candidates=candidates,
                field_key=field_key,
                preferred_anchor=demographic_anchor if field_key.lower() in _HEADER_DEMOGRAPHIC_FIELDS else None,
            )
            if block_match is not None:
                spans, matched_excerpt = block_match
                excerpt = llm_excerpt or matched_excerpt
                if spans:
                    start = spans[0].get("start")
                    end = spans[0].get("end")

            if excerpt is None and llm_excerpt:
                excerpt, start, end = _find_tight_excerpt(doc.text, llm_excerpt)
            if excerpt is None and val_str:
                excerpt, start, end = _find_tight_excerpt(doc.text, val_str)

            if not spans and start is not None and end is not None and doc.text_blocks:
                raw_spans = locate_in_document(doc, start, end)
                if raw_spans:
                    spans = raw_spans

            entry: dict[str, Any] = {"value": value}
            source_value, canonicalization = _derive_source_value_metadata(
                value=value,
                llm_excerpt=llm_excerpt,
                field_evidence=field_ev if isinstance(field_ev, dict) else None,
            )
            if source_value is not None:
                entry["source_value"] = source_value
            if excerpt:
                entry["excerpt"] = excerpt
            entry["grounded"] = excerpt is not None
            entry["spans"] = spans
            # Merge reasoning and LLM excerpt from deep think evidence
            if isinstance(field_ev, dict):
                if field_ev.get("reasoning"):
                    entry["reasoning"] = field_ev["reasoning"]
                if llm_excerpt:
                    entry["llm_excerpt"] = llm_excerpt
                if isinstance(field_ev.get("evidence_selection"), dict):
                    entry["evidence_selection"] = dict(field_ev["evidence_selection"])
            if canonicalization is not None:
                entry["canonicalization"] = canonicalization
            source_fields[field_key] = entry

            lowered = field_key.lower()
            if entry["grounded"] and spans and lowered in {"patient_name", "name", "patient_id", "mrn", "uhid", "ip_no", "sex", "gender"}:
                demographic_anchor = _union_anchor(demographic_anchor, spans)

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


def _build_phi_summary(phi_items: list[dict[str, Any]]) -> str:
    """Build a one-line summary of detected PHI items."""
    from collections import Counter

    counts = Counter(item.get("type", "OTHER") for item in phi_items)
    parts = []
    for phi_type in sorted(counts):
        n = counts[phi_type]
        label = phi_type.lower() + ("s" if n > 1 else "")
        parts.append(f"{n} {label}")
    total = len(phi_items)
    return f"{total} PHI item{'s' if total != 1 else ''} redacted: {', '.join(parts)}"


def strip_for_open_source(output_data: dict[str, Any]) -> dict[str, Any]:
    """Strip a full extraction/deidentification JSON to open-source tier.

    **Extraction:** keeps ``extracted`` + ``_evidence`` (dict of field
    name to reasoning + excerpt).

    **Deidentification:** keeps ``redacted_text`` + ``phi`` (list of
    detected PHI items with original, type, replacement, excerpt,
    reasoning) + ``summary``.

    Removes ``_source`` (coordinates), ``_extraction_contract``,
    ``_planner``, ``redaction_map``, and other internal diagnostics.
    """
    source = output_data.get("_source", {})
    fields = source.get("fields", {})
    is_deid = "redacted_text" in output_data

    if is_deid:
        # Deidentification: flat list of PHI items
        result: dict[str, Any] = {
            "redacted_text": output_data["redacted_text"],
            "mode": output_data.get("mode", "remove"),
        }
        phi_items: list[dict[str, Any]] = []
        for info in fields.values():
            item: dict[str, Any] = {
                "original": info.get("value", ""),
                "type": info.get("phi_type", "OTHER"),
                "replacement": info.get("replacement", "[REDACTED]"),
            }
            if info.get("excerpt"):
                item["excerpt"] = info["excerpt"]
            if info.get("reasoning"):
                item["reasoning"] = info["reasoning"]
            phi_items.append(item)

        if phi_items:
            result["summary"] = _build_phi_summary(phi_items)
            result["phi_types"] = sorted({item["type"] for item in phi_items})
            result["phi"] = phi_items

        return result

    # Extraction: dict keyed by field name
    result = {}
    if "extracted" in output_data:
        result["extracted"] = output_data["extracted"]
    if isinstance(output_data.get("_run"), dict):
        result["_run"] = output_data["_run"]

    if fields:
        evidence: dict[str, dict[str, str]] = {}
        for key, info in fields.items():
            ev: dict[str, str] = {}
            if info.get("llm_excerpt"):
                ev["excerpt"] = info["llm_excerpt"]
            elif info.get("excerpt"):
                ev["excerpt"] = info["excerpt"]
            if info.get("reasoning"):
                ev["reasoning"] = info["reasoning"]
            if ev:
                evidence[key] = ev
        if evidence:
            result["_evidence"] = evidence

    return result
