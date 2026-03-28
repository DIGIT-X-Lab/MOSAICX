# mosaicx/pipelines/provenance.py
"""Coordinate-based source provenance utilities."""

from __future__ import annotations

import bisect
import difflib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mosaicx.documents.models import LoadedDocument


def locate_in_document(
    doc: LoadedDocument,
    start: int,
    end: int,
) -> list[dict] | None:
    """Map a char range to page bboxes in a LoadedDocument.

    Uses binary search to find all TextBlocks that overlap [start, end),
    groups them by page, and returns the union bbox for each page.

    Parameters
    ----------
    doc:
        A LoadedDocument with a populated ``text_blocks`` list.
    start:
        Inclusive start char offset in the full document text.
    end:
        Exclusive end char offset in the full document text.

    Returns
    -------
    list[dict] | None
        One entry per page touched by the range, each with keys
        ``page`` (int), ``bbox`` (tuple), ``start`` (int), ``end`` (int).
        Returns ``None`` if ``doc.text_blocks`` is empty.
    """
    if not doc.text_blocks:
        return None

    blocks = doc.text_blocks

    # Build a sorted list of block start offsets for binary search.
    # Blocks are assumed to be sorted by start; if not, sort defensively.
    starts = [b.start for b in blocks]

    # Find the first block whose *end* is > query start (i.e., could overlap).
    # We look for blocks where b.end > start AND b.start < end.
    # Binary search to skip blocks that end before our range.
    # bisect_left finds first index where starts[i] >= start;
    # but blocks before that index might still have end > start, so
    # we walk back one position to catch partial overlaps.
    right_idx = bisect.bisect_left(starts, end)  # first block starting at or after end
    left_idx = bisect.bisect_right(starts, start) - 1  # last block starting <= start
    left_idx = max(left_idx, 0)

    overlapping: list = []
    for block in blocks[left_idx:right_idx]:
        # Overlap condition: block.start < end AND block.end > start
        if block.start < end and block.end > start:
            overlapping.append(block)

    if not overlapping:
        return None

    # Group by page and compute union bbox per page.
    by_page: dict[int, list] = {}
    for block in overlapping:
        by_page.setdefault(block.page, []).append(block)

    result: list[dict] = []
    for page_num in sorted(by_page):
        page_blocks = by_page[page_num]
        x0 = min(b.bbox[0] for b in page_blocks)
        y0 = min(b.bbox[1] for b in page_blocks)
        x1 = max(b.bbox[2] for b in page_blocks)
        y1 = max(b.bbox[3] for b in page_blocks)
        result.append(
            {
                "page": page_num,
                "bbox": (x0, y0, x1, y1),
                "start": start,
                "end": end,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Task 5: resolve_provenance() — exact + fuzzy evidence resolution
# ---------------------------------------------------------------------------

_FUZZY_THRESHOLD_SHORT = 0.90  # excerpts < 40 chars
_FUZZY_THRESHOLD_LONG = 0.80   # excerpts >= 40 chars
_SHORT_EXCERPT_LEN = 40


def _fuzzy_find(
    text: str,
    excerpt: str,
    claimed: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Slide a window over *text* looking for the best fuzzy match for *excerpt*.

    Uses :class:`difflib.SequenceMatcher` with a similarity threshold that
    depends on the excerpt length: 0.90 for short excerpts (< 40 chars) and
    0.80 for longer ones.

    Already-claimed ranges are skipped so that duplicate excerpts each land
    on a distinct location.

    Parameters
    ----------
    text:
        Full document text to search.
    excerpt:
        The excerpt string to match.
    claimed:
        List of (start, end) ranges already assigned to other fields.

    Returns
    -------
    tuple[int, int] | None
        (start, end) of the best-matching window, or ``None`` if no window
        exceeds the similarity threshold.
    """
    n = len(excerpt)
    if n == 0:
        return None

    threshold = _FUZZY_THRESHOLD_SHORT if n < _SHORT_EXCERPT_LEN else _FUZZY_THRESHOLD_LONG

    best_ratio = 0.0
    best_pos: tuple[int, int] | None = None

    # Slide a window of the same length as the excerpt across the text.
    # Allow a small size variation (±20%) to handle insertions/deletions.
    min_win = max(1, int(n * 0.8))
    max_win = int(n * 1.2) + 1

    for win_len in range(min_win, min(max_win, len(text) + 1)):
        for i in range(len(text) - win_len + 1):
            # Skip windows that overlap an already-claimed range
            w_end = i + win_len
            overlaps = any(
                not (w_end <= cs or i >= ce) for cs, ce in claimed
            )
            if overlaps:
                continue

            window = text[i:w_end]
            ratio = difflib.SequenceMatcher(None, excerpt, window, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = (i, w_end)

    if best_ratio >= threshold and best_pos is not None:
        return best_pos
    return None


def _resolve_single(
    doc: LoadedDocument,
    excerpt: str,
    claimed: list[tuple[int, int]],
) -> dict[str, Any]:
    """Resolve a single *excerpt* to source coordinates within *doc*.

    Tries exact match first, then fuzzy match. If neither succeeds, returns
    an unresolved entry.

    Parameters
    ----------
    doc:
        The loaded document with full text and optional text blocks.
    excerpt:
        The excerpt string to locate.
    claimed:
        Mutable list of (start, end) ranges already assigned; updated in place
        when a match is found to prevent duplicate assignments.

    Returns
    -------
    dict
        Keys: ``excerpt``, ``start``, ``end``, ``spans``, ``resolution``.
    """
    # --- Tier 1: exact match ------------------------------------------------
    pos = doc.text.find(excerpt)
    if pos != -1:
        start, end = pos, pos + len(excerpt)
        # Check for duplicate claim — if this exact range is already taken,
        # search for the next occurrence.
        search_from = 0
        while pos != -1:
            already = any(
                not (end <= cs or start >= ce) for cs, ce in claimed
            )
            if not already:
                break
            search_from = pos + 1
            pos = doc.text.find(excerpt, search_from)
            if pos != -1:
                start, end = pos, pos + len(excerpt)
        else:
            # All occurrences claimed — fall through to fuzzy
            pos = -1

        if pos != -1:
            claimed.append((start, end))
            spans = locate_in_document(doc, start, end) or []
            return {
                "excerpt": excerpt,
                "start": start,
                "end": end,
                "spans": spans,
                "resolution": "exact",
            }

    # --- Tier 2: fuzzy match ------------------------------------------------
    match = _fuzzy_find(doc.text, excerpt, claimed)
    if match is not None:
        start, end = match
        claimed.append((start, end))
        spans = locate_in_document(doc, start, end) or []
        return {
            "excerpt": excerpt,
            "start": start,
            "end": end,
            "spans": spans,
            "resolution": "fuzzy",
        }

    # --- Tier 3: unresolved -------------------------------------------------
    return {
        "excerpt": excerpt,
        "start": -1,
        "end": -1,
        "spans": [],
        "resolution": "unresolved",
    }


def resolve_provenance(
    doc: LoadedDocument,
    evidence: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Resolve a mapping of field names to excerpt strings into source coordinates.

    For each field in *evidence*, locates the excerpt within *doc* using
    exact match (tier 1), fuzzy match (tier 2), or marks it as unresolved
    (tier 3).

    Parameters
    ----------
    doc:
        A :class:`~mosaicx.documents.models.LoadedDocument`.
    evidence:
        Dict mapping field names to verbatim excerpt strings from the document.

    Returns
    -------
    dict[str, dict]
        One entry per field with keys: ``excerpt``, ``start``, ``end``,
        ``spans`` (list of {page, bbox}), and ``resolution``
        (``"exact"``, ``"fuzzy"``, or ``"unresolved"``).
    """
    claimed: list[tuple[int, int]] = []
    result: dict[str, dict[str, Any]] = {}
    for field_name, excerpt in evidence.items():
        result[field_name] = _resolve_single(doc, excerpt, claimed)
    return result


# ---------------------------------------------------------------------------
# Task 7: enrich_redaction_map — add coordinates to existing map entries
# ---------------------------------------------------------------------------


def enrich_redaction_map(
    doc: LoadedDocument,
    redaction_map: list[dict[str, Any]],
    context_chars: int = 30,
) -> list[dict[str, Any]]:
    """Add source coordinates to each entry in a redaction map.

    For each entry, calls :func:`locate_in_document` using the entry's
    existing ``start``/``end`` offsets, adds ``spans``, ``excerpt`` (the
    redacted text with surrounding context), and ``resolution``.

    Parameters
    ----------
    doc:
        A :class:`~mosaicx.documents.models.LoadedDocument`.
    redaction_map:
        List of redaction mapping dicts as produced by
        :class:`~mosaicx.pipelines.deidentifier.Deidentifier`.  Each entry
        must have ``start`` and ``end`` keys referencing offsets in
        ``doc.text``.
    context_chars:
        Number of characters of surrounding context to include in the
        ``excerpt`` field.

    Returns
    -------
    list[dict]
        A new list of enriched dicts; the originals are not mutated.
    """
    enriched: list[dict[str, Any]] = []
    for entry in redaction_map:
        start: int = entry["start"]
        end: int = entry["end"]

        ctx_start = max(0, start - context_chars)
        ctx_end = min(len(doc.text), end + context_chars)
        excerpt = doc.text[ctx_start:ctx_end]

        spans = locate_in_document(doc, start, end) or []

        enriched_entry = dict(entry)
        enriched_entry["spans"] = spans
        enriched_entry["excerpt"] = excerpt
        enriched_entry["resolution"] = "located"
        enriched.append(enriched_entry)

    return enriched


# ---------------------------------------------------------------------------
# Task 6: GatherEvidence DSPy Signature (lazy loading)
# ---------------------------------------------------------------------------


def _build_dspy_classes() -> dict[str, type]:
    """Lazily build and return the GatherEvidence DSPy Signature class."""
    import dspy

    class GatherEvidence(dspy.Signature):
        """Given a source document and extracted fields, find the verbatim
        excerpt from the source document that supports each field value."""

        document_text: str = dspy.InputField(
            desc="Full text of the source document"
        )
        extracted_fields: str = dspy.InputField(
            desc="JSON of field names and their extracted values"
        )
        evidence: str = dspy.OutputField(
            desc="JSON mapping each field name to the verbatim excerpt from document_text"
        )

    return {"GatherEvidence": GatherEvidence}


_provenance_dspy_classes: dict[str, type] | None = None
_PROVENANCE_CLASS_NAMES = frozenset({"GatherEvidence"})


def __getattr__(name: str) -> Any:
    """Module-level __getattr__ for lazy loading of provenance DSPy classes."""
    global _provenance_dspy_classes
    if name in _PROVENANCE_CLASS_NAMES:
        if _provenance_dspy_classes is None:
            _provenance_dspy_classes = _build_dspy_classes()
        return _provenance_dspy_classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# gather_evidence() convenience function
# ---------------------------------------------------------------------------


def gather_evidence(
    document_text: str,
    extracted_fields: dict[str, Any],
) -> dict[str, str]:
    """Run the GatherEvidence DSPy step and return parsed evidence dict.

    Parameters
    ----------
    document_text:
        Full text of the source document.
    extracted_fields:
        Dict of field names to their extracted values.

    Returns
    -------
    dict[str, str]
        Mapping of field names to verbatim excerpt strings.  Returns an empty
        dict if the model output cannot be parsed as JSON.
    """
    import dspy

    # Access lazy-loaded class via module __getattr__
    import mosaicx.pipelines.provenance as _self
    _GatherEvidence = getattr(_self, "GatherEvidence")
    predict = dspy.Predict(_GatherEvidence)
    result = predict(
        document_text=document_text,
        extracted_fields=json.dumps(extracted_fields),
    )
    try:
        return json.loads(result.evidence)
    except (json.JSONDecodeError, AttributeError):
        return {}
