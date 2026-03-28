# mosaicx/pipelines/deidentifier.py
"""De-identification pipeline with LLM + regex belt-and-suspenders.

A 2-layer approach to removing Protected Health Information (PHI) from
medical documents:

    1. **LLM redaction** (via DSPy) -- A language model identifies and
       removes context-dependent PHI (names, addresses, dates of birth,
       etc.) that cannot be reliably caught by patterns alone.
    2. **Regex guard** -- A deterministic regex sweep catches
       format-based PHI (SSNs, phone numbers, MRNs, emails) that the
       LLM might miss.

This "belt-and-suspenders" strategy ensures that even if the LLM fails
to catch a phone number or SSN, the regex layer will scrub it.

Three de-identification modes are supported:

    - **remove** -- Replace PHI with ``[REDACTED]``.
    - **pseudonymize** -- Replace PHI with realistic but fake values.
    - **dateshift** -- Shift all dates by a consistent random offset.

Key components:
    - PHI_PATTERNS: Compiled regex patterns for common PHI formats.
    - PHI_PATTERN_TYPES: Type label for each pattern in PHI_PATTERNS.
    - regex_scrub_phi(text): Deterministic regex-based PHI scrubber.
    - regex_scrub_phi_with_mappings(text): Enhanced scrubber returning mappings.
    - _compute_redaction_mappings(original, redacted): Diff-based mapping.
    - _label_phi_types(original, mappings): Label mappings with PHI types.
    - RedactPHI: DSPy Signature for LLM-based redaction.
    - Deidentifier: DSPy Module orchestrating both layers.

The DSPy-dependent classes are lazily imported so that PHI_PATTERNS and
regex_scrub_phi remain importable even when dspy is not installed.
This follows the same pattern established in
``mosaicx.pipelines.extraction``.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# PHI regex patterns (no dspy dependency)
# ---------------------------------------------------------------------------

PHI_PATTERNS: list[re.Pattern[str]] = [
    # SSN: 123-45-6789
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Phone: (555) 123-4567 or 555-123-4567 or 555.123.4567
    re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"),
    # MRN: MRN: 12345678 (case-insensitive)
    re.compile(r"\bMRN\s*:?\s*\d{6,}\b", re.IGNORECASE),
    # Email: john.doe@hospital.com
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # US/EU dates with slashes: 1/2/2024 or 01/02/24
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    # Dot-separated dates: 27.02.2026 or 27.02.26
    re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"),
    # ISO dates: 2026-02-27
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    # German/English abbreviated month dates: 27.Feb.2026, 03.Mär.2024
    re.compile(
        r"\b\d{1,2}\."
        r"(?:Jan|Feb|Mär|Mar|Apr|Mai|May|Jun|Jul|Aug|Sep|Okt|Oct|Nov|Dez|Dec)\."
        r"\d{2,4}\b",
        re.IGNORECASE,
    ),
]

PHI_PATTERN_TYPES: list[str] = [
    "SSN",
    "PHONE",
    "MRN",
    "EMAIL",
    "DATE",
    "DATE",
    "DATE",
    "DATE",
]

_REDACTED = "[REDACTED]"


def regex_scrub_phi(text: str) -> str:
    """Replace all PHI pattern matches in *text* with ``[REDACTED]``.

    This is a deterministic, regex-only scrubber intended as a safety net
    after LLM-based redaction.  It catches format-based PHI (SSNs, phone
    numbers, MRNs, emails, US-format dates) that the LLM might overlook.

    Parameters
    ----------
    text:
        The input text, possibly already partially redacted by the LLM.

    Returns
    -------
    str
        Text with all regex-matched PHI replaced by ``[REDACTED]``.
    """
    for pattern in PHI_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


def regex_scrub_phi_with_mappings(
    text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Replace PHI pattern matches and return the scrubbed text with mappings.

    Collects all matches with their positions in the *original* text first,
    then applies replacements.  This ensures positions reference the input
    text, not an intermediate modified version.

    Parameters
    ----------
    text:
        The input text to scrub.

    Returns
    -------
    tuple[str, list[dict]]
        A 2-tuple of (scrubbed_text, mappings).  Each mapping dict has keys:
        ``original``, ``replacement``, ``start``, ``end``, ``phi_type``,
        ``method``.
    """
    # Collect all matches across all patterns
    raw_matches: list[tuple[int, int, str, str]] = []  # (start, end, text, phi_type)
    for pattern, phi_type in zip(PHI_PATTERNS, PHI_PATTERN_TYPES, strict=True):
        for m in pattern.finditer(text):
            raw_matches.append((m.start(), m.end(), m.group(), phi_type))

    # Sort by start position, then by longest match first (for overlaps)
    raw_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Deduplicate overlapping ranges: keep the first (earliest/longest) match
    merged: list[tuple[int, int, str, str]] = []
    for start, end, matched_text, phi_type in raw_matches:
        if merged and start < merged[-1][1]:
            # Overlaps with previous -- skip
            continue
        merged.append((start, end, matched_text, phi_type))

    # Build mappings and apply replacements (right to left to preserve offsets)
    mappings: list[dict[str, Any]] = []
    result = text
    for start, end, matched_text, phi_type in reversed(merged):
        result = result[:start] + _REDACTED + result[end:]
        mappings.append({
            "original": matched_text,
            "replacement": _REDACTED,
            "start": start,
            "end": end,
            "phi_type": phi_type,
            "method": "regex",
        })

    # Reverse so mappings are in document order (start ascending)
    mappings.reverse()

    return result, mappings


# ---------------------------------------------------------------------------
# Redaction mapping helpers (no dspy dependency)
# ---------------------------------------------------------------------------


def _compute_redaction_mappings(
    original: str,
    redacted: str,
) -> list[dict[str, Any]]:
    """Compare *original* and *redacted* text to find all redaction positions.

    Walks both strings looking for ``[REDACTED]`` tokens in the redacted
    version.  When found, determines what span of the original text was
    replaced by aligning the text that follows the token.

    Returns a list of mapping dicts with keys: ``original``, ``replacement``,
    ``start``, ``end`` (positions in the *original* text).
    """
    mappings: list[dict[str, Any]] = []
    oi = 0  # index into original
    ri = 0  # index into redacted
    token_len = len(_REDACTED)

    while ri < len(redacted) and oi < len(original):
        # Check if redacted text has a [REDACTED] token at current position
        if redacted[ri:ri + token_len] == _REDACTED:
            ri_after = ri + token_len
            orig_start = oi

            if ri_after >= len(redacted):
                # Token is at the end -- the rest of the original was redacted.
                orig_end = len(original)
            else:
                # Extract the literal text between this [REDACTED] and the
                # next one (or end of string).  This is the "anchor" we
                # search for in the original.
                next_token = redacted.find(_REDACTED, ri_after)
                if next_token == -1:
                    anchor = redacted[ri_after:]
                else:
                    anchor = redacted[ri_after:next_token]

                if not anchor:
                    # Two [REDACTED] tokens are adjacent -- use a minimal
                    # scan: find the shortest non-empty span in the original
                    # such that after it the original could still match the
                    # remainder of the redacted string.
                    # Heuristic: advance original by 1 until we find a match
                    # for what follows the pair of tokens.
                    ri_after2 = ri_after + token_len
                    if ri_after2 >= len(redacted):
                        after_anchor = ""
                    else:
                        nt2 = redacted.find(_REDACTED, ri_after2)
                        if nt2 == -1:
                            after_anchor = redacted[ri_after2:]
                        else:
                            after_anchor = redacted[ri_after2:nt2]
                    if after_anchor:
                        # Find after_anchor in original starting from oi
                        pos = original.find(after_anchor, oi)
                        if pos == -1:
                            orig_end = len(original)
                        else:
                            # Split the span between the two adjacent tokens
                            # evenly as a heuristic
                            orig_end = oi + (pos - oi) // 2
                            if orig_end == oi:
                                orig_end = oi + 1
                    else:
                        orig_end = len(original)
                else:
                    # Find the anchor text in the original
                    pos = original.find(anchor, oi)
                    if pos == -1:
                        orig_end = len(original)
                    else:
                        orig_end = pos

            redacted_span = original[orig_start:orig_end]
            if redacted_span:
                mappings.append({
                    "original": redacted_span,
                    "replacement": _REDACTED,
                    "start": orig_start,
                    "end": orig_end,
                })

            oi = orig_end
            ri = ri_after
        elif original[oi] == redacted[ri]:
            oi += 1
            ri += 1
        else:
            # Characters differ but no [REDACTED] token -- the LLM may have
            # made a small edit.  Try to re-sync by scanning ahead.
            found_sync = False
            for delta in range(1, min(200, max(len(redacted) - ri,
                                               len(original) - oi))):
                if (ri + delta < len(redacted)
                        and redacted[ri + delta:ri + delta + token_len] == _REDACTED):
                    oi += delta
                    ri += delta
                    found_sync = True
                    break
                if (oi + delta < len(original)
                        and ri + delta < len(redacted)
                        and original[oi + delta] == redacted[ri + delta]):
                    match_len = 0
                    for k in range(min(5, len(original) - oi - delta,
                                       len(redacted) - ri - delta)):
                        if original[oi + delta + k] == redacted[ri + delta + k]:
                            match_len += 1
                        else:
                            break
                    if match_len >= 3:
                        oi += delta
                        ri += delta
                        found_sync = True
                        break
            if not found_sync:
                oi += 1
                ri += 1

    return mappings


def _label_phi_types(
    original: str,
    mappings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Label each mapping with a PHI type based on regex pattern matching.

    For each mapping, test whether the original text fragment matches any
    known PHI regex pattern.  If so, label with the corresponding PHI type
    and ``method: "regex"``.  Otherwise, label with ``phi_type: "OTHER"``
    and ``method: "llm"``.

    Parameters
    ----------
    original:
        The full original text (used for context but not strictly needed;
        the mapping's ``original`` field is matched against patterns).
    mappings:
        List of mapping dicts from :func:`_compute_redaction_mappings`.

    Returns
    -------
    list[dict]
        The input mappings, each augmented with ``phi_type`` and ``method``.
    """
    for mapping in mappings:
        fragment = mapping["original"]
        phi_type = "OTHER"
        method = "llm"
        for pattern, ptype in zip(PHI_PATTERNS, PHI_PATTERN_TYPES, strict=True):
            if pattern.fullmatch(fragment) or pattern.search(fragment):
                phi_type = ptype
                method = "regex"
                break
        mapping["phi_type"] = phi_type
        mapping["method"] = method
    return mappings


def _merge_mappings(
    diff_mappings: list[dict[str, Any]],
    regex_mappings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge diff-based and regex-based mappings, deduplicating overlaps.

    Regex mappings take priority for labeling when ranges overlap.
    The result is sorted by ``start`` position.

    Parameters
    ----------
    diff_mappings:
        Mappings derived from diffing original vs final redacted text.
    regex_mappings:
        Mappings derived from running regex on the original text.

    Returns
    -------
    list[dict]
        Merged, deduplicated list of mappings sorted by start position.
    """
    # Build a set of (start, end) from regex mappings for fast overlap check
    regex_ranges = {(m["start"], m["end"]) for m in regex_mappings}

    # Start with regex mappings (they have precise phi_type labels)
    merged: list[dict[str, Any]] = list(regex_mappings)

    # Add diff mappings that don't overlap with regex ranges
    for dm in diff_mappings:
        ds, de = dm["start"], dm["end"]
        overlaps = False
        for rs, re_ in regex_ranges:
            # Check overlap: two intervals [ds, de) and [rs, re_) overlap
            # if ds < re_ and rs < de
            if ds < re_ and rs < de:
                overlaps = True
                break
        if not overlaps:
            merged.append(dm)

    merged.sort(key=lambda m: m["start"])
    return merged


# ---------------------------------------------------------------------------
# DSPy Signature & Module (lazy)
# ---------------------------------------------------------------------------
# We defer all dspy imports so that PHI_PATTERNS and regex_scrub_phi
# remain importable even when dspy is not installed.


def _build_dspy_classes():
    """Lazily define and return DSPy signature and the Deidentifier module.

    Called on first access via module-level ``__getattr__``.
    """
    import dspy  # noqa: F811 -- intentional lazy import

    # -- Signature ---------------------------------------------------------

    class RedactPHI(dspy.Signature):
        """Remove or replace Protected Health Information (PHI) from a
        medical document while preserving clinical content.

        Modes:
            - remove: Replace PHI with [REDACTED].
            - pseudonymize: Replace PHI with realistic fake values.
            - dateshift: Shift all dates by a consistent random offset.
        """

        document_text: str = dspy.InputField(
            desc="Full text of the medical document to de-identify"
        )
        mode: str = dspy.InputField(
            desc="De-identification mode: 'remove', 'pseudonymize', or 'dateshift'",
            default="remove",
        )
        redacted_text: str = dspy.OutputField(
            desc="De-identified text with PHI removed or replaced"
        )

    # -- Module ------------------------------------------------------------

    class Deidentifier(dspy.Module):
        """DSPy Module implementing the belt-and-suspenders de-identifier.

        Sub-modules:
            - ``redact`` -- ChainOfThought for LLM-based PHI redaction.

        The ``forward`` method runs LLM redaction first (catches
        context-dependent PHI like names and addresses), then applies
        ``regex_scrub_phi`` as a deterministic safety net for
        format-based PHI.

        The returned Prediction includes a ``redaction_map`` list that
        records every PHI span found, its position in the original text,
        the PHI type, and whether it was caught by the LLM or the regex
        guard.
        """

        def __init__(self) -> None:
            super().__init__()
            self.redact = dspy.ChainOfThought(RedactPHI)

        def forward(
            self, document_text: str, mode: str = "remove"
        ) -> dspy.Prediction:
            """Run the de-identification pipeline.

            Parameters
            ----------
            document_text:
                Full text of the medical document to de-identify.
            mode:
                De-identification mode.  One of ``"remove"``
                (default), ``"pseudonymize"``, or ``"dateshift"``.

            Returns
            -------
            dspy.Prediction
                Keys:
                - ``redacted_text`` (str) -- the de-identified text.
                - ``redaction_map`` (list[dict]) -- one entry per PHI
                  span with ``original``, ``replacement``, ``start``,
                  ``end``, ``phi_type``, and ``method``.
            """
            from mosaicx.metrics import PipelineMetrics, get_tracker, track_step

            metrics = PipelineMetrics()
            tracker = get_tracker()

            # Layer 1: LLM-based redaction
            with track_step(metrics, "LLM redaction", tracker):
                llm_result = self.redact(
                    document_text=document_text,
                    mode=mode,
                )
            llm_text: str = llm_result.redacted_text

            # Layer 2: regex safety net (only in "remove" mode -- in
            # pseudonymize/dateshift the LLM output intentionally
            # contains date-like and phone-like fake values that the
            # regex would incorrectly redact).
            if mode == "remove":
                with track_step(metrics, "Regex guard", tracker):
                    scrubbed_text = regex_scrub_phi(llm_text)
            else:
                scrubbed_text = llm_text

            # -- Build redaction map ------------------------------------------
            # Strategy:
            # 1. Run regex on the ORIGINAL text to get regex mappings with
            #    original-text positions and precise PHI type labels.
            # 2. Diff original vs final scrubbed_text to find ALL redactions
            #    (both LLM and regex) with original-text positions.
            # 3. Label the diff mappings with PHI types using regex matching.
            # 4. Merge: regex mappings take priority for type labeling;
            #    diff mappings add LLM-only detections.
            with track_step(metrics, "Redaction mapping", tracker):
                if mode == "remove":
                    _, regex_mappings = regex_scrub_phi_with_mappings(
                        document_text,
                    )
                    diff_mappings = _compute_redaction_mappings(
                        document_text, scrubbed_text,
                    )
                    diff_mappings = _label_phi_types(
                        document_text, diff_mappings,
                    )
                    redaction_map = _merge_mappings(
                        diff_mappings, regex_mappings,
                    )
                else:
                    # For pseudonymize/dateshift, diff to find LLM changes
                    diff_mappings = _compute_redaction_mappings(
                        document_text, scrubbed_text,
                    )
                    redaction_map = _label_phi_types(
                        document_text, diff_mappings,
                    )

            self._last_metrics = metrics

            return dspy.Prediction(
                redacted_text=scrubbed_text,
                redaction_map=redaction_map,
            )

    return {
        "RedactPHI": RedactPHI,
        "Deidentifier": Deidentifier,
    }


# Cache for lazily-built DSPy classes
_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({
    "RedactPHI",
    "Deidentifier",
})


def __getattr__(name: str):
    """Module-level __getattr__ for lazy loading of DSPy classes."""
    global _dspy_classes

    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
