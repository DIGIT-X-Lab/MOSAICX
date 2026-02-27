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
    - regex_scrub_phi(text): Deterministic regex-based PHI scrubber.
    - RedactPHI: DSPy Signature for LLM-based redaction.
    - Deidentifier: DSPy Module orchestrating both layers.

The DSPy-dependent classes are lazily imported so that PHI_PATTERNS and
regex_scrub_phi remain importable even when dspy is not installed.
This follows the same pattern established in
``mosaicx.pipelines.extraction``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List


# ---------------------------------------------------------------------------
# PHI regex patterns (no dspy dependency)
# ---------------------------------------------------------------------------

PHI_PATTERNS: List[re.Pattern[str]] = [
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
                Keys: ``redacted_text`` (str) -- the de-identified text.
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

            # Layer 2: regex safety net
            with track_step(metrics, "Regex guard", tracker):
                scrubbed_text = regex_scrub_phi(llm_text)

            self._last_metrics = metrics

            return dspy.Prediction(redacted_text=scrubbed_text)

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
