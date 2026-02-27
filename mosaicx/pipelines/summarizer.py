# mosaicx/pipelines/summarizer.py
"""Report summarizer pipeline with timeline synthesis.

A 2-step DSPy chain that summarizes a collection of medical reports into
a coherent patient timeline narrative:

    1. **ExtractTimelineEvent** -- Extract a single structured timeline event
       from each report (parallelizable across reports).
    2. **SynthesizeTimeline** -- Synthesize all extracted events into a
       unified narrative summary.

This replaces the v1 summarizer that concatenated all reports into one
massive prompt, enabling better per-report extraction and more controllable
synthesis.

Key components:
    - TimelineEvent: Pydantic model for a single timeline event (no dspy dep).
    - ExtractTimelineEvent: DSPy Signature for per-report event extraction.
    - SynthesizeTimeline: DSPy Signature for narrative synthesis.
    - ReportSummarizer: DSPy Module orchestrating the full pipeline.

The DSPy-dependent classes are lazily imported so that the Pydantic model
remains importable even when dspy is not installed.  This follows the same
pattern established in ``mosaicx.pipelines.extraction``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic output model (no dspy dependency)
# ---------------------------------------------------------------------------

class TimelineEvent(BaseModel):
    """A single event on a patient's clinical timeline."""

    date: str = Field(
        "not present",
        description=(
            "Date of the exam or event (ISO 8601, e.g. 2025-06-15). "
            "Use 'not present' if no explicit date appears in the report. "
            "Never guess or infer a date."
        ),
    )
    exam_type: str = Field(..., description="Type of exam (e.g., CT chest, MRI brain)")
    key_finding: str = Field(..., description="Primary finding from the report")
    clinical_context: Optional[str] = Field(
        None, description="Clinical context or indication for the exam"
    )
    change_from_prior: Optional[str] = Field(
        None, description="Change compared to prior exam, if applicable"
    )


# ---------------------------------------------------------------------------
# DSPy Signatures & Module (lazy)
# ---------------------------------------------------------------------------
# We defer all dspy imports so the Pydantic model above stays importable
# even when dspy is not installed or has import issues.


def _build_dspy_classes():
    """Lazily define and return DSPy signatures and the ReportSummarizer module.

    Called on first access via module-level ``__getattr__``.
    """
    import dspy  # noqa: F811 -- intentional lazy import

    # -- Signatures --------------------------------------------------------

    class ExtractTimelineEvent(dspy.Signature):
        """Extract a structured timeline event from a single medical report.

        For the date field: only use a date that is explicitly written in the
        report text. If no date is stated, set date to 'not present'. Never
        guess or infer dates from context clues like 'recently'.
        """

        report_text: str = dspy.InputField(
            desc="Full text of a single medical report"
        )
        patient_id: str = dspy.InputField(
            desc="Patient identifier for context",
            default="",
        )
        event: TimelineEvent = dspy.OutputField(
            desc="Structured timeline event extracted from the report"
        )

    class SynthesizeTimeline(dspy.Signature):
        """Synthesize multiple timeline events into a coherent narrative."""

        events_json: str = dspy.InputField(
            desc="JSON array of TimelineEvent objects"
        )
        patient_id: str = dspy.InputField(
            desc="Patient identifier for context",
            default="",
        )
        narrative: str = dspy.OutputField(
            desc="Coherent narrative summary synthesizing all timeline events"
        )

    # -- Module ------------------------------------------------------------

    class ReportSummarizer(dspy.Module):
        """DSPy Module implementing the 2-step report summarizer.

        Sub-modules:
            - ``extract_event`` -- ChainOfThought for per-report event extraction.
            - ``synthesize`` -- ChainOfThought for narrative synthesis.

        The ``forward`` method extracts one timeline event per report, then
        synthesizes all events into a unified narrative.
        """

        def __init__(self) -> None:
            super().__init__()
            self.extract_event = dspy.ChainOfThought(ExtractTimelineEvent)
            self.synthesize = dspy.ChainOfThought(SynthesizeTimeline)

        def forward(
            self, reports: list[str], patient_id: str = ""
        ) -> dspy.Prediction:
            """Run the summarizer pipeline.

            Parameters
            ----------
            reports:
                List of report texts to summarize.
            patient_id:
                Optional patient identifier for context.

            Returns
            -------
            dspy.Prediction
                Keys: ``events`` (list[TimelineEvent]), ``narrative`` (str).
            """
            from mosaicx.metrics import PipelineMetrics, get_tracker, track_step

            metrics = PipelineMetrics()
            tracker = get_tracker()

            # Step 1: extract one event per report
            events: list[TimelineEvent] = []
            for i, report_text in enumerate(reports, 1):
                with track_step(metrics, f"Extract event ({i}/{len(reports)})", tracker):
                    result = self.extract_event(
                        report_text=report_text,
                        patient_id=patient_id,
                    )
                events.append(result.event)

            # Step 2: synthesize into narrative
            events_json = (
                "[" + ", ".join(e.model_dump_json() for e in events) + "]"
            )
            with track_step(metrics, "Synthesize narrative", tracker):
                synth_result = self.synthesize(
                    events_json=events_json,
                    patient_id=patient_id,
                )

            self._last_metrics = metrics

            return dspy.Prediction(
                events=events,
                narrative=synth_result.narrative,
            )

    return {
        "ExtractTimelineEvent": ExtractTimelineEvent,
        "SynthesizeTimeline": SynthesizeTimeline,
        "ReportSummarizer": ReportSummarizer,
    }


# Cache for lazily-built DSPy classes
_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({
    "ExtractTimelineEvent",
    "SynthesizeTimeline",
    "ReportSummarizer",
})


def __getattr__(name: str):
    """Module-level __getattr__ for lazy loading of DSPy classes."""
    global _dspy_classes

    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
