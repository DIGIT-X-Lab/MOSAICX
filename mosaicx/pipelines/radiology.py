# mosaicx/pipelines/radiology.py
"""Radiology report structurer pipeline.

A 5-step DSPy chain that converts free-text radiology reports into
structured data:

    1. **ClassifyExamType** -- Identify exam modality / body region from the
       report header.
    2. **ParseReportSections** -- Split the raw report text into standard
       sections (indication, comparison, technique, findings, impression).
    3. **ExtractTechnique** -- Parse the technique section for modality,
       body region, contrast, and protocol details.
    4. **ExtractRadFindings** -- Extract structured findings from the
       findings section.
    5. **ExtractImpression** -- Extract actionable impression items from the
       impression section.

Each step is independently optimizable via GEPA.

DSPy-dependent classes are lazily imported via module-level ``__getattr__``
so that the module can be imported even when dspy is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from mosaicx.schemas.radreport.base import (
    ImpressionItem,
    RadReportFinding,
    ReportSections,
)

from mosaicx.pipelines.modes import register_mode_info

register_mode_info("radiology", "5-step radiology report structurer (findings, measurements, scoring)")


# ---------------------------------------------------------------------------
# DSPy Signatures & Module (lazy)
# ---------------------------------------------------------------------------

def _build_dspy_classes():
    """Lazily define and return all DSPy signatures and the pipeline module.

    Called on first access via module-level ``__getattr__``.
    """
    import dspy  # noqa: F811 -- intentional lazy import

    # -- Step 1: Classify exam type ----------------------------------------

    class ClassifyExamType(dspy.Signature):
        """Classify the type of radiology exam from the report header."""

        report_header: str = dspy.InputField(
            desc="Header / title portion of the radiology report"
        )
        exam_type: str = dspy.OutputField(
            desc="Identified exam type (e.g., CT Chest, MRI Brain, X-ray Chest)"
        )

    # -- Step 2: Parse report sections -------------------------------------

    class ParseReportSections(dspy.Signature):
        """Split a radiology report into its standard sections."""

        report_text: str = dspy.InputField(
            desc="Full text of the radiology report"
        )
        sections: ReportSections = dspy.OutputField(
            desc="Parsed report sections (indication, comparison, technique, findings, impression)"
        )

    # -- Step 3: Extract technique -----------------------------------------

    class ExtractTechnique(dspy.Signature):
        """Extract imaging technique details from the technique section."""

        technique_text: str = dspy.InputField(
            desc="Text of the technique / protocol section"
        )
        modality: str = dspy.OutputField(
            desc="Imaging modality (e.g., CT, MRI, US, XR)"
        )
        body_region: str = dspy.OutputField(
            desc="Body region imaged (e.g., chest, abdomen, brain)"
        )
        contrast: str = dspy.OutputField(
            desc="Contrast agent details, or 'none' if non-contrast"
        )
        protocol: str = dspy.OutputField(
            desc="Protocol name or description"
        )

    # -- Step 4: Extract findings ------------------------------------------

    class ExtractRadFindings(dspy.Signature):
        """Extract structured radiology findings from the findings section."""

        findings_text: str = dspy.InputField(
            desc="Text of the findings section"
        )
        exam_type: str = dspy.InputField(
            desc="Exam type for context (e.g., CT Chest)"
        )
        findings: List[RadReportFinding] = dspy.OutputField(
            desc="List of structured radiology findings"
        )

    # -- Step 5: Extract impression ----------------------------------------

    class ExtractImpression(dspy.Signature):
        """Extract actionable impression items from the impression section."""

        impression_text: str = dspy.InputField(
            desc="Text of the impression / conclusion section"
        )
        exam_type: str = dspy.InputField(
            desc="Exam type for context"
        )
        findings_context: str = dspy.InputField(
            desc="JSON summary of previously extracted findings for grounding",
            default="",
        )
        impressions: List[ImpressionItem] = dspy.OutputField(
            desc="List of structured impression items"
        )

    # -- Pipeline module ---------------------------------------------------

    class RadiologyReportStructurer(dspy.Module):
        """DSPy Module implementing the 5-step radiology report structurer.

        Sub-modules:
            - ``classify_exam`` -- Predict for exam type classification.
            - ``parse_sections`` -- Predict for section parsing.
            - ``extract_technique`` -- Predict for technique extraction.
            - ``extract_findings`` -- ChainOfThought for finding extraction.
            - ``extract_impression`` -- ChainOfThought for impression extraction.
        """

        def __init__(self) -> None:
            super().__init__()
            from mosaicx.config import get_config

            cfg = get_config()

            self.classify_exam = dspy.Predict(ClassifyExamType)
            self.parse_sections = dspy.Predict(ParseReportSections)
            self.extract_technique = dspy.Predict(ExtractTechnique)

            if cfg.use_refine:
                from mosaicx.pipelines.rewards import findings_reward as _fr

                def _findings_reward_fn(args, pred):
                    findings = pred.findings
                    finding_dicts = [
                        f.model_dump() if hasattr(f, "model_dump") else f
                        for f in findings
                    ]
                    return _fr(findings=finding_dicts)

                self.extract_findings = dspy.Refine(
                    module=dspy.ChainOfThought(ExtractRadFindings),
                    N=3,
                    reward_fn=_findings_reward_fn,
                    threshold=0.7,
                )
                self.extract_impression = dspy.Refine(
                    module=dspy.ChainOfThought(ExtractImpression),
                    N=3,
                    reward_fn=lambda args, pred: 0.8 if pred.impressions else 0.0,
                    threshold=0.5,
                )
            else:
                self.extract_findings = dspy.ChainOfThought(ExtractRadFindings)
                self.extract_impression = dspy.ChainOfThought(ExtractImpression)

        def forward(self, report_text: str, report_header: str = "") -> dspy.Prediction:
            """Run the full 5-step structuring pipeline.

            Parameters
            ----------
            report_text:
                Full text of the radiology report.
            report_header:
                Optional header / title to use for exam classification.
                If empty, the first 200 characters of report_text are used.

            Returns
            -------
            dspy.Prediction
                Keys: exam_type, sections, modality, body_region, contrast,
                protocol, findings, impressions.
            """
            from mosaicx.metrics import PipelineMetrics, get_tracker, track_step

            metrics = PipelineMetrics()
            tracker = get_tracker()

            # Step 1: classify exam type
            header = report_header or report_text[:200]
            with track_step(metrics, "Classify exam type", tracker):
                classify_result = self.classify_exam(report_header=header)
            exam_type: str = classify_result.exam_type

            # Step 2: parse sections
            with track_step(metrics, "Parse sections", tracker):
                sections_result = self.parse_sections(report_text=report_text)
            sections: ReportSections = sections_result.sections

            # Step 3: extract technique
            with track_step(metrics, "Extract technique", tracker):
                technique_result = self.extract_technique(
                    technique_text=sections.technique
                )

            # Step 4: extract findings
            with track_step(metrics, "Extract findings", tracker):
                findings_result = self.extract_findings(
                    findings_text=sections.findings,
                    exam_type=exam_type,
                )
            findings: List[RadReportFinding] = findings_result.findings

            # Step 5: extract impression
            findings_json = "[" + ", ".join(
                f.model_dump_json() for f in findings
            ) + "]"
            with track_step(metrics, "Extract impression", tracker):
                impression_result = self.extract_impression(
                    impression_text=sections.impression,
                    exam_type=exam_type,
                    findings_context=findings_json,
                )

            self._last_metrics = metrics

            return dspy.Prediction(
                exam_type=exam_type,
                sections=sections,
                modality=technique_result.modality,
                body_region=technique_result.body_region,
                contrast=technique_result.contrast,
                protocol=technique_result.protocol,
                findings=findings,
                impressions=impression_result.impressions,
            )


    # Register as extraction mode
    from mosaicx.pipelines.modes import register_mode
    register_mode("radiology", "5-step radiology report structurer (findings, measurements, scoring)")(RadiologyReportStructurer)
    return {
        "ClassifyExamType": ClassifyExamType,
        "ParseReportSections": ParseReportSections,
        "ExtractTechnique": ExtractTechnique,
        "ExtractRadFindings": ExtractRadFindings,
        "ExtractImpression": ExtractImpression,
        "RadiologyReportStructurer": RadiologyReportStructurer,
    }


# Cache for lazily-built DSPy classes
_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({
    "ClassifyExamType",
    "ParseReportSections",
    "ExtractTechnique",
    "ExtractRadFindings",
    "ExtractImpression",
    "RadiologyReportStructurer",
})


def __getattr__(name: str):
    """Module-level __getattr__ for lazy loading of DSPy classes."""
    global _dspy_classes

    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
