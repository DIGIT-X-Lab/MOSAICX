"""Pathology report structurer pipeline.

A 5-step DSPy chain that converts free-text pathology reports into
structured data:

    1. **ClassifySpecimenType** -- Identify specimen / procedure type.
    2. **ParsePathSections** -- Split the report into standard sections.
    3. **ExtractSpecimenDetails** -- Parse specimen site, laterality,
       procedure, dimensions.
    4. **ExtractMicroscopicFindings** -- Extract structured findings.
    5. **ExtractPathDiagnosis** -- Extract diagnoses with WHO
       classification, TNM staging, ICD-O codes, and biomarkers.

DSPy-dependent classes are lazily imported via module-level ``__getattr__``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from mosaicx.schemas.pathreport.base import (
    Biomarker,
    PathDiagnosis,
    PathFinding,
    PathSections,
)


def _build_dspy_classes():
    import dspy

    class ClassifySpecimenType(dspy.Signature):
        """Classify the specimen type from the pathology report header."""
        report_header: str = dspy.InputField(desc="Header / title portion of the pathology report")
        specimen_type: str = dspy.OutputField(desc="Specimen type (e.g., Biopsy - Prostate, Resection - Colon, Cytology - Thyroid FNA)")

    class ParsePathSections(dspy.Signature):
        """Split a pathology report into its standard sections."""
        report_text: str = dspy.InputField(desc="Full text of the pathology report")
        sections: PathSections = dspy.OutputField(desc="Parsed report sections (clinical history, gross, microscopic, diagnosis, ancillary)")

    class ExtractSpecimenDetails(dspy.Signature):
        """Extract specimen details from the gross description."""
        gross_text: str = dspy.InputField(desc="Text of the gross description section")
        site: str = dspy.OutputField(desc="Anatomical site of the specimen")
        laterality: str = dspy.OutputField(desc="Laterality (left/right/bilateral/N/A)")
        procedure: str = dspy.OutputField(desc="Procedure type (biopsy, excision, resection)")
        dimensions: str = dspy.OutputField(desc="Specimen dimensions (e.g., 3.2 x 2.1 x 1.5 cm)")
        specimens_received: int = dspy.OutputField(desc="Number of specimens / parts received")

    class ExtractMicroscopicFindings(dspy.Signature):
        """Extract structured findings from the microscopic description."""
        microscopic_text: str = dspy.InputField(desc="Text of the microscopic description")
        specimen_type: str = dspy.InputField(desc="Specimen type for context")
        findings: List[PathFinding] = dspy.OutputField(desc="List of structured microscopic findings")

    class ExtractPathDiagnosis(dspy.Signature):
        """Extract diagnoses with staging and biomarker data."""
        diagnosis_text: str = dspy.InputField(desc="Text of the diagnosis section")
        findings_context: str = dspy.InputField(desc="JSON summary of extracted findings for grounding", default="")
        ancillary_text: str = dspy.InputField(desc="Text of the ancillary studies section", default="")
        diagnoses: List[PathDiagnosis] = dspy.OutputField(desc="List of structured pathologic diagnoses")

    class PathologyReportStructurer(dspy.Module):
        """DSPy Module implementing the 5-step pathology report structurer."""

        def __init__(self) -> None:
            super().__init__()
            self.classify_specimen = dspy.Predict(ClassifySpecimenType)
            self.parse_sections = dspy.Predict(ParsePathSections)
            self.extract_specimen_details = dspy.Predict(ExtractSpecimenDetails)
            self.extract_findings = dspy.ChainOfThought(ExtractMicroscopicFindings)
            self.extract_diagnosis = dspy.ChainOfThought(ExtractPathDiagnosis)

        def forward(self, report_text: str, report_header: str = "") -> dspy.Prediction:
            header = report_header or report_text[:200]
            classify_result = self.classify_specimen(report_header=header)
            specimen_type: str = classify_result.specimen_type

            sections_result = self.parse_sections(report_text=report_text)
            sections: PathSections = sections_result.sections

            specimen_result = self.extract_specimen_details(gross_text=sections.gross_description)

            findings_result = self.extract_findings(microscopic_text=sections.microscopic, specimen_type=specimen_type)
            findings: List[PathFinding] = findings_result.findings

            findings_json = "[" + ", ".join(f.model_dump_json() for f in findings) + "]"
            diagnosis_result = self.extract_diagnosis(
                diagnosis_text=sections.diagnosis,
                findings_context=findings_json,
                ancillary_text=sections.ancillary_studies,
            )

            return dspy.Prediction(
                specimen_type=specimen_type,
                sections=sections,
                site=specimen_result.site,
                laterality=specimen_result.laterality,
                procedure=specimen_result.procedure,
                dimensions=specimen_result.dimensions,
                specimens_received=specimen_result.specimens_received,
                findings=findings,
                diagnoses=diagnosis_result.diagnoses,
            )

    # Register as extraction mode
    from mosaicx.pipelines.modes import register_mode
    register_mode("pathology", "5-step pathology report structurer (histology, staging, biomarkers)")(PathologyReportStructurer)

    return {
        "ClassifySpecimenType": ClassifySpecimenType,
        "ParsePathSections": ParsePathSections,
        "ExtractSpecimenDetails": ExtractSpecimenDetails,
        "ExtractMicroscopicFindings": ExtractMicroscopicFindings,
        "ExtractPathDiagnosis": ExtractPathDiagnosis,
        "PathologyReportStructurer": PathologyReportStructurer,
    }


_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({
    "ClassifySpecimenType",
    "ParsePathSections",
    "ExtractSpecimenDetails",
    "ExtractMicroscopicFindings",
    "ExtractPathDiagnosis",
    "PathologyReportStructurer",
})


def __getattr__(name: str):
    global _dspy_classes
    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
