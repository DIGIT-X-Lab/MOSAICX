# mosaicx/pipelines/extraction.py
"""Generic document extraction pipeline.

Provides a 3-step DSPy chain for extracting structured data from medical
documents: demographics, findings, and diagnoses.  Also supports a custom
schema mode where an arbitrary Pydantic model is used as the output type.

Key components:
    - Demographics, Finding, Diagnosis: Pydantic output models (no dspy dep).
    - ExtractDemographics / ExtractFindings / ExtractDiagnoses: DSPy Signatures.
    - DocumentExtractor: DSPy Module orchestrating the full pipeline.

The DSPy-dependent classes are lazily imported so that the Pydantic models
remain importable even when dspy is not installed.  This follows the same
pattern established in ``mosaicx.pipelines.schema_gen``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic output models (no dspy dependency)
# ---------------------------------------------------------------------------

class Demographics(BaseModel):
    """Patient demographic information extracted from a document."""

    patient_id: Optional[str] = Field(None, description="Patient identifier")
    name: Optional[str] = Field(None, description="Patient name")
    age: Optional[int] = Field(None, description="Patient age in years")
    gender: Optional[str] = Field(None, description="Patient gender")
    date_of_birth: Optional[str] = Field(
        None, description="Patient date of birth"
    )


class Finding(BaseModel):
    """A single clinical finding extracted from a document."""

    description: str = Field(..., description="Description of the finding")
    location: Optional[str] = Field(
        None, description="Anatomical location of the finding"
    )
    severity: Optional[str] = Field(
        None, description="Severity level (e.g., mild, moderate, severe)"
    )
    status: Optional[str] = Field(
        None,
        description="Status of the finding (e.g., new, stable, resolved)",
    )


class Diagnosis(BaseModel):
    """A clinical diagnosis extracted from a document."""

    name: str = Field(..., description="Name of the diagnosis")
    icd10_code: Optional[str] = Field(
        None, description="ICD-10 code for the diagnosis"
    )
    confidence: Optional[float] = Field(
        None, description="Confidence score (0-1) for the diagnosis"
    )


# ---------------------------------------------------------------------------
# DSPy Signatures & Module (lazy)
# ---------------------------------------------------------------------------
# We defer all dspy imports so the Pydantic models above stay importable
# even when dspy is not installed or has import issues.


def _build_dspy_classes():
    """Lazily define and return DSPy signatures and the DocumentExtractor module.

    Called on first access via module-level ``__getattr__``.
    """
    import dspy  # noqa: F811  -- intentional lazy import

    # -- Signatures --------------------------------------------------------

    class ExtractDemographics(dspy.Signature):
        """Extract patient demographic information from a medical document."""

        document_text: str = dspy.InputField(
            desc="Full text of the medical document"
        )
        demographics: Demographics = dspy.OutputField(
            desc="Extracted patient demographics"
        )

    class ExtractFindings(dspy.Signature):
        """Extract clinical findings from a medical document."""

        document_text: str = dspy.InputField(
            desc="Full text of the medical document"
        )
        demographics_context: str = dspy.InputField(
            desc="Previously extracted demographics context for grounding",
            default="",
        )
        findings: List[Finding] = dspy.OutputField(
            desc="List of clinical findings extracted from the document"
        )

    class ExtractDiagnoses(dspy.Signature):
        """Extract clinical diagnoses from a medical document."""

        document_text: str = dspy.InputField(
            desc="Full text of the medical document"
        )
        findings_context: str = dspy.InputField(
            desc="Previously extracted findings context for grounding",
            default="",
        )
        diagnoses: List[Diagnosis] = dspy.OutputField(
            desc="List of clinical diagnoses extracted from the document"
        )

    # -- Module ------------------------------------------------------------

    class DocumentExtractor(dspy.Module):
        """DSPy Module for generic medical document extraction.

        Operates in one of two modes:

        **Default mode** (no ``output_schema``):
            3-step chain:
            1. Extract demographics from the document text.
            2. Extract findings, using demographics as context.
            3. Extract diagnoses, using findings as context.

        **Custom schema mode** (``output_schema`` provided):
            A single ChainOfThought step that outputs the provided
            Pydantic model type, exposed as ``extract_custom``.
        """

        def __init__(self, output_schema: type[BaseModel] | None = None) -> None:
            super().__init__()

            if output_schema is not None:
                # Custom schema mode: single extraction step
                custom_sig = dspy.Signature(
                    "document_text -> extracted",
                    instructions=(
                        f"Extract structured data matching the "
                        f"{output_schema.__name__} schema from the document."
                    ),
                ).with_updated_fields(
                    "document_text",
                    desc="Full text of the document",
                    type_=str,
                ).with_updated_fields(
                    "extracted",
                    desc=f"Extracted {output_schema.__name__} data",
                    type_=output_schema,
                )
                self.extract_custom = dspy.ChainOfThought(custom_sig)
            else:
                # Default 3-step chain mode
                self.extract_demographics = dspy.ChainOfThought(
                    ExtractDemographics
                )
                self.extract_findings = dspy.ChainOfThought(ExtractFindings)
                self.extract_diagnoses = dspy.ChainOfThought(ExtractDiagnoses)

        def forward(self, document_text: str) -> dspy.Prediction:
            """Run the extraction pipeline.

            Returns a ``dspy.Prediction`` with the following keys:

            - Default mode: ``demographics``, ``findings``, ``diagnoses``
            - Custom mode: ``extracted``
            """
            if hasattr(self, "extract_custom"):
                result = self.extract_custom(document_text=document_text)
                return dspy.Prediction(extracted=result.extracted)

            # Step 1: demographics
            demo_result = self.extract_demographics(
                document_text=document_text
            )
            demographics: Demographics = demo_result.demographics

            # Step 2: findings (demographics as context)
            findings_result = self.extract_findings(
                document_text=document_text,
                demographics_context=demographics.model_dump_json(),
            )
            findings: List[Finding] = findings_result.findings

            # Step 3: diagnoses (findings as context)
            findings_json = "[" + ", ".join(
                f.model_dump_json() for f in findings
            ) + "]"
            diagnoses_result = self.extract_diagnoses(
                document_text=document_text,
                findings_context=findings_json,
            )
            diagnoses: List[Diagnosis] = diagnoses_result.diagnoses

            return dspy.Prediction(
                demographics=demographics,
                findings=findings,
                diagnoses=diagnoses,
            )

    return {
        "ExtractDemographics": ExtractDemographics,
        "ExtractFindings": ExtractFindings,
        "ExtractDiagnoses": ExtractDiagnoses,
        "DocumentExtractor": DocumentExtractor,
    }


# Cache for lazily-built DSPy classes
_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({
    "ExtractDemographics",
    "ExtractFindings",
    "ExtractDiagnoses",
    "DocumentExtractor",
})


def __getattr__(name: str):
    """Module-level __getattr__ for lazy loading of DSPy classes."""
    global _dspy_classes

    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
