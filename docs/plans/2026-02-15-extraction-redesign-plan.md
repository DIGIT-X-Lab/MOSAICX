# Extraction Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the hardcoded 3-step extraction with schema-first extraction, an extensible mode system (radiology + pathology), and working batch processing.

**Architecture:** `mosaicx extract` supports three paths: auto (LLM infers schema), schema-first (user-provided schema), and mode (specialized pipeline). Modes are registered via a decorator into a dict. Batch processing runs extraction in parallel with checkpointing.

**Tech Stack:** DSPy (ChainOfThought, Predict), Pydantic (BaseModel, create_model), Click (CLI), concurrent.futures (batch parallelism)

---

### Task 1: Mode Registry

**Files:**
- Create: `mosaicx/pipelines/modes.py`
- Test: `tests/test_modes.py`

**Step 1: Write the failing tests**

```python
# tests/test_modes.py
"""Tests for the extraction mode registry."""
import pytest


class TestModeRegistry:
    def test_register_and_get_mode(self):
        from mosaicx.pipelines.modes import MODES, register_mode, get_mode

        @register_mode("test_mode", "A test mode")
        class TestMode:
            mode_name: str
            mode_description: str

        assert "test_mode" in MODES
        assert get_mode("test_mode") is TestMode

    def test_get_unknown_mode_raises(self):
        from mosaicx.pipelines.modes import get_mode
        with pytest.raises(ValueError, match="Unknown mode"):
            get_mode("nonexistent_mode_xyz")

    def test_list_modes_returns_tuples(self):
        from mosaicx.pipelines.modes import list_modes
        modes = list_modes()
        assert isinstance(modes, list)
        for name, desc in modes:
            assert isinstance(name, str)
            assert isinstance(desc, str)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_modes.py -v`
Expected: FAIL (module not found)

**Step 3: Write the implementation**

```python
# mosaicx/pipelines/modes.py
"""Extraction mode registry.

Modes are specialized multi-step pipelines for specific document domains.
Each mode is a DSPy Module registered via the ``@register_mode`` decorator.
"""
from __future__ import annotations

from typing import Any

MODES: dict[str, type] = {}


def register_mode(name: str, description: str):
    """Decorator to register an extraction mode."""
    def decorator(cls):
        cls.mode_name = name
        cls.mode_description = description
        MODES[name] = cls
        return cls
    return decorator


def get_mode(name: str) -> type:
    """Get a registered mode by name. Raises ValueError if not found."""
    if name not in MODES:
        available = ", ".join(sorted(MODES)) or "(none)"
        raise ValueError(f"Unknown mode: {name!r}. Available: {available}")
    return MODES[name]


def list_modes() -> list[tuple[str, str]]:
    """Return list of (name, description) tuples for all registered modes."""
    return [(name, cls.mode_description) for name, cls in sorted(MODES.items())]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_modes.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mosaicx/pipelines/modes.py tests/test_modes.py
git commit -m "feat: add extraction mode registry"
```

---

### Task 2: Pathology Base Models

**Files:**
- Create: `mosaicx/schemas/pathreport/__init__.py`
- Create: `mosaicx/schemas/pathreport/base.py`
- Test: `tests/test_pathology_pipeline.py` (models only for now)

**Step 1: Write the failing tests**

```python
# tests/test_pathology_pipeline.py
"""Tests for the pathology report structurer pipeline."""
import pytest


class TestPathReportBaseModels:
    def test_path_sections_construction(self):
        from mosaicx.schemas.pathreport.base import PathSections
        s = PathSections(
            clinical_history="56-year-old male with rectal mass",
            gross_description="Received in formalin...",
            microscopic="Sections show moderately differentiated adenocarcinoma",
            diagnosis="Adenocarcinoma of the rectum",
        )
        assert s.clinical_history == "56-year-old male with rectal mass"
        assert s.ancillary_studies == ""  # default empty

    def test_path_finding_construction(self):
        from mosaicx.schemas.pathreport.base import PathFinding
        f = PathFinding(
            description="Moderately differentiated adenocarcinoma",
            histologic_type="adenocarcinoma",
            grade="G2",
            margins="negative, closest margin 0.3 cm",
            lymphovascular_invasion="present",
        )
        assert f.histologic_type == "adenocarcinoma"
        assert f.perineural_invasion is None

    def test_biomarker_construction(self):
        from mosaicx.schemas.pathreport.base import Biomarker
        b = Biomarker(name="ER", result="positive (95%)", method="IHC")
        assert b.name == "ER"
        assert b.method == "IHC"

    def test_path_diagnosis_construction(self):
        from mosaicx.schemas.pathreport.base import PathDiagnosis
        d = PathDiagnosis(
            diagnosis="Invasive ductal carcinoma, grade 2",
            who_classification="Invasive carcinoma of no special type",
            tnm_stage="pT2 pN1a",
            biomarkers=[],
        )
        assert d.tnm_stage == "pT2 pN1a"
        assert d.icd_o_morphology is None

    def test_path_diagnosis_with_biomarkers(self):
        from mosaicx.schemas.pathreport.base import Biomarker, PathDiagnosis
        d = PathDiagnosis(
            diagnosis="Breast carcinoma",
            biomarkers=[
                Biomarker(name="ER", result="positive"),
                Biomarker(name="PR", result="positive"),
                Biomarker(name="HER2", result="negative"),
                Biomarker(name="Ki-67", result="30%"),
            ],
        )
        assert len(d.biomarkers) == 4
        assert d.biomarkers[0].name == "ER"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pathology_pipeline.py::TestPathReportBaseModels -v`
Expected: FAIL (module not found)

**Step 3: Write the implementation**

```python
# mosaicx/schemas/pathreport/__init__.py
```

```python
# mosaicx/schemas/pathreport/base.py
"""Base Pydantic models for structured pathology reports.

These models capture the core data structures used to represent parsed
pathology reports: sections, findings, biomarkers, and diagnoses.
They are intentionally free of any DSPy dependency so they can be
imported and used independently for validation, serialization, or export.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

__all__ = [
    "PathSections",
    "PathFinding",
    "Biomarker",
    "PathDiagnosis",
]


class PathSections(BaseModel):
    """Top-level sections parsed from a pathology report."""

    clinical_history: str = Field(
        "", description="Clinical history / reason for biopsy"
    )
    gross_description: str = Field(
        "", description="Gross / macroscopic description of the specimen"
    )
    microscopic: str = Field(
        "", description="Microscopic description"
    )
    diagnosis: str = Field(
        "", description="Final diagnosis section text"
    )
    ancillary_studies: str = Field(
        "", description="Ancillary studies (IHC, molecular, flow cytometry)"
    )
    comment: str = Field(
        "", description="Additional comments or notes"
    )


class PathFinding(BaseModel):
    """A single microscopic finding from a pathology report."""

    description: str = Field(
        ..., description="Description of the finding"
    )
    histologic_type: Optional[str] = Field(
        None, description="Histologic type (e.g., adenocarcinoma, squamous cell)"
    )
    grade: Optional[str] = Field(
        None, description="Histologic grade (e.g., G2, Gleason 3+4=7, Nottingham 2)"
    )
    margins: Optional[str] = Field(
        None, description="Margin status (positive/negative/close + distance)"
    )
    lymphovascular_invasion: Optional[str] = Field(
        None, description="Lymphovascular invasion (present/absent/indeterminate)"
    )
    perineural_invasion: Optional[str] = Field(
        None, description="Perineural invasion (present/absent/indeterminate)"
    )


class Biomarker(BaseModel):
    """A single biomarker result (IHC, molecular, etc.)."""

    name: str = Field(
        ..., description="Biomarker name (e.g., ER, PR, HER2, Ki-67, EGFR)"
    )
    result: str = Field(
        ..., description="Result (e.g., positive (95%), negative, 3+)"
    )
    method: Optional[str] = Field(
        None, description="Method used (e.g., IHC, FISH, PCR)"
    )


class PathDiagnosis(BaseModel):
    """A final pathologic diagnosis with staging and biomarker data."""

    diagnosis: str = Field(
        ..., description="Primary diagnosis text"
    )
    who_classification: Optional[str] = Field(
        None, description="WHO tumor classification"
    )
    tnm_stage: Optional[str] = Field(
        None, description="Pathologic TNM stage (e.g., pT2 pN1a)"
    )
    icd_o_morphology: Optional[str] = Field(
        None, description="ICD-O morphology code (e.g., 8500/3)"
    )
    biomarkers: List[Biomarker] = Field(
        default_factory=list,
        description="Biomarker results (IHC, molecular)"
    )
    ancillary_results: Optional[str] = Field(
        None, description="Summary of ancillary study results"
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pathology_pipeline.py::TestPathReportBaseModels -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mosaicx/schemas/pathreport/ tests/test_pathology_pipeline.py
git commit -m "feat: add pathology report base models"
```

---

### Task 3: Pathology Pipeline

**Files:**
- Create: `mosaicx/pipelines/pathology.py`
- Modify: `tests/test_pathology_pipeline.py` (add signature + module tests)

**Step 1: Add failing tests for signatures and module**

Append to `tests/test_pathology_pipeline.py`:

```python
class TestPathologyPipelineSignatures:
    def test_classify_specimen_type_signature(self):
        from mosaicx.pipelines.pathology import ClassifySpecimenType
        assert "report_header" in ClassifySpecimenType.input_fields
        assert "specimen_type" in ClassifySpecimenType.output_fields

    def test_parse_path_sections_signature(self):
        from mosaicx.pipelines.pathology import ParsePathSections
        assert "report_text" in ParsePathSections.input_fields
        assert "sections" in ParsePathSections.output_fields

    def test_extract_specimen_details_signature(self):
        from mosaicx.pipelines.pathology import ExtractSpecimenDetails
        assert "gross_text" in ExtractSpecimenDetails.input_fields
        assert "site" in ExtractSpecimenDetails.output_fields

    def test_extract_microscopic_findings_signature(self):
        from mosaicx.pipelines.pathology import ExtractMicroscopicFindings
        assert "microscopic_text" in ExtractMicroscopicFindings.input_fields
        assert "findings" in ExtractMicroscopicFindings.output_fields

    def test_extract_path_diagnosis_signature(self):
        from mosaicx.pipelines.pathology import ExtractPathDiagnosis
        assert "diagnosis_text" in ExtractPathDiagnosis.input_fields
        assert "diagnoses" in ExtractPathDiagnosis.output_fields


class TestPathologyStructurerModule:
    def test_module_has_submodules(self):
        from mosaicx.pipelines.pathology import PathologyReportStructurer
        pipeline = PathologyReportStructurer()
        assert hasattr(pipeline, "classify_specimen")
        assert hasattr(pipeline, "parse_sections")
        assert hasattr(pipeline, "extract_specimen_details")
        assert hasattr(pipeline, "extract_findings")
        assert hasattr(pipeline, "extract_diagnosis")
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pathology_pipeline.py -v`
Expected: Models pass, signature + module tests FAIL

**Step 3: Write the implementation**

```python
# mosaicx/pipelines/pathology.py
"""Pathology report structurer pipeline.

A 5-step DSPy chain that converts free-text pathology reports into
structured data:

    1. **ClassifySpecimenType** -- Identify specimen / procedure type.
    2. **ParsePathSections** -- Split the report into standard sections
       (clinical history, gross, microscopic, diagnosis, ancillary).
    3. **ExtractSpecimenDetails** -- Parse specimen site, laterality,
       procedure, dimensions.
    4. **ExtractMicroscopicFindings** -- Extract structured findings
       with histology, grade, margins, invasion status.
    5. **ExtractPathDiagnosis** -- Extract diagnoses with WHO
       classification, TNM staging, ICD-O codes, and biomarkers.

Each step is independently optimizable via GEPA.

DSPy-dependent classes are lazily imported via module-level ``__getattr__``
so that the module can be imported even when dspy is not installed.
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
    """Lazily define and return all DSPy signatures and the pipeline module."""
    import dspy

    # -- Step 1: Classify specimen type ------------------------------------

    class ClassifySpecimenType(dspy.Signature):
        """Classify the specimen type from the pathology report header."""

        report_header: str = dspy.InputField(
            desc="Header / title portion of the pathology report"
        )
        specimen_type: str = dspy.OutputField(
            desc="Specimen type (e.g., Biopsy - Prostate, Resection - Colon, Cytology - Thyroid FNA)"
        )

    # -- Step 2: Parse report sections -------------------------------------

    class ParsePathSections(dspy.Signature):
        """Split a pathology report into its standard sections."""

        report_text: str = dspy.InputField(
            desc="Full text of the pathology report"
        )
        sections: PathSections = dspy.OutputField(
            desc="Parsed report sections (clinical history, gross, microscopic, diagnosis, ancillary)"
        )

    # -- Step 3: Extract specimen details ----------------------------------

    class ExtractSpecimenDetails(dspy.Signature):
        """Extract specimen details from the gross description."""

        gross_text: str = dspy.InputField(
            desc="Text of the gross description section"
        )
        site: str = dspy.OutputField(desc="Anatomical site of the specimen")
        laterality: str = dspy.OutputField(
            desc="Laterality (left/right/bilateral/N/A)"
        )
        procedure: str = dspy.OutputField(
            desc="Procedure type (biopsy, excision, resection)"
        )
        dimensions: str = dspy.OutputField(
            desc="Specimen dimensions (e.g., 3.2 x 2.1 x 1.5 cm)"
        )
        specimens_received: int = dspy.OutputField(
            desc="Number of specimens / parts received"
        )

    # -- Step 4: Extract microscopic findings ------------------------------

    class ExtractMicroscopicFindings(dspy.Signature):
        """Extract structured findings from the microscopic description."""

        microscopic_text: str = dspy.InputField(
            desc="Text of the microscopic description"
        )
        specimen_type: str = dspy.InputField(
            desc="Specimen type for context"
        )
        findings: List[PathFinding] = dspy.OutputField(
            desc="List of structured microscopic findings"
        )

    # -- Step 5: Extract diagnosis -----------------------------------------

    class ExtractPathDiagnosis(dspy.Signature):
        """Extract diagnoses with staging and biomarker data."""

        diagnosis_text: str = dspy.InputField(
            desc="Text of the diagnosis section"
        )
        findings_context: str = dspy.InputField(
            desc="JSON summary of previously extracted findings for grounding",
            default="",
        )
        ancillary_text: str = dspy.InputField(
            desc="Text of the ancillary studies section",
            default="",
        )
        diagnoses: List[PathDiagnosis] = dspy.OutputField(
            desc="List of structured pathologic diagnoses"
        )

    # -- Pipeline module ---------------------------------------------------

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
            """Run the full 5-step pathology structuring pipeline."""
            # Step 1: classify specimen type
            header = report_header or report_text[:200]
            classify_result = self.classify_specimen(report_header=header)
            specimen_type: str = classify_result.specimen_type

            # Step 2: parse sections
            sections_result = self.parse_sections(report_text=report_text)
            sections: PathSections = sections_result.sections

            # Step 3: extract specimen details
            specimen_result = self.extract_specimen_details(
                gross_text=sections.gross_description,
            )

            # Step 4: extract microscopic findings
            findings_result = self.extract_findings(
                microscopic_text=sections.microscopic,
                specimen_type=specimen_type,
            )
            findings: List[PathFinding] = findings_result.findings

            # Step 5: extract diagnosis
            findings_json = "[" + ", ".join(
                f.model_dump_json() for f in findings
            ) + "]"
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

    return {
        "ClassifySpecimenType": ClassifySpecimenType,
        "ParsePathSections": ParsePathSections,
        "ExtractSpecimenDetails": ExtractSpecimenDetails,
        "ExtractMicroscopicFindings": ExtractMicroscopicFindings,
        "ExtractPathDiagnosis": ExtractPathDiagnosis,
        "PathologyReportStructurer": PathologyReportStructurer,
    }


# Cache for lazily-built DSPy classes
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
    """Module-level __getattr__ for lazy loading of DSPy classes."""
    global _dspy_classes

    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pathology_pipeline.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add mosaicx/pipelines/pathology.py tests/test_pathology_pipeline.py
git commit -m "feat: add pathology report structurer pipeline"
```

---

### Task 4: Register Modes

**Files:**
- Modify: `mosaicx/pipelines/radiology.py` — register as mode
- Modify: `mosaicx/pipelines/pathology.py` — register as mode
- Modify: `tests/test_modes.py` — test both modes are registered

**Step 1: Add failing test**

Add to `tests/test_modes.py`:

```python
class TestBuiltInModes:
    def test_radiology_mode_registered(self):
        # Import triggers registration
        import mosaicx.pipelines.radiology  # noqa: F401
        from mosaicx.pipelines.modes import MODES
        assert "radiology" in MODES

    def test_pathology_mode_registered(self):
        import mosaicx.pipelines.pathology  # noqa: F401
        from mosaicx.pipelines.modes import MODES
        assert "pathology" in MODES

    def test_list_modes_includes_both(self):
        import mosaicx.pipelines.radiology  # noqa: F401
        import mosaicx.pipelines.pathology  # noqa: F401
        from mosaicx.pipelines.modes import list_modes
        names = [name for name, _ in list_modes()]
        assert "radiology" in names
        assert "pathology" in names
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_modes.py::TestBuiltInModes -v`
Expected: FAIL (modes not registered)

**Step 3: Register modes**

In `mosaicx/pipelines/radiology.py`, inside `_build_dspy_classes()` after defining `RadiologyReportStructurer`, register it:

```python
    from mosaicx.pipelines.modes import register_mode
    register_mode("radiology", "5-step radiology report structurer (findings, measurements, scoring)")(RadiologyReportStructurer)
```

In `mosaicx/pipelines/pathology.py`, inside `_build_dspy_classes()` after defining `PathologyReportStructurer`, register it:

```python
    from mosaicx.pipelines.modes import register_mode
    register_mode("pathology", "5-step pathology report structurer (histology, staging, biomarkers)")(PathologyReportStructurer)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_modes.py -v`
Expected: ALL PASS

**Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/ --ignore=tests/test_ocr_integration.py -q`
Expected: All pass (255 + new tests)

**Step 6: Commit**

```bash
git add mosaicx/pipelines/radiology.py mosaicx/pipelines/pathology.py tests/test_modes.py
git commit -m "feat: register radiology and pathology as extraction modes"
```

---

### Task 5: Rewrite Extraction Pipeline

**Files:**
- Modify: `mosaicx/pipelines/extraction.py` — replace 3-step chain with auto-infer + schema-first
- Modify: `tests/test_extraction_pipeline.py` — update tests

**Step 1: Write the failing tests**

Replace `tests/test_extraction_pipeline.py` entirely:

```python
# tests/test_extraction_pipeline.py
"""Tests for the extraction pipeline."""
import pytest


class TestInferSchemaSignature:
    """Test the auto-infer schema signature."""

    def test_infer_schema_signature_fields(self):
        from mosaicx.pipelines.extraction import InferSchemaFromDocument
        assert "document_text" in InferSchemaFromDocument.input_fields
        assert "schema_spec" in InferSchemaFromDocument.output_fields


class TestDocumentExtractorModule:
    """Test the DocumentExtractor DSPy module."""

    def test_auto_mode_has_infer_and_extract(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        extractor = DocumentExtractor()
        assert hasattr(extractor, "infer_schema")
        assert hasattr(extractor, "extract_custom")

    def test_schema_mode_has_extract_custom(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        from pydantic import BaseModel

        class CustomReport(BaseModel):
            summary: str
            category: str

        extractor = DocumentExtractor(output_schema=CustomReport)
        assert hasattr(extractor, "extract_custom")
        assert not hasattr(extractor, "infer_schema")

    def test_extract_with_schema_function(self):
        """Test the extract_with_schema convenience function exists."""
        from mosaicx.pipelines.extraction import extract_with_schema
        assert callable(extract_with_schema)

    def test_extract_with_mode_function(self):
        """Test the extract_with_mode convenience function exists."""
        from mosaicx.pipelines.extraction import extract_with_mode
        assert callable(extract_with_mode)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extraction_pipeline.py -v`
Expected: FAIL

**Step 3: Rewrite the extraction pipeline**

Replace `mosaicx/pipelines/extraction.py` with the new implementation. Key changes:
- Remove `Demographics`, `Finding`, `Diagnosis` models (hardcoded defaults)
- Remove `ExtractDemographics`, `ExtractFindings`, `ExtractDiagnoses` signatures
- Add `InferSchemaFromDocument` signature that outputs a `SchemaSpec`
- Rewrite `DocumentExtractor` to support two modes: auto-infer and schema-first
- Add `extract_with_schema()` and `extract_with_mode()` convenience functions

The new `DocumentExtractor.__init__` takes an optional `output_schema`:
- If provided: single-step extraction into that schema (same as current custom mode)
- If not provided: two-step — infer schema from doc, then extract into it

Add convenience functions:
```python
def extract_with_schema(document_text: str, schema_name: str, schema_dir: Path) -> dict:
    """Load a schema by name and extract into it."""
    from .schema_gen import load_schema, compile_schema
    spec = load_schema(schema_name, schema_dir)
    model = compile_schema(spec)
    extractor = DocumentExtractor(output_schema=model)
    result = extractor(document_text=document_text)
    return result.extracted.model_dump() if hasattr(result.extracted, "model_dump") else result.extracted

def extract_with_mode(document_text: str, mode_name: str) -> dict:
    """Run a registered extraction mode."""
    from .modes import get_mode
    mode_cls = get_mode(mode_name)
    pipeline = mode_cls()
    result = pipeline(report_text=document_text)
    # Convert dspy.Prediction to dict
    output = {}
    for key in result.keys():
        val = getattr(result, key)
        if hasattr(val, "model_dump"):
            output[key] = val.model_dump()
        elif isinstance(val, list):
            output[key] = [v.model_dump() if hasattr(v, "model_dump") else v for v in val]
        else:
            output[key] = val
    return output
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extraction_pipeline.py -v`
Expected: ALL PASS

**Step 5: Run full test suite**

Run: `uv run pytest tests/ --ignore=tests/test_ocr_integration.py -q`
Expected: Check for failures in other tests that import from extraction.py. Update any imports of the removed `Demographics`, `Finding`, `Diagnosis` models.

Known files that reference old extraction models:
- `tests/test_public_api.py` — may reference old output structure
- `tests/test_cli_integration.py` — extract command tests
- `mosaicx/__init__.py` — `extract()` public API
- `mosaicx/cli.py` — extract command

Fix any broken imports/references. The old models (`Demographics`, `Finding`, `Diagnosis`) are deleted — update all code that references them.

**Step 6: Commit**

```bash
git add mosaicx/pipelines/extraction.py tests/test_extraction_pipeline.py
git commit -m "feat: rewrite extraction pipeline with auto-infer and schema-first modes"
```

---

### Task 6: Rewrite CLI Extract Command

**Files:**
- Modify: `mosaicx/cli.py` — rewrite extract command with --schema, --mode, --list-modes
- Modify: `tests/test_cli_integration.py` — update extract tests

**Step 1: Update CLI extract tests**

Update `tests/test_cli_integration.py` class `TestLLMCommandsGracefulFailure`:

- Remove `test_extract_bad_template` (template registry lookup is removed)
- Add `test_extract_with_schema_no_api_key` — `extract --document X --schema Y` fails gracefully
- Add `test_extract_with_mode_no_api_key` — `extract --document X --mode radiology` fails gracefully
- Add `test_extract_unknown_mode` — `extract --document X --mode nonexistent` shows error
- Add `test_extract_list_modes` — `extract --list-modes` shows available modes without error
- Keep `test_extract_no_api_key`, `test_extract_no_document`, `test_extract_missing_document`

**Step 2: Rewrite the extract CLI command**

Replace the `extract` command in `mosaicx/cli.py`. Key changes:

- Remove `--template` option for registry lookup (keep YAML file path support via `--template`)
- Add `--schema` option — loads schema by name from `~/.mosaicx/schemas/`
- Add `--mode` option — `click.Choice` from registered modes
- Add `--list-modes` flag — prints available modes and exits
- Mutually exclusive: `--schema` and `--mode` cannot be used together
- Auto mode: when neither `--schema` nor `--mode` is specified, run auto-infer

**Step 3: Remove `template create` stub**

Delete the `template_create` command from CLI.

**Step 4: Run tests**

Run: `uv run pytest tests/test_cli_integration.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add mosaicx/cli.py tests/test_cli_integration.py
git commit -m "feat: rewrite extract command with --schema and --mode flags"
```

---

### Task 7: Implement Batch Processing

**Files:**
- Modify: `mosaicx/batch.py` — implement `process_directory()`
- Modify: `mosaicx/cli.py` — wire batch with --schema/--mode, real processing
- Modify: `tests/test_batch.py` — add real processing tests

**Step 1: Write the failing tests**

Add to `tests/test_batch.py`:

```python
class TestBatchProcessorReal:
    def test_process_empty_directory(self, tmp_path):
        from mosaicx.batch import BatchProcessor
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        proc = BatchProcessor(workers=1)
        result = proc.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            process_fn=lambda text: {"extracted": "test"},
        )
        assert result["total"] == 0
        assert result["succeeded"] == 0

    def test_process_txt_files(self, tmp_path):
        from mosaicx.batch import BatchProcessor
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create some test documents
        (input_dir / "doc1.txt").write_text("Patient report 1")
        (input_dir / "doc2.txt").write_text("Patient report 2")
        (input_dir / "not_a_doc.xyz").write_text("skip this")

        proc = BatchProcessor(workers=1)
        result = proc.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            process_fn=lambda text: {"summary": "extracted"},
        )
        assert result["total"] == 2  # only .txt files
        assert result["succeeded"] == 2
        assert output_dir.exists()

    def test_process_with_error_isolation(self, tmp_path):
        from mosaicx.batch import BatchProcessor
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        (input_dir / "good.txt").write_text("Good report")
        (input_dir / "bad.txt").write_text("Bad report")

        call_count = 0
        def flaky_fn(text):
            nonlocal call_count
            call_count += 1
            if "Bad" in text:
                raise ValueError("Simulated extraction failure")
            return {"result": "ok"}

        proc = BatchProcessor(workers=1)
        result = proc.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            process_fn=flaky_fn,
        )
        assert result["succeeded"] == 1
        assert result["failed"] == 1

    def test_process_with_resume(self, tmp_path):
        from mosaicx.batch import BatchProcessor, BatchCheckpoint
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        checkpoint_dir = tmp_path / "checkpoints"

        (input_dir / "doc1.txt").write_text("Report 1")
        (input_dir / "doc2.txt").write_text("Report 2")

        # Pre-populate checkpoint as if doc1 was already processed
        cp = BatchCheckpoint(batch_id="resume", checkpoint_dir=checkpoint_dir)
        cp.mark_completed("doc1.txt", {"status": "ok"})
        cp.save()

        call_count = 0
        def counting_fn(text):
            nonlocal call_count
            call_count += 1
            return {"result": "ok"}

        proc = BatchProcessor(workers=1)
        result = proc.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            process_fn=counting_fn,
            resume_id="resume",
            checkpoint_dir=checkpoint_dir,
        )
        assert call_count == 1  # only doc2 was processed
        assert result["skipped"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_batch.py::TestBatchProcessorReal -v`
Expected: FAIL (stub returns wrong format)

**Step 3: Implement `process_directory()`**

Rewrite `mosaicx/batch.py` `BatchProcessor.process_directory()`:

```python
def process_directory(
    self,
    input_dir: Path,
    output_dir: Path,
    process_fn: Callable[[str], dict[str, Any]],
    resume_id: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Process all documents in a directory.

    Args:
        input_dir: Directory of input documents.
        output_dir: Directory for output JSON files.
        process_fn: Function that takes document text and returns a dict.
        resume_id: If set, resume from checkpoint with this ID.
        checkpoint_dir: Directory for checkpoint files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover documents
    supported = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    docs = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported
    )

    # Load checkpoint if resuming
    checkpoint: BatchCheckpoint | None = None
    skipped = 0
    if resume_id and checkpoint_dir:
        cp_path = checkpoint_dir / f"{resume_id}.json"
        if cp_path.exists():
            checkpoint = BatchCheckpoint.load(cp_path)
            # Filter out already-completed docs
            remaining = []
            for d in docs:
                if checkpoint.is_completed(d.name):
                    skipped += 1
                else:
                    remaining.append(d)
            docs = remaining
        else:
            checkpoint = BatchCheckpoint(
                batch_id=resume_id, checkpoint_dir=checkpoint_dir
            )
    elif resume_id:
        # Use output_dir for checkpoints if no checkpoint_dir specified
        ckpt_dir = checkpoint_dir or output_dir
        checkpoint = BatchCheckpoint(batch_id=resume_id, checkpoint_dir=ckpt_dir)

    succeeded = 0
    failed = 0
    errors: list[dict[str, str]] = []

    def _process_one(doc_path: Path) -> tuple[str, dict | None, str | None]:
        try:
            text = doc_path.read_text(encoding="utf-8")
            result = process_fn(text)
            # Write result
            out_path = output_dir / f"{doc_path.stem}.json"
            out_path.write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )
            return doc_path.name, result, None
        except Exception as exc:
            return doc_path.name, None, str(exc)

    # Process with thread pool
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=self.workers) as pool:
        futures = {pool.submit(_process_one, d): d for d in docs}
        processed = 0
        for future in as_completed(futures):
            name, result, error = future.result()
            processed += 1
            if error:
                failed += 1
                errors.append({"file": name, "error": error})
            else:
                succeeded += 1
                if checkpoint:
                    checkpoint.mark_completed(name, result or {})
                    if processed % self.checkpoint_every == 0:
                        checkpoint.save()

    # Final checkpoint save
    if checkpoint:
        checkpoint.save()

    return {
        "total": len(docs) + skipped,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
    }
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_batch.py -v`
Expected: ALL PASS

**Step 5: Wire batch CLI with --schema and --mode**

Update the `batch` CLI command in `mosaicx/cli.py`:
- Add `--schema` and `--mode` options (same as extract)
- Replace `process_fn=lambda doc: doc` with real extraction logic
- Add `--checkpoint-dir` option (defaults to config)
- Show summary after processing (total, succeeded, failed, skipped)

**Step 6: Run full test suite**

Run: `uv run pytest tests/ --ignore=tests/test_ocr_integration.py -q`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add mosaicx/batch.py mosaicx/cli.py tests/test_batch.py
git commit -m "feat: implement batch processing with parallel workers and checkpointing"
```

---

### Task 8: Update Public API and README

**Files:**
- Modify: `mosaicx/__init__.py` — update `extract()` to accept schema/mode
- Modify: `tests/test_public_api.py` — update extract tests
- Modify: `README.md` — document new extraction paths and modes

**Step 1: Update public API**

Update `mosaicx/__init__.py` `extract()` function:
- Add `schema: str | None = None` parameter
- Add `mode: str | None = None` parameter
- Remove `template` parameter (or keep for YAML file paths only)
- When `schema` is provided: load schema, compile, extract into it
- When `mode` is provided: run the mode pipeline
- When neither: auto-infer schema from document

**Step 2: Update public API tests**

Update `tests/test_public_api.py` to test new signature.

**Step 3: Update README**

- Update the command table with new flags
- Rewrite the extraction section to document auto/schema/mode paths
- Add the pathology mode
- Document batch processing with --schema/--mode
- Remove references to template registry lookup
- Remove `template create` from docs

**Step 4: Run full test suite**

Run: `uv run pytest tests/ --ignore=tests/test_ocr_integration.py -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add mosaicx/__init__.py tests/test_public_api.py README.md
git commit -m "feat: update public API and docs for extraction redesign"
```

---

### Task 9: Final Verification and Cleanup

**Step 1: Run full test suite**

Run: `uv run pytest tests/ --ignore=tests/test_ocr_integration.py -q`
Expected: ALL PASS

**Step 2: Check for any remaining references to old code**

Search for:
- `Demographics` imports from extraction.py (should be removed)
- `Finding` / `Diagnosis` imports from extraction.py (should be removed)
- `template create` references
- `template_create` function
- `process_fn=lambda doc: doc`

**Step 3: Verify CLI commands work**

Run: `uv run mosaicx extract --help` — should show --schema, --mode, --list-modes
Run: `uv run mosaicx extract --list-modes` — should show radiology and pathology
Run: `uv run mosaicx batch --help` — should show --schema, --mode
Run: `uv run mosaicx schema --help` — should work as before

**Step 4: Commit any final fixes**

```bash
git add -A && git commit -m "chore: cleanup old extraction references"
```

**Step 5: Push**

```bash
git push
```
