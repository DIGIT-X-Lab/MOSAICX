# Extraction Redesign

## Problem

`mosaicx extract` has a hardcoded 3-step chain (demographics → findings → diagnoses) as its default path. This ignores the schema system entirely — if the whole point of `schema generate` is to define *what* to extract, then `extract` should use those schemas. The current default is a rigid fallback that doesn't adapt to the document.

Additionally:
- The `RadiologyReportStructurer` pipeline exists but is never wired to the CLI.
- The template registry (`--template chest_ct`) looks up metadata but never compiles a schema — it silently falls back to the default 3-step chain.
- `BatchProcessor.process_directory()` is a stub.
- `template create` is a stub with no clear purpose (redundant with `schema generate`).

## Goals

1. **Schema-first extraction** — `mosaicx extract --schema X` extracts into a user-generated schema.
2. **Smart default** — when no schema is given, the LLM auto-infers a schema from the document content, then extracts into it.
3. **Extensible mode system** — `--mode radiology` runs a specialized multi-step pipeline. New modes (pathology, etc.) can be added with one class + one registry line.
4. **Working batch processing** — `mosaicx batch` actually processes documents with the same three paths (auto/schema/mode).
5. **Clean up dead code** — remove broken template registry integration and `template create` stub.

## Design

### Extraction paths

```
mosaicx extract --document report.pdf [--schema X] [--mode Y]
```

| Flags | Behavior |
|-------|----------|
| (none) | **Auto mode** — LLM reads the doc, infers a schema, extracts into it. Two LLM calls. |
| `--schema EchoReport` | **Schema mode** — loads schema from `~/.mosaicx/schemas/`, single-step extraction into that shape. |
| `--mode radiology` | **Mode** — runs the specialized 5-step radiology pipeline. |
| `--mode pathology` | **Mode** — runs the specialized 5-step pathology pipeline. |

`--schema` and `--mode` are mutually exclusive. `--template custom.yaml` still works (compiles YAML → Pydantic model → schema mode).

### Auto mode (no schema, no mode)

Two-step process:

1. **Infer schema** — new DSPy signature `InferSchemaFromDocument`:
   - Input: first 2000 chars of document text
   - Output: `SchemaSpec` (same model used by `schema generate`)
   - Compiles the SchemaSpec into a Pydantic model via `compile_schema()`

2. **Extract** — single-step `ChainOfThought` extraction into the inferred model (same as current custom schema mode).

The inferred schema is shown to the user but not saved to disk (ephemeral). Users who want to reuse it can run `schema generate` explicitly.

### Mode system

Each mode is a DSPy Module subclass registered in a dict:

```python
# mosaicx/pipelines/modes.py

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
    if name not in MODES:
        available = ", ".join(sorted(MODES))
        raise ValueError(f"Unknown mode: {name!r}. Available: {available}")
    return MODES[name]

def list_modes() -> list[tuple[str, str]]:
    return [(name, cls.mode_description) for name, cls in sorted(MODES.items())]
```

Each mode class must implement `forward(self, document_text: str) -> dict[str, Any]` returning a JSON-serializable dict.

### Radiology mode

Already exists in `mosaicx/pipelines/radiology.py`. Wire it as:

```python
@register_mode("radiology", "5-step radiology report structurer (findings, measurements, scoring)")
class RadiologyMode(RadiologyReportStructurer):
    ...
```

The `forward()` method already exists. We just need to wrap the `dspy.Prediction` output into a plain dict for the CLI.

### Pathology mode

New file: `mosaicx/pipelines/pathology.py`

5-step DSPy chain mirroring the radiology pipeline pattern:

**Step 1: ClassifySpecimenType**
- Input: report header / first 200 chars
- Output: `specimen_type: str` (e.g., "Biopsy - Prostate", "Resection - Colon", "Cytology - Thyroid FNA")

**Step 2: ParsePathSections**
- Input: full report text
- Output: `PathSections` Pydantic model:
  - `clinical_history: str`
  - `gross_description: str`
  - `microscopic: str`
  - `diagnosis: str`
  - `ancillary_studies: str` (IHC, molecular, flow cytometry)
  - `comment: str`

**Step 3: ExtractSpecimenDetails**
- Input: gross description text
- Output:
  - `site: str` (anatomical site)
  - `laterality: str` (left/right/bilateral/N/A)
  - `procedure: str` (biopsy, excision, resection)
  - `dimensions: str` (e.g., "3.2 x 2.1 x 1.5 cm")
  - `specimens_received: int`

**Step 4: ExtractMicroscopicFindings**
- Input: microscopic text + specimen type for context
- Output: `List[PathFinding]` Pydantic model:
  - `description: str`
  - `histologic_type: str` (e.g., "invasive ductal carcinoma")
  - `grade: str` (e.g., "Nottingham grade 2", "Gleason 3+4=7")
  - `margins: str` (positive/negative/close + distance)
  - `lymphovascular_invasion: str` (present/absent/indeterminate)
  - `perineural_invasion: str` (present/absent/indeterminate)

**Step 5: ExtractPathDiagnosis**
- Input: diagnosis text + findings context
- Output: `List[PathDiagnosis]` Pydantic model:
  - `diagnosis: str` (primary diagnosis text)
  - `who_classification: str` (WHO tumor classification if applicable)
  - `tnm_stage: str` (pTNM staging)
  - `icd_o_morphology: str` (ICD-O morphology code)
  - `biomarkers: list[Biomarker]` (name, result, method — e.g., ER+, PR+, HER2-, Ki-67 30%)
  - `ancillary_results: str` (molecular/IHC summary)

### Pydantic models for pathology

New file: `mosaicx/schemas/pathreport/base.py`

```python
class PathSections(BaseModel):
    clinical_history: str = ""
    gross_description: str = ""
    microscopic: str = ""
    diagnosis: str = ""
    ancillary_studies: str = ""
    comment: str = ""

class PathFinding(BaseModel):
    description: str
    histologic_type: Optional[str] = None
    grade: Optional[str] = None
    margins: Optional[str] = None
    lymphovascular_invasion: Optional[str] = None
    perineural_invasion: Optional[str] = None

class Biomarker(BaseModel):
    name: str
    result: str
    method: Optional[str] = None

class PathDiagnosis(BaseModel):
    diagnosis: str
    who_classification: Optional[str] = None
    tnm_stage: Optional[str] = None
    icd_o_morphology: Optional[str] = None
    biomarkers: list[Biomarker] = []
    ancillary_results: Optional[str] = None
```

### Batch processing

Wire `BatchProcessor.process_directory()` to:

1. Discover documents in `input_dir` (PDF, DOCX, TXT, images)
2. For each document (in parallel with `concurrent.futures.ThreadPoolExecutor`):
   a. Load via `load_document()` with OCR config
   b. Run extraction (auto / schema / mode — same flags as `mosaicx extract`)
   c. Write result JSON to `output_dir/{filename}.json`
   d. Mark completed in checkpoint
3. After all documents: run export (CSV, Parquet, JSONL, FHIR, Markdown) on collected results
4. Print summary (total, succeeded, failed, skipped)

Checkpoint resume: on `--resume`, load checkpoint, skip already-completed documents.

Error isolation: if one document fails, log the error and continue with the next.

### CLI changes

**`mosaicx extract` (modify):**
```bash
# Auto mode (LLM infers schema)
mosaicx extract --document report.pdf

# Schema mode
mosaicx extract --document report.pdf --schema EchoReport

# Template mode (YAML file → compiled schema)
mosaicx extract --document report.pdf --template ./custom.yaml

# Specialized mode
mosaicx extract --document report.pdf --mode radiology
mosaicx extract --document report.pdf --mode pathology
```

New flags: `--schema`, `--mode`. Remove `--template` registry lookup (keep YAML file support). Remove default demographics/findings/diagnoses chain.

**`mosaicx batch` (modify):**
```bash
mosaicx batch \
  --input-dir ./reports \
  --output-dir ./structured \
  --schema EchoReport \          # or --mode radiology, or neither for auto
  --format parquet jsonl \
  --workers 4 \
  --resume
```

Add `--schema` and `--mode` flags (same as extract). Wire to real implementation.

**`mosaicx template create` (remove):**
Redundant with `schema generate`. Delete the stub.

**`mosaicx extract --list-modes` (new):**
Show available modes with descriptions.

## Files

| File | Change |
|------|--------|
| `mosaicx/pipelines/modes.py` | **New** — mode registry (`MODES` dict, `register_mode`, `get_mode`, `list_modes`) |
| `mosaicx/pipelines/extraction.py` | Rewrite — auto-infer schema path, schema-first path, remove hardcoded 3-step chain |
| `mosaicx/pipelines/radiology.py` | Add `@register_mode("radiology", ...)` decorator, add result-to-dict helper |
| `mosaicx/pipelines/pathology.py` | **New** — 5-step pathology pipeline, registered as mode |
| `mosaicx/schemas/pathreport/__init__.py` | **New** |
| `mosaicx/schemas/pathreport/base.py` | **New** — PathSections, PathFinding, Biomarker, PathDiagnosis models |
| `mosaicx/batch.py` | Implement `process_directory()` with parallel workers, checkpointing, export |
| `mosaicx/cli.py` | Rewrite extract command, wire batch, remove template create, add --schema/--mode flags |
| `mosaicx/__init__.py` | Update `extract()` public API to accept schema/mode |
| `tests/test_extraction_pipeline.py` | Update tests for new extraction paths |
| `tests/test_pathology_pipeline.py` | **New** — tests for pathology pipeline |
| `tests/test_batch.py` | Update tests for real batch processing |
| `tests/test_modes.py` | **New** — tests for mode registry |

## Verification

1. `mosaicx extract --document report.pdf` — auto mode infers schema and extracts
2. `mosaicx extract --document report.pdf --schema PatientInfo` — extracts into user schema
3. `mosaicx extract --document report.pdf --mode radiology` — runs 5-step radiology chain
4. `mosaicx extract --document report.pdf --mode pathology` — runs 5-step pathology chain
5. `mosaicx extract --list-modes` — shows radiology and pathology with descriptions
6. `mosaicx batch --input-dir ./reports --output-dir ./out --mode radiology --workers 4` — batch processes
7. `mosaicx batch --input-dir ./reports --output-dir ./out --resume` — resumes from checkpoint
8. All existing tests pass
