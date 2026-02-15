# MOSAICX v2 — DSPy-Powered Medical Document Structuring Platform

**Date**: 2026-02-15
**Author**: DIGIT-X Lab, LMU Radiology
**Status**: Design approved, pending implementation plan

## 1. Overview

MOSAICX v2 is a ground-up rewrite of the medical document structuring platform. It replaces the current fragmented LLM integration (11 call sites across 3 libraries with hand-crafted prompts) with a unified DSPy-powered pipeline that is composable, optimizable, and self-improving.

**Target audience**: Clinical researchers who need structured, ML-ready datasets from unstructured radiology reports and clinical documents.

**Core promise**: One command to go from a folder of PDFs to a structured, coded, quality-scored Parquet/CSV/FHIR dataset.

### What changes

| Aspect | v1 (current) | v2 (rewrite) |
|--------|-------------|--------------|
| LLM integration | OpenAI client + Instructor + Outlines + raw Ollama | DSPy modules with typed signatures |
| Prompt engineering | Hand-crafted strings, no optimization | GEPA/MIPROv2 auto-optimized instructions |
| Quality assurance | None — no metrics, no evaluation | Completeness scoring + reward-driven refinement |
| Backend support | Brittle URL pattern matching | `dspy.LM()` via LiteLLM (50+ providers) |
| Schema generation | LLM generates Python code, `exec()`s it | Structured `SchemaSpec` → `pydantic.create_model()` |
| Radiology support | Generic extraction only | RadReport-aligned templates with scoring systems |
| Ontology coding | None | Multi-ontology resolution (RadLex, SNOMED, ICD-10, LOINC) |
| Interoperability | JSON export only | FHIR R4 DiagnosticReport bundles |
| Batch processing | None | Checkpointed parallel processing with quality dashboard |
| De-identification | None | Belt-and-suspenders LLM + regex PHI removal |
| Webapp | FastAPI shelling out to CLI via subprocess | Dropped (CLI + Python API only) |
| Dependencies | Instructor, Outlines, json-repair, raw OpenAI, Ollama client | DSPy (handles all LLM orchestration) |

### What stays

- **Docling** for document loading (PDF/DOCX/PPTX → Markdown)
- **Pydantic** for data validation and schema models
- **Click + Rich** for CLI and terminal UI
- **ReportLab / python-docx** for PDF/DOCX export
- **AGPL-3.0** license

## 2. Architecture

```
mosaicx/
├── __init__.py                    # Public API: extract(), summarize(), generate_schema()
├── cli.py                         # Click CLI (thin wrapper over API)
├── config.py                      # Pydantic Settings — single config source
├── display.py                     # Rich terminal UI (kept, simplified)
│
├── pipelines/                     # DSPy module pipelines
│   ├── __init__.py
│   ├── extraction.py              # DocumentExtractor — generic multi-step extraction
│   ├── radiology.py               # RadiologyReportStructurer — RadReport-aligned
│   ├── schema_gen.py              # SchemaGenerator — schema from natural language
│   ├── summarizer.py              # ReportSummarizer — timeline generation
│   └── deidentifier.py            # Deidentifier — PHI removal
│
├── documents/                     # Document loading
│   ├── __init__.py
│   ├── loader.py                  # Docling DocumentConverter → LoadedDocument
│   └── vision.py                  # dspy.Image VLM fallback for scanned docs
│
├── schemas/                       # Schema management & ontologies
│   ├── __init__.py
│   ├── registry.py                # Schema registry (simplified)
│   ├── template_compiler.py       # YAML template → Pydantic model → DSPy pipeline
│   ├── ontology.py                # Multi-ontology resolution (RadLex/SNOMED/ICD-10/LOINC)
│   ├── fhir.py                    # FHIR R4 resource builders
│   ├── radreport/
│   │   ├── __init__.py
│   │   ├── base.py                # Base RadReport models (Finding, Measurement, etc.)
│   │   ├── registry.py            # Template discovery & auto-selection
│   │   ├── scoring.py             # BI-RADS, Lung-RADS, TI-RADS, etc.
│   │   └── templates/
│   │       ├── chest_ct.py
│   │       ├── chest_xr.py
│   │       ├── brain_mri.py
│   │       ├── abdomen_ct.py
│   │       ├── mammography.py
│   │       ├── thyroid_us.py
│   │       ├── lung_ct.py
│   │       ├── msk_mri.py
│   │       ├── cardiac_mri.py
│   │       ├── pet_ct.py
│   │       └── generic.py
│   └── terminologies/
│       ├── radlex_core.json.gz    # ~46K terms, ~3MB
│       ├── snomed_radiology.json.gz
│       ├── icd10_cm.json.gz
│       └── loinc_radiology.json.gz
│
├── evaluation/                    # Quality, metrics & optimization
│   ├── __init__.py
│   ├── completeness.py            # CompletenessEvaluator — 3-layer scoring
│   ├── metrics.py                 # DSPy metrics (extraction, deidentification, summarization)
│   ├── rewards.py                 # Reward functions for dspy.Refine
│   └── optimize.py                # GEPA/MIPROv2/BootstrapFewShot workflows
│
├── export/                        # Output formats
│   ├── __init__.py
│   ├── tabular.py                 # CSV, Parquet, HuggingFace Datasets, JSONL
│   ├── fhir_bundle.py             # FHIR R4 Bundle assembly
│   └── report.py                  # PDF/DOCX/Markdown narrative reports
│
├── batch.py                       # Batch processing engine with checkpointing
│
└── utils/
    ├── __init__.py
    └── logging.py                 # Simplified session logging
```

## 3. DSPy Pipeline Design

### 3.1 Radiology Report Structurer (`pipelines/radiology.py`)

The flagship pipeline. A 5-step DSPy module chain with RadReport template auto-selection:

**Step 1 — Exam classification** (`dspy.Predict`): Classify the report into an exam type (chest_ct, brain_mri, mammography, etc.) from the first 500 characters. Routes to the correct template. Cheap model sufficient.

**Step 2 — Section parsing** (`dspy.Predict`): Segment the report into standard radiology sections (indication, comparison, technique, findings, impression). Deterministic-grade task, cheap model.

**Step 3 — Technique classification** (`dspy.Predict`): Extract modality, body region, contrast, protocol from the technique section. Constrained output via `Literal` types.

**Step 4 — Findings extraction** (`dspy.ChainOfThought`): The hardest step. Extract structured findings with anatomy (RadLex-coded), measurements, change-from-prior tracking, and severity. Chain-of-thought reasoning helps with ambiguous anatomy and implicit measurements. Template-specific — the chest CT extractor has different output fields than the brain MRI extractor.

**Step 5 — Impression extraction** (`dspy.ChainOfThought`): Extract impression items with standardized scoring (BI-RADS, Lung-RADS, TI-RADS) and actionable flags. Cross-references findings from step 4.

Each step is a separate DSPy module with its own signature, independently optimizable via GEPA. Steps 1-3 can run on a cheap/fast model (3B), steps 4-5 require a strong model (70B+).

The pipeline is wrapped with `dspy.Refine` at inference time — if the reward function scores the extraction below threshold, the system retries with auto-generated feedback.

### 3.2 Generic Document Extractor (`pipelines/extraction.py`)

A 3-module chain for non-radiology documents:

1. **ExtractDemographics** (`dspy.ChainOfThought`) — patient demographics
2. **ExtractFindings** (`dspy.ChainOfThought`) — clinical findings with demographics as context
3. **ExtractDiagnoses** (`dspy.ChainOfThought`) — diagnoses with findings as context

Also supports user-defined schemas: if a custom Pydantic model or YAML template is provided, a dynamic signature is generated and used instead of the built-in 3-step chain.

### 3.3 Schema Generator (`pipelines/schema_gen.py`)

Replaces the current approach (LLM generates Python code → `exec()`) with a safe structured approach:

1. LLM outputs a `SchemaSpec` — a Pydantic model containing `class_name`, `description`, and `fields` (each with name, type, description, constraints)
2. MOSAICX calls `pydantic.create_model()` to build the class programmatically
3. No code generation, no exec, no injection risk
4. The `SchemaSpec` is JSON-serializable and version-controllable

### 3.4 Report Summarizer (`pipelines/summarizer.py`)

Two-step pipeline with parallel per-report processing:

1. **ExtractTimelineEvent** (`dspy.ChainOfThought`) — one event per report, run in parallel via `dspy.Parallel`
2. **SynthesizeTimeline** (`dspy.ChainOfThought`) — weave events into a longitudinal patient narrative

Scales to arbitrary numbers of reports without hitting context limits (unlike v1 which concatenates all reports into one prompt capped at 6000 chars each).

### 3.5 De-identifier (`pipelines/deidentifier.py`)

Belt-and-suspenders pattern:

1. **LLM redaction** via `dspy.ChainOfThought(RedactPHI)` — removes names, locations, context-dependent identifiers
2. **`dspy.Refine`** with `threshold=1.0` — zero tolerance, retries if any PHI detected by reward function
3. **Deterministic regex guard** — final safety net catching dates, SSNs, MRNs, phone numbers

Three modes: full removal, pseudonymization (consistent fake IDs), date-shifting (preserving temporal relationships).

## 4. Custom Template System

Users define extraction templates in YAML — no Python required:

```yaml
name: CardiacMRIReport
description: "Structured cardiac MRI report"
radreport_id: RDES214  # optional RSNA RadReport link

sections:
  - name: indication
    type: str
    required: true
  - name: findings
    type: list
    item:
      type: object
      fields:
        - name: category
          type: enum
          values: ["ventricular_function", "wall_motion", "valvular", "other"]
        - name: description
          type: str
        - name: measurement
          type: object
          required: false
          fields:
            - name: name
              type: str
            - name: value
              type: float
            - name: unit
              type: str
```

The `TemplateCompiler` converts YAML → Pydantic model (via `pydantic.create_model()`) → DSPy signatures → extraction pipeline, all at runtime. Users get the same power as built-in RadReport templates without writing code.

CLI:
```bash
mosaicx template create my_template.yaml
mosaicx extract --document report.pdf --template my_template
```

## 5. RadReport Template Integration

MOSAICX ships with pre-built templates aligned to RSNA RadReport standards for the most common exam types:

| Template | RadReport ID | Scoring System |
|----------|-------------|----------------|
| Chest CT | RDES3 | Lung-RADS |
| Chest X-ray | RDES2 | — |
| Brain MRI | RDES28 | — |
| Abdomen CT | RDES44 | LI-RADS (liver) |
| Mammography | RDES4 | BI-RADS |
| Thyroid US | RDES72 | TI-RADS |
| Lung CT Screening | RDES195 | Lung-RADS |
| MSK MRI | varies | — |
| Cardiac MRI | RDES214 | AHA 17-segment |
| PET/CT | RDES76 | Deauville (lymphoma) |

### Base data models

All templates share common base models:

- **RadReportFinding**: template_field_id, anatomy, radlex_id, observation, value, measurement, change_from_prior
- **Measurement**: value, unit, dimension, prior_value (for change tracking)
- **ChangeType**: status (new/stable/increased/decreased/resolved), prior_date, prior_measurement
- **ImpressionItem**: statement, category (scoring system value), icd10_code, actionable flag

### Template auto-selection

A `ClassifyExamType` DSPy signature reads the first 500 characters of a report and classifies it into one of the template categories. Unknown exam types fall back to the generic extractor.

### Per-template GEPA optimization

Each template's findings extractor is a separate DSPy module, independently optimizable. The chest CT extractor learns different patterns than the brain MRI extractor. 10-15 annotated examples per template are sufficient for GEPA to achieve 20+ percentage point accuracy improvements.

## 6. Completeness Scoring

Every extraction includes a completeness score answering: "how much of the original report did we capture?"

### Three layers

**Layer 1 — Field coverage** (deterministic, zero cost): Are required fields populated? Are lists non-empty? Are nested objects complete? Per-field scores between 0.0 and 1.0.

**Layer 2 — Semantic completeness** (1 LLM call): A `dspy.ChainOfThought(AssessCompleteness)` module identifies clinically meaningful text spans in the original report that are NOT represented in the structured output. Each uncaptured span incurs a penalty.

**Layer 3 — Information density** (deterministic, zero cost): Ratio of structured content tokens to source text tokens. Catches superficial extractions where fields are populated but with minimal content.

### Output

```json
{
  "overall": 0.82,
  "field_coverage": {"indication": 1.0, "findings": 0.90, "impressions": 1.0},
  "missing_required": [],
  "uncaptured_content": ["Patient positioned supine with arms above head"],
  "information_density": 0.68
}
```

Reports with `overall < 0.7` are flagged for manual review in batch processing.

## 7. Ontology Resolution

### Hybrid approach: lookup table + LLM fallback

**Fast path**: Local terminology tables (shipped as compressed JSON, ~14MB total) handle exact and fuzzy matching for common terms. Sub-millisecond per lookup. Covers ~80% of terms.

**Slow path**: For ambiguous or novel terms, DSPy modules (`ResolveAnatomy`, `ResolveDiagnosis`) use LLM reasoning with modality context for disambiguation.

**Validation layer**: All LLM-proposed codes are verified against the local terminology tables. Hallucinated codes are flagged, not silently accepted.

### Vocabularies

| Vocabulary | Coverage | Use |
|-----------|---------|-----|
| RadLex | ~46K terms | Anatomy, imaging techniques, radiology-specific terms |
| SNOMED CT | Radiology subset | Clinical interoperability, cross-system coding |
| ICD-10-CM | Full | Diagnosis coding, billing, research cohort selection |
| LOINC | Radiology procedures | Procedure coding (unified with RadLex Playbook) |

Each extracted entity carries codes from all applicable vocabularies plus a confidence score.

## 8. FHIR R4 Export

Structured reports export as FHIR R4 `DiagnosticReport` bundles:

- **DiagnosticReport** — container with coded procedure (LOINC), conclusion text, references to Observations
- **Observation** (one per finding) — coded with SNOMED CT + RadLex, `valueQuantity` for measurements, components for scoring systems (BI-RADS, Lung-RADS)
- **ImagingStudy** reference (when DICOM metadata available)
- **Patient** reference (pseudonymized ID)

Output is a JSON bundle consumable by any FHIR-compatible system (Epic, Cerner, HAPI FHIR).

## 9. Evaluation & Optimization

### Metrics

Each pipeline has a dedicated metric function serving dual purpose: evaluation scoring AND GEPA optimization feedback.

**Radiology extraction metric** (5 dimensions, weighted):
- Findings recall: 0.30 — did we capture all findings?
- Measurement accuracy: 0.25 — did we get the numbers right?
- Impression completeness: 0.20 — SemanticF1 against ground truth
- Scoring system accuracy: 0.15 — BI-RADS/Lung-RADS correct?
- Completeness score: 0.10 — overall completeness

**De-identification metric**: Zero-tolerance PHI detection + content preservation (penalizes over-redaction).

**Summarization metric**: Event recall (0.40) + date accuracy (0.30) + narrative quality via LM-as-judge (0.30).

All metrics return `{"score": float, "feedback": str}` when called by GEPA (module-specific feedback for reflective optimization) and plain `float` when called by `dspy.Evaluate`.

### Reward functions

Simpler than metrics — score a single prediction without gold labels, used by `dspy.Refine` at inference time:

- **extraction_reward**: Penalizes empty required fields, findings without anatomy, suspiciously short content. Bonus for measurements.
- **phi_leak_reward**: Regex-based PHI detection. Any match → 0.0, triggering retry.

### Optimization workflow

Progressive strategy — start cheap, upgrade if quality is insufficient:

1. **BootstrapFewShot** (~$0.50, ~5 min) — needs ~10 examples
2. **MIPROv2** (~$3, ~20 min) — Bayesian instruction + few-shot optimization
3. **GEPA** (~$10, ~45 min) — reflective prompt evolution with module-specific feedback

GEPA configuration:
- `reflection_lm`: Strong model (GPT-4o or Claude) for reflective mutation
- `candidate_selection_strategy`: "pareto" for maintaining diverse solutions
- `use_merge`: True for crossover between successful variants
- `auto`: "light"/"medium"/"heavy" budget presets

### Distillation

After optimization with a strong model, `dspy.BootstrapFinetune` distills the optimized pipeline into a cheaper model (e.g., Llama-3.2-8B) for cost-efficient local deployment.

### The optimization flywheel

```
Run batch extraction → get completeness scores
→ Flag low-completeness reports for manual review
→ Researcher corrects → corrections become training examples
→ Run GEPA optimization → pipeline improves
→ Re-run batch → higher completeness scores
→ Repeat
```

## 10. Export Formats

### Tabular (`export/tabular.py`)

Three flattening strategies for nested report data:

| Strategy | Shape | Use case |
|----------|-------|----------|
| `one_row` | 1 row per report, findings as JSON column | Quick overview, report-level analysis |
| `findings_rows` | 1 row per finding, report metadata repeated | ML pipelines, per-finding analysis |
| `long` | 1 row per field value, fully normalized | Maximum flexibility |

Supported formats: **CSV**, **Parquet** (PyArrow), **HuggingFace Datasets**, **JSONL**.

### Narrative (`export/report.py`)

Human-readable reports in **PDF** (ReportLab), **DOCX** (python-docx), and **Markdown** — with completeness badge, coded findings, and scoring system categories.

### FHIR (`export/fhir_bundle.py`)

FHIR R4 Bundle as JSON — see section 8.

## 11. Batch Processing

### Engine (`batch.py`)

- **Parallel document processing** via `ThreadPoolExecutor` — each document is independent
- **Configurable workers** — default 4, scale based on LLM backend capacity
- **Checkpointing** every N documents (default 50) — crash-safe resume via `--resume`
- **Progress tracking** — Rich progress bar with ETA
- **Error isolation** — individual document failures don't stop the batch

### Cost-efficient model routing

For large batches, `dspy.context` routes different pipeline steps to different models:
- Steps 1-3 (classification, parsing, technique): cheap/fast model (3B)
- Steps 4-5 (findings, impressions): strong model (70B+)
- Completeness assessment: cheap model (3B)

~60% cost reduction vs. using the strong model for all steps.

### Quality dashboard

After batch completion, a Rich terminal dashboard shows:
- Success/failure counts and duration
- Completeness distribution (mean, std, below-threshold count)
- Exam type distribution
- Ontology resolution rate
- Findings and measurements per report averages
- Export file locations
- List of failures with error details

## 12. CLI Commands

```bash
# --- Extraction ---
mosaicx extract --document report.pdf --template auto
mosaicx extract --document report.pdf --template radreport:chest_ct
mosaicx extract --document report.pdf --template my_custom.yaml
mosaicx extract --document report.pdf --template auto --optimized optimized/v2.json

# --- Batch processing ---
mosaicx batch --input-dir /data/reports/ --template auto --output-dir /data/structured/ \
  --format parquet jsonl fhir --workers 8 --completeness-threshold 0.7
mosaicx batch --resume batch_20260215_143022

# --- Template management ---
mosaicx template create my_template.yaml
mosaicx template list
mosaicx template validate my_template.yaml

# --- Schema generation ---
mosaicx schema generate --desc "Cardiac MRI report with ejection fraction and wall motion"
mosaicx schema list
mosaicx schema refine my_schema.yaml --instruction "Add a field for pericardial effusion"

# --- Summarization ---
mosaicx summarize --dir /data/patient_001/ --patient P001 --format json pdf

# --- De-identification ---
mosaicx deidentify --document report.pdf --mode pseudonymize
mosaicx deidentify --dir /data/reports/ --mode remove --workers 4

# --- Optimization ---
mosaicx optimize --pipeline radiology --trainset train.json --valset val.json \
  --budget medium --save optimized/radiology_v1.json
mosaicx distill --optimized optimized/v1.json --student ollama_chat/llama3.2:8b \
  --trainset train.json --save distilled/v1.json
mosaicx evaluate --pipeline radiology --testset test.json --optimized optimized/v1.json

# --- Configuration ---
mosaicx config show
mosaicx config set lm "ollama_chat/llama3.1:70b"
```

## 13. Configuration

Single source of truth via Pydantic Settings:

```python
class MosaicxConfig(pydantic.BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOSAICX_")

    # LLM
    lm: str = "ollama_chat/llama3.1:70b"
    lm_cheap: str = "ollama_chat/llama3.2:3b"
    api_key: str = "ollama"

    # Processing
    default_template: str = "auto"
    completeness_threshold: float = 0.7
    batch_workers: int = 4
    checkpoint_every: int = 50

    # Paths
    home_dir: Path = Path("~/.mosaicx").expanduser()
    schema_dir: Path = home_dir / "schemas"
    optimized_dir: Path = home_dir / "optimized"
    checkpoint_dir: Path = home_dir / "checkpoints"
    log_dir: Path = home_dir / "logs"

    # De-identification
    deidentify_mode: Literal["remove", "pseudonymize", "dateshift"] = "remove"

    # Export
    default_export_formats: list[str] = ["parquet", "jsonl"]

    # Document loading
    force_ocr: bool = False
    ocr_langs: list[str] = ["en", "de"]
    vlm_model: str = "gemma3:27b"
```

Resolution order: CLI flags > environment variables (`MOSAICX_LM`, `MOSAICX_API_KEY`, etc.) > config file (`~/.mosaicx/config.yaml`) > defaults.

## 14. Dependencies

### Core (required)

| Package | Purpose |
|---------|---------|
| `dspy` (>=2.6) | LLM orchestration, optimization, all module abstractions |
| `pydantic` (>=2.0) | Data validation, schema models, settings |
| `click` (>=8.1) | CLI framework |
| `rich` (>=13.0) | Terminal UI |
| `docling` (>=2.0) | Document conversion (PDF/DOCX/PPTX → Markdown) |
| `pyyaml` (>=6.0) | YAML template parsing |
| `pandas` (>=2.0) | DataFrame export |
| `pyarrow` (>=14.0) | Parquet export |

### Optional

| Package | Purpose | Install extra |
|---------|---------|---------------|
| `reportlab` (>=4.4) | PDF export | `mosaicx[pdf]` |
| `python-docx` (>=1.0) | DOCX export | `mosaicx[docx]` |
| `datasets` (>=2.0) | HuggingFace Datasets export | `mosaicx[hf]` |

### Removed (vs. v1)

| Package | Why removed |
|---------|------------|
| `instructor` | Replaced by DSPy typed predictors |
| `outlines` | Replaced by DSPy structured output |
| `json-repair` | No longer needed — DSPy handles output parsing |
| `openai` | DSPy uses LiteLLM internally |
| `ollama` | DSPy uses LiteLLM internally |
| `httpx` | Was unused in v1 |
| `requests` | DSPy handles HTTP internally |
| `python-cfonts` | Simplified display |
| `dspy-ai` | Was listed but never used; replaced by `dspy` |

## 15. Testing Strategy

### Unit tests

- Every DSPy signature: verify output types match schema
- Template compiler: YAML → Pydantic model correctness
- Ontology resolver: local lookup accuracy, code validation
- Completeness evaluator: deterministic field coverage scoring
- FHIR exporter: valid FHIR R4 bundle structure
- Tabular exporter: correct flattening for all strategies
- Batch processor: checkpoint save/load, error isolation

### Integration tests

- End-to-end extraction: PDF → structured report → export (mocked LLM)
- Template auto-selection accuracy
- Ontology resolution with LLM fallback
- Batch processing with resume

### Evaluation tests

- Metric functions: known gold/pred pairs → expected scores
- Reward functions: edge cases (empty output, PHI patterns)
- GEPA integration: smoke test with tiny dataset (2-3 examples)

### Fixtures

- Sample radiology reports (chest CT, brain MRI, mammography) as text fixtures
- Pre-built gold-standard structured outputs for metric testing
- Sample YAML templates for compiler testing

## 16. Migration Path

### Phase 1: Core infrastructure
Config, document loading, display, CLI skeleton. All DSPy module stubs.

### Phase 2: Pipelines
Generic extractor, radiology structurer, schema generator, summarizer, deidentifier. All with typed signatures and basic Pydantic output models.

### Phase 3: RadReport templates
Base models, 6 initial templates (chest CT, chest XR, brain MRI, abdomen CT, mammography, PET/CT), template auto-selection, template compiler for custom YAML.

### Phase 4: Quality & ontology
Completeness evaluator, metrics, reward functions, ontology resolver with terminology tables, FHIR export.

### Phase 5: Optimization & batch
GEPA/MIPROv2 optimization workflow, distillation, batch processor with checkpointing, quality dashboard, cost-efficient model routing.

### Phase 6: Export & polish
All export formats (CSV, Parquet, HuggingFace, JSONL, PDF, DOCX, Markdown, FHIR bundles), CLI completion, documentation, packaging.

## 17. Success Criteria

1. **All existing v1 tests pass** — backward-compatible Python API for `extract_document()`, `generate_schema()`, `summarize_reports()`
2. **Radiology extraction accuracy** — F1 > 0.85 on a held-out test set of 20 annotated chest CT reports after GEPA optimization with 15 training examples
3. **Completeness scoring** — overall score correlates with manual quality assessment (Pearson r > 0.7)
4. **Batch throughput** — 1,000 reports processed in < 2 hours with 4 workers on consumer hardware (M-series Mac or NVIDIA 4090)
5. **Zero PHI leaks** — de-identified output passes automated PHI detection on test set
6. **Ontology resolution rate** — > 90% of extracted entities resolved to at least one vocabulary code
7. **FHIR compliance** — exported bundles validate against FHIR R4 schema
8. **Dependency reduction** — core install from ~15 packages to ~8 packages
9. **Code reduction** — ~8,000 lines → ~4,000-5,000 lines (less code, more capability)
