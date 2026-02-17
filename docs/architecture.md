# MOSAICX Architecture

## System Overview

MOSAICX (Medical cOmputational Suite for Advanced Intelligent eXtraction) converts unstructured clinical documents (PDFs, images, text) into structured JSON using DSPy pipelines backed by local LLMs. Documents pass through a dual-engine OCR layer (Surya + Chandra), then into domain-specific DSPy Modules that chain multiple LLM calls with typed Pydantic schemas. A unified template system resolves extraction schemas from YAML template files (built-in, user-created, or file paths), auto-detects pipeline modes, and scores output completeness. The system supports DSPy prompt optimization (BootstrapFewShot, MIPROv2, GEPA) to tune pipeline prompts against labeled datasets.

## Architecture Diagram

```
                        MOSAICX Architecture

  Documents (.pdf/.png/.txt)
       |
       v
  +------------------+
  | Document Loader  |    ocr_engine: "both" | "surya" | "chandra"
  | (dual-engine OCR)|    Surya (local) + Chandra (VLM)
  +------------------+    Quality scoring selects best result
       |
       v  plain text
  +--------------------+     +---------------------+
  | Template Resolution|     | Evaluation Loop     |
  | (report.py)        |     |                     |
  |                    |     | JSONL dataset        |
  | resolve_template() |     |   -> dspy.Example   |
  |   -> Pydantic model|     |   -> optimizer       |
  | detect_mode()      |     |   -> tuned prompts   |
  |   -> pipeline mode |     |                     |
  +--------------------+     | Metrics:             |
       |                     |   metric(ex, pred)   |
       v                     |   -> float [0, 1]    |
  +------------------+       +---------------------+
  | DSPy Pipeline    |
  | (Mode Module)    |
  |                  |
  | Signature 1      |
  |   -> Predict     |
  | Signature 2      |
  |   -> CoT         |
  | ...              |
  +------------------+
       |
       v
  dspy.Prediction
       |
       v
  +--------------------+
  | Completeness       |
  | Scoring (--score)  |
  +--------------------+
       |
       v
  +------------------+
  | Export           |
  | JSON / JSONL /   |
  | Parquet / YAML   |
  +------------------+
```

## Unified Template System

The template system is the central abstraction for specifying what to extract from a document. Templates are YAML files that define a name, description, optional pipeline mode, and a list of typed sections. The system compiles YAML templates into Pydantic models at runtime via `pydantic.create_model` -- no `exec()` is used.

### Template Resolution Chain (`report.py`)

`resolve_template(template)` resolves a template specification into a Pydantic model class using the following chain:

```
1. File path       template ends in .yaml/.yml and exists on disk
                     -> compile_template_file(path)

2. User template   ~/.mosaicx/templates/<name>.yaml exists
                     -> compile_template_file(user_path)

3. Built-in YAML   mosaicx/schemas/radreport/templates/<name>.yaml exists
                     -> compile_template_file(builtin_path)

4. Legacy schema   ~/.mosaicx/schemas/<name>.json exists (saved SchemaSpec)
                     -> load_schema() + compile_schema()

5. Error           ValueError with suggestions
```

The first match wins. This means a user template with the same name as a built-in will override it, and a file path always takes precedence.

### Mode Auto-Detection

`detect_mode(template_name)` reads the `mode` field from the resolved YAML template (user or built-in) to determine which pipeline to run. For example, `chest_ct.yaml` declares `mode: radiology`, so `--template chest_ct` automatically uses the radiology pipeline without requiring `--mode radiology`.

### Built-in Templates (`schemas/radreport/templates/`)

Eleven YAML template files ship with the package:

| Template | Mode | RDES |
|---|---|---|
| `chest_ct` | radiology | RDES3 |
| `chest_xr` | radiology | RDES44 |
| `brain_mri` | radiology | RDES29 |
| `abdomen_ct` | radiology | RDES28 |
| `mammography` | radiology | RDES154 |
| `thyroid_us` | radiology | RDES70 |
| `lung_ct` | radiology | RDES206 |
| `msk_mri` | radiology | -- |
| `cardiac_mri` | radiology | -- |
| `pet_ct` | radiology | -- |
| `generic` | -- | -- |

Adding a new built-in template requires only a new `.yaml` file in this directory -- no Python changes are needed. The registry discovers templates by scanning the directory at first access.

### Template YAML Format

Each template YAML file defines:

```yaml
name: ChestCTReport                    # Pydantic model class name
description: Structured chest CT ...   # Human-readable description
radreport_id: RDES3                    # Optional RadReport.org identifier
mode: radiology                        # Optional pipeline mode for auto-detection
sections:
  - name: indication
    type: str
    required: true
    description: Clinical indication for chest CT
  - name: nodules
    type: list
    required: false
    description: Pulmonary nodules
    item:
      type: object
      fields:
        - name: location
          type: str
        - name: size_mm
          type: float
```

Supported types: `str`, `int`, `float`, `bool`, `enum` (with `values` list), `list` (with `item` descriptor), `object` (with `fields` list, recursively nested).

### Template Compiler (`schemas/template_compiler.py`)

Key functions:

- `parse_template(yaml_str)` -- parse YAML into a `TemplateMeta` descriptor
- `compile_template(yaml_str)` -- compile YAML into a Pydantic `BaseModel` subclass
- `compile_template_file(path)` -- read a file and compile it
- `schema_spec_to_template_yaml(spec)` -- bridge from legacy `SchemaSpec` to YAML format

### Template Registry (`schemas/radreport/registry.py`)

The registry was rewritten from a hardcoded list of `TemplateInfo` entries to a YAML-scanning approach. On first access (`list_templates()` or `get_template()`), it scans `schemas/radreport/templates/*.yaml`, parses each file's metadata, and caches the results. The public API is unchanged:

- `TemplateInfo` -- dataclass with `name`, `exam_type`, `radreport_id`, `description`, `mode`
- `list_templates()` -- returns all registered templates
- `get_template(name)` -- returns a single template by name, or `None`

### RadReport Fetcher (`schemas/radreport/fetcher.py`)

Fetches and parses templates from the RSNA RadReport Template Library API (`https://api3.rsna.org/radreport/v1`). Used by `mosaicx template create --from-radreport <ID>`. Parses the embedded HTML to extract section names, LOINC codes, and default text, then returns a structured context string for the SchemaGenerator LLM.

## Report Orchestrator (`report.py`)

The `report.py` module wires together template resolution, pipeline execution, and completeness scoring into a single call. It is used by both the CLI `extract` command and the SDK.

### `ReportResult` Dataclass

Container for structured report output:

```python
@dataclass
class ReportResult:
    extracted: dict[str, Any]       # Extracted data
    completeness: dict[str, Any]    # Completeness scores
    template_name: str | None       # Resolved template name
    mode_used: str | None           # Pipeline mode that ran
    metrics: dict[str, Any] | None  # Duration, tokens, step count
```

### `run_report()` Flow

```
1. Determine mode     mode arg > detect_mode(template_name)
2. Route extraction
   a. Mode pipeline   extract_with_mode_raw() -> output_data, metrics, prediction
   b. Template model   DocumentExtractor(output_schema=model) -> result
   c. Auto mode        DocumentExtractor() -> result (LLM infers schema)
3. Find primary model  _find_primary_model(prediction) -> first BaseModel in prediction
4. Score completeness  compute_report_completeness(model_instance, text, model_class)
5. Return              ReportResult(extracted, completeness, template_name, mode, metrics)
```

### `_find_primary_model()`

Scans a `dspy.Prediction` for the first `BaseModel` instance among its values. This is the natural "completeness target" (e.g., `ReportSections` for radiology) -- the Pydantic model that can be scored for field coverage.

## Data Flow: `extract --template chest_ct --score`

End-to-end example of the unified template system:

```
1. CLI receives     --template chest_ct --score
2. Resolve          resolve_template("chest_ct")
                      -> finds built-in YAML: schemas/radreport/templates/chest_ct.yaml
                      -> compile_template_file() -> ChestCTReport Pydantic model
3. Detect mode      detect_mode("chest_ct")
                      -> reads mode: radiology from YAML
4. Configure DSPy   _configure_dspy() -> HarmonyLM wraps configured LM
5. Extract          extract_with_mode_raw(text, "radiology")
                      -> RadiologyReportStructurer.forward()
                      -> dspy.Prediction with ReportSections, findings, etc.
6. Serialize        output_data dict from prediction values
7. Score            _find_primary_model() -> ReportSections instance
                      -> compute_report_completeness()
8. Display          Rich-rendered extracted data + completeness table
9. Export           Optional: --output file.json
```

## Pipeline Architecture

Every pipeline follows the same pattern. `radiology.py` is the canonical reference.

### Lazy Loading Pattern

DSPy classes are defined inside `_build_dspy_classes()` and exposed via module-level `__getattr__`. This avoids importing DSPy at module load time, which matters because: (a) DSPy is heavy and slow to import, (b) listing available modes should work without DSPy installed, (c) only the requested pipeline's classes need to load.

```python
# mosaicx/pipelines/radiology.py (condensed)

# Eager: register metadata without importing DSPy
register_mode_info("radiology", "5-step radiology report structurer ...")

def _build_dspy_classes():
    import dspy  # lazy

    class ClassifyExamType(dspy.Signature):
        report_header: str = dspy.InputField(desc="...")
        exam_type: str = dspy.OutputField(desc="...")

    class RadiologyReportStructurer(dspy.Module):
        def __init__(self):
            self.classify_exam = dspy.Predict(ClassifyExamType)
            # ... more steps

        def forward(self, report_text: str, report_header: str = "") -> dspy.Prediction:
            # Chain steps, return dspy.Prediction with all outputs
            ...

    register_mode("radiology", "...")(RadiologyReportStructurer)
    return {"ClassifyExamType": ClassifyExamType, "RadiologyReportStructurer": ..., ...}

_dspy_classes: dict | None = None
_DSPY_CLASS_NAMES = frozenset({"ClassifyExamType", ..., "RadiologyReportStructurer"})

def __getattr__(name: str):
    global _dspy_classes
    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]
    raise AttributeError(...)
```

### DSPy Signatures

Each step defines a `dspy.Signature` with typed `InputField`/`OutputField`. Output types can be Pydantic models (e.g., `List[RadReportFinding]`), which DSPy parses from LLM output.

### DSPy Module

The pipeline's `dspy.Module` wires steps together in `forward()`. Each step is either `dspy.Predict` (simple) or `dspy.ChainOfThought` (adds reasoning). Steps pass outputs forward as inputs to subsequent steps.

### Mode Registration (Two-Phase)

```
Phase 1 (module load):  register_mode_info("radiology", "description")
                         -> _MODE_DESCRIPTIONS dict (no DSPy needed)
                         -> list_modes() works immediately

Phase 2 (first use):    get_mode("radiology")
                         -> _trigger_lazy_load()
                         -> imports module, accesses class via __getattr__
                         -> _build_dspy_classes() runs, calls register_mode()
                         -> MODES dict populated
```

`_MODE_MODULES` in `modes.py` maps mode names to their module paths for lazy loading.

## Configuration System

`MosaicxConfig` extends `pydantic_settings.BaseSettings`:

```python
class MosaicxConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOSAICX_", env_file=".env")

    lm: str = "openai/gpt-oss:120b"
    api_base: str = "http://localhost:11434/v1"
    ocr_engine: Literal["both", "surya", "chandra"] = "both"
    # ...
```

**Resolution order:** CLI flags > environment variables (`MOSAICX_*`) > `.env` file > defaults.

Accessed via `get_config()` which returns an `lru_cache(maxsize=1)` singleton. Derived paths (`schema_dir`, `optimized_dir`, `checkpoint_dir`, `log_dir`, `templates_dir`) are `@property` methods under `~/.mosaicx/`.

## Evaluation System

Three registries must stay in sync when adding a pipeline:

| Registry | File | Purpose |
|---|---|---|
| `PIPELINE_INPUT_FIELDS` | `evaluation/dataset.py` | Which JSONL keys are inputs vs. gold labels |
| `_METRIC_REGISTRY` | `evaluation/metrics.py` | Scoring function per pipeline |
| `_PIPELINE_REGISTRY` | `evaluation/optimize.py` | Module path + class name for lazy import |

### How DSPy Optimization Works

1. `load_jsonl()` reads a JSONL file into `dspy.Example` objects, calling `.with_inputs()` to partition input fields from gold labels.
2. An optimizer (`BootstrapFewShot`, `MIPROv2`, or `GEPA`) compiles the module against the training set using the metric function.
3. The metric follows `metric(example, prediction, trace=None) -> float` in range `[0.0, 1.0]`.
4. The optimized module (with tuned prompts/demos) is saved to disk via `module.save()`.
5. Budget presets control optimizer selection: `light` (BootstrapFewShot), `medium` (MIPROv2), `heavy` (GEPA).

## CLI Architecture

Built on Click with custom `MosaicxGroup`/`MosaicxCommand` classes that re-render help text with the coral/greige theme.

**Command structure:**

```
mosaicx
  extract       --document, --template, --mode, --score, --optimized, -o
  batch         --input-dir, --output-dir, --template, --mode, --format, --workers
  summarize     --document, --dir, --patient
  deidentify    --document, --dir, --mode, --regex-only
  optimize      --pipeline, --trainset, --valset, --budget, --save
  eval          --pipeline, --testset, --optimized
  template
    list                                    # list built-in + user templates
    validate    --file                      # validate a YAML template
    create      --describe, --from-document, --from-url, --from-radreport, --from-json
    show        TEMPLATE_NAME               # display template details
    refine      --template, --instruction   # LLM-assisted template refinement
    migrate     --schema                    # convert saved SchemaSpec JSON to YAML
    history     TEMPLATE_NAME               # version history
    revert      TEMPLATE_NAME --version     # revert to version
    diff        TEMPLATE_NAME --version     # diff against version
  pipeline
    new         --name, --description       # scaffold a new pipeline
  config
    show
    set         KEY VALUE
  mcp
    serve                                   # start MCP server
```

**Theme system:** `cli_theme.py` defines colors (`CORAL=#E87461`, `GREIGE=#B5A89A`, `MUTED=#8B8178`), helper functions (`section()`, `spinner()`, `badge()`, `progress()`), and the banner. `cli_display.py` handles Rich rendering of extracted data, metrics, and completeness scores.

## Data Flow

```
1. Input       PDF / image / DOCX / text file
2. Load        documents/loader.py: route by suffix
3. OCR         .pdf/.png/.tiff -> parallel Surya + Chandra -> quality scoring
               .txt/.md -> direct read (no OCR)
4. Text        LoadedDocument with .text, .ocr_confidence, .quality_warning
5. Resolve     report.resolve_template(): template name -> Pydantic model class
6. Detect      report.detect_mode(): template name -> pipeline mode from YAML
7. DSPy init   _configure_dspy(): HarmonyLM wraps the configured LM, strips tokens
8. Pipeline    Instantiate DSPy Module -> forward(text) -> chained Predict/CoT calls
9. Prediction  dspy.Prediction with typed outputs (Pydantic models, lists, strings)
10. Score      (optional) compute_report_completeness() -> field coverage metrics
11. Serialize  .model_dump() on Pydantic outputs -> dict
12. Export     JSON file, or batch -> JSONL + Parquet (via pandas)
```

## Adding a New Pipeline

Checklist -- six files to touch:

1. **`mosaicx/pipelines/<name>.py`** -- Define DSPy Signatures and Module inside `_build_dspy_classes()`. Register with `register_mode_info()` at top level and `register_mode()` inside the builder. Expose class names via `__getattr__`.

2. **`mosaicx/pipelines/modes.py`** -- Add entry to `_MODE_MODULES` dict and `_LAZY_CLASS_NAMES` dict inside `_trigger_lazy_load()`.

3. **`mosaicx/evaluation/optimize.py`** -- Add entry to `_PIPELINE_REGISTRY` mapping name to `(module_path, class_name)`.

4. **`mosaicx/evaluation/metrics.py`** -- Write a `<name>_metric(example, prediction, trace=None) -> float` function and add it to `_METRIC_REGISTRY`.

5. **`mosaicx/evaluation/dataset.py`** -- Add entry to `PIPELINE_INPUT_FIELDS` listing which JSONL keys are pipeline inputs (everything else becomes gold labels).

6. **`mosaicx/schemas/<name>/`** -- (Optional) Define Pydantic models for typed OutputFields if the pipeline produces structured domain objects.

### Adding a New Built-in Template

To add a new built-in extraction template (no pipeline code changes needed):

1. Create `mosaicx/schemas/radreport/templates/<name>.yaml` with `name`, `description`, `mode`, and `sections`.
2. The registry auto-discovers the file on next access. No Python code changes required.
3. The template is immediately available via `mosaicx extract --template <name>` and `mosaicx template list`.

## Key Design Decisions

**Unified template system:** Templates are the single concept for specifying extraction structure. YAML files are the source of truth -- they define the output schema (compiled to Pydantic models at runtime), the pipeline mode (auto-detected from the `mode` field), and metadata (description, RadReport ID). The resolution chain (file path > user templates > built-in > legacy schema) provides flexibility while keeping the common case simple. This replaced the earlier dual system of separate `--schema` and `--template` flags.

**YAML-scanned registry:** The template registry discovers built-in templates by scanning `templates/*.yaml` at first access, with lazy caching. Adding a new built-in template requires only a YAML file drop -- no Python changes. This replaced a hardcoded list of eleven `TemplateInfo` entries.

**Report orchestrator as single entry point:** `report.py` centralizes the template resolution, mode detection, pipeline dispatch, and completeness scoring logic. Both the CLI and SDK call into this module, ensuring consistent behavior. The `ReportResult` dataclass provides a uniform return type with extracted data, completeness scores, and metadata.

**Lazy loading via `__getattr__`:** DSPy import is ~2s and pulls in torch/transformers. Listing modes, showing help, and running non-DSPy commands (config, template list) must be fast. The two-phase registration (eager metadata, lazy classes) keeps `mosaicx --help` under 200ms.

**DSPy as the pipeline framework:** DSPy separates prompt engineering from pipeline logic. Each step is a typed Signature; DSPy handles prompt construction, output parsing, and optimization. Pipelines are optimizable without code changes -- just provide labeled JSONL and run `mosaicx optimize`.

**Pydantic-settings for configuration:** Single `MosaicxConfig` class provides typed defaults, env var binding (`MOSAICX_*` prefix), `.env` file support, and CLI override -- all with validation. The `lru_cache` singleton avoids re-parsing on every access.

**Dual OCR (Surya + Chandra):** Surya is a lightweight local OCR model; Chandra is a vision-language model (VLM) that handles complex layouts better. Running both in parallel and quality-scoring the results gives the best text extraction across document types. The `ocr_engine` config lets users choose one if resource-constrained.

**HarmonyLM wrapper:** Strips Harmony formatting tokens from LLM responses before DSPy parses them. This avoids parse failures when using models that emit special tokens (e.g., local fine-tuned models served via vLLM/MLX).
