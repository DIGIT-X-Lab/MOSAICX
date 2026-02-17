# MOSAICX Architecture

## System Overview

MOSAICX (Medical cOmputational Suite for Advanced Intelligent eXtraction) converts unstructured clinical documents (PDFs, images, text) into structured JSON using DSPy pipelines backed by local LLMs. Documents pass through a dual-engine OCR layer (Surya + Chandra), then into domain-specific DSPy Modules that chain multiple LLM calls with typed Pydantic schemas. The system supports DSPy prompt optimization (BootstrapFewShot, MIPROv2, GEPA) to tune pipeline prompts against labeled datasets.

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
  +------------------+     +---------------------+
  | DSPy Pipeline    |     | Evaluation Loop     |
  | (Mode Module)    |     |                     |
  |                  |     | JSONL dataset        |
  | Signature 1      |     |   -> dspy.Example   |
  |   -> Predict     |     |   -> optimizer       |
  | Signature 2      |     |   -> tuned prompts   |
  |   -> CoT         |     |                     |
  | ...              |     | Metrics:             |
  +------------------+     |   metric(ex, pred)   |
       |                   |   -> float [0, 1]    |
       v                   +---------------------+
  dspy.Prediction
       |
       v
  +------------------+
  | Export           |
  | JSON / JSONL /   |
  | Parquet / YAML   |
  +------------------+
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

Accessed via `get_config()` which returns an `lru_cache(maxsize=1)` singleton. Derived paths (`schema_dir`, `optimized_dir`, `checkpoint_dir`, `log_dir`) are `@property` methods under `~/.mosaicx/`.

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
  extract       --document, --schema, --mode, --template, --optimized, -o
  batch         --input-dir, --output-dir, --schema, --mode, --format, --workers
  summarize     --document, --dir, --patient
  deidentify    --document, --dir, --mode, --regex-only
  optimize      --pipeline, --trainset, --valset, --budget, --save
  eval          --pipeline, --testset, --optimized
  template
    list
    validate    --file
  schema
    generate    --description, --from-document, --name
    list
    show        SCHEMA_NAME
    refine      --schema, --instruction, --add, --remove, --rename
    history     SCHEMA_NAME
    revert      SCHEMA_NAME --version
    diff        SCHEMA_NAME --version
  config
    show
    set         KEY VALUE
```

**Theme system:** `cli_theme.py` defines colors (`CORAL=#E87461`, `GREIGE=#B5A89A`, `MUTED=#8B8178`), helper functions (`section()`, `spinner()`, `badge()`, `progress()`), and the banner. `cli_display.py` handles Rich rendering of extracted data and metrics.

## Data Flow

```
1. Input       PDF / image / DOCX / text file
2. Load        documents/loader.py: route by suffix
3. OCR         .pdf/.png/.tiff -> parallel Surya + Chandra -> quality scoring
               .txt/.md -> direct read (no OCR)
4. Text        LoadedDocument with .text, .ocr_confidence, .quality_warning
5. DSPy init   _configure_dspy(): HarmonyLM wraps the configured LM, strips tokens
6. Pipeline    Instantiate DSPy Module -> forward(text) -> chained Predict/CoT calls
7. Prediction  dspy.Prediction with typed outputs (Pydantic models, lists, strings)
8. Serialize   .model_dump() on Pydantic outputs -> dict
9. Export      JSON file, or batch -> JSONL + Parquet (via pandas)
```

## Adding a New Pipeline

Checklist -- six files to touch:

1. **`mosaicx/pipelines/<name>.py`** -- Define DSPy Signatures and Module inside `_build_dspy_classes()`. Register with `register_mode_info()` at top level and `register_mode()` inside the builder. Expose class names via `__getattr__`.

2. **`mosaicx/pipelines/modes.py`** -- Add entry to `_MODE_MODULES` dict and `_LAZY_CLASS_NAMES` dict inside `_trigger_lazy_load()`.

3. **`mosaicx/evaluation/optimize.py`** -- Add entry to `_PIPELINE_REGISTRY` mapping name to `(module_path, class_name)`.

4. **`mosaicx/evaluation/metrics.py`** -- Write a `<name>_metric(example, prediction, trace=None) -> float` function and add it to `_METRIC_REGISTRY`.

5. **`mosaicx/evaluation/dataset.py`** -- Add entry to `PIPELINE_INPUT_FIELDS` listing which JSONL keys are pipeline inputs (everything else becomes gold labels).

6. **`mosaicx/schemas/<name>/`** -- (Optional) Define Pydantic models for typed OutputFields if the pipeline produces structured domain objects.

## Key Design Decisions

**Lazy loading via `__getattr__`:** DSPy import is ~2s and pulls in torch/transformers. Listing modes, showing help, and running non-DSPy commands (config, template list) must be fast. The two-phase registration (eager metadata, lazy classes) keeps `mosaicx --help` under 200ms.

**DSPy as the pipeline framework:** DSPy separates prompt engineering from pipeline logic. Each step is a typed Signature; DSPy handles prompt construction, output parsing, and optimization. Pipelines are optimizable without code changes -- just provide labeled JSONL and run `mosaicx optimize`.

**Pydantic-settings for configuration:** Single `MosaicxConfig` class provides typed defaults, env var binding (`MOSAICX_*` prefix), `.env` file support, and CLI override -- all with validation. The `lru_cache` singleton avoids re-parsing on every access.

**Dual OCR (Surya + Chandra):** Surya is a lightweight local OCR model; Chandra is a vision-language model (VLM) that handles complex layouts better. Running both in parallel and quality-scoring the results gives the best text extraction across document types. The `ocr_engine` config lets users choose one if resource-constrained.

**HarmonyLM wrapper:** Strips Harmony formatting tokens from LLM responses before DSPy parses them. This avoids parse failures when using models that emit special tokens (e.g., local fine-tuned models served via vLLM/MLX).
