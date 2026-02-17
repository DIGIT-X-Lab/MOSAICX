# CLAUDE.md — Agent Guide for MOSAICX

## What is MOSAICX?

Medical cOmputational Suite for Advanced Intelligent eXtraction. Converts unstructured medical documents (radiology reports, pathology summaries, clinical notes) into structured JSON using DSPy pipelines backed by local LLMs via Ollama.

## Quick Commands

```bash
# Setup
pip install -e ".[dev]"           # install in dev mode
make install                      # same thing

# Testing
make test                         # run all tests
make test-unit                    # unit tests only
pytest tests/ -q                  # direct pytest

# Quality
make lint                         # ruff check + fix
make typecheck                    # mypy
make check                        # lint + typecheck + test

# Run
mosaicx extract --document report.pdf                         # auto extraction
mosaicx extract --document report.pdf --template chest_ct     # with template
mosaicx extract --document report.pdf --mode radiology        # explicit mode
mosaicx extract --document report.pdf --template echo --score # + completeness
mosaicx template create --describe "echo with EF"            # create template
mosaicx template list                                         # list templates
mosaicx eval --pipeline radiology --testset test.jsonl        # evaluate
```

## Project Structure

```
mosaicx/
├── cli.py              # Click CLI (extract, batch, template, optimize, eval, ...)
├── cli_display.py      # Rich display helpers for CLI output
├── cli_theme.py        # Theme colors: coral (#E87461), greige (#B5A89A)
├── config.py           # MosaicxConfig (pydantic-settings, MOSAICX_* env vars)
├── batch.py            # Batch processing with checkpointing
├── display.py          # Display utilities
├── metrics.py          # PipelineMetrics tracking (latency per step)
│
├── pipelines/
│   ├── modes.py        # Mode registry (register_mode, get_mode, list_modes)
│   ├── radiology.py    # 5-step radiology pipeline (DSPy Module)
│   ├── pathology.py    # 5-step pathology pipeline (DSPy Module)
│   ├── extraction.py   # Generic document extraction pipeline
│   ├── summarizer.py   # Patient timeline summarizer
│   ├── deidentifier.py # PHI removal (LLM + regex)
│   └── schema_gen.py   # Schema generation from description
│
├── report.py           # Structured report orchestrator (resolve_template, run_report)
│
├── schemas/            # Pydantic models for structured output
│   └── radreport/      # RadReportFinding, ImpressionItem, ReportSections, etc.
│       └── templates/  # Built-in YAML templates (chest_ct.yaml, brain_mri.yaml, ...)
│
├── evaluation/
│   ├── dataset.py      # JSONL loader (load_jsonl) + PIPELINE_INPUT_FIELDS registry
│   ├── metrics.py      # DSPy metric functions (one per pipeline)
│   ├── optimize.py     # Optimizer execution + pipeline registry
│   ├── rewards.py      # Scalar reward functions (extraction_reward, phi_leak_reward)
│   └── completeness.py # Field coverage scoring
│
├── documents/          # Document reading (PDF, text)
└── export/             # Export to JSONL, Parquet, CSV
```

## Key Patterns

### Pipeline Pattern (DSPy Modules)

Every pipeline follows this pattern. **Do not deviate.**

1. **Lazy loading** — DSPy classes are defined inside `_build_dspy_classes()` and loaded via module-level `__getattr__`. This allows importing the module without DSPy installed.

2. **Signature classes** — Each step defines a `dspy.Signature` with typed `InputField` and `OutputField`.

3. **Module class** — The main pipeline is a `dspy.Module` with `forward()` method that chains steps. Uses `dspy.Predict` for simple steps, `dspy.ChainOfThought` for complex reasoning.

4. **Mode registration** — Pipelines register eagerly via `register_mode_info(name, desc)` at module top-level, and register the class via `@register_mode` inside `_build_dspy_classes()`.

5. **Metrics tracking** — `forward()` wraps each step in `track_step(metrics, "Step name", tracker)`.

```python
# Template: mosaicx/pipelines/<name>.py
from mosaicx.pipelines.modes import register_mode_info
register_mode_info("<name>", "<description>")

def _build_dspy_classes():
    import dspy

    class MySignature(dspy.Signature):
        """..."""
        input_field: str = dspy.InputField(desc="...")
        output_field: str = dspy.OutputField(desc="...")

    class MyPipeline(dspy.Module):
        def __init__(self):
            super().__init__()
            self.step = dspy.ChainOfThought(MySignature)

        def forward(self, **kwargs) -> dspy.Prediction:
            result = self.step(**kwargs)
            return dspy.Prediction(...)

    from mosaicx.pipelines.modes import register_mode
    register_mode("<name>", "<description>")(MyPipeline)
    return {"MySignature": MySignature, "MyPipeline": MyPipeline}

_dspy_classes = None
_DSPY_CLASS_NAMES = frozenset({"MySignature", "MyPipeline"})

def __getattr__(name):
    global _dspy_classes
    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### Configuration

- All settings via `MosaicxConfig` in `config.py` (pydantic-settings)
- Env prefix: `MOSAICX_` (e.g., `MOSAICX_LM`, `MOSAICX_API_BASE`)
- Access via `get_config()` (cached singleton)
- CLI flags override env vars

### CLI

- Framework: Click
- Theme: coral `#E87461` + greige `#B5A89A` (see `cli_theme.py`)
- Display: Rich tables, panels, progress bars
- Entry point: `mosaicx.cli:cli`

### Evaluation System

When adding a new pipeline, register it in **three places**:

1. `evaluation/dataset.py` — add to `PIPELINE_INPUT_FIELDS` dict
2. `evaluation/metrics.py` — add metric function + register in `_METRIC_REGISTRY`
3. `evaluation/optimize.py` — add to `_PIPELINE_REGISTRY` dict

### Testing

- Framework: pytest with markers (`@pytest.mark.unit`, `@pytest.mark.integration`)
- Tests in `tests/` mirroring source structure
- Sample data in `tests/datasets/`
- Run: `pytest tests/ -q`

## Conventions

- Python 3.11+
- Line length: 88 (ruff/black)
- Type hints on all public functions
- Imports: `from __future__ import annotations` at top of every file
- Linter: ruff (E, W, F, I, B, C4, UP rules)
- No emojis in code or output
- All DSPy pipelines use lazy loading via `__getattr__`

## Common Tasks

### Add a new pipeline

1. Create `mosaicx/pipelines/<name>.py` following the pipeline pattern above
2. Add input fields to `evaluation/dataset.py` `PIPELINE_INPUT_FIELDS`
3. Add metric function to `evaluation/metrics.py` `_METRIC_REGISTRY`
4. Add to `evaluation/optimize.py` `_PIPELINE_REGISTRY`
5. Add mode to `pipelines/modes.py` `_MODE_MODULES` dict
6. Wire CLI commands in `cli.py`

### Add a new CLI command

1. Define function with `@cli.command()` decorator in `cli.py`
2. Use `@click.option()` for flags
3. Use theme functions from `cli_theme.py` for output
4. Use Rich for tables and progress display

## File Locations

- Templates: `~/.mosaicx/templates/` (user-created YAML templates)
- Template history: `~/.mosaicx/templates/.history/` (versioned backups)
- Legacy schemas: `~/.mosaicx/schemas/` (migrate with `mosaicx template migrate`)
- Optimized programs: `~/.mosaicx/optimized/`
- Checkpoints: `~/.mosaicx/checkpoints/`
- Logs: `~/.mosaicx/logs/`
