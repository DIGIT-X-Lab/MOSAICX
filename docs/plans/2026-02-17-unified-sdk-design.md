# Unified SDK API Design

## Goal

One parameter (`documents`), same inputs everywhere, predictable outputs. Make the MOSAICX SDK a delight for developers to integrate.

## Current Problems

1. **Six functions doing variations of "give me files, give me results"**: `extract`, `batch_extract`, `process_file`, `process_files`, plus `deidentify` and `summarize` with inconsistent input handling.
2. **Inconsistent parameter naming**: `document` (singular) on extract/deidentify, `documents` (plural) on summarize, `file` on process_file, `files` on process_files, `texts` on batch_extract.
3. **Inconsistent input acceptance**: extract accepts file paths, deidentify accepts file paths, summarize accepts file paths, but each was added separately with different patterns.
4. **No batch support for deidentify/summarize in SDK**: `batch_extract` exists but no `batch_deidentify`.
5. **Documentation scattered**: SDK reference is buried inside the 1500-line developer guide.

## Design

### Three Core Functions

After cleanup, the SDK surface is three core functions + utilities:

```python
import mosaicx

# ---------- Extract ----------
result  = mosaicx.extract(documents="scan.pdf", template="chest_ct")
results = mosaicx.extract(documents=["a.pdf", "b.pdf"], mode="radiology")
results = mosaicx.extract(documents=Path("reports/"))
result  = mosaicx.extract("Patient presents with chest pain...")
result  = mosaicx.extract(documents=uploaded_bytes, filename="scan.pdf")
results = mosaicx.extract(documents=["a.pdf", "b.pdf"], workers=4,
                          on_progress=lambda name, ok, res: print(f"{name}: {'ok' if ok else 'fail'}"))

# ---------- Deidentify ----------
clean   = mosaicx.deidentify(documents="record.pdf")
batch   = mosaicx.deidentify(documents=["a.pdf", "b.pdf"], workers=4)
clean   = mosaicx.deidentify("John Doe SSN 123-45-6789", mode="regex")

# ---------- Summarize ----------
summary = mosaicx.summarize(documents=["r1.pdf", "r2.pdf"], patient_id="P001")
summary = mosaicx.summarize(documents=Path("patient_folder/"))
summary = mosaicx.summarize(["Report 1 text...", "Report 2 text..."])

# ---------- Utilities (unchanged) ----------
mosaicx.health()
mosaicx.list_templates()
mosaicx.list_modes()
mosaicx.generate_schema("echo report with EF and chambers")
mosaicx.evaluate("radiology", "test.jsonl")
```

### Removed (clean break, no deprecation)

Since MOSAICX has no external users yet, these are removed outright:

| Removed | Replaced by |
|---------|-------------|
| `batch_extract()` | `extract(documents=[...])` |
| `process_file()` | `extract(documents="file.pdf")` |
| `process_files()` | `extract(documents=[...], workers=4)` |
| `report()` | `extract(score=True)` |

### Function Signatures

```python
def extract(
    text: str | None = None,
    *,
    documents: str | Path | bytes | list[str | Path] | None = None,
    filename: str | None = None,
    template: str | Path | None = None,
    mode: str = "auto",
    score: bool = False,
    optimized: str | Path | None = None,
    workers: int = 1,
    on_progress: Callable[[str, bool, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:

def deidentify(
    text: str | None = None,
    *,
    documents: str | Path | bytes | list[str | Path] | None = None,
    filename: str | None = None,
    mode: str = "remove",
    workers: int = 1,
    on_progress: Callable[[str, bool, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:

def summarize(
    reports: list[str] | None = None,
    *,
    documents: str | Path | list[str | Path] | None = None,
    patient_id: str = "unknown",
    optimized: str | Path | None = None,
) -> dict[str, Any]:
```

### Input/Output Matrix

| Input type | `extract` | `deidentify` | `summarize` |
|-----------|-----------|-------------|-------------|
| Single path `"file.pdf"` | `dict` | `dict` | `dict` |
| List of paths `["a.pdf", "b.pdf"]` | `list[dict]` | `list[dict]` | `dict` |
| Directory `Path("dir/")` | `list[dict]` | `list[dict]` | `dict` |
| Bytes `b"..." + filename` | `dict` | `dict` | N/A |
| Raw text via `text=`/`reports=` | `dict` | `dict` | `dict` |

**Smart return**: single input returns `dict`, multiple inputs returns `list[dict]`. `summarize` always returns `dict` (merges into one timeline).

### Document Resolution

When `documents` is provided:

1. **`bytes`** -- write to temp file (extension from `filename`), load, extract, cleanup.
2. **`str` or `Path` pointing to a file** -- load directly via `load_document()`.
3. **`str` or `Path` pointing to a directory** -- discover all supported files (`.pdf`, `.txt`, `.docx`, `.png`, `.jpg`, etc.), process each.
4. **`list[str | Path]`** -- process each path in the list.

For multi-file processing (`extract` and `deidentify`):
- Documents are loaded sequentially (pypdfium2 is not thread-safe).
- Extraction/deidentification is parallelized via `ThreadPoolExecutor(max_workers=workers)`.
- `on_progress(filename, success, result_or_none)` is called after each file completes.
- Errors are isolated per file -- one failure doesn't stop the batch.

Each result dict includes `_document` metadata when loaded from a file:
```python
{
    "extracted": {...},
    "_document": {
        "file": "scan.pdf",
        "format": "pdf",
        "page_count": 3,
        "ocr_engine_used": "surya",
        "quality_warning": None
    }
}
```

### CLI Changes

#### `extract` command

Add `--dir` flag (absorb batch command functionality):

```bash
mosaicx extract --document scan.pdf                          # single file
mosaicx extract --document scan.pdf --template chest_ct      # with template
mosaicx extract --dir reports/ --template chest_ct            # directory batch
mosaicx extract --dir reports/ --workers 4 --output results/  # parallel + output dir
mosaicx extract --dir reports/ --format json csv              # multiple output formats
mosaicx extract --dir reports/ --resume                       # resume from checkpoint
```

New flags on `extract` (only relevant when `--dir` is used):
- `--dir` -- directory of documents to process
- `--workers` -- parallel workers (default: 1)
- `--output` / `--output-dir` -- output directory for results
- `--format` -- output format(s): json, jsonl, csv, parquet
- `--resume` -- resume from checkpoint

#### `batch` command

Removed entirely (no users).

#### `summarize` and `deidentify`

Already have `--document` and `--dir`. No changes needed.

### Documentation

Split the current monolithic developer guide into focused pages:

| Page | Content |
|------|---------|
| `docs/sdk-reference.md` (NEW) | Complete SDK API reference -- function signatures, parameter tables, input/output matrix, usage examples for every function |
| `docs/developer-guide.md` (TRIMMED) | Architecture, pipeline development, evaluation system, contributing guidelines -- remove SDK reference section |

The SDK reference page structure:
1. Quick start (5-line example)
2. Core functions (extract, deidentify, summarize) with full parameter tables + examples
3. Input types explained (text, file path, directory, bytes, list)
4. Smart return behavior
5. Progress callbacks and parallel processing
6. Utility functions (health, list_templates, list_modes, generate_schema, evaluate)
7. Configuration

### What Stays Unchanged

- CLI commands `summarize`, `deidentify`, `template`, `optimize`, `eval`, `config`, `pipeline`, `mcp`
- DSPy pipeline internals
- Template system
- Completeness scoring
- Document loading / OCR
- Configuration system
