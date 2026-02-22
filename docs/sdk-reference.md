# SDK Reference

The MOSAICX Python SDK provides programmatic access to extraction, de-identification, and summarization without the CLI. Every SDK function accepts plain Python types and returns plain Python dicts.

---

## Quick Start

```python
import mosaicx

result = mosaicx.extract("CT CHEST: 2.3cm RUL nodule...", mode="radiology")
print(result["exam_type"])       # "CT Chest"
print(result["findings"])        # [{"structure": "RUL", ...}]

# Verify the extraction
check = mosaicx.verify(extraction=result, source_text="CT CHEST: 2.3cm RUL nodule...")
print(check["verdict"])       # "verified"
```

Install with `pip install mosaicx`. Configure your LLM backend with environment variables (see [Configuration](#configuration) below).

---

## Core Functions

### `extract()`

Extract structured data from one or more documents. Accepts raw text, file paths, directories, byte content, or a list of paths.

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
    on_progress: Callable | None = None,
) -> dict[str, Any] | list[dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| None` | `None` | Document text to extract from. Mutually exclusive with `documents`. |
| `documents` | `str \| Path \| bytes \| list[str \| Path] \| None` | `None` | One or more documents to process. See [Input Types](#input-types). Mutually exclusive with `text`. |
| `filename` | `str \| None` | `None` | Original filename. Required when `documents` is `bytes` (used for format detection from the file extension). |
| `template` | `str \| Path \| None` | `None` | Template name (built-in or user-created) or path to a YAML template file. When provided, `mode` is ignored -- the template determines the extraction pipeline. |
| `mode` | `str` | `"auto"` | Extraction mode. `"auto"` lets the LLM infer the schema. `"radiology"` and `"pathology"` run specialized multi-step pipelines. Custom modes use their registered name. Ignored when `template` is provided. |
| `score` | `bool` | `False` | If `True`, compute completeness scoring against the template and include it in the output under `"completeness"`. |
| `optimized` | `str \| Path \| None` | `None` | Path to an optimized DSPy program. Applicable for `mode="auto"` or template-based extraction. |
| `workers` | `int` | `1` | Number of parallel extraction workers for multi-document input. See [Parallel Processing](#progress-callbacks-parallel-processing). |
| `on_progress` | `Callable \| None` | `None` | Callback `(filename, success, result_or_none)` called after each document completes. Only used with multi-document input. |

**Returns:** `dict[str, Any]` for single-document input, `list[dict[str, Any]]` for multi-document input. See [Smart Return](#smart-return).

**Raises:**

- `ValueError` -- if both `text` and `documents` are provided, if neither is provided, or if the template/mode is unknown.
- `FileNotFoundError` -- if a document path does not exist.

#### Examples

**Extract from raw text:**

```python
import mosaicx

result = mosaicx.extract("CT CHEST WITH CONTRAST\nFindings: 2.3cm RUL nodule...")
print(result["extracted"])
```

**Extract from a single file:**

```python
import mosaicx

result = mosaicx.extract(documents="scan.pdf", mode="radiology")
print(result["exam_type"])
print(result["findings"])
print(result["_document"]["page_count"])
```

**Extract from multiple files:**

```python
import mosaicx

results = mosaicx.extract(
    documents=["report_a.pdf", "report_b.pdf"],
    mode="radiology",
)
for r in results:
    print(r["_document"]["file"], r.get("exam_type"))
```

**Extract from a directory:**

```python
from pathlib import Path
import mosaicx

results = mosaicx.extract(documents=Path("reports/"), template="chest_ct")
print(f"Processed {len(results)} documents")
```

**Extract from bytes (e.g., web upload):**

```python
import mosaicx

uploaded_content = uploaded_file.read()
result = mosaicx.extract(
    documents=uploaded_content,
    filename="scan.pdf",
    template="chest_ct",
)
print(result["extracted"])
```

**Extract with template and completeness scoring:**

```python
import mosaicx

result = mosaicx.extract(
    "CT CHEST WITH CONTRAST\nFindings: 2.3cm RUL nodule...",
    template="chest_ct",
    score=True,
)
print(result["extracted"])
print(result["completeness"])
# {"overall": 0.85, "missing_required": [...], ...}
```

**Parallel extraction with progress callback:**

```python
import mosaicx

def on_progress(filename, success, result):
    status = "done" if success else "FAILED"
    print(f"[{status}] {filename}")

results = mosaicx.extract(
    documents=["a.pdf", "b.pdf", "c.pdf"],
    mode="radiology",
    workers=4,
    on_progress=on_progress,
)
print(f"{len(results)} documents processed")
```

**Extract with an optimized program:**

```python
import mosaicx

result = mosaicx.extract(
    "CT ABDOMEN WITH CONTRAST...",
    mode="auto",
    optimized="~/.mosaicx/optimized/extract_optimized.json",
)
```

---

### `deidentify()`

Remove Protected Health Information (PHI) from one or more documents.

```python
def deidentify(
    text: str | None = None,
    *,
    documents: str | Path | bytes | list[str | Path] | None = None,
    filename: str | None = None,
    mode: str = "remove",
    workers: int = 1,
    on_progress: Callable | None = None,
) -> dict[str, Any] | list[dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| None` | `None` | Text containing PHI. Mutually exclusive with `documents`. |
| `documents` | `str \| Path \| bytes \| list[str \| Path] \| None` | `None` | One or more documents to de-identify. See [Input Types](#input-types). Mutually exclusive with `text`. |
| `filename` | `str \| None` | `None` | Original filename. Required when `documents` is `bytes`. |
| `mode` | `str` | `"remove"` | De-identification strategy. See table below. |
| `workers` | `int` | `1` | Number of parallel workers for multi-document input. |
| `on_progress` | `Callable \| None` | `None` | Callback `(filename, success, result_or_none)` called after each document completes. |

**De-identification modes:**

| Mode | Description | Requires LLM |
|------|-------------|:------------:|
| `"remove"` | Replace PHI with `[REDACTED]` | Yes |
| `"pseudonymize"` | Replace PHI with realistic fake values | Yes |
| `"dateshift"` | Shift dates by a consistent random offset | Yes |
| `"regex"` | Regex-only pattern matching (fastest) | No |

**Returns:** `dict[str, Any]` for single input, `list[dict[str, Any]]` for multiple inputs. Each result contains a `"redacted_text"` key.

**Raises:**

- `ValueError` -- if `mode` is not one of the supported values, or if input parameters conflict.

#### Examples

**Remove PHI with default mode:**

```python
import mosaicx

result = mosaicx.deidentify("Patient John Doe, SSN 123-45-6789")
print(result["redacted_text"])
# "Patient [REDACTED], SSN [REDACTED]"
```

**Pseudonymize (replace with fake values):**

```python
import mosaicx

result = mosaicx.deidentify(
    "Patient Jane Smith, DOB 03/22/1975",
    mode="pseudonymize",
)
print(result["redacted_text"])
# "Patient Maria Garcia, DOB 07/14/1982"
```

**Regex-only mode (no LLM, fastest):**

```python
import mosaicx

result = mosaicx.deidentify(
    "SSN: 123-45-6789, MRN: 12345678",
    mode="regex",
)
print(result["redacted_text"])
```

**De-identify a PDF:**

```python
import mosaicx

result = mosaicx.deidentify(documents="clinical_note.pdf")
print(result["redacted_text"])
```

**Batch de-identification with parallel processing:**

```python
import mosaicx

results = mosaicx.deidentify(
    documents=["note_a.pdf", "note_b.pdf", "note_c.pdf"],
    workers=4,
)
for r in results:
    print(r["_document"]["file"], len(r["redacted_text"]))
```

---

### `summarize()`

Summarize multiple clinical reports into a patient timeline. Always returns a single `dict`, regardless of how many reports are provided.

```python
def summarize(
    reports: list[str] | None = None,
    *,
    documents: str | Path | list[str | Path] | None = None,
    patient_id: str = "unknown",
    optimized: str | Path | None = None,
) -> dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reports` | `list[str] \| None` | `None` | List of report texts. Mutually exclusive with `documents`. |
| `documents` | `str \| Path \| list[str \| Path] \| None` | `None` | File paths or directory to load reports from. Mutually exclusive with `reports`. |
| `patient_id` | `str` | `"unknown"` | Patient identifier included in the summary output. |
| `optimized` | `str \| Path \| None` | `None` | Path to an optimized DSPy program. |

**Returns:** `dict[str, Any]` with keys `"narrative"` (str) and `"events"` (list of event dicts). When `documents` is used, includes `"_document"` with a list of loading metadata dicts.

**Raises:**

- `ValueError` -- if both `reports` and `documents` are provided, if neither is provided, or if the report list is empty.
- `FileNotFoundError` -- if any document path does not exist.

#### Examples

**Summarize from text:**

```python
import mosaicx

reports = [
    "2024-01-15: CT Chest showing 2.3cm RUL nodule. Recommend follow-up.",
    "2024-04-20: Follow-up CT: RUL nodule stable at 2.3cm. Continue surveillance.",
    "2024-10-10: PET/CT: RUL nodule with low-grade uptake. SUVmax 2.1.",
]

result = mosaicx.summarize(reports, patient_id="PAT-001")
print(result["narrative"])
# "Patient PAT-001 was found to have a 2.3cm right upper lobe nodule..."
print(result["events"])
# [{"date": "2024-01-15", "exam_type": "CT Chest", "key_finding": "2.3cm RUL nodule", ...}, ...]
```

**Summarize from files:**

```python
from pathlib import Path
import mosaicx

result = mosaicx.summarize(
    documents=["report_jan.pdf", "report_apr.pdf", "report_oct.pdf"],
    patient_id="PAT-001",
)
print(result["narrative"])
```

**Summarize from a patient folder:**

```python
from pathlib import Path
import mosaicx

result = mosaicx.summarize(
    documents=Path("patients/PAT-001/"),
    patient_id="PAT-001",
)
print(f"Timeline has {len(result['events'])} events")
```

**Summarize with an optimized program:**

```python
import mosaicx

result = mosaicx.summarize(
    ["Report 1...", "Report 2..."],
    optimized="~/.mosaicx/optimized/summarize_optimized.json",
)
```

---

### `verify()`

Verify extractions or free-text claims against source text. Useful as a post-extraction quality gate to catch hallucinated or misattributed fields before downstream consumption.

```python
def verify(
    *,
    extraction: dict[str, Any] | None = None,
    claim: str | None = None,
    source_text: str | None = None,
    document: str | Path | None = None,
    level: str = "quick",
) -> dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extraction` | `dict[str, Any] \| None` | `None` | Extraction output dict to verify against the source. |
| `claim` | `str \| None` | `None` | Free-text claim to verify. At least one of `extraction` or `claim` must be provided. |
| `source_text` | `str \| None` | `None` | Source document text. Mutually exclusive with `document`. |
| `document` | `str \| Path \| None` | `None` | Path to a source document file. Mutually exclusive with `source_text`. |
| `level` | `str` | `"quick"` | Verification depth: `"quick"` (deterministic), `"standard"`, `"thorough"`. |

**Returns:** `dict[str, Any]` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `verdict` | `str` | Raw engine verdict (`verified`, `partially_supported`, `contradicted`, `insufficient_evidence`). |
| `decision` | `str` | Normalized final decision used for display and gating. |
| `confidence` | `float` | Engine confidence score between 0 and 1. |
| `support_score` | `float` | Claim-support score (claim mode) or confidence proxy (extraction mode). |
| `verification_mode` | `str` | `"claim"` or `"extraction"`. |
| `level` | `str` | The verification level that was used. |
| `issues` | `list[dict]` | List of issue dicts, each with keys `severity`, `field`, and `detail`. Empty when fully verified. |
| `field_verdicts` | `list[dict]` | Per-field verification results. |
| `requested_level` | `str` | Requested level (`quick`/`standard`/`thorough`). |
| `effective_level` | `str` | Actually executed level (`deterministic`/`spot_check`/`audit`). |
| `fallback_used` | `bool` | Whether verification fell back due model/tool unavailability. |
| `fallback_reason` | `str` | Present when fallback occurred. |
| `claim_comparison` | `dict` | Claim mode only: `claimed`, `source`, `evidence`, `grounded`. |

**Raises:**

- `ValueError` -- if neither `extraction` nor `claim` is provided, or if neither `source_text` nor `document` is provided.

#### Examples

**Verify a claim against text:**

```python
import mosaicx

result = mosaicx.verify(
    claim="2.3cm nodule in right upper lobe",
    source_text="FINDINGS: 2.3 cm spiculated nodule in the RUL.",
)
print(result["verdict"])      # "verified"
print(result["confidence"])   # 1.0
```

**Verify an extraction against a document file:**

```python
import mosaicx

extraction = {"exam_type": "CT Chest", "findings": [{"anatomy": "RUL", "observation": "nodule"}]}
result = mosaicx.verify(
    extraction=extraction,
    document="ct_report.pdf",
)
if result["verdict"] != "verified":
    for issue in result["issues"]:
        print(f"  {issue['severity']}: {issue['field']} - {issue['detail']}")
```

!!! tip
    Use `level="quick"` (the default) for fast deterministic checks during development. Switch to `level="thorough"` for production validation where accuracy matters more than speed.

---

### `query()`

Open a query session over one or more documents for RLM-powered natural-language Q&A. The session loads and indexes the provided sources, then exposes an `ask()` method for iterative questioning.

```python
def query(
    sources: list[str | Path] | None = None,
) -> QuerySession
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sources` | `list[str \| Path] \| None` | `None` | List of file paths to load into the query session. Supports CSV, JSON, Parquet, Excel, PDF, text files. |

**Returns:** `QuerySession` object with the following interface:

| Member | Type | Description |
|--------|------|-------------|
| `catalog` | `list` | List of source metadata objects (each has `name`, `format`, `size`). |
| `data` | `dict` | Dict of loaded data keyed by source name. |
| `ask(question)` | method | Ask a natural-language question; returns an answer string. |
| `ask_structured(question)` | method | Ask and return answer metadata: `answer`, `citations`, `confidence`, fallback flags. |
| `close()` | method | Release resources held by the session. |

**Raises:**

- `ValueError` -- if `sources` is empty or `None`.

#### Examples

**Open a query session and inspect sources:**

```python
import mosaicx

session = mosaicx.query(sources=["patient_data.csv", "notes.pdf"])

for src in session.catalog:
    print(f"{src.name}: {src.format} ({src.size:,} bytes)")

session.close()
```

**Ask questions with the query engine:**

```python
import mosaicx
from mosaicx.query.engine import QueryEngine

session = mosaicx.query(sources=["patient_data.csv", "notes.pdf"])
engine = QueryEngine(session=session)
answer = engine.ask("What is the mean patient age?")
print(answer)

session.close()
```

**Structured query response with citations:**

```python
import mosaicx

session = mosaicx.query(sources=["patient_data.csv", "notes.pdf"])
payload = session.ask_structured("What are the highest-risk findings?")
print(payload["answer"])
for c in payload["citations"]:
    print(c["source"], c["score"], c["snippet"][:120])
print("confidence:", payload["confidence"])
session.close()
```

!!! note
    The `ask()` method requires DSPy and Deno to be available for the RLM sandbox. See [Configuration](configuration.md) for setup details.

---

## Input Types

The `documents` parameter on `extract()` and `deidentify()` accepts several input types. The `summarize()` function accepts either `reports` (list of text strings) or `documents` (file paths).

| Input | Type | Example | Description |
|-------|------|---------|-------------|
| Raw text | `text="..."` | `extract(text="CT Chest...")` | Process a text string directly. |
| Single file | `documents="file.pdf"` | `extract(documents="scan.pdf")` | Load and process one file. |
| Multiple files | `documents=[...]` | `extract(documents=["a.pdf", "b.pdf"])` | Process a list of file paths. |
| Directory | `documents=Path("dir/")` | `extract(documents=Path("reports/"))` | Discover and process all supported files in the directory. |
| Bytes | `documents=b"..." + filename` | `extract(documents=content, filename="scan.pdf")` | Process raw bytes (e.g., from a web upload). The `filename` parameter is required for format detection. |

Supported file formats include `.pdf`, `.txt`, `.docx`, `.png`, `.jpg`, and other common document and image formats. PDFs and images are automatically OCR'd when needed.

### Document Resolution

When `documents` is provided, MOSAICX resolves the input as follows:

1. **`bytes`** -- Written to a temporary file (extension from `filename`), loaded, processed, and cleaned up.
2. **`str` or `Path` pointing to a file** -- Loaded directly via the document loader.
3. **`str` or `Path` pointing to a directory** -- All supported files are discovered and processed.
4. **`list[str | Path]`** -- Each path in the list is processed.

---

## Smart Return

The return type depends on whether the input is a single document or multiple documents:

| Input | `extract()` | `deidentify()` | `summarize()` |
|-------|-------------|----------------|---------------|
| Single path `"file.pdf"` | `dict` | `dict` | `dict` |
| List of paths `["a.pdf", "b.pdf"]` | `list[dict]` | `list[dict]` | `dict` |
| Directory `Path("dir/")` | `list[dict]` | `list[dict]` | `dict` |
| Bytes `b"..." + filename` | `dict` | `dict` | N/A |
| Raw text via `text=` / `reports=` | `dict` | `dict` | `dict` |

`summarize()` always returns a single `dict` because it merges all reports into one patient timeline.

!!! note
    When you need to distinguish single vs. multi-document results programmatically, check `isinstance(result, list)`.

---

## Progress Callbacks & Parallel Processing

When processing multiple documents with `extract()` or `deidentify()`, you can control parallelism and receive progress updates.

### Workers

The `workers` parameter controls how many extraction tasks run in parallel. Documents are loaded sequentially (the OCR engine is not thread-safe), but extraction itself is parallelized via a thread pool.

```python
import mosaicx

# Process 4 documents at a time
results = mosaicx.extract(
    documents=["a.pdf", "b.pdf", "c.pdf", "d.pdf"],
    workers=4,
)
```

### Progress Callback

The `on_progress` callback is called after each document completes (whether it succeeded or failed). The callback signature is:

```python
def on_progress(filename: str, success: bool, result: dict | None) -> None:
    ...
```

| Argument | Type | Description |
|----------|------|-------------|
| `filename` | `str` | Name of the processed file. |
| `success` | `bool` | `True` if extraction succeeded, `False` on error. |
| `result` | `dict \| None` | The extraction result dict on success, `None` on failure. |

```python
import mosaicx

def progress(filename, success, result):
    if success:
        print(f"  Extracted: {filename}")
    else:
        print(f"  FAILED: {filename}")

results = mosaicx.extract(
    documents=["a.pdf", "b.pdf", "c.pdf"],
    workers=4,
    on_progress=progress,
)
```

!!! note
    Errors are isolated per file -- one document failing does not stop the rest of the batch. Failed documents produce a result dict with an `"error"` key.

---

## Document Metadata

When a document is loaded from a file (via the `documents` parameter), each result dict includes a `_document` key with loading metadata:

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

| Field | Type | Description |
|-------|------|-------------|
| `file` | `str` | Filename of the source document. |
| `format` | `str` | Detected format (`"pdf"`, `"txt"`, `"docx"`, `"png"`, etc.). |
| `page_count` | `int` | Number of pages in the document. |
| `ocr_engine_used` | `str` | Which OCR engine was used (`"surya"`, `"chandra"`, `"both"`, or `None` for text files). |
| `quality_warning` | `str \| None` | Warning message if OCR quality was below the configured threshold, or `None`. |

---

## Utility Functions

### `health()`

Check MOSAICX configuration status and available capabilities. Does not make any LLM calls -- suitable for service health endpoints.

```python
import mosaicx

status = mosaicx.health()
print(f"MOSAICX v{status['version']}")
print(f"LLM: {status['lm_model']} at {status['api_base']}")
print(f"Modes: {status['available_modes']}")
print(f"Templates: {status['available_templates']}")
```

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `version` | `str` | Installed MOSAICX version. |
| `configured` | `bool` | Whether an API key is set. |
| `lm_model` | `str` | Configured language model identifier. |
| `api_base` | `str` | LLM API base URL. |
| `available_modes` | `list[str]` | Registered extraction modes (e.g., `["radiology", "pathology"]`). |
| `available_templates` | `list[str]` | Available template names (built-in + user-created). |
| `ocr_engine` | `str` | Configured OCR engine (`"both"`, `"surya"`, or `"chandra"`). |

---

### `list_templates()`

List all available extraction templates -- both built-in templates that ship with MOSAICX and user-created templates from `~/.mosaicx/templates/`.

```python
import mosaicx

for tpl in mosaicx.list_templates():
    print(f"{tpl['name']:20s} [{tpl['source']}] {tpl['description']}")
```

**Returns:** `list[dict]` -- Each dict has keys:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Template name (use with `extract(template=name)`). |
| `description` | `str` | Human-readable description. |
| `mode` | `str \| None` | Associated pipeline mode (e.g., `"radiology"`), or `None`. |
| `source` | `str` | `"built-in"` or `"user"`. |

---

### `list_modes()`

List available extraction modes with descriptions.

```python
import mosaicx

for m in mosaicx.list_modes():
    print(f"{m['name']}: {m['description']}")
# radiology: 5-step radiology report structurer
# pathology: 5-step pathology report structurer
```

**Returns:** `list[dict]` -- Each dict has keys `"name"` (str) and `"description"` (str).

---

### `generate_schema()`

Generate a Pydantic schema from a plain-English description. For most use cases, the CLI command `mosaicx template create` is the preferred way to create templates, since it integrates with the unified template system. This SDK function is available for programmatic schema generation.

```python
import mosaicx

schema = mosaicx.generate_schema(
    "Echocardiogram report with LVEF, chamber dimensions, valve grades",
    name="EchoReport",
    save=True,
)
print(schema["name"])        # "EchoReport"
print(schema["fields"])      # [{"name": "lvef", "type": "str", ...}, ...]
print(schema["saved_to"])    # "~/.mosaicx/schemas/EchoReport.json"
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `description` | `str` | (required) | Natural language description of desired fields. |
| `name` | `str \| None` | `None` | Optional schema name. If omitted the LLM will choose one. |
| `example_text` | `str \| None` | `None` | Optional example document text to guide schema generation. |
| `save` | `bool` | `False` | If `True`, persist the schema to `~/.mosaicx/schemas/`. |

**Returns:** `dict` with keys `"name"` (str), `"fields"` (list of field dicts), `"json_schema"` (dict). If `save=True`, also includes `"saved_to"` (str).

---

### `evaluate()`

Evaluate a pipeline against a labeled test set.

```python
import mosaicx

results = mosaicx.evaluate("radiology", "tests/datasets/radiology_test.jsonl")
print(f"Mean score: {results['mean']:.3f}")
print(f"Std dev: {results['std']:.3f}")

# Compare baseline vs. optimized
optimized_results = mosaicx.evaluate(
    "radiology",
    "tests/datasets/radiology_test.jsonl",
    optimized="~/.mosaicx/optimized/radiology_optimized.json",
)
print(f"Baseline: {results['mean']:.3f}")
print(f"Optimized: {optimized_results['mean']:.3f}")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | `str` | (required) | Pipeline name (`"radiology"`, `"pathology"`, `"extract"`, `"summarize"`, `"deidentify"`, `"schema"`). |
| `testset_path` | `str \| Path` | (required) | Path to a `.jsonl` file with labeled examples. |
| `optimized` | `str \| Path \| None` | `None` | Path to an optimized DSPy program. If `None`, evaluates the baseline. |

**Returns:** `dict` with keys `"mean"`, `"median"`, `"std"` (or `None` if fewer than 2 examples), `"min"`, `"max"`, `"count"`, `"scores"` (list of floats).

**Raises:**

- `ValueError` -- if `pipeline` is not recognized.
- `FileNotFoundError` -- if `testset_path` does not exist.

---

## Configuration

The SDK uses the same configuration system as the CLI. Configuration is resolved in this order (highest priority first):

1. Environment variables with the `MOSAICX_` prefix
2. Values in a `.env` file in the current directory
3. Built-in defaults

### Setting Configuration

Set environment variables before importing or calling SDK functions:

```bash
export MOSAICX_LM="openai/gpt-oss:120b"
export MOSAICX_API_BASE="http://localhost:11434/v1"
export MOSAICX_API_KEY="your-api-key"
```

Or use a `.env` file in your project root:

```ini
MOSAICX_LM=openai/gpt-oss:120b
MOSAICX_API_BASE=http://localhost:11434/v1
MOSAICX_API_KEY=your-api-key
MOSAICX_OCR_ENGINE=both
```

Or set them in Python before your first SDK call:

```python
import os
os.environ["MOSAICX_LM"] = "openai/gpt-oss:120b"
os.environ["MOSAICX_API_BASE"] = "http://localhost:11434/v1"
os.environ["MOSAICX_API_KEY"] = "your-api-key"

import mosaicx
result = mosaicx.extract("Patient presents with...")
```

DSPy is configured automatically on the first SDK call that needs it. The `deidentify()` function with `mode="regex"` does not require DSPy or an API key.

!!! tip
    For the full list of configuration options (LLM backends, OCR engines, export formats, file paths), see the [Configuration](configuration.md) guide.

---

## Error Handling

SDK functions raise standard Python exceptions:

### Common Exceptions

```python
import mosaicx

# RuntimeError: no API key or DSPy not installed
try:
    result = mosaicx.extract("some text")
except RuntimeError as e:
    if "No API key" in str(e):
        print("Set MOSAICX_API_KEY before calling SDK functions")
    elif "DSPy is required" in str(e):
        print("Install DSPy: pip install dspy")

# ValueError: unknown mode or invalid parameters
try:
    result = mosaicx.extract("some text", mode="nonexistent")
except ValueError as e:
    print(f"Invalid mode: {e}")

# FileNotFoundError: missing document or template
try:
    result = mosaicx.extract(documents="missing_file.pdf")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

### Batch Error Recovery

When processing multiple documents, errors are isolated per file. Failed documents do not stop the batch -- they produce a result dict with an `"error"` key:

```python
import mosaicx

results = mosaicx.extract(
    documents=["valid.pdf", "corrupt.pdf", "another_valid.pdf"],
    workers=4,
)

for r in results:
    if "error" in r:
        print(f"FAILED {r['_document']['file']}: {r['error']}")
    else:
        print(f"OK {r['_document']['file']}")
```

---

## Integration Examples

### Verify After Extract

```python
import mosaicx

report_text = "CT CHEST: 2.3 cm spiculated RUL nodule. Impression: Suspicious for malignancy."

# Step 1: Extract
result = mosaicx.extract(report_text, mode="radiology")

# Step 2: Verify
check = mosaicx.verify(extraction=result, source_text=report_text)
if check["verdict"] == "verified":
    print("Extraction verified against source")
else:
    print(f"Issues found: {len(check['issues'])}")
    for issue in check["issues"]:
        print(f"  {issue['severity']}: {issue['detail']}")
```

### pandas DataFrame

```python
import pandas as pd
import mosaicx

df = pd.read_csv("reports.csv")

results = []
for _, row in df.iterrows():
    try:
        result = mosaicx.extract(row["report_text"], mode="radiology")
        result["source_id"] = row["report_id"]
        results.append(result)
    except Exception as e:
        results.append({"source_id": row["report_id"], "error": str(e)})

results_df = pd.json_normalize(results, sep="_")
results_df.to_csv("extracted_results.csv", index=False)
```

### FastAPI

```python
import os
os.environ["MOSAICX_API_KEY"] = "your-api-key"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mosaicx

app = FastAPI(title="MOSAICX API")

class ExtractRequest(BaseModel):
    text: str
    template: str | None = None
    mode: str = "auto"
    score: bool = False

@app.post("/extract")
def run_extract(req: ExtractRequest):
    try:
        return mosaicx.extract(
            req.text,
            template=req.template,
            mode=req.mode,
            score=req.score,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deidentify")
def run_deidentify(text: str, mode: str = "remove"):
    return mosaicx.deidentify(text, mode=mode)
```

### Jupyter Notebook

```python
# Cell 1: Configure
import os
os.environ["MOSAICX_LM"] = "openai/gpt-oss:120b"
os.environ["MOSAICX_API_BASE"] = "http://localhost:11434/v1"
os.environ["MOSAICX_API_KEY"] = "your-api-key"

# Cell 2: Extract
import mosaicx

report = """
CT CHEST WITH CONTRAST
Clinical indication: Cough, weight loss

FINDINGS:
Right upper lobe: 2.3 cm spiculated nodule (series 4, image 67).
Left lung: Clear.

IMPRESSION:
1. 2.3 cm spiculated RUL nodule, suspicious for malignancy.
"""

result = mosaicx.extract(report, mode="radiology")
result

# Cell 3: Explore results
import json
print(json.dumps(result, indent=2, default=str))
```
