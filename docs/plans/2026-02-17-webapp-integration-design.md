# Design: MOSAICX SDK Enhancements for Web Integration

**Date:** 2026-02-17
**Status:** Approved

## Context

MOSAICX is an open-source medical document structuring engine with three interface layers: CLI, Python SDK, and MCP server. A clinical web application (Zenta) is being built on top of MOSAICX by a separate frontend and backend team.

**Problem:** The backend team needs to integrate MOSAICX into a web application but the current SDK requires manual document loading (OCR) separate from extraction. There is no single call that takes a file and returns structured data.

**Constraint:** MOSAICX stays open-source. The production API server, auth, and frontend are part of the private Zenta product. We only add SDK improvements and a reference example to the open-source repo.

## Decision

Enhance `mosaicx.sdk` with three new methods that cover common webapp patterns, and provide a minimal FastAPI integration example in `examples/`.

## New SDK Methods

### `sdk.process_file()` -- File to structured data in one call

```python
def process_file(
    file: Path | bytes,
    *,
    filename: str | None = None,
    template: str | None = None,
    mode: str = "auto",
    score: bool = False,
    ocr_engine: str | None = None,
    force_ocr: bool = False,
) -> dict[str, Any]:
```

**Behavior:**
1. If `file` is `bytes`, write to a temp file using `filename` for format detection.
2. Load document via `documents.loader.load_document()` (handles OCR).
3. Call `sdk.extract()` with the loaded text.
4. Return extraction result dict with additional `_document` metadata (page_count, ocr_engine_used, quality_warning).

**Why:** Backend devs receive file uploads as bytes. This method handles the full pipeline without requiring them to understand OCR internals.

### `sdk.process_files()` -- Batch processing with progress callbacks

```python
def process_files(
    files: list[Path] | Path,
    *,
    template: str | None = None,
    mode: str = "auto",
    score: bool = False,
    workers: int = 4,
    on_progress: Callable[[str, bool, dict | None], None] | None = None,
) -> dict[str, Any]:
```

**Behavior:**
1. If `files` is a `Path` to a directory, discover all supported documents in it.
2. Load documents sequentially (OCR is not thread-safe for pypdfium2).
3. Run extraction in parallel using `ThreadPoolExecutor` (LLM calls are I/O-bound).
4. Call `on_progress(filename, success, result_or_none)` after each document.
5. Return summary dict: `total`, `succeeded`, `failed`, `results` (list), `errors` (list).

**Why:** Batch is the primary workflow. The `on_progress` callback lets the backend team wire up WebSocket/SSE progress updates.

### `sdk.health()` -- Health check for integrations

```python
def health() -> dict[str, Any]:
```

**Returns:**
```json
{
  "version": "2.0.0a1",
  "configured": true,
  "lm_model": "openai/gpt-oss:120b",
  "api_base": "http://localhost:11434/v1",
  "available_modes": ["radiology", "pathology"],
  "available_templates": ["chest_ct", "brain_mri", ...],
  "ocr_engine": "both"
}
```

**Why:** Every web service needs a health endpoint. This checks that MOSAICX configuration is valid without making an LLM call.

## Integration Example

A single file at `examples/fastapi_integration.py` (~80 lines) demonstrating the wrapping pattern:

```python
"""Minimal FastAPI server wrapping mosaicx.sdk -- reference only.

This is a teaching example, NOT production code. For production:
- Add authentication and authorization
- Add rate limiting
- Use a task queue (Celery, ARQ) for batch processing
- Add proper error handling and logging
- Use async workers (uvicorn with multiple workers)
"""
from fastapi import FastAPI, UploadFile
from mosaicx import sdk

app = FastAPI(title="MOSAICX API Example")

@app.get("/health")
def health():
    return sdk.health()

@app.post("/extract")
def extract(text: str, template: str | None = None, mode: str = "auto"):
    return sdk.extract(text, template=template, mode=mode)

@app.post("/extract/file")
async def extract_file(file: UploadFile, template: str | None = None):
    content = await file.read()
    return sdk.process_file(content, filename=file.filename, template=template)

@app.post("/deidentify")
def deidentify(text: str, mode: str = "remove"):
    return sdk.deidentify(text, mode=mode)

@app.post("/summarize")
def summarize(reports: list[str]):
    return sdk.summarize(reports)

@app.get("/templates")
def list_templates():
    return sdk.list_templates()

@app.get("/modes")
def list_modes():
    return sdk.list_modes()
```

## Boundary: MOSAICX vs Zenta

| Layer | Location | Notes |
|-------|----------|-------|
| DSPy pipelines | MOSAICX (open-source) | Core engine |
| CLI, SDK, MCP | MOSAICX (open-source) | Interface layers |
| `process_file`, `process_files`, `health` | MOSAICX SDK (open-source) | Convenience methods -- benefit all users |
| FastAPI example | MOSAICX `examples/` | Teaching reference |
| Production API server | Zenta (private) | Auth, queue, deployment |
| OpenAPI specification | Zenta (private) | Product-specific contract |
| Frontend | Zenta (private) | Product differentiator |

## Implementation Steps

1. Add `sdk.process_file()` -- wraps document loader + extract
2. Add `sdk.process_files()` -- wraps batch processor with SDK integration
3. Add `sdk.health()` -- configuration introspection
4. Add `sdk.list_templates()` -- convenience wrapper (already have `list_schemas`)
5. Create `examples/fastapi_integration.py` -- reference example
6. Update docs and run tests
