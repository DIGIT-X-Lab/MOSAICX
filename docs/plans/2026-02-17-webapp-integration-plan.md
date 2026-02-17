# Webapp Integration — SDK Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `process_file()`, `process_files()`, `health()`, and `list_templates()` to `mosaicx.sdk`, then create a FastAPI reference example.

**Architecture:** New SDK functions wrap existing document loading (`documents.loader`) and extraction (`sdk.extract()`) into single-call convenience methods. `process_file()` accepts Path or bytes. `process_files()` delegates to `BatchProcessor` internally. No new dependencies in core.

**Tech Stack:** Python, pytest, existing mosaicx internals. FastAPI example is documentation only (not a dependency).

---

### Task 1: Add `sdk.health()` — configuration introspection

**Files:**
- Modify: `mosaicx/sdk.py` (append after `batch_extract` function, ~line 746)
- Test: `tests/test_public_api.py`

**Step 1: Write the failing test**

Add to `tests/test_public_api.py` at the end:

```python
class TestHealth:
    """Test sdk.health() — no LLM needed."""

    def test_returns_dict(self):
        from mosaicx.sdk import health

        result = health()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        from mosaicx.sdk import health

        result = health()
        for key in ("version", "configured", "lm_model", "api_base", "available_modes", "available_templates", "ocr_engine"):
            assert key in result, f"Missing key: {key}"

    def test_version_is_string(self):
        from mosaicx.sdk import health

        result = health()
        assert isinstance(result["version"], str)

    def test_available_modes_is_list(self):
        from mosaicx.sdk import health

        result = health()
        assert isinstance(result["available_modes"], list)

    def test_available_templates_is_list(self):
        from mosaicx.sdk import health

        result = health()
        assert isinstance(result["available_templates"], list)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_public_api.py::TestHealth -v`
Expected: FAIL with `ImportError` or `cannot import name 'health'`

**Step 3: Write the implementation**

Add to `mosaicx/sdk.py` after the `batch_extract` function (after line 745):

```python
# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def health() -> dict[str, Any]:
    """Check MOSAICX configuration status and available capabilities.

    Does NOT make an LLM call. Reads configuration and scans available
    modes/templates to report what the system can do.

    Returns
    -------
    dict
        Keys: ``"version"``, ``"configured"``, ``"lm_model"``,
        ``"api_base"``, ``"available_modes"``, ``"available_templates"``,
        ``"ocr_engine"``.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    from .config import get_config

    try:
        version = _pkg_version("mosaicx")
    except PackageNotFoundError:
        version = "2.0.0a1"

    cfg = get_config()

    # Modes (no DSPy needed — just registry scan)
    import mosaicx.pipelines.pathology  # noqa: F401
    import mosaicx.pipelines.radiology  # noqa: F401
    from .pipelines.modes import list_modes as _list_modes

    modes = [name for name, _desc in _list_modes()]

    # Templates (built-in + user)
    from .schemas.radreport.registry import list_templates as _list_builtin

    templates = [t.name for t in _list_builtin()]
    if cfg.templates_dir.is_dir():
        for f in sorted(cfg.templates_dir.glob("*.yaml")) + sorted(cfg.templates_dir.glob("*.yml")):
            templates.append(f.stem)

    return {
        "version": version,
        "configured": bool(cfg.api_key),
        "lm_model": cfg.lm,
        "api_base": cfg.api_base,
        "available_modes": modes,
        "available_templates": templates,
        "ocr_engine": cfg.ocr_engine,
    }
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_public_api.py::TestHealth -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add mosaicx/sdk.py tests/test_public_api.py
git commit -m "feat(sdk): add health() for configuration introspection"
```

---

### Task 2: Add `sdk.list_templates()` — template listing

**Files:**
- Modify: `mosaicx/sdk.py` (append after `list_modes` function, ~line 623)
- Modify: `mosaicx/__init__.py` (add to imports and `__all__`)
- Test: `tests/test_public_api.py`

**Step 1: Write the failing test**

Add to `tests/test_public_api.py`:

```python
class TestListTemplates:
    """Test sdk.list_templates() — no LLM needed."""

    def test_returns_list(self):
        from mosaicx.sdk import list_templates

        result = list_templates()
        assert isinstance(result, list)

    def test_items_are_dicts(self):
        from mosaicx.sdk import list_templates

        result = list_templates()
        for item in result:
            assert isinstance(item, dict)
            assert "name" in item
            assert "source" in item
```

Also add `"list_templates"` to the `test_all_exports` assertion list, and add a signature test:

```python
    def test_list_templates_signature(self):
        from mosaicx import list_templates

        sig = inspect.signature(list_templates)
        # No required params
        for param in sig.parameters.values():
            assert param.default is not inspect.Parameter.empty or param.kind in (
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
            )
```

And update `test_all_exports` to include `"list_templates"`.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_public_api.py::TestListTemplates -v`
Expected: FAIL with `ImportError`

**Step 3: Write the implementation**

Add to `mosaicx/sdk.py` after the `list_modes` function (after line 623):

```python
# ---------------------------------------------------------------------------
# list_templates
# ---------------------------------------------------------------------------


def list_templates() -> list[dict[str, Any]]:
    """List available extraction templates (built-in and user-created).

    Returns
    -------
    list[dict]
        Each dict has keys ``"name"``, ``"description"``, ``"mode"``,
        and ``"source"`` (``"built-in"`` or ``"user"``).
    """
    from .config import get_config
    from .schemas.radreport.registry import list_templates as _list_builtin

    cfg = get_config()
    templates: list[dict[str, Any]] = []

    for tpl in _list_builtin():
        templates.append({
            "name": tpl.name,
            "description": tpl.description,
            "mode": tpl.mode,
            "source": "built-in",
        })

    if cfg.templates_dir.is_dir():
        from .schemas.template_compiler import parse_template

        for f in sorted(cfg.templates_dir.glob("*.yaml")) + sorted(cfg.templates_dir.glob("*.yml")):
            try:
                meta = parse_template(f.read_text(encoding="utf-8"))
                templates.append({
                    "name": f.stem,
                    "description": meta.description or "",
                    "mode": meta.mode,
                    "source": "user",
                })
            except Exception:
                templates.append({
                    "name": f.stem,
                    "description": "(invalid YAML)",
                    "mode": None,
                    "source": "user",
                })

    return templates
```

Update `mosaicx/__init__.py`:
- Add `list_templates` to the `from mosaicx.sdk import (...)` block (line 238)
- Add `"list_templates"` to `__all__` (line 250)

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_public_api.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add mosaicx/sdk.py mosaicx/__init__.py tests/test_public_api.py
git commit -m "feat(sdk): add list_templates() for template discovery"
```

---

### Task 3: Add `sdk.process_file()` — file-to-structured in one call

**Files:**
- Modify: `mosaicx/sdk.py` (append after `health` function)
- Modify: `mosaicx/__init__.py` (add to imports and `__all__`)
- Test: `tests/test_public_api.py`

**Step 1: Write the failing tests**

Add to `tests/test_public_api.py`:

```python
class TestProcessFile:
    """Test sdk.process_file() signature and basic validation."""

    def test_importable(self):
        from mosaicx.sdk import process_file
        assert callable(process_file)

    def test_signature_has_expected_params(self):
        from mosaicx.sdk import process_file

        sig = inspect.signature(process_file)
        params = list(sig.parameters.keys())
        assert "file" in params
        assert "filename" in params
        assert "template" in params
        assert "mode" in params
        assert "score" in params

    def test_txt_file(self, tmp_path, monkeypatch):
        """Process a .txt file — no OCR, no LLM (mock extract)."""
        from mosaicx import sdk

        txt = tmp_path / "report.txt"
        txt.write_text("Normal chest radiograph. No acute findings.")

        # Mock sdk.extract to avoid needing LLM
        monkeypatch.setattr(sdk, "extract", lambda text, **kw: {"extracted": {"summary": text[:20]}})
        # Mock _ensure_configured to skip DSPy setup
        monkeypatch.setattr(sdk, "_ensure_configured", lambda: None)

        result = sdk.process_file(txt, mode="auto")
        assert "extracted" in result
        assert "_document" in result
        assert result["_document"]["format"] == "txt"

    def test_bytes_input_with_filename(self, tmp_path, monkeypatch):
        """Process bytes input with a filename for format detection."""
        from mosaicx import sdk

        content = b"Normal chest radiograph. No acute findings."

        monkeypatch.setattr(sdk, "extract", lambda text, **kw: {"extracted": {"summary": "ok"}})
        monkeypatch.setattr(sdk, "_ensure_configured", lambda: None)

        result = sdk.process_file(content, filename="report.txt", mode="auto")
        assert "extracted" in result
        assert "_document" in result

    def test_bytes_without_filename_raises(self):
        """Bytes input without filename should raise ValueError."""
        from mosaicx.sdk import process_file

        with pytest.raises(ValueError, match="filename"):
            process_file(b"some bytes")

    def test_nonexistent_file_raises(self, tmp_path):
        """Nonexistent file path should raise FileNotFoundError."""
        from mosaicx.sdk import process_file

        with pytest.raises(FileNotFoundError):
            process_file(tmp_path / "nonexistent.txt")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_public_api.py::TestProcessFile -v`
Expected: FAIL with `ImportError`

**Step 3: Write the implementation**

Add to `mosaicx/sdk.py`:

```python
# ---------------------------------------------------------------------------
# process_file
# ---------------------------------------------------------------------------


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
    """Load a document and extract structured data in one call.

    Handles OCR for PDFs and images, then runs the extraction pipeline.
    Accepts a file path or raw bytes (e.g., from a web upload).

    Parameters
    ----------
    file:
        Path to a document file, or raw bytes of the file content.
    filename:
        Original filename. Required when *file* is ``bytes`` so the
        format can be detected from the extension.
    template:
        Template name or YAML file path for targeted extraction.
    mode:
        Extraction mode (``"auto"``, ``"radiology"``, ``"pathology"``).
    score:
        If ``True``, include completeness scoring in the result.
    ocr_engine:
        Override the configured OCR engine (``"both"``, ``"surya"``,
        ``"chandra"``).  If ``None``, uses the config default.
    force_ocr:
        Force OCR even on PDFs with a native text layer.

    Returns
    -------
    dict
        Extraction result from :func:`extract`, plus a ``"_document"``
        key with loading metadata (format, page_count, ocr_engine_used,
        quality_warning).

    Raises
    ------
    ValueError
        If *file* is ``bytes`` and *filename* is not provided.
    FileNotFoundError
        If *file* is a path that does not exist.
    """
    import tempfile

    from .config import get_config
    from .documents.loader import load_document

    cfg = get_config()

    if isinstance(file, bytes):
        if not filename:
            raise ValueError(
                "filename is required when file is bytes "
                "(needed for format detection from extension)."
            )
        # Write to temp file for the loader
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file)
            tmp_path = Path(tmp.name)
        try:
            doc = load_document(
                tmp_path,
                ocr_engine=ocr_engine or cfg.ocr_engine,
                force_ocr=force_ocr or cfg.force_ocr,
                ocr_langs=cfg.ocr_langs,
                quality_threshold=cfg.quality_threshold,
                page_timeout=cfg.ocr_page_timeout,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        doc = load_document(
            file_path,
            ocr_engine=ocr_engine or cfg.ocr_engine,
            force_ocr=force_ocr or cfg.force_ocr,
            ocr_langs=cfg.ocr_langs,
            quality_threshold=cfg.quality_threshold,
            page_timeout=cfg.ocr_page_timeout,
        )

    # Extract structured data
    result = extract(doc.text, template=template, mode=mode, score=score)

    # Attach document metadata
    result["_document"] = {
        "format": doc.format,
        "page_count": doc.page_count,
        "ocr_engine_used": doc.ocr_engine_used,
        "quality_warning": doc.quality_warning,
    }

    return result
```

Update `mosaicx/__init__.py`:
- Add `process_file` to the `from mosaicx.sdk import (...)` block
- Add `"process_file"` to `__all__`

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_public_api.py::TestProcessFile -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add mosaicx/sdk.py mosaicx/__init__.py tests/test_public_api.py
git commit -m "feat(sdk): add process_file() -- file-to-structured in one call"
```

---

### Task 4: Add `sdk.process_files()` — batch with progress callbacks

**Files:**
- Modify: `mosaicx/sdk.py` (append after `process_file`)
- Modify: `mosaicx/__init__.py` (add to imports and `__all__`)
- Test: `tests/test_public_api.py`

**Step 1: Write the failing tests**

Add to `tests/test_public_api.py`:

```python
class TestProcessFiles:
    """Test sdk.process_files() with real files but mocked extraction."""

    def test_importable(self):
        from mosaicx.sdk import process_files
        assert callable(process_files)

    def test_signature_has_expected_params(self):
        from mosaicx.sdk import process_files

        sig = inspect.signature(process_files)
        params = list(sig.parameters.keys())
        assert "files" in params
        assert "template" in params
        assert "mode" in params
        assert "workers" in params
        assert "on_progress" in params

    def test_process_txt_directory(self, tmp_path, monkeypatch):
        """Process a directory of .txt files with mocked extraction."""
        from mosaicx import sdk

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "doc1.txt").write_text("Report one content")
        (input_dir / "doc2.txt").write_text("Report two content")

        monkeypatch.setattr(sdk, "extract", lambda text, **kw: {"extracted": {"text": text[:10]}})
        monkeypatch.setattr(sdk, "_ensure_configured", lambda: None)

        result = sdk.process_files(input_dir)
        assert result["total"] == 2
        assert result["succeeded"] == 2
        assert result["failed"] == 0
        assert len(result["results"]) == 2

    def test_process_file_list(self, tmp_path, monkeypatch):
        """Process an explicit list of file paths."""
        from mosaicx import sdk

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("Report A")
        f2.write_text("Report B")

        monkeypatch.setattr(sdk, "extract", lambda text, **kw: {"extracted": "ok"})
        monkeypatch.setattr(sdk, "_ensure_configured", lambda: None)

        result = sdk.process_files([f1, f2])
        assert result["total"] == 2
        assert result["succeeded"] == 2

    def test_progress_callback(self, tmp_path, monkeypatch):
        """Verify on_progress is called for each document."""
        from mosaicx import sdk

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "doc1.txt").write_text("Report one")

        monkeypatch.setattr(sdk, "extract", lambda text, **kw: {"extracted": "ok"})
        monkeypatch.setattr(sdk, "_ensure_configured", lambda: None)

        progress_calls = []
        result = sdk.process_files(
            input_dir,
            on_progress=lambda name, ok, res: progress_calls.append((name, ok)),
        )
        assert len(progress_calls) == 1
        assert progress_calls[0][1] is True  # success

    def test_error_isolation(self, tmp_path, monkeypatch):
        """One failing document should not stop the others."""
        from mosaicx import sdk

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "good.txt").write_text("Good report")
        (input_dir / "bad.txt").write_text("Bad report")

        def flaky_extract(text, **kw):
            if "Bad" in text:
                raise ValueError("Simulated failure")
            return {"extracted": "ok"}

        monkeypatch.setattr(sdk, "extract", flaky_extract)
        monkeypatch.setattr(sdk, "_ensure_configured", lambda: None)

        result = sdk.process_files(input_dir)
        assert result["succeeded"] == 1
        assert result["failed"] == 1
        assert len(result["errors"]) == 1

    def test_empty_directory(self, tmp_path, monkeypatch):
        """Empty directory returns zero counts."""
        from mosaicx import sdk

        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        monkeypatch.setattr(sdk, "_ensure_configured", lambda: None)

        result = sdk.process_files(input_dir)
        assert result["total"] == 0
        assert result["succeeded"] == 0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_public_api.py::TestProcessFiles -v`
Expected: FAIL with `ImportError`

**Step 3: Write the implementation**

Add to `mosaicx/sdk.py` after `process_file`:

```python
# ---------------------------------------------------------------------------
# process_files
# ---------------------------------------------------------------------------


def process_files(
    files: list[Path] | Path,
    *,
    template: str | None = None,
    mode: str = "auto",
    score: bool = False,
    workers: int = 4,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """Process multiple documents with parallel extraction.

    Accepts a directory path (discovers all supported documents) or an
    explicit list of file paths.

    Parameters
    ----------
    files:
        Directory path or list of file paths.
    template:
        Template name for targeted extraction.
    mode:
        Extraction mode (``"auto"``, ``"radiology"``, ``"pathology"``).
    score:
        Include completeness scoring.
    workers:
        Number of parallel extraction workers.
    on_progress:
        Optional callback ``(filename: str, success: bool, result: dict | None) -> None``
        called after each document completes.

    Returns
    -------
    dict
        Keys: ``"total"``, ``"succeeded"``, ``"failed"``,
        ``"results"`` (list of result dicts), ``"errors"`` (list of
        error dicts with ``"file"`` and ``"error"`` keys).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .config import get_config
    from .documents.engines.base import SUPPORTED_FORMATS
    from .documents.loader import load_document

    cfg = get_config()

    # Resolve file list
    if isinstance(files, Path) and files.is_dir():
        file_list = sorted(
            p for p in files.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
        )
    elif isinstance(files, Path):
        file_list = [files]
    else:
        file_list = [Path(f) for f in files]

    if not file_list:
        return {"total": 0, "succeeded": 0, "failed": 0, "results": [], "errors": []}

    # Load documents sequentially (pypdfium2 is not thread-safe)
    loaded: list[tuple[Path, str | None, str | None]] = []
    for path in file_list:
        try:
            doc = load_document(
                path,
                ocr_engine=cfg.ocr_engine,
                force_ocr=cfg.force_ocr,
                ocr_langs=cfg.ocr_langs,
                quality_threshold=cfg.quality_threshold,
                page_timeout=cfg.ocr_page_timeout,
            )
            loaded.append((path, doc.text, None))
        except Exception as exc:
            loaded.append((path, None, f"{type(exc).__name__}: {exc}"))

    succeeded = 0
    failed = 0
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    # Handle load failures
    to_extract = []
    for path, text, err in loaded:
        if err is not None:
            failed += 1
            errors.append({"file": path.name, "error": err})
            if on_progress:
                on_progress(path.name, False, None)
        elif text:
            to_extract.append((path, text))

    # Parallel extraction
    def _do_extract(path: Path, text: str) -> tuple[str, dict | None, str | None]:
        try:
            result = extract(text, template=template, mode=mode, score=score)
            return path.name, result, None
        except Exception as exc:
            return path.name, None, f"{type(exc).__name__}: {exc}"

    max_w = min(max(1, workers), 32)
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        futures = {pool.submit(_do_extract, p, t): p for p, t in to_extract}
        for future in as_completed(futures):
            name, result, error = future.result()
            if error:
                failed += 1
                errors.append({"file": name, "error": error})
                if on_progress:
                    on_progress(name, False, None)
            else:
                succeeded += 1
                results.append({"file": name, **(result or {})})
                if on_progress:
                    on_progress(name, True, result)

    return {
        "total": len(file_list),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
        "errors": errors,
    }
```

Update `mosaicx/__init__.py`:
- Add `process_files` to the `from mosaicx.sdk import (...)` block
- Add `"process_files"` to `__all__`

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_public_api.py::TestProcessFiles -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add mosaicx/sdk.py mosaicx/__init__.py tests/test_public_api.py
git commit -m "feat(sdk): add process_files() -- batch with progress callbacks"
```

---

### Task 5: Create `examples/fastapi_integration.py` — reference example

**Files:**
- Create: `examples/fastapi_integration.py`

**Step 1: Write the example file**

```python
"""Minimal FastAPI server wrapping mosaicx.sdk -- reference example.

This is a TEACHING EXAMPLE, not production code. It demonstrates how
to wrap the MOSAICX SDK in a web API. For production deployments:

- Add authentication and authorization
- Add rate limiting and request validation
- Use a task queue (Celery, ARQ) for batch processing
- Add structured logging and error tracking
- Run with multiple uvicorn workers

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn examples.fastapi_integration:app --reload

The server runs at http://localhost:8000 with auto-generated docs
at http://localhost:8000/docs (Swagger UI).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from mosaicx import sdk

app = FastAPI(
    title="MOSAICX API",
    description="Reference API wrapping the MOSAICX medical document structuring SDK.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    text: str
    template: str | None = None
    mode: str = "auto"
    score: bool = False


class DeidentifyRequest(BaseModel):
    text: str
    mode: str = "remove"


class SummarizeRequest(BaseModel):
    reports: list[str]
    patient_id: str = "unknown"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Check MOSAICX configuration and available capabilities."""
    return sdk.health()


@app.post("/extract")
def extract_text(req: ExtractRequest):
    """Extract structured data from document text."""
    try:
        return sdk.extract(
            req.text, template=req.template, mode=req.mode, score=req.score,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/extract/file")
async def extract_file(
    file: UploadFile,
    template: str | None = None,
    mode: str = "auto",
    score: bool = False,
):
    """Upload a document file (PDF, image, text) and extract structured data."""
    content = await file.read()
    try:
        return sdk.process_file(
            content,
            filename=file.filename,
            template=template,
            mode=mode,
            score=score,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/deidentify")
def deidentify(req: DeidentifyRequest):
    """Remove Protected Health Information from text."""
    try:
        return sdk.deidentify(req.text, mode=req.mode)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    """Summarize multiple clinical reports into a patient timeline."""
    try:
        return sdk.summarize(req.reports, patient_id=req.patient_id)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/templates")
def templates():
    """List available extraction templates."""
    return sdk.list_templates()


@app.get("/modes")
def modes():
    """List available extraction modes."""
    return sdk.list_modes()
```

**Step 2: Verify it parses (no syntax errors)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('examples/fastapi_integration.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add examples/fastapi_integration.py
git commit -m "docs: add FastAPI integration example for web teams"
```

---

### Task 6: Update `__init__.py` exports and run full test suite

**Files:**
- Modify: `mosaicx/__init__.py` (final cleanup of imports and `__all__`)
- Test: all tests

This task ensures all new SDK functions are properly exported and nothing is broken.

**Step 1: Verify `__init__.py` has all new imports**

The `from mosaicx.sdk import (...)` block should include:
- `health`
- `list_templates`
- `process_file`
- `process_files`

The `__all__` list should include all four names.

**Step 2: Run ruff**

Run: `.venv/bin/python -m ruff check mosaicx/sdk.py mosaicx/__init__.py`
Expected: No errors

**Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests pass (existing + new)

**Step 4: Final commit if any fixups needed**

```bash
git add -A
git commit -m "chore: fixups from full test suite run"
```
