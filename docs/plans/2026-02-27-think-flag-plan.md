# `--think` Extraction Strategy Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `--think fast|standard|deep` flag to CLI, SDK, and MCP that controls extraction reasoning depth.

**Architecture:** Thread a `think: Literal["fast", "standard", "deep"]` parameter from the interface layer (CLI/SDK/MCP) through `report.py` and `DocumentExtractor` down to `_extract_schema_with_structured_chain()`, where it controls which branches of the extraction cascade execute. Default `"standard"` preserves current behavior with zero breaking changes.

**Tech Stack:** Click (CLI), DSPy (extraction), Pydantic (schemas), pytest (tests)

**Design doc:** `docs/plans/2026-02-27-think-flag-design.md`

---

### Task 1: Add `think` parameter to extraction core

**Files:**
- Modify: `mosaicx/pipelines/extraction.py:2173` (`_extract_schema_with_structured_chain`)
- Modify: `mosaicx/pipelines/extraction.py:2514` (`DocumentExtractor.__init__`)
- Modify: `mosaicx/pipelines/extraction.py:2557` (`DocumentExtractor.forward`)
- Test: `tests/test_extraction_pipeline.py`

**Step 1: Write the failing tests**

Add to `tests/test_extraction_pipeline.py`:

```python
class TestThinkParameter:
    """DocumentExtractor respects the think parameter."""

    def test_default_think_is_standard(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        from pydantic import BaseModel

        class SimpleReport(BaseModel):
            summary: str

        extractor = DocumentExtractor(output_schema=SimpleReport)
        assert extractor._think == "standard"

    def test_think_fast_stored(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        from pydantic import BaseModel

        class SimpleReport(BaseModel):
            summary: str

        extractor = DocumentExtractor(output_schema=SimpleReport, think="fast")
        assert extractor._think == "fast"

    def test_think_deep_stored(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        from pydantic import BaseModel

        class SimpleReport(BaseModel):
            summary: str

        extractor = DocumentExtractor(output_schema=SimpleReport, think="deep")
        assert extractor._think == "deep"

    def test_think_invalid_raises(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        from pydantic import BaseModel

        class SimpleReport(BaseModel):
            summary: str

        with pytest.raises(ValueError, match="think"):
            DocumentExtractor(output_schema=SimpleReport, think="invalid")
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py::TestThinkParameter -v`
Expected: FAIL — `__init__` doesn't accept `think` parameter

**Step 3: Implement — modify DocumentExtractor.__init__**

In `mosaicx/pipelines/extraction.py`, change `DocumentExtractor.__init__` (line 2514):

```python
def __init__(
    self,
    output_schema: type[BaseModel] | None = None,
    think: str = "standard",
) -> None:
    super().__init__()
    if think not in ("fast", "standard", "deep"):
        raise ValueError(
            f"think must be 'fast', 'standard', or 'deep', got {think!r}"
        )
    self._think = think
    self._output_schema = output_schema
    # ... rest unchanged
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py::TestThinkParameter -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add mosaicx/pipelines/extraction.py tests/test_extraction_pipeline.py
git commit -m "feat: add think parameter to DocumentExtractor"
```

---

### Task 2: Wire `think` through `_extract_schema_with_structured_chain` — fast mode

**Files:**
- Modify: `mosaicx/pipelines/extraction.py:2173` (`_extract_schema_with_structured_chain`)
- Modify: `mosaicx/pipelines/extraction.py:2557` (`DocumentExtractor.forward`)
- Test: `tests/test_extraction_pipeline.py`

**Step 1: Write the failing test**

```python
class TestThinkFastMode:
    """think=fast skips ChainOfThought, uses Predict fallback only."""

    def test_fast_mode_skips_outlines_and_cot(self, monkeypatch):
        """Fast mode: Outlines -> if fails, dspy.Predict -> coerce."""
        from mosaicx.pipelines.extraction import DocumentExtractor

        class SimpleReport(BaseModel):
            summary: str
            category: str

        extractor = DocumentExtractor(output_schema=SimpleReport, think="fast")

        # Mock extract_custom (ChainOfThought) to track if called
        cot_called = False
        original_extract = extractor.extract_custom

        def _tracking_extract(**kwargs):
            nonlocal cot_called
            cot_called = True
            return original_extract(**kwargs)

        # Mock Outlines to fail (so we test the Predict fallback)
        monkeypatch.setattr(
            "mosaicx.pipelines.extraction._recover_schema_instance_with_outlines",
            lambda **kwargs: None,
        )

        # Mock the JSON fallback (Predict) to return valid data
        class _FallbackPred:
            extracted_json = '{"summary": "test", "category": "radiology"}'

        monkeypatch.setattr(
            extractor,
            "extract_json_fallback",
            lambda **kwargs: _FallbackPred(),
        )

        # Patch extract_custom to track calls
        monkeypatch.setattr(extractor, "extract_custom", _tracking_extract)

        result = extractor.forward("test document")
        assert result.extracted.summary == "test"
        # ChainOfThought should NOT have been called in fast mode
        assert not cot_called
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py::TestThinkFastMode -v`
Expected: FAIL — `_extract_schema_with_structured_chain` doesn't accept `think`

**Step 3: Implement — add think parameter to chain function and forward**

In `_extract_schema_with_structured_chain` (line 2173), add the parameter:

```python
def _extract_schema_with_structured_chain(
    *,
    document_text: str,
    schema_class: type[BaseModel],
    typed_extract: Any,
    json_extract: Any | None = None,
    planner_diag: dict[str, Any] | None = None,
    think: str = "standard",
) -> tuple[BaseModel, dict[str, Any]]:
```

Add fast-mode logic after the existing `has_lm` check (around line 2220):

```python
    # Fast mode: Outlines only, fallback to json_extract (Predict, no reasoning)
    if think == "fast":
        if has_lm:
            outlines_result = _recover_schema_instance_with_outlines(
                document_text=document_text,
                schema_class=schema_class,
                error_hint="fast_mode",
            )
            if outlines_result is not None:
                _record("outlines_fast", True)
                return outlines_result, {
                    "selected_path": "outlines_fast",
                    "fallback_used": False,
                    "attempts": attempts,
                    "bestofn": bestofn_info,
                    "adjudication": adjudication_info,
                }
            _record("outlines_fast", False, ValueError("outlines_fast_unavailable"))

        # Predict fallback (no reasoning)
        if json_extract is not None:
            try:
                fallback_pred = json_extract(document_text=document_text)
                raw = getattr(fallback_pred, "extracted_json", "")
                model_instance = _recover_schema_instance_from_raw(raw, schema_class)
                _record("predict_fast", True)
                return model_instance, {
                    "selected_path": "predict_fast",
                    "fallback_used": True,
                    "attempts": attempts,
                    "bestofn": bestofn_info,
                    "adjudication": adjudication_info,
                }
            except Exception as exc:
                _record("predict_fast", False, exc)

        raise ValueError("Fast mode: both Outlines and Predict failed")
```

In `DocumentExtractor.forward` (line 2597), pass `think`:

```python
model_instance, chain_diag = _extract_schema_with_structured_chain(
    document_text=planned_text,
    schema_class=schema,
    typed_extract=lambda *, document_text: self.extract_custom(
        document_text=document_text
    ),
    json_extract=lambda *, document_text: self.extract_json_fallback(
        document_text=document_text
    ),
    planner_diag=planner_diag,
    think=self._think,
)
```

Do the same for the rescue call (~line 2611) and the auto-mode calls (~line 2690).

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py::TestThinkFastMode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mosaicx/pipelines/extraction.py tests/test_extraction_pipeline.py
git commit -m "feat: implement think=fast mode in extraction chain"
```

---

### Task 3: Implement `think=deep` mode — both paths, score, pick best

**Files:**
- Modify: `mosaicx/pipelines/extraction.py:2173` (`_extract_schema_with_structured_chain`)
- Test: `tests/test_extraction_pipeline.py`

**Step 1: Write the failing test**

```python
class TestThinkDeepMode:
    """think=deep runs both Outlines and ChainOfThought, picks best."""

    def test_deep_mode_runs_both_and_picks_higher_score(self, monkeypatch):
        from mosaicx.pipelines.extraction import DocumentExtractor

        class SimpleReport(BaseModel):
            summary: str
            category: str

        extractor = DocumentExtractor(output_schema=SimpleReport, think="deep")

        outlines_called = False
        cot_called = False

        # Outlines returns a result with low evidence overlap
        def _mock_outlines(**kwargs):
            nonlocal outlines_called
            outlines_called = True
            return SimpleReport(summary="wrong", category="wrong")

        monkeypatch.setattr(
            "mosaicx.pipelines.extraction._recover_schema_instance_with_outlines",
            _mock_outlines,
        )

        # ChainOfThought returns a result with values from the source text
        class _CotPred:
            extracted = SimpleReport(summary="chest pain", category="radiology")

        def _mock_cot(**kwargs):
            nonlocal cot_called
            cot_called = True
            return _CotPred()

        monkeypatch.setattr(extractor, "extract_custom", _mock_cot)

        result = extractor.forward("Patient presents with chest pain. Radiology report.")

        assert outlines_called, "Deep mode should try Outlines"
        assert cot_called, "Deep mode should always run ChainOfThought"
        # CoT result has evidence overlap ("chest pain" in source), Outlines doesn't
        assert result.extracted.summary == "chest pain"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py::TestThinkDeepMode -v`
Expected: FAIL — deep mode not yet implemented

**Step 3: Implement deep mode in _extract_schema_with_structured_chain**

Add after the fast-mode block:

```python
    # Deep mode: run both Outlines and ChainOfThought, score, pick best
    if think == "deep":
        candidates: list[tuple[str, BaseModel, float, dict[str, float]]] = []

        # Candidate 1: Outlines
        if has_lm:
            outlines_result = _recover_schema_instance_with_outlines(
                document_text=document_text,
                schema_class=schema_class,
                error_hint="deep_mode_baseline",
            )
            if outlines_result is not None:
                score, components = _score_extraction_candidate(
                    extracted=outlines_result,
                    schema_class=schema_class,
                    source_text=document_text,
                )
                candidates.append(("outlines_deep", outlines_result, score, components))
                _record("outlines_deep", True)
            else:
                _record("outlines_deep", False, ValueError("outlines_deep_unavailable"))

        # Candidate 2: ChainOfThought (always runs in deep mode)
        try:
            cot_pred = typed_extract(document_text=document_text)
            cot_extracted = getattr(cot_pred, "extracted", cot_pred)
            cot_instance = _coerce_extracted_to_model_instance(
                extracted=cot_extracted,
                schema_class=schema_class,
            )
            score, components = _score_extraction_candidate(
                extracted=cot_instance,
                schema_class=schema_class,
                source_text=document_text,
            )
            candidates.append(("cot_deep", cot_instance, score, components))
            _record("cot_deep", True)
        except Exception as exc:
            _record("cot_deep", False, exc)

        if not candidates:
            raise ValueError("Deep mode: both Outlines and ChainOfThought failed")

        # Pick highest-scoring candidate
        candidates.sort(key=lambda c: c[2], reverse=True)
        best_path, best_instance, best_score, best_components = candidates[0]

        return best_instance, {
            "selected_path": best_path,
            "fallback_used": len(candidates) > 1 and best_path != "cot_deep",
            "attempts": attempts,
            "bestofn": bestofn_info,
            "adjudication": {
                "deep_mode": True,
                "candidates": [
                    {"path": c[0], "score": c[2], "components": c[3]}
                    for c in candidates
                ],
                "chosen": best_path,
            },
        }
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py::TestThinkDeepMode -v`
Expected: PASS

**Step 5: Also run existing tests to confirm standard mode unchanged**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add mosaicx/pipelines/extraction.py tests/test_extraction_pipeline.py
git commit -m "feat: implement think=deep mode with dual-path scoring"
```

---

### Task 4: Wire `think` through `report.py`

**Files:**
- Modify: `mosaicx/report.py:231` (`run_report`)
- Test: `tests/test_extraction_pipeline.py`

**Step 1: Write the failing test**

```python
class TestRunReportThink:
    def test_run_report_accepts_think_parameter(self):
        """run_report should accept and forward think parameter."""
        import inspect
        from mosaicx.report import run_report
        sig = inspect.signature(run_report)
        assert "think" in sig.parameters
        assert sig.parameters["think"].default == "standard"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py::TestRunReportThink -v`
Expected: FAIL — `think` not in `run_report` signature

**Step 3: Implement — add think to run_report**

In `mosaicx/report.py:231`, add the parameter:

```python
def run_report(
    document_text: str,
    template_model: type[BaseModel] | None = None,
    template_name: str | None = None,
    mode: str | None = None,
    think: str = "standard",
) -> ReportResult:
```

At line 291 (and line 311), pass it to DocumentExtractor:

```python
extractor = DocumentExtractor(output_schema=template_model, think=think)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py::TestRunReportThink -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mosaicx/report.py tests/test_extraction_pipeline.py
git commit -m "feat: thread think parameter through run_report"
```

---

### Task 5: Wire `think` through SDK

**Files:**
- Modify: `mosaicx/sdk.py:1258` (`extract`)
- Modify: `mosaicx/sdk.py:1046` (`_extract_single_text`)
- Test: `tests/test_sdk_envelope.py`

**Step 1: Write the failing test**

Add to `tests/test_sdk_envelope.py`:

```python
class TestSdkThinkParameter:
    def test_extract_accepts_think_parameter(self):
        import inspect
        from mosaicx.sdk import extract
        sig = inspect.signature(extract)
        assert "think" in sig.parameters
        assert sig.parameters["think"].default == "standard"

    def test_extract_single_text_accepts_think(self):
        import inspect
        from mosaicx.sdk import _extract_single_text
        sig = inspect.signature(_extract_single_text)
        assert "think" in sig.parameters
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sdk_envelope.py::TestSdkThinkParameter -v`
Expected: FAIL

**Step 3: Implement — add think to SDK functions**

In `mosaicx/sdk.py:1258` (`extract`), add parameter:

```python
def extract(
    text: str | None = None,
    *,
    # ... existing params ...
    on_progress: Callable[[str, bool, dict[str, Any] | None], None] | None = None,
    think: str = "standard",
) -> dict[str, Any] | list[dict[str, Any]]:
```

Pass `think=think` to `_extract_single_text()` call (~line 1346).

In `mosaicx/sdk.py:1046` (`_extract_single_text`), add parameter:

```python
def _extract_single_text(
    text: str,
    *,
    # ... existing params ...
    provenance: bool,
    think: str = "standard",
) -> dict[str, Any]:
```

Pass `think=think` to `DocumentExtractor()` at lines 1162 and 1231, and to `run_report()` where called.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_sdk_envelope.py::TestSdkThinkParameter -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mosaicx/sdk.py tests/test_sdk_envelope.py
git commit -m "feat: add think parameter to SDK extract()"
```

---

### Task 6: Wire `think` through MCP server

**Files:**
- Modify: `mosaicx/mcp_server.py:159` (`extract_document`)
- Test: `tests/test_mcp_server_dspy_config.py`

**Step 1: Write the failing test**

```python
class TestMcpExtractThink:
    def test_extract_document_accepts_think(self):
        import inspect
        from mosaicx.mcp_server import extract_document
        sig = inspect.signature(extract_document)
        assert "think" in sig.parameters
        assert sig.parameters["think"].default == "standard"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mcp_server_dspy_config.py::TestMcpExtractThink -v`
Expected: FAIL

**Step 3: Implement — add think to MCP handler**

In `mosaicx/mcp_server.py:159`:

```python
def extract_document(
    document_text: str,
    mode: str = "auto",
    template: str | None = None,
    score: bool = False,
    think: str = "standard",
) -> str:
```

Pass `think=think` to `DocumentExtractor()` at lines 243 and 305.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_mcp_server_dspy_config.py::TestMcpExtractThink -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mosaicx/mcp_server.py tests/test_mcp_server_dspy_config.py
git commit -m "feat: add think parameter to MCP extract tool"
```

---

### Task 7: Wire `think` through CLI

**Files:**
- Modify: `mosaicx/cli.py:534` (extract command options)
- Modify: `mosaicx/cli.py:305` (`_extract_batch`)
- Test: `tests/test_cli_extract.py`

**Step 1: Write the failing test**

Add to `tests/test_cli_extract.py`:

```python
class TestExtractThinkFlag:
    def test_extract_help_shows_think_flag(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--think" in result.output
        assert "fast" in result.output
        assert "standard" in result.output
        assert "deep" in result.output

    def test_extract_rejects_invalid_think(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "extract", "--document", "nonexistent.pdf", "--think", "invalid"
        ])
        assert result.exit_code != 0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_extract.py::TestExtractThinkFlag -v`
Expected: FAIL — no `--think` option

**Step 3: Implement — add --think to CLI**

In `mosaicx/cli.py`, add the option after line 555 (before `@click.pass_context`):

```python
@click.option(
    "--think",
    type=click.Choice(["fast", "standard", "deep"], case_sensitive=False),
    default="standard",
    show_default=True,
    help="Reasoning depth: fast (no reasoning), standard (cascade), deep (full reasoning).",
)
```

Add `think: str` to the `extract` function signature (line 557).

Thread `think` to:
- `_extract_batch()` call (~line 648)
- `DocumentExtractor()` instantiations (~lines 752, 812)
- `run_report()` calls where applicable

In `_extract_batch` (line 305), add `think: str = "standard"` parameter and capture in the `process_fn` closure.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_cli_extract.py::TestExtractThinkFlag -v`
Expected: PASS

**Step 5: Run full CLI test suite**

Run: `.venv/bin/python -m pytest tests/test_cli_extract.py tests/test_cli.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add mosaicx/cli.py tests/test_cli_extract.py
git commit -m "feat: add --think flag to CLI extract command"
```

---

### Task 8: Deep mode — enable dspy.Refine for repair

**Files:**
- Modify: `mosaicx/pipelines/extraction.py` (forward method, repair section)
- Test: `tests/test_extraction_pipeline.py`

**Step 1: Write the failing test**

```python
class TestThinkDeepRepair:
    def test_deep_mode_enables_refine(self, monkeypatch):
        """Deep mode should set use_refine=True for the repair phase."""
        from mosaicx.pipelines.extraction import DocumentExtractor

        class SimpleReport(BaseModel):
            summary: str

        extractor = DocumentExtractor(output_schema=SimpleReport, think="deep")

        # Track whether _repair_failed_critical_fields_with_refine is called with use_refine=True
        repair_kwargs_captured = {}

        original_repair = None
        import mosaicx.pipelines.extraction as ext_mod

        def _capture_repair(**kwargs):
            repair_kwargs_captured.update(kwargs)
            # Return the model unchanged
            return kwargs.get("model_instance"), {}

        monkeypatch.setattr(
            ext_mod,
            "_repair_failed_critical_fields_with_refine",
            _capture_repair,
        )

        # Mock the extraction to succeed immediately
        class _Pred:
            extracted = SimpleReport(summary="test")

        monkeypatch.setattr(extractor, "extract_custom", lambda **kw: _Pred())
        monkeypatch.setattr(
            ext_mod, "_recover_schema_instance_with_outlines", lambda **kw: None,
        )

        result = extractor.forward("test doc")
        assert repair_kwargs_captured.get("use_refine") is True
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py::TestThinkDeepRepair -v`
Expected: FAIL

**Step 3: Implement**

In `DocumentExtractor.forward()`, in the repair section after extraction, check `self._think`:

```python
use_refine = self._think == "deep" or cfg.use_refine
```

Pass this to the repair function call.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py::TestThinkDeepRepair -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mosaicx/pipelines/extraction.py tests/test_extraction_pipeline.py
git commit -m "feat: deep mode enables dspy.Refine for field repair"
```

---

### Task 9: Full integration test and final commit

**Files:**
- Test: `tests/test_extraction_pipeline.py`

**Step 1: Run the complete test suite**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: ALL PASS, no regressions.

**Step 2: Run linter**

```bash
.venv/bin/python -m ruff check mosaicx/pipelines/extraction.py mosaicx/cli.py mosaicx/sdk.py mosaicx/mcp_server.py mosaicx/report.py
```

Expected: No errors.

**Step 3: Run type checker**

```bash
.venv/bin/python -m mypy mosaicx/pipelines/extraction.py mosaicx/cli.py mosaicx/sdk.py mosaicx/mcp_server.py mosaicx/report.py --ignore-missing-imports
```

Expected: No new errors.

**Step 4: Manual smoke test (if LLM server running)**

```bash
mosaicx extract --document tests/datasets/sample_report.txt --template chest_ct --think fast
mosaicx extract --document tests/datasets/sample_report.txt --template chest_ct --think deep
```

**Step 5: Final commit if any fixups needed**

```bash
git add -A
git commit -m "test: add integration tests for --think flag"
```
