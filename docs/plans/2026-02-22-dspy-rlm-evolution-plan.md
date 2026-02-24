# MOSAICX DSPy 3.x + RLM Evolution — Implementation Plan

**Status:** Historical implementation plan (non-canonical)
**Canonical Status Board:** `docs/plans/2026-02-24-roadmap-status-audit.md`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evolve MOSAICX into a medical document intelligence platform with self-healing extraction, verification, provenance tracking, and conversational query — all powered by DSPy 3.x and RLM.

**Architecture:** Four phases. Phase 1 adds the `_mosaicx` metadata envelope and upgrades extraction with `dspy.Refine`/`BestOfN`. Phase 2 adds inline provenance and the 3-level verification engine. Phase 3 adds RLM-powered conversational query. Phase 4 is polish. Each phase is independently shippable.

**Tech Stack:** DSPy 3.1.3+, dspy.RLM, dspy.Refine, dspy.BestOfN, Pydantic v2, Click, Rich, pytest, Deno (for RLM sandbox)

**Design doc:** `docs/plans/2026-02-22-dspy-rlm-evolution-design.md`

---

## Phase 1: Metadata Envelope + Self-Healing Extraction

---

### Task 1: Metadata Envelope Models

**Files:**
- Create: `mosaicx/envelope.py`
- Test: `tests/test_envelope.py`

**Step 1: Write the failing test**

```python
# tests/test_envelope.py
from __future__ import annotations

import pytest


class TestMosaicxEnvelope:
    def test_build_envelope_returns_required_keys(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology", template="chest_ct")
        assert "version" in env
        assert "pipeline" in env
        assert "template" in env
        assert "model" in env
        assert "model_temperature" in env
        assert "timestamp" in env
        assert "duration_s" in env
        assert "tokens" in env
        assert "provenance" in env
        assert "verification" in env
        assert "document" in env

    def test_build_envelope_defaults(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology")
        assert env["template"] is None
        assert env["provenance"] is False
        assert env["verification"] is None
        assert env["document"] is None
        assert env["duration_s"] is None
        assert isinstance(env["timestamp"], str)

    def test_build_envelope_with_document_meta(self):
        from mosaicx.envelope import build_envelope

        doc_meta = {"file": "report.pdf", "pages": 2, "ocr_engine": "surya", "quality_warning": None}
        env = build_envelope(pipeline="extraction", document=doc_meta)
        assert env["document"]["file"] == "report.pdf"
        assert env["document"]["pages"] == 2

    def test_build_envelope_with_tokens(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology", tokens={"input": 100, "output": 50})
        assert env["tokens"]["input"] == 100
        assert env["tokens"]["output"] == 50

    def test_build_envelope_serializable(self):
        import json
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology")
        # Must be JSON-serializable (no Python objects)
        json_str = json.dumps(env)
        assert isinstance(json_str, str)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_envelope.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mosaicx.envelope'`

**Step 3: Write minimal implementation**

```python
# mosaicx/envelope.py
"""Standard _mosaicx metadata envelope for all MOSAICX outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _get_version() -> str:
    """Return the installed MOSAICX version."""
    try:
        from importlib.metadata import version
        return version("mosaicx")
    except Exception:
        return "2.0.0a1"


def _get_model_info() -> tuple[str, float]:
    """Read current LM model and temperature from config."""
    try:
        from mosaicx.config import get_config
        cfg = get_config()
        return cfg.lm, cfg.lm_temperature
    except Exception:
        return "unknown", 0.0


def build_envelope(
    *,
    pipeline: str,
    template: str | None = None,
    template_version: str | None = None,
    duration_s: float | None = None,
    tokens: dict[str, int] | None = None,
    provenance: bool = False,
    verification: dict[str, Any] | None = None,
    document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the standard _mosaicx metadata envelope.

    This is attached to every MOSAICX output dict under the ``"_mosaicx"`` key.

    Parameters
    ----------
    pipeline:
        Name of the pipeline that produced the output.
    template:
        Template name used for extraction, if any.
    template_version:
        Template version string, if available.
    duration_s:
        Total processing duration in seconds.
    tokens:
        Token usage dict with ``"input"`` and ``"output"`` keys.
    provenance:
        Whether provenance tracking was enabled.
    verification:
        Verification summary dict, if verification was run.
    document:
        Document metadata dict (file, pages, ocr_engine, etc.).

    Returns
    -------
    dict[str, Any]
        JSON-serializable metadata envelope.
    """
    model, temperature = _get_model_info()

    return {
        "version": _get_version(),
        "pipeline": pipeline,
        "template": template,
        "template_version": template_version,
        "model": model,
        "model_temperature": temperature,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_s": duration_s,
        "tokens": tokens or {"input": 0, "output": 0},
        "provenance": provenance,
        "verification": verification,
        "document": document,
    }
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_envelope.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add mosaicx/envelope.py tests/test_envelope.py
git commit -m "feat: add _mosaicx metadata envelope builder"
```

---

### Task 2: Wire Envelope into SDK extract()

**Files:**
- Modify: `mosaicx/sdk.py` (functions `_extract_single_text`, `_deidentify_single_text`, `summarize`)
- Test: `tests/test_sdk_envelope.py`

**Step 1: Write the failing test**

```python
# tests/test_sdk_envelope.py
from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest


class TestSDKEnvelope:
    def test_extract_output_has_mosaicx_key(self):
        """Every extract() output must have _mosaicx metadata."""
        from mosaicx.envelope import build_envelope

        # We test the envelope attachment logic without calling LLMs.
        # Mock _extract_single_text to return a plain dict, then verify
        # the envelope would be attached.
        env = build_envelope(pipeline="radiology", template="chest_ct")
        result = {"exam_type": "CT Chest", "_mosaicx": env}
        assert "_mosaicx" in result
        assert result["_mosaicx"]["pipeline"] == "radiology"
        assert result["_mosaicx"]["template"] == "chest_ct"
        assert result["_mosaicx"]["version"] is not None
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sdk_envelope.py -v`
Expected: PASS (this is a unit test of the integration pattern)

**Step 3: Modify SDK to attach envelope**

In `mosaicx/sdk.py`, at the end of each `_extract_single_text`, `_deidentify_single_text`, and `summarize` function, add envelope attachment. The key change pattern:

```python
# At the end of _extract_single_text, before returning output_data:
from mosaicx.envelope import build_envelope

envelope = build_envelope(
    pipeline=effective_mode or "auto",
    template=str(template) if template else None,
    duration_s=metrics.total_duration_s if metrics else None,
    tokens={"input": metrics.total_input_tokens, "output": metrics.total_output_tokens} if metrics else None,
    document=doc_meta if 'doc_meta' in locals() else None,
)
output_data["_mosaicx"] = envelope
```

Replace existing `_metrics` and `_document` attachment with the unified `_mosaicx` envelope. Maintain backward compatibility by keeping `_metrics` as a nested key inside the envelope.

**Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All existing tests pass. Some tests checking for `_metrics` key may need updating.

**Step 5: Commit**

```bash
git add mosaicx/sdk.py tests/test_sdk_envelope.py
git commit -m "feat: wire _mosaicx envelope into SDK extract/deidentify/summarize"
```

---

### Task 3: Reward Functions for Refine/BestOfN

**Files:**
- Create: `mosaicx/pipelines/rewards.py`
- Test: `tests/test_pipeline_rewards.py`

These are the reward functions that `dspy.Refine` and `dspy.BestOfN` will use inside extraction pipelines. Separate from `evaluation/rewards.py` (which scores gold vs predicted). These score a single prediction's quality.

**Step 1: Write the failing test**

```python
# tests/test_pipeline_rewards.py
from __future__ import annotations

import pytest


class TestFindingsReward:
    def test_empty_findings_scores_low(self):
        from mosaicx.pipelines.rewards import findings_reward

        score = findings_reward(findings=[])
        assert score < 0.3

    def test_findings_with_anatomy_scores_higher(self):
        from mosaicx.pipelines.rewards import findings_reward

        findings = [
            {"anatomy": "right upper lobe", "observation": "nodule", "description": "5mm nodule"},
            {"anatomy": "left lower lobe", "observation": "atelectasis", "description": "mild"},
        ]
        score = findings_reward(findings=findings)
        assert score > 0.5

    def test_findings_with_measurements_get_bonus(self):
        from mosaicx.pipelines.rewards import findings_reward

        no_measure = [{"anatomy": "RUL", "observation": "nodule", "description": "nodule"}]
        with_measure = [{"anatomy": "RUL", "observation": "nodule", "description": "5mm nodule"}]
        assert findings_reward(findings=with_measure) > findings_reward(findings=no_measure)

    def test_reward_capped_at_one(self):
        from mosaicx.pipelines.rewards import findings_reward

        many_findings = [
            {"anatomy": f"loc{i}", "observation": "finding", "description": f"{i}mm mass"}
            for i in range(20)
        ]
        score = findings_reward(findings=many_findings)
        assert score <= 1.0


class TestImpressionReward:
    def test_empty_impression_scores_zero(self):
        from mosaicx.pipelines.rewards import impression_reward

        assert impression_reward(impressions=[]) == 0.0

    def test_nonempty_impression_scores_positive(self):
        from mosaicx.pipelines.rewards import impression_reward

        impressions = [{"statement": "Pulmonary nodule, recommend follow-up.", "actionable": True}]
        assert impression_reward(impressions=impressions) > 0.0


class TestDiagnosisReward:
    def test_empty_diagnosis_scores_zero(self):
        from mosaicx.pipelines.rewards import diagnosis_reward

        assert diagnosis_reward(diagnoses=[]) == 0.0

    def test_diagnosis_with_detail_scores_high(self):
        from mosaicx.pipelines.rewards import diagnosis_reward

        diagnoses = [{"diagnosis": "Adenocarcinoma", "grade": "well-differentiated", "margin": "negative"}]
        assert diagnosis_reward(diagnoses=diagnoses) > 0.5


class TestSchemaComplianceReward:
    def test_all_required_fields_present(self):
        from mosaicx.pipelines.rewards import schema_compliance_reward

        schema_fields = ["indication", "findings", "impression"]
        extraction = {"indication": "cough", "findings": "nodule", "impression": "follow-up"}
        assert schema_compliance_reward(extraction=extraction, required_fields=schema_fields) == 1.0

    def test_missing_required_field_penalized(self):
        from mosaicx.pipelines.rewards import schema_compliance_reward

        schema_fields = ["indication", "findings", "impression"]
        extraction = {"indication": "cough", "findings": "nodule"}
        score = schema_compliance_reward(extraction=extraction, required_fields=schema_fields)
        assert score < 1.0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline_rewards.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# mosaicx/pipelines/rewards.py
"""Reward functions for dspy.Refine and dspy.BestOfN inside extraction pipelines.

These score a single prediction's quality (no gold label needed).
Separate from evaluation/rewards.py which compares against gold labels.
"""

from __future__ import annotations

import re
from typing import Any

_MEASUREMENT_RE = re.compile(r"\d+\s*(mm|cm|m|cc|ml|mg|g|kg)", re.IGNORECASE)


def findings_reward(findings: list[dict[str, Any]]) -> float:
    """Score extraction quality of a findings list."""
    if not findings:
        return 0.0

    score = 0.3  # non-empty findings
    anatomy_bonus = sum(
        0.1 for f in findings
        if f.get("anatomy") and str(f["anatomy"]).strip()
    )
    score += min(anatomy_bonus, 0.4)

    measure_bonus = sum(
        0.1 for f in findings
        if _MEASUREMENT_RE.search(str(f.get("description", "")))
    )
    score += min(measure_bonus, 0.2)

    return min(score, 1.0)


def impression_reward(impressions: list[dict[str, Any]]) -> float:
    """Score extraction quality of an impressions list."""
    if not impressions:
        return 0.0

    score = 0.3  # non-empty
    for imp in impressions:
        stmt = str(imp.get("statement", ""))
        if len(stmt) > 10:
            score += 0.2
        if imp.get("actionable"):
            score += 0.1
    return min(score, 1.0)


def diagnosis_reward(diagnoses: list[dict[str, Any]]) -> float:
    """Score extraction quality of a diagnosis list."""
    if not diagnoses:
        return 0.0

    score = 0.3
    for dx in diagnoses:
        if dx.get("diagnosis") and str(dx["diagnosis"]).strip():
            score += 0.2
        if dx.get("grade"):
            score += 0.1
        if dx.get("margin"):
            score += 0.1
    return min(score, 1.0)


def schema_compliance_reward(
    extraction: dict[str, Any],
    required_fields: list[str],
) -> float:
    """Score how well an extraction matches the expected schema fields."""
    if not required_fields:
        return 1.0

    present = sum(
        1 for f in required_fields
        if f in extraction and extraction[f] is not None and str(extraction[f]).strip()
    )
    return present / len(required_fields)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline_rewards.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add mosaicx/pipelines/rewards.py tests/test_pipeline_rewards.py
git commit -m "feat: add pipeline-level reward functions for Refine/BestOfN"
```

---

### Task 4: Wrap Radiology Pipeline with dspy.Refine

**Files:**
- Modify: `mosaicx/pipelines/radiology.py` (inside `_build_dspy_classes()`)
- Modify: `mosaicx/config.py` (add `use_refine` flag)
- Test: `tests/test_radiology_refine.py`

**Step 1: Write the failing test**

```python
# tests/test_radiology_refine.py
from __future__ import annotations

import pytest


class TestRadiologyRefineConfig:
    def test_config_has_use_refine_flag(self):
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig(api_key="test")
        assert hasattr(cfg, "use_refine")
        assert isinstance(cfg.use_refine, bool)

    def test_use_refine_defaults_false(self):
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig(api_key="test")
        assert cfg.use_refine is False


class TestRadiologyPipelineRefineWiring:
    def test_pipeline_has_extract_findings_module(self):
        """RadiologyReportStructurer always has extract_findings, whether Refine or ChainOfThought."""
        from mosaicx.pipelines.radiology import RadiologyReportStructurer

        pipeline = RadiologyReportStructurer()
        assert hasattr(pipeline, "extract_findings")

    def test_pipeline_has_extract_impression_module(self):
        from mosaicx.pipelines.radiology import RadiologyReportStructurer

        pipeline = RadiologyReportStructurer()
        assert hasattr(pipeline, "extract_impression")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_radiology_refine.py -v`
Expected: FAIL on `test_config_has_use_refine_flag` (attribute doesn't exist yet)

**Step 3: Add `use_refine` to config**

In `mosaicx/config.py`, add to `MosaicxConfig`:

```python
    # --- Self-healing ---
    use_refine: bool = False
```

**Step 4: Modify radiology pipeline to optionally use Refine**

In `mosaicx/pipelines/radiology.py`, inside `_build_dspy_classes()`, modify the `RadiologyReportStructurer.__init__` to check config:

```python
class RadiologyReportStructurer(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        from mosaicx.config import get_config
        cfg = get_config()

        self.classify_exam = dspy.Predict(ClassifyExamType)
        self.parse_sections = dspy.Predict(ParseReportSections)
        self.extract_technique = dspy.Predict(ExtractTechnique)

        if cfg.use_refine:
            from mosaicx.pipelines.rewards import findings_reward as _fr

            def _findings_reward_fn(args, pred):
                findings = pred.findings
                finding_dicts = [f.model_dump() if hasattr(f, "model_dump") else f for f in findings]
                return _fr(findings=finding_dicts)

            self.extract_findings = dspy.Refine(
                module=dspy.ChainOfThought(ExtractRadFindings),
                N=3,
                reward_fn=_findings_reward_fn,
                threshold=0.7,
            )
            self.extract_impression = dspy.Refine(
                module=dspy.ChainOfThought(ExtractImpression),
                N=3,
                reward_fn=lambda args, pred: 0.8 if pred.impressions else 0.0,
                threshold=0.5,
            )
        else:
            self.extract_findings = dspy.ChainOfThought(ExtractRadFindings)
            self.extract_impression = dspy.ChainOfThought(ExtractImpression)
```

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_radiology_refine.py tests/test_radiology_pipeline.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add mosaicx/config.py mosaicx/pipelines/radiology.py tests/test_radiology_refine.py
git commit -m "feat: add dspy.Refine wrappers to radiology pipeline (opt-in via use_refine)"
```

---

### Task 5: Wrap Pathology Pipeline with dspy.Refine

**Files:**
- Modify: `mosaicx/pipelines/pathology.py` (inside `_build_dspy_classes()`)
- Test: `tests/test_pathology_refine.py`

Same pattern as Task 4 but for pathology. Wrap `ExtractPathDiagnosis` with `dspy.Refine` using `diagnosis_reward`, and `ExtractMicroscopicFindings` with `findings_reward`. Gate on `cfg.use_refine`.

**Step 1-6:** Follow same TDD pattern as Task 4.

**Commit message:** `feat: add dspy.Refine wrappers to pathology pipeline`

---

### Task 6: Add SIMBA to Optimizer Budget Presets

**Files:**
- Modify: `mosaicx/evaluation/optimize.py` (update `_BUDGET_PRESETS`, `_build_optimizer`)
- Test: `tests/test_optimize_simba.py`

**Step 1: Write the failing test**

```python
# tests/test_optimize_simba.py
from __future__ import annotations

import pytest


class TestSIMBABudget:
    def test_medium_budget_uses_simba(self):
        from mosaicx.evaluation.optimize import get_optimizer_config

        config = get_optimizer_config("medium")
        assert config["strategy"] == "SIMBA"

    def test_budget_presets_all_exist(self):
        from mosaicx.evaluation.optimize import get_optimizer_config

        for budget in ("light", "medium", "heavy"):
            config = get_optimizer_config(budget)
            assert "strategy" in config
            assert "max_iterations" in config
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_optimize_simba.py -v`
Expected: FAIL — medium preset currently uses `"MIPROv2"`

**Step 3: Update budget presets**

In `mosaicx/evaluation/optimize.py`, change `_BUDGET_PRESETS`:

```python
_BUDGET_PRESETS: dict[str, dict[str, Any]] = {
    "light": {"max_iterations": 10, "strategy": "BootstrapFewShot", "num_candidates": 5},
    "medium": {"max_iterations": 50, "strategy": "SIMBA", "num_candidates": 10},
    "heavy": {"max_iterations": 150, "strategy": "GEPA", "num_candidates": 20,
              "reflection_lm": None, "candidate_selection_strategy": "pareto", "use_merge": True},
}
```

Also update `OPTIMIZATION_STRATEGY` list at the top.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_optimize_simba.py tests/test_optimize.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mosaicx/evaluation/optimize.py tests/test_optimize_simba.py
git commit -m "feat: replace MIPROv2 with SIMBA for medium budget preset"
```

---

## Phase 2: Provenance + Verification Engine

---

### Task 7: Provenance Data Models

**Files:**
- Create: `mosaicx/provenance/__init__.py`
- Create: `mosaicx/provenance/models.py`
- Test: `tests/test_provenance_models.py`

**Step 1: Write the failing test**

```python
# tests/test_provenance_models.py
from __future__ import annotations

import json
import pytest


class TestSourceSpan:
    def test_construction(self):
        from mosaicx.provenance.models import SourceSpan

        span = SourceSpan(page=1, line_start=14, char_start=847, char_end=912)
        assert span.page == 1
        assert span.char_start == 847

    def test_serializable(self):
        import json
        from mosaicx.provenance.models import SourceSpan

        span = SourceSpan(page=1, line_start=14, char_start=847, char_end=912)
        data = span.model_dump()
        assert json.dumps(data)  # must be JSON-serializable


class TestFieldEvidence:
    def test_construction(self):
        from mosaicx.provenance.models import FieldEvidence, SourceSpan

        ev = FieldEvidence(
            field_path="findings[0].severity",
            source_excerpt="mild bilateral bony neural foraminal narrowing",
            source_location=SourceSpan(page=1, line_start=14, char_start=847, char_end=912),
            confidence=0.95,
        )
        assert ev.field_path == "findings[0].severity"
        assert ev.confidence == 0.95

    def test_optional_location(self):
        from mosaicx.provenance.models import FieldEvidence

        ev = FieldEvidence(
            field_path="exam_type",
            source_excerpt="CT Chest",
            confidence=0.99,
        )
        assert ev.source_location is None


class TestProvenanceMap:
    def test_to_dict(self):
        from mosaicx.provenance.models import FieldEvidence, ProvenanceMap

        pm = ProvenanceMap(fields=[
            FieldEvidence(field_path="findings[0].severity", source_excerpt="mild", confidence=0.95),
            FieldEvidence(field_path="findings[0].anatomy", source_excerpt="C6-7", confidence=0.98),
        ])
        d = pm.to_dict()
        assert "findings[0].severity" in d
        assert d["findings[0].severity"]["source_excerpt"] == "mild"
        assert d["findings[0].severity"]["confidence"] == 0.95
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_provenance_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# mosaicx/provenance/__init__.py
"""Provenance tracking for MOSAICX extractions."""

# mosaicx/provenance/models.py
"""Data models for field-level provenance tracking."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceSpan(BaseModel):
    """Location of evidence in a source document."""

    page: int | None = None
    line_start: int | None = None
    char_start: int
    char_end: int


class FieldEvidence(BaseModel):
    """Evidence linking an extracted field to its source text."""

    field_path: str = Field(description="Dotted path e.g. 'findings[0].severity'")
    source_excerpt: str = Field(description="Exact text from source document")
    source_location: SourceSpan | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ProvenanceMap(BaseModel):
    """Collection of field evidence for an entire extraction."""

    fields: list[FieldEvidence] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the _provenance dict format keyed by field_path."""
        result = {}
        for ev in self.fields:
            entry: dict[str, Any] = {
                "source_excerpt": ev.source_excerpt,
                "confidence": ev.confidence,
            }
            if ev.source_location is not None:
                entry["source_location"] = ev.source_location.model_dump()
            result[ev.field_path] = entry
        return result
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_provenance_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mosaicx/provenance/__init__.py mosaicx/provenance/models.py tests/test_provenance_models.py
git commit -m "feat: add provenance data models (SourceSpan, FieldEvidence, ProvenanceMap)"
```

---

### Task 8: Verification Data Models

**Files:**
- Create: `mosaicx/verify/__init__.py`
- Create: `mosaicx/verify/models.py`
- Test: `tests/test_verify_models.py`

**Step 1: Write the failing test**

```python
# tests/test_verify_models.py
from __future__ import annotations

import pytest


class TestVerificationReport:
    def test_construction(self):
        from mosaicx.verify.models import VerificationReport

        report = VerificationReport(
            verdict="verified",
            confidence=0.95,
            level="deterministic",
        )
        assert report.verdict == "verified"
        assert report.confidence == 0.95

    def test_to_dict(self):
        from mosaicx.verify.models import VerificationReport, Issue

        report = VerificationReport(
            verdict="partially_supported",
            confidence=0.6,
            level="spot_check",
            issues=[Issue(type="value_mismatch", field="findings[0].measurement", detail="Claimed 22mm, source says 14mm", severity="critical")],
        )
        d = report.to_dict()
        assert d["verdict"] == "partially_supported"
        assert len(d["issues"]) == 1
        assert d["issues"][0]["severity"] == "critical"


class TestIssue:
    def test_severity_values(self):
        from mosaicx.verify.models import Issue

        for sev in ("info", "warning", "critical"):
            issue = Issue(type="test", field="x", detail="y", severity=sev)
            assert issue.severity == sev


class TestFieldVerdict:
    def test_construction(self):
        from mosaicx.verify.models import FieldVerdict

        fv = FieldVerdict(
            status="mismatch",
            claimed_value="22mm",
            source_value="14mm",
            evidence_excerpt="nodule now measures 14mm",
            severity="critical",
        )
        assert fv.status == "mismatch"
```

**Step 2-5:** Same TDD pattern. Create `mosaicx/verify/__init__.py` and `mosaicx/verify/models.py` with dataclasses matching the design doc.

**Commit message:** `feat: add verification data models (VerificationReport, Issue, FieldVerdict, Evidence)`

---

### Task 9: Deterministic Verification (Level 1)

**Files:**
- Create: `mosaicx/verify/deterministic.py`
- Test: `tests/test_verify_deterministic.py`

**Step 1: Write the failing test**

```python
# tests/test_verify_deterministic.py
from __future__ import annotations

import pytest


class TestDeterministicVerify:
    def test_measurement_found_in_source(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {"findings": [{"measurement": {"value": 12.0, "unit": "mm"}}]}
        source = "Right external iliac node enlarged (short axis 12 mm)."
        report = verify_deterministic(extraction, source)
        assert report.verdict == "verified"
        assert len(report.issues) == 0

    def test_measurement_not_in_source(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {"findings": [{"measurement": {"value": 22.0, "unit": "mm"}}]}
        source = "Right external iliac node enlarged (short axis 12 mm)."
        report = verify_deterministic(extraction, source)
        assert report.verdict != "verified"
        assert any(i.type == "value_not_found" for i in report.issues)

    def test_empty_extraction_passes(self):
        from mosaicx.verify.deterministic import verify_deterministic

        report = verify_deterministic({}, "Some source text.")
        assert report.verdict == "verified"

    def test_invalid_finding_ref(self):
        from mosaicx.verify.deterministic import verify_deterministic

        extraction = {
            "findings": [{"anatomy": "RUL"}],
            "impressions": [{"finding_refs": [5]}],  # index 5 doesn't exist
        }
        source = "Some source."
        report = verify_deterministic(extraction, source)
        assert any(i.type == "invalid_reference" for i in report.issues)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_verify_deterministic.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# mosaicx/verify/deterministic.py
"""Level 1: Deterministic verification (no LLM, < 1 second)."""

from __future__ import annotations

import re
from typing import Any

from .models import VerificationReport, Issue


def verify_deterministic(
    extraction: dict[str, Any],
    source_text: str,
) -> VerificationReport:
    """Verify an extraction against source text using deterministic checks.

    Checks:
    - Measurements: are claimed values found in source text?
    - Finding refs: do impression finding_refs point to valid indices?
    - Enum consistency: do severity/modality values make sense?
    """
    issues: list[Issue] = []
    findings = extraction.get("findings", [])

    # Check measurements
    for i, finding in enumerate(findings):
        measurement = finding.get("measurement")
        if measurement and isinstance(measurement, dict):
            value = measurement.get("value")
            if value is not None:
                # Search for the number in source text
                value_str = str(int(value)) if float(value) == int(value) else str(value)
                if value_str not in source_text:
                    issues.append(Issue(
                        type="value_not_found",
                        field=f"findings[{i}].measurement",
                        detail=f"Claimed value '{value_str}' not found in source text",
                        severity="warning",
                    ))

    # Check finding_refs validity
    impressions = extraction.get("impressions", [])
    for j, imp in enumerate(impressions):
        refs = imp.get("finding_refs", [])
        if isinstance(refs, list):
            for ref_idx in refs:
                if isinstance(ref_idx, int) and ref_idx >= len(findings):
                    issues.append(Issue(
                        type="invalid_reference",
                        field=f"impressions[{j}].finding_refs",
                        detail=f"References finding[{ref_idx}] but only {len(findings)} findings exist",
                        severity="warning",
                    ))

    verdict = "verified" if not issues else "partially_supported"
    confidence = max(0.0, 1.0 - len(issues) * 0.15)

    return VerificationReport(
        verdict=verdict,
        confidence=confidence,
        level="deterministic",
        issues=issues,
    )
```

**Step 4: Run test**

Run: `.venv/bin/python -m pytest tests/test_verify_deterministic.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mosaicx/verify/deterministic.py tests/test_verify_deterministic.py
git commit -m "feat: add level 1 deterministic verification"
```

---

### Task 10: LLM Spot-Check Verification (Level 2)

**Files:**
- Create: `mosaicx/verify/spot_check.py`
- Test: `tests/test_verify_spot_check.py`

This follows the lazy-loading DSPy pattern. Define a `VerifyClaim` Signature and a `SpotChecker` Module. Test only the Signature structure (no LLM calls in tests).

**Step 1: Write the failing test**

```python
# tests/test_verify_spot_check.py
from __future__ import annotations

import pytest


class TestSpotCheckSignature:
    def test_verify_claim_has_expected_fields(self):
        from mosaicx.verify.spot_check import VerifyClaim

        assert "source_text" in VerifyClaim.input_fields
        assert "claims" in VerifyClaim.input_fields
        assert "verdicts" in VerifyClaim.output_fields


class TestHighRiskFieldSelector:
    def test_selects_measurements(self):
        from mosaicx.verify.spot_check import select_high_risk_fields

        extraction = {
            "exam_type": "CT Chest",
            "findings": [
                {"anatomy": "RUL", "measurement": {"value": 12, "unit": "mm"}},
                {"anatomy": "LLL", "measurement": None},
            ],
        }
        fields = select_high_risk_fields(extraction)
        assert any("measurement" in f for f in fields)

    def test_selects_severity_fields(self):
        from mosaicx.verify.spot_check import select_high_risk_fields

        extraction = {
            "findings": [{"severity": "severe", "anatomy": "C6-7"}],
        }
        fields = select_high_risk_fields(extraction)
        assert any("severity" in f for f in fields)
```

**Step 2-5:** Implement with lazy-loaded DSPy Signature and a `select_high_risk_fields()` helper. Follow the MOSAICX lazy-loading pattern.

**Commit message:** `feat: add level 2 LLM spot-check verification`

---

### Task 11: Verification Engine Orchestrator

**Files:**
- Create: `mosaicx/verify/engine.py`
- Test: `tests/test_verify_engine.py`

The orchestrator routes to the correct verification level and provides the unified `verify()` function.

**Step 1: Write the failing test**

```python
# tests/test_verify_engine.py
from __future__ import annotations

import pytest


class TestVerifyEngine:
    def test_quick_level_uses_deterministic(self):
        from mosaicx.verify.engine import verify

        extraction = {"findings": []}
        source = "Normal exam."
        report = verify(extraction=extraction, source_text=source, level="quick")
        assert report.level == "deterministic"

    def test_invalid_level_raises(self):
        from mosaicx.verify.engine import verify

        with pytest.raises(ValueError, match="Unknown verification level"):
            verify(extraction={}, source_text="", level="invalid")

    def test_claim_based_verify(self):
        from mosaicx.verify.engine import verify

        report = verify(claim="EF was 45%", source_text="EF estimated at 45%.", level="quick")
        assert report.verdict == "verified"
```

**Step 2-5:** Implement `verify()` function that dispatches to `verify_deterministic`, `verify_spot_check`, or `verify_audit` based on level.

**Commit message:** `feat: add verification engine orchestrator`

---

### Task 12: Wire Verify into SDK and CLI

**Files:**
- Modify: `mosaicx/sdk.py` (add `verify()` function)
- Modify: `mosaicx/cli.py` (add `verify` command, add `--verify` to extract)
- Test: `tests/test_sdk_verify.py`
- Test: `tests/test_cli_verify.py`

**Step 1: Write SDK test**

```python
# tests/test_sdk_verify.py
from __future__ import annotations

import pytest


class TestSDKVerify:
    def test_verify_function_exists(self):
        from mosaicx.sdk import verify

        assert callable(verify)

    def test_verify_quick_no_llm(self):
        """Quick verification should work without DSPy configured."""
        from mosaicx.sdk import verify

        result = verify(
            extraction={"findings": []},
            source_text="Normal report.",
            level="quick",
        )
        assert "verdict" in result
```

**Step 2: Write CLI test**

```python
# tests/test_cli_verify.py
from __future__ import annotations

import pytest
from click.testing import CliRunner


class TestVerifyCommand:
    def test_verify_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "--help"])
        assert result.exit_code == 0
        assert "claim" in result.output.lower() or "source" in result.output.lower()
```

**Step 3-5:** Implement `mosaicx.sdk.verify()` and `mosaicx verify` CLI command.

**Commit message:** `feat: wire verify into SDK and CLI`

---

## Phase 3: Query Engine

---

### Task 13: Query Source Loaders

**Files:**
- Create: `mosaicx/query/__init__.py`
- Create: `mosaicx/query/loaders.py`
- Test: `tests/test_query_loaders.py`

**Step 1: Write the failing test**

```python
# tests/test_query_loaders.py
from __future__ import annotations

import json
import pytest
from pathlib import Path


class TestJSONLoader:
    def test_load_json_file(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.json"
        f.write_text(json.dumps({"key": "value"}))
        meta, data = load_source(f)
        assert meta.source_type == "json"
        assert data["key"] == "value"


class TestCSVLoader:
    def test_load_csv_file(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")
        meta, data = load_source(f)
        assert meta.source_type == "dataframe"
        assert len(data) == 2  # 2 rows


class TestParquetLoader:
    def test_load_parquet_file(self, tmp_path: Path):
        import pandas as pd
        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.parquet"
        pd.DataFrame({"col": [1, 2, 3]}).to_parquet(f)
        meta, data = load_source(f)
        assert meta.source_type == "dataframe"
        assert len(data) == 3


class TestTextLoader:
    def test_load_txt_file(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "report.txt"
        f.write_text("Patient presents with cough.")
        meta, data = load_source(f)
        assert meta.source_type == "document"
        assert "cough" in data


class TestSourceMeta:
    def test_meta_has_required_fields(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        meta, _ = load_source(f)
        assert meta.name == "data.csv"
        assert meta.format == "csv"
        assert meta.size > 0
        assert meta.preview is not None
```

**Step 2-5:** Implement `load_source()` dispatcher and per-format loaders.

**Commit message:** `feat: add query source loaders (JSON, CSV, parquet, text, Excel)`

---

### Task 14: Query Session Model

**Files:**
- Create: `mosaicx/query/session.py`
- Test: `tests/test_query_session.py`

**Step 1: Write the failing test**

```python
# tests/test_query_session.py
from __future__ import annotations

import json
import pytest
from pathlib import Path


class TestQuerySession:
    def test_session_loads_sources(self, tmp_path: Path):
        from mosaicx.query.session import QuerySession

        f = tmp_path / "data.json"
        f.write_text(json.dumps({"key": "value"}))
        session = QuerySession(sources=[f])
        assert len(session.catalog) == 1
        assert session.catalog[0].name == "data.json"

    def test_session_catalog_metadata(self, tmp_path: Path):
        from mosaicx.query.session import QuerySession

        f1 = tmp_path / "report.txt"
        f1.write_text("Patient has cough.")
        f2 = tmp_path / "data.csv"
        f2.write_text("a,b\n1,2\n")
        session = QuerySession(sources=[f1, f2])
        assert len(session.catalog) == 2
        types = {m.source_type for m in session.catalog}
        assert "document" in types
        assert "dataframe" in types

    def test_session_conversation_starts_empty(self, tmp_path: Path):
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("text")
        session = QuerySession(sources=[f])
        assert session.conversation == []

    def test_session_close(self, tmp_path: Path):
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("text")
        session = QuerySession(sources=[f])
        session.close()
        assert session.closed
```

**Step 2-5:** Implement `QuerySession` class that loads sources, builds catalog, manages conversation history.

**Commit message:** `feat: add query session model with source loading and catalog`

---

### Task 15: Query RLM Tools

**Files:**
- Create: `mosaicx/query/tools.py`
- Test: `tests/test_query_tools.py`

**Step 1: Write the failing test**

```python
# tests/test_query_tools.py
from __future__ import annotations

import json
import pytest
from pathlib import Path


class TestSearchDocuments:
    def test_keyword_search(self):
        from mosaicx.query.tools import search_documents

        docs = {"report1.txt": "Patient has 5mm nodule in RUL.", "report2.txt": "Normal chest X-ray."}
        results = search_documents("nodule", documents=docs, top_k=5)
        assert len(results) >= 1
        assert "report1.txt" in results[0]["source"]


class TestGetDocument:
    def test_get_by_name(self):
        from mosaicx.query.tools import get_document

        docs = {"report1.txt": "Patient has cough."}
        text = get_document("report1.txt", documents=docs)
        assert "cough" in text

    def test_get_missing_raises(self):
        from mosaicx.query.tools import get_document

        with pytest.raises(KeyError):
            get_document("nonexistent.txt", documents={})


class TestSaveArtifact:
    def test_save_csv(self, tmp_path: Path):
        from mosaicx.query.tools import save_artifact

        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        path = save_artifact(data, tmp_path / "output.csv", format="csv")
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Alice" in content
```

**Step 2-5:** Implement the tool functions.

**Commit message:** `feat: add MOSAICX-specific RLM tools for query engine`

---

### Task 16: Query Engine (RLM Integration)

**Files:**
- Create: `mosaicx/query/engine.py`
- Test: `tests/test_query_engine.py`

This is the core RLM integration. The engine configures `dspy.RLM` with MOSAICX tools and handles the ask/response cycle.

**Step 1: Write the failing test**

```python
# tests/test_query_engine.py
from __future__ import annotations

import pytest


class TestQueryEngineStructure:
    def test_engine_has_ask_method(self):
        """QueryEngine must expose an ask() method."""
        from mosaicx.query.engine import QueryEngine

        assert hasattr(QueryEngine, "ask")

    def test_engine_requires_catalog(self):
        from mosaicx.query.engine import QueryEngine

        with pytest.raises(TypeError):
            QueryEngine()  # must provide catalog/sources
```

**Note:** Full integration testing of RLM requires a running LLM and Deno. Unit tests verify structure and configuration only. Integration tests should be marked `@pytest.mark.integration`.

**Step 2-5:** Implement `QueryEngine` that wraps `dspy.RLM`.

**Commit message:** `feat: add RLM-powered query engine`

---

### Task 17: Wire Query into SDK and CLI

**Files:**
- Modify: `mosaicx/sdk.py` (add `query()` function)
- Modify: `mosaicx/cli.py` (add `query` command)
- Test: `tests/test_sdk_query.py`
- Test: `tests/test_cli_query.py`

**Step 1: Write SDK test**

```python
# tests/test_sdk_query.py
from __future__ import annotations

import pytest


class TestSDKQuery:
    def test_query_function_exists(self):
        from mosaicx.sdk import query

        assert callable(query)

    def test_query_returns_session(self, tmp_path):
        from mosaicx.sdk import query
        from mosaicx.query.session import QuerySession

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        session = query(sources=[f])
        assert isinstance(session, QuerySession)
        session.close()
```

**Step 2: Write CLI test**

```python
# tests/test_cli_query.py
from __future__ import annotations

import pytest
from click.testing import CliRunner


class TestQueryCommand:
    def test_query_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])
        assert result.exit_code == 0
```

**Step 3-5:** Wire into SDK and CLI.

**Commit message:** `feat: wire query into SDK and CLI`

---

### Task 18: Wire Query into MCP Server

**Files:**
- Modify: `mosaicx/mcp_server.py` (add `query_start`, `query_ask`, `query_close` tools)
- Test: `tests/test_mcp_query.py`

**Step 1-5:** Add three MCP tools for session-based query. Follow existing MCP tool patterns.

**Commit message:** `feat: add query tools to MCP server`

---

## Phase 4: Polish

---

### Task 19: Wire Verify into MCP Server

**Files:**
- Modify: `mosaicx/mcp_server.py` (add `verify_output` tool)
- Test: `tests/test_mcp_verify.py`

**Commit message:** `feat: add verify_output tool to MCP server`

---

### Task 20: Add `query` Optional Dependency Group

**Files:**
- Modify: `pyproject.toml`

Add:

```toml
[project.optional-dependencies]
query = ["openpyxl>=3.1.0", "matplotlib>=3.7.0"]
```

Update `all`:

```toml
all = ["mosaicx[hf]", "mosaicx[mcp]", "mosaicx[query]"]
```

Note: `pandas` and `pyarrow` are already in core dependencies.

**Commit message:** `chore: add query optional dependency group`

---

### Task 21: Update pyproject.toml Markers

**Files:**
- Modify: `pyproject.toml`

Add marker for RLM-dependent tests:

```toml
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "contract: Contract tests",
    "slow: Slow running tests",
    "rlm: Tests requiring Deno + RLM sandbox",
]
```

**Commit message:** `chore: add rlm test marker to pytest config`

---

### Task 22: Final Integration Test

**Files:**
- Create: `tests/integration/test_full_pipeline.py`

```python
# tests/integration/test_full_pipeline.py
from __future__ import annotations

import json
import pytest
from pathlib import Path


@pytest.mark.integration
class TestExtractVerifyPipeline:
    """Test the extract -> verify pipeline end-to-end (deterministic only)."""

    def test_extract_then_verify_quick(self, tmp_path: Path):
        """Extract from text, then verify the extraction deterministically."""
        from mosaicx.verify.engine import verify

        # Simulate extraction output (no LLM needed)
        extraction = {
            "findings": [
                {"anatomy": "RUL", "measurement": {"value": 5, "unit": "mm"}, "description": "5mm nodule"},
            ],
            "impressions": [{"statement": "Pulmonary nodule", "finding_refs": [0]}],
        }
        source_text = "Findings: 5mm ground glass nodule in the right upper lobe. Impression: Pulmonary nodule."

        report = verify(extraction=extraction, source_text=source_text, level="quick")
        assert report.verdict == "verified"
        assert report.level == "deterministic"
        assert len(report.issues) == 0

    def test_extract_with_mosaicx_envelope(self):
        """Every output should have the _mosaicx envelope."""
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology", template="chest_ct")
        output = {"exam_type": "CT Chest", "_mosaicx": env}
        assert output["_mosaicx"]["pipeline"] == "radiology"
        assert output["_mosaicx"]["version"] is not None
```

**Commit message:** `test: add integration test for extract + verify pipeline`

---

## Summary

| Phase | Tasks | New files | Modified files |
|---|---|---|---|
| 1: Envelope + Self-healing | 1-6 | `envelope.py`, `pipelines/rewards.py`, 4 test files | `sdk.py`, `config.py`, `radiology.py`, `pathology.py`, `optimize.py` |
| 2: Provenance + Verify | 7-12 | `provenance/` (3 files), `verify/` (5 files), 6 test files | `sdk.py`, `cli.py` |
| 3: Query | 13-18 | `query/` (5 files), 5 test files | `sdk.py`, `cli.py`, `mcp_server.py` |
| 4: Polish | 19-22 | 2 test files | `mcp_server.py`, `pyproject.toml` |

**Total:** 22 tasks, ~20 new files, ~8 modified files

Each task is independently testable and committable. Run `make test` after each commit to ensure nothing breaks.
