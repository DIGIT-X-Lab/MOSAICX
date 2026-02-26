# tests/test_extraction_pipeline.py
"""Tests for the extraction pipeline."""
from types import SimpleNamespace

import pytest
from pydantic import BaseModel


class TestInferSchemaSignature:
    def test_infer_schema_signature_fields(self):
        from mosaicx.pipelines.extraction import InferSchemaFromDocument
        assert "document_text" in InferSchemaFromDocument.input_fields
        assert "schema_spec" in InferSchemaFromDocument.output_fields


class TestDocumentExtractorModule:
    def test_auto_mode_has_infer_and_extract(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        extractor = DocumentExtractor()
        assert hasattr(extractor, "infer_schema")

    def test_schema_mode_has_extract_custom(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        from pydantic import BaseModel

        class CustomReport(BaseModel):
            summary: str
            category: str

        extractor = DocumentExtractor(output_schema=CustomReport)
        assert hasattr(extractor, "extract_custom")
        assert not hasattr(extractor, "infer_schema")

    def test_schema_mode_coerces_model_instance_output(self, monkeypatch):
        from enum import Enum

        from pydantic import BaseModel

        from mosaicx.pipelines.extraction import DocumentExtractor

        class OptionalSeverity(str, Enum):
            Mild = "Mild"
            None_ = "None"

        class Finding(BaseModel):
            level: str | None = None
            severity: OptionalSeverity | None = None

        class CustomReport(BaseModel):
            finding: Finding | None = None

        extractor = DocumentExtractor(output_schema=CustomReport)

        extracted_instance = CustomReport(
            finding=Finding(level="C2‑3", severity=None)
        )

        class _DummyPrediction:
            extracted = extracted_instance

        monkeypatch.setattr(
            extractor,
            "extract_custom",
            lambda document_text: _DummyPrediction(),
        )

        result = extractor.forward("synthetic report text")
        assert result.extracted.finding is not None
        assert result.extracted.finding.level == "C2-C3"
        assert result.extracted.finding.severity == OptionalSeverity.None_


class TestConvenienceFunctions:
    def test_extract_with_schema_function_exists(self):
        from mosaicx.pipelines.extraction import extract_with_schema
        assert callable(extract_with_schema)

    def test_extract_with_mode_function_exists(self):
        from mosaicx.pipelines.extraction import extract_with_mode
        assert callable(extract_with_mode)


class TestExtractionPlannerRouting:
    def test_plan_routes_fallback_assigns_easy_and_hard_sections(self, monkeypatch):
        from mosaicx.pipelines.extraction import (
            _plan_section_routes_with_react,
            _split_document_sections,
        )

        # Force deterministic fallback path so route selection is stable in tests.
        monkeypatch.setattr(
            "mosaicx.runtime_env.import_dspy",
            lambda: (_ for _ in ()).throw(RuntimeError("no dspy")),
        )

        doc = (
            "Clinical Information:\n"
            "Neck pain.\n\n"
            "Findings:\n"
            "C3-4 bulge with 12 mm lesion and 16 mm follow-up measurement. "
            "Multilevel narrowing. " * 8
        )
        sections = _split_document_sections(doc)
        routes, meta = _plan_section_routes_with_react(
            schema_name="MRICervicalSpineV3",
            sections=sections,
        )

        assert meta["planner"] == "deterministic_fallback"
        by_section = {r["section"].lower(): r for r in routes}
        assert by_section["clinical information"]["strategy"] in {
            "deterministic",
            "constrained_extract",
        }
        assert by_section["findings"]["strategy"] in {"heavy_extract", "repair"}

    def test_plan_extraction_document_text_returns_diagnostics(self, monkeypatch):
        from mosaicx.pipelines.extraction import _plan_extraction_document_text

        monkeypatch.setattr(
            "mosaicx.runtime_env.import_dspy",
            lambda: (_ for _ in ()).throw(RuntimeError("no dspy")),
        )

        doc = (
            "Clinical Information:\n"
            "Neck pain.\n\n"
            "Findings:\n"
            "Bulge at C3-4 with foraminal narrowing.\n\n"
            "Impression:\n"
            "Mild progression."
        )
        planned_text, diag = _plan_extraction_document_text(
            document_text=doc,
            schema_name="MRICervicalSpineV3",
        )

        assert isinstance(planned_text, str) and planned_text.strip()
        assert isinstance(diag.get("routes"), list) and diag["routes"]
        assert "strategy_counts" in diag
        assert diag.get("planned_chars", 0) > 0
        assert diag.get("original_chars", 0) >= diag.get("planned_chars", 0)

    def test_schema_mode_uses_planned_text_and_exposes_planner(self, monkeypatch):
        from mosaicx.pipelines.extraction import DocumentExtractor

        class CustomReport(BaseModel):
            summary: str

        extractor = DocumentExtractor(output_schema=CustomReport)

        planned = "Findings:\nRouted concise context."
        planner_diag = {
            "planner": "react",
            "react_used": True,
            "routes": [{"section": "Findings", "strategy": "constrained_extract"}],
            "strategy_counts": {
                "deterministic": 0,
                "constrained_extract": 1,
                "heavy_extract": 0,
                "repair": 0,
            },
            "original_chars": 120,
            "planned_chars": 40,
            "compression_ratio": 0.33,
            "sections": [{"name": "findings", "title": "Findings", "chars": 120}],
        }

        monkeypatch.setattr(
            "mosaicx.pipelines.extraction._plan_extraction_document_text",
            lambda *, document_text, schema_name: (planned, planner_diag),
        )

        captured: dict[str, str] = {}

        class _DummyPrediction:
            extracted = CustomReport(summary="stable")

        def _fake_extract_custom(*, document_text: str):
            captured["document_text"] = document_text
            return _DummyPrediction()

        monkeypatch.setattr(extractor, "extract_custom", _fake_extract_custom)

        result = extractor.forward("Original long source text that should be routed.")
        assert captured["document_text"] == planned
        assert getattr(result, "planner") == planner_diag
        assert extractor._last_planner == planner_diag


class TestStructuredExtractionChain:
    def test_chain_prefers_outlines_primary_when_available(self, monkeypatch):
        from mosaicx.pipelines.extraction import _extract_schema_with_structured_chain

        class Report(BaseModel):
            summary: str

        called = {"typed": 0}

        def _typed_extract(*, document_text: str):
            called["typed"] += 1
            return SimpleNamespace(extracted={"summary": "typed"})

        monkeypatch.setattr(
            "mosaicx.pipelines.extraction._recover_schema_instance_with_outlines",
            lambda **kwargs: Report(summary="outlines"),
        )
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
                return False

        fake_dspy = SimpleNamespace(
            settings=SimpleNamespace(lm=object()),
            context=lambda adapter=None: _Ctx(),
        )
        monkeypatch.setattr("mosaicx.runtime_env.import_dspy", lambda: fake_dspy)

        model, diag = _extract_schema_with_structured_chain(
            document_text="sample",
            schema_class=Report,
            typed_extract=_typed_extract,
            json_extract=None,
        )

        assert model.summary == "outlines"
        assert diag["selected_path"] == "outlines_primary"
        assert diag["fallback_used"] is False
        assert called["typed"] == 0
        assert diag["attempts"][0]["step"] == "outlines_primary"
        assert diag["attempts"][0]["ok"] is True

    def test_chain_order_json_adapter_then_two_step(self, monkeypatch):
        from mosaicx.pipelines.extraction import _extract_schema_with_structured_chain

        class Report(BaseModel):
            summary: str

        monkeypatch.setattr(
            "mosaicx.pipelines.extraction._recover_schema_instance_with_outlines",
            lambda **kwargs: None,
        )

        active_adapter = {"name": ""}

        class JSONAdapter:
            pass

        class TwoStepAdapter:
            pass

        class _Context:
            def __init__(self, adapter):
                self.adapter = adapter

            def __enter__(self):
                active_adapter["name"] = type(self.adapter).__name__
                return None

            def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
                active_adapter["name"] = ""
                return False

        fake_dspy = SimpleNamespace(
            settings=SimpleNamespace(lm=object()),
            JSONAdapter=JSONAdapter,
            TwoStepAdapter=TwoStepAdapter,
            context=lambda adapter=None: _Context(adapter),
        )
        monkeypatch.setattr(
            "mosaicx.runtime_env.import_dspy",
            lambda: fake_dspy,
        )

        def _typed_extract(*, document_text: str):  # noqa: ARG001
            if active_adapter["name"] == "JSONAdapter":
                raise ValueError("json adapter failed")
            if active_adapter["name"] == "TwoStepAdapter":
                return SimpleNamespace(extracted={"summary": "two-step"})
            raise AssertionError("unexpected adapter path")

        model, diag = _extract_schema_with_structured_chain(
            document_text="sample",
            schema_class=Report,
            typed_extract=_typed_extract,
            json_extract=lambda **kwargs: SimpleNamespace(extracted_json='{"summary":"fallback"}'),
        )

        assert model.summary == "two-step"
        assert diag["selected_path"] == "dspy_two_step_adapter"
        steps = [a["step"] for a in diag["attempts"]]
        assert steps[:4] == [
            "outlines_primary",
            "dspy_typed_direct",
            "dspy_json_adapter",
            "dspy_two_step_adapter",
        ]
        assert diag["attempts"][1]["ok"] is False
        assert diag["attempts"][2]["ok"] is False
        assert diag["attempts"][3]["ok"] is True

    def test_chain_uses_existing_rescue_on_malformed_json(self, monkeypatch):
        from mosaicx.pipelines.extraction import _extract_schema_with_structured_chain

        class Report(BaseModel):
            summary: str

        outlines_calls = {"count": 0}

        def _fake_outlines(**kwargs):  # noqa: ARG001
            outlines_calls["count"] += 1
            if outlines_calls["count"] == 1:
                return None
            return Report(summary="rescued")

        monkeypatch.setattr(
            "mosaicx.pipelines.extraction._recover_schema_instance_with_outlines",
            _fake_outlines,
        )
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
                return False

        fake_dspy = SimpleNamespace(
            settings=SimpleNamespace(lm=object()),
            context=lambda adapter=None: _Ctx(),
            JSONAdapter=None,
            TwoStepAdapter=None,
        )
        monkeypatch.setattr("mosaicx.runtime_env.import_dspy", lambda: fake_dspy)

        model, diag = _extract_schema_with_structured_chain(
            document_text="sample",
            schema_class=Report,
            typed_extract=lambda **kwargs: (_ for _ in ()).throw(ValueError("typed failed")),
            json_extract=lambda **kwargs: SimpleNamespace(extracted_json="not json at all"),
        )

        assert model.summary == "rescued"
        assert diag["selected_path"] == "existing_outlines_rescue"
        assert any(a["step"] == "existing_json_fallback" and a["ok"] is False for a in diag["attempts"])


class TestBestOfNSelectiveRouting:
    def test_score_components_reward_grounded_complete_candidate(self):
        from mosaicx.pipelines.extraction import _score_extraction_candidate

        class Report(BaseModel):
            finding: str
            level: str

        score, components = _score_extraction_candidate(
            extracted={"finding": "disc bulge", "level": "C3-C4"},
            schema_class=Report,
            source_text="Findings: disc bulge at C3-C4 level.",
        )

        assert score > 0.7
        assert components["schema_compliance"] == 1.0
        assert components["critical_completeness"] == 1.0
        assert components["evidence_overlap"] > 0.5

    def test_score_components_penalize_null_overuse(self):
        from mosaicx.pipelines.extraction import _score_extraction_candidate

        class Report(BaseModel):
            finding: str | None = None
            level: str | None = None
            impression: str | None = None
            note: str | None = None

        score, components = _score_extraction_candidate(
            extracted={"finding": None, "level": None, "impression": None, "note": "stable"},
            schema_class=Report,
            source_text="Impression: stable.",
        )

        assert components["null_overuse_penalty"] > 0.0
        assert score < 0.8

    def test_bestofn_only_triggers_for_uncertain_routes(self, monkeypatch):
        from mosaicx.pipelines.extraction import _try_bestofn_for_uncertain_sections

        class Report(BaseModel):
            summary: str

        fake_dspy = SimpleNamespace(settings=SimpleNamespace(lm=object()))
        monkeypatch.setattr("mosaicx.runtime_env.import_dspy", lambda: fake_dspy)

        model, info = _try_bestofn_for_uncertain_sections(
            document_text="sample",
            schema_class=Report,
            typed_extract=lambda **kwargs: SimpleNamespace(extracted={"summary": "ok"}),
            planner_diag={"routes": [{"section": "Findings", "strategy": "deterministic"}]},
        )

        assert model is None
        assert info["triggered"] is False
        assert info["reason"] == "no_uncertain_sections"

    def test_bestofn_triggers_and_returns_candidate_on_uncertain_routes(self, monkeypatch):
        from mosaicx.pipelines.extraction import _try_bestofn_for_uncertain_sections

        class Report(BaseModel):
            summary: str

        class _FakeBestOfN:
            def __init__(self, module, N, reward_fn, threshold):  # noqa: N803
                self.module = module
                self.reward_fn = reward_fn

            def __call__(self, **kwargs):
                pred = self.module(**kwargs)
                # Ensure reward function is exercised deterministically.
                _ = self.reward_fn(kwargs, pred)
                return pred

        fake_dspy = SimpleNamespace(
            settings=SimpleNamespace(lm=object()),
            BestOfN=_FakeBestOfN,
        )
        monkeypatch.setattr("mosaicx.runtime_env.import_dspy", lambda: fake_dspy)

        model, info = _try_bestofn_for_uncertain_sections(
            document_text="Findings: disc bulge at C3-C4.",
            schema_class=Report,
            typed_extract=lambda **kwargs: SimpleNamespace(extracted={"summary": "disc bulge"}),
            planner_diag={"routes": [{"section": "Findings", "strategy": "heavy_extract"}]},
        )

        assert model is not None
        assert model.summary == "disc bulge"
        assert info["triggered"] is True
        assert info["used"] is True
        assert info["reason"] == "bestofn_selected"


class TestAdjudicationAndRepair:
    def test_conflicting_candidates_are_adjudicated_with_mcc(self, monkeypatch):
        from mosaicx.pipelines.extraction import _adjudicate_conflicting_candidates

        class Report(BaseModel):
            summary: str

        class _FakeMCC:
            def __init__(self, *args, **kwargs):  # noqa: ARG002
                pass

            def __call__(self, completions, source_text):  # noqa: ARG002
                # Prefer the second candidate deterministically.
                return SimpleNamespace(chosen_path=completions[1]["chosen_path"], rationale="mcc_selected")

        fake_dspy = SimpleNamespace(
            settings=SimpleNamespace(lm=object()),
            MultiChainComparison=_FakeMCC,
        )
        monkeypatch.setattr("mosaicx.runtime_env.import_dspy", lambda: fake_dspy)

        candidates = [
            {
                "path": "candidate_a",
                "model": Report(summary="mild bulge"),
                "score": 0.62,
                "components": {"evidence_overlap": 0.4, "null_overuse_penalty": 0.0},
            },
            {
                "path": "candidate_b",
                "model": Report(summary="disc bulge at C3-C4"),
                "score": 0.59,
                "components": {"evidence_overlap": 0.8, "null_overuse_penalty": 0.0},
            },
        ]

        chosen, diag = _adjudicate_conflicting_candidates(
            candidates=candidates,
            source_text="Findings: disc bulge at C3-C4.",
        )

        assert chosen is not None
        assert chosen["path"] == "candidate_b"
        assert diag["conflict_detected"] is True
        assert diag["method"] == "mcc"
        assert diag["chosen_path"] == "candidate_b"
        assert "mcc" in diag["rationale"].lower()

    def test_refine_repair_only_touches_failed_fields(self, monkeypatch):
        from mosaicx.pipelines.extraction import _repair_failed_critical_fields_with_refine

        class Report(BaseModel):
            finding: str | None = None
            impression: str | None = None

        # Enable use_refine so the repair logic actually runs
        from mosaicx.config import MosaicxConfig

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(use_refine=True),
        )

        class _FakeDspy:
            settings = SimpleNamespace(lm=object())

            class Signature:
                pass

            @staticmethod
            def InputField(**kwargs):  # noqa: ARG004
                return None

            @staticmethod
            def OutputField(**kwargs):  # noqa: ARG004
                return None

            class _RepairModule:
                def __call__(self, **kwargs):
                    field_name = str(kwargs.get("field_name") or "")
                    if field_name == "finding":
                        return SimpleNamespace(repaired_value="disc bulge")
                    return SimpleNamespace(repaired_value="")

            @staticmethod
            def ChainOfThought(_sig):  # noqa: ARG004
                return _FakeDspy._RepairModule()

            class _Refine:
                def __init__(self, module, N, reward_fn, threshold):  # noqa: N803, ARG002
                    self.module = module

                def __call__(self, **kwargs):
                    return self.module(**kwargs)

            Refine = _Refine

        monkeypatch.setattr("mosaicx.runtime_env.import_dspy", lambda: _FakeDspy())

        base = Report(finding=None, impression="stable")
        repaired, diag = _repair_failed_critical_fields_with_refine(
            model_instance=base,
            schema_class=Report,
            source_text="Findings: disc bulge. Impression: stable.",
        )

        assert repaired.finding == "disc bulge"
        assert repaired.impression == "stable"
        assert diag["triggered"] is True
        assert any(item["field"] == "finding" for item in diag["repaired_fields"])
        assert all(item["field"] != "impression" for item in diag["repaired_fields"])

    def test_refine_repair_skipped_when_use_refine_disabled(self, monkeypatch):
        """When use_refine=False (default), repair should return early without LLM calls."""
        from mosaicx.pipelines.extraction import _repair_failed_critical_fields_with_refine

        class Report(BaseModel):
            finding: str
            impression: str

        # Ensure use_refine is False (the default)
        from mosaicx.config import MosaicxConfig

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(use_refine=False),
        )

        base = Report(finding="", impression="stable")
        repaired, diag = _repair_failed_critical_fields_with_refine(
            model_instance=base,
            schema_class=Report,
            source_text="No grounded finding text is available in this document.",
        )

        # Should return the original instance unmodified
        assert repaired is base
        assert repaired.finding == ""
        assert diag["reason"] == "use_refine_disabled"
        assert diag["repaired_fields"] == []

    def test_deterministic_backfill_runs_even_when_use_refine_disabled(self, monkeypatch):
        """Missing fields should be repaired from labeled text blocks before LLM refine."""
        from mosaicx.pipelines.extraction import _repair_failed_critical_fields_with_refine

        class Report(BaseModel):
            clinical_information: str
            comparison: str

        from mosaicx.config import MosaicxConfig

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(use_refine=False),
        )

        base = Report(clinical_information="", comparison="")
        repaired, diag = _repair_failed_critical_fields_with_refine(
            model_instance=base,
            schema_class=Report,
            source_text=(
                "Clinical Information: Evaluation of stable acute chest pain.\n"
                "Comparison: No prior imaging is available for comparison."
            ),
        )

        assert repaired.clinical_information == "Evaluation of stable acute chest pain."
        assert repaired.comparison == "No prior imaging is available for comparison."
        assert diag["reason"] == "deterministic_backfill_applied"
        assert len(diag["remaining_failed_fields"]) == 0
        assert {entry["field"] for entry in diag["repaired_fields"]} == {
            "clinical_information",
            "comparison",
        }

    def test_refine_repair_proceeds_when_use_refine_enabled(self, monkeypatch):
        """When use_refine=True, repair should proceed past the gate."""
        from mosaicx.pipelines.extraction import _repair_failed_critical_fields_with_refine

        class Report(BaseModel):
            finding: str | None = None

        from mosaicx.config import MosaicxConfig

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(use_refine=True),
        )

        # Mock DSPy import to fail — this tests the gate was passed
        monkeypatch.setattr(
            "mosaicx.runtime_env.import_dspy",
            lambda: (_ for _ in ()).throw(RuntimeError("no dspy")),
        )

        base = Report(finding=None)
        repaired, diag = _repair_failed_critical_fields_with_refine(
            model_instance=base,
            schema_class=Report,
            source_text="Findings: disc bulge.",
        )

        # Should have passed the use_refine gate and hit dspy import failure
        assert diag["reason"] == "dspy_import_failed:RuntimeError"


class TestPlannerShortDocBypass:
    def test_short_doc_bypasses_react_planner(self, monkeypatch):
        """Documents under planner_min_chars should skip ReAct entirely."""
        from mosaicx.pipelines.extraction import _plan_extraction_document_text

        from mosaicx.config import MosaicxConfig

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(planner_min_chars=4000),
        )

        # Short document (well under 4000 chars)
        doc = (
            "Clinical Information:\n"
            "Chest pain.\n\n"
            "Findings:\n"
            "Normal coronary arteries.\n\n"
            "Impression:\n"
            "No significant stenosis."
        )
        assert len(doc) < 4000

        planned_text, diag = _plan_extraction_document_text(
            document_text=doc,
            schema_name="CTAHeartV2",
        )

        assert diag["planner"] == "short_doc_bypass"
        assert diag["react_used"] is False
        assert isinstance(diag["routes"], list)
        assert all(r["reason"] == "short_doc_bypass" for r in diag["routes"])
        assert planned_text.strip()

    def test_long_doc_uses_react_planner(self, monkeypatch):
        """Documents over planner_min_chars should use the ReAct path."""
        from mosaicx.pipelines.extraction import _plan_extraction_document_text

        from mosaicx.config import MosaicxConfig

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(planner_min_chars=100),
        )

        # Force deterministic fallback so we don't need actual DSPy
        monkeypatch.setattr(
            "mosaicx.runtime_env.import_dspy",
            lambda: (_ for _ in ()).throw(RuntimeError("no dspy")),
        )

        # Long document (over 100 chars threshold)
        doc = "Findings:\n" + "Multilevel disc disease. " * 20
        assert len(doc) > 100

        planned_text, diag = _plan_extraction_document_text(
            document_text=doc,
            schema_name="SpineV1",
        )

        # Should have attempted ReAct (fallen back to deterministic due to no dspy)
        assert diag["planner"] == "deterministic_fallback"
        assert diag["react_used"] is False


class TestOutlinesTimeout:
    def test_outlines_timeout_prevents_hang(self, monkeypatch):
        """Outlines generator should be killed after timeout seconds."""
        import time

        from mosaicx.pipelines.extraction import _recover_schema_instance_with_outlines

        class Report(BaseModel):
            summary: str

        # Mock outlines + openai to simulate a hanging generator
        class _FakeModel:
            pass

        class _FakeGenerator:
            def __call__(self, prompt, temperature=0.0, max_tokens=2200):
                time.sleep(10)  # Simulate hang
                return '{"summary": "should not reach"}'

        fake_outlines = SimpleNamespace(
            from_openai=lambda client, model_name: _FakeModel(),
            Generator=lambda model, schema: _FakeGenerator(),
            json_schema=lambda cls: "fake_schema",
        )
        fake_openai = SimpleNamespace(
            OpenAI=lambda base_url, api_key: None,
        )

        monkeypatch.setitem(__import__("sys").modules, "outlines", fake_outlines)
        monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

        from mosaicx.config import MosaicxConfig

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(outlines_timeout=1),
        )

        start = time.monotonic()
        result = _recover_schema_instance_with_outlines(
            document_text="test document",
            schema_class=Report,
        )
        elapsed = time.monotonic() - start

        assert result is None  # Should have timed out
        assert elapsed < 5  # Should not wait the full 10s
