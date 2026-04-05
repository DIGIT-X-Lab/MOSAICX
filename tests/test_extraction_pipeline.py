# tests/test_extraction_pipeline.py
"""Tests for the extraction pipeline."""
import json
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

        extractor = DocumentExtractor(output_schema=CustomReport, think="standard")

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

        extractor = DocumentExtractor(output_schema=CustomReport, think="standard")

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
        actual_planner = getattr(result, "planner")
        for key, value in planner_diag.items():
            assert actual_planner[key] == value, f"planner[{key!r}] mismatch"
        for key, value in planner_diag.items():
            assert extractor._last_planner[key] == value


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
        assert diag["attempts"][0]["complete"] is True

    def test_chain_accepts_incomplete_outlines_for_standard_mode(self, monkeypatch):
        from mosaicx.pipelines.extraction import _extract_schema_with_structured_chain

        class Report(BaseModel):
            summary: str | None

        monkeypatch.setattr(
            "mosaicx.pipelines.extraction._recover_schema_instance_with_outlines",
            lambda **kwargs: Report(summary=None),
        )
        called = {"typed": 0}

        def _typed_extract(*, document_text: str):  # noqa: ARG001
            called["typed"] += 1
            return SimpleNamespace(extracted={"summary": "typed"})

        model, diag = _extract_schema_with_structured_chain(
            document_text="sample",
            schema_class=Report,
            typed_extract=_typed_extract,
            json_extract=None,
        )

        assert model.summary is None
        assert called["typed"] == 0
        assert diag["selected_path"] == "outlines_primary"
        steps = [a["step"] for a in diag["attempts"]]
        assert steps == ["outlines_primary"]
        assert diag["attempts"][0]["ok"] is True
        assert diag["attempts"][0]["reason"] == "accepted_incomplete"
        assert diag["attempts"][0]["valid"] is True
        assert diag["attempts"][0]["complete"] is False

    def test_chain_uses_optional_json_text_fallback(self, monkeypatch):
        from mosaicx.pipelines.extraction import _extract_schema_with_structured_chain
        from mosaicx.config import MosaicxConfig

        class Report(BaseModel):
            summary: str

        monkeypatch.setattr(
            "mosaicx.pipelines.extraction._recover_schema_instance_with_outlines",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(structured_json_fallback=True),
        )

        model, diag = _extract_schema_with_structured_chain(
            document_text="sample",
            schema_class=Report,
            typed_extract=lambda **kwargs: (_ for _ in ()).throw(ValueError("typed failed")),
            json_extract=lambda **kwargs: SimpleNamespace(extracted_json='{"summary":"fallback"}'),
        )

        assert model.summary == "fallback"
        assert diag["selected_path"] == "json_text_fallback"
        assert [a["step"] for a in diag["attempts"]] == [
            "outlines_primary",
            "dspy_typed_direct",
            "json_text_fallback",
        ]
        assert diag["attempts"][2]["ok"] is True

    def test_chain_accepts_supported_absent_required_field(self, monkeypatch):
        from mosaicx.pipelines.extraction import (
            _augment_schema_with_evidence,
            _extract_schema_with_structured_chain,
        )

        class Report(BaseModel):
            summary: str | None

        AugmentedReport = _augment_schema_with_evidence(Report)
        monkeypatch.setattr(
            "mosaicx.pipelines.extraction._recover_schema_instance_with_outlines",
            lambda **kwargs: AugmentedReport(
                summary="",
                summary_excerpt="",
                summary_reasoning="Summary not provided in the document.",
            ),
        )

        called = {"typed": 0}
        model, diag = _extract_schema_with_structured_chain(
            document_text="sample",
            schema_class=AugmentedReport,
            typed_extract=lambda **kwargs: called.__setitem__("typed", called["typed"] + 1),
            json_extract=None,
        )

        assert model.summary == ""
        assert called["typed"] == 0
        assert diag["selected_path"] == "outlines_primary"
        assert diag["attempts"][0]["ok"] is True
        assert diag["attempts"][0]["reason"] == "accepted_with_supported_absence"
        assert diag["attempts"][0]["supported_absent_required"] == ["summary"]


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
    def test_extract_labeled_blocks_parses_markdown_label_value_pairs(self):
        from mosaicx.pipelines.extraction import _extract_labeled_blocks

        blocks = _extract_labeled_blocks(
            "Requested Date\n"
            "**28-03-2026 12:27:17**\n\n"
            "Reported Date\n"
            "**28-03-2026 13:34:17**\n"
        )

        assert blocks[0]["label"] == "Requested Date"
        assert blocks[0]["text"] == "28-03-2026 12:27:17"
        assert blocks[1]["label"] == "Reported Date"
        assert blocks[1]["text"] == "28-03-2026 13:34:17"

    def test_identify_suspicious_fields_marks_exam_date_as_ambiguous(self):
        from mosaicx.pipelines.extraction import _identify_suspicious_fields

        class Report(BaseModel):
            exam_date: str | None = None
            date_of_birth: str | None = None

        model = Report(exam_date="2026-03-28", date_of_birth=None)
        suspicious = _identify_suspicious_fields(
            model_instance=model,
            schema_class=Report,
            source_text=(
                "Requested Date\n"
                "**28-03-2026 12:27:17**\n\n"
                "Collected Date\n"
                "**28-03-2026 12:49:23**\n\n"
                "Reported Date\n"
                "**28-03-2026 13:34:17**\n"
            ),
            field_evidence={
                "exam_date": {
                    "excerpt": "28-03-2026",
                    "reasoning": "Collected Date field shows the date of examination.",
                },
                "date_of_birth": {
                    "excerpt": "",
                    "reasoning": "Date of birth not provided in the document.",
                },
            },
        )

        exam_date_issue = next(item for item in suspicious if item["field"] == "exam_date")
        assert exam_date_issue["issue"] == "ambiguous_label"
        assert exam_date_issue["current_label"] == "Collected Date"
        assert exam_date_issue["preferred_label"] == "Reported Date"
        assert all(item["field"] != "date_of_birth" for item in suspicious)

    def test_targeted_field_repair_prefers_reported_date_label(self):
        from mosaicx.pipelines.extraction import _targeted_field_repair

        class Report(BaseModel):
            exam_date: str | None = None

        model = Report(exam_date="2026-03-28")
        field_evidence = {
            "exam_date": {
                "excerpt": "28-03-2026",
                "reasoning": "Collected Date field shows the date of examination.",
            }
        }
        repaired, diag = _targeted_field_repair(
            model_instance=model,
            schema_class=Report,
            suspicious_fields=[
                {
                    "field": "exam_date",
                    "current_value": "2026-03-28",
                    "issue": "ambiguous_label",
                    "current_label": "Collected Date",
                    "preferred_label": "Reported Date",
                }
            ],
            source_text=(
                "Requested Date\n"
                "**28-03-2026 12:27:17**\n\n"
                "Collected Date\n"
                "**28-03-2026 12:49:23**\n\n"
                "Reported Date\n"
                "**28-03-2026 13:34:17**\n"
            ),
            field_evidence=field_evidence,
        )

        assert repaired.exam_date == "2026-03-28"
        assert diag["repaired_count"] == 1
        assert diag["fields"][0]["method"] == "label_block"
        assert diag["fields"][0]["label"] == "Reported Date"
        assert field_evidence["exam_date"]["excerpt"] == "28-03-2026 13:34:17"
        assert "Reported Date" in field_evidence["exam_date"]["reasoning"]

    def test_select_semantic_canonicalization_candidates_uses_mismatched_excerpt(self):
        from mosaicx.pipelines.extraction import _select_semantic_canonicalization_candidates

        class Report(BaseModel):
            sex: str | None = None
            note: str | None = None

        model = Report(sex="Female", note="free text")
        candidates = _select_semantic_canonicalization_candidates(
            model_instance=model,
            schema_class=Report,
            field_evidence={
                "sex": {
                    "excerpt": "F",
                    "reasoning": "The source token F denotes female sex.",
                },
                "note": {
                    "excerpt": "free text",
                    "reasoning": "Matches exactly.",
                },
            },
        )

        assert [candidate["field"] for candidate in candidates] == ["sex"]
        assert candidates[0]["source_value"] == "F"

    def test_apply_semantic_canonicalization_batches_candidate_fields(self):
        from mosaicx.pipelines.extraction import _apply_semantic_canonicalization

        class Report(BaseModel):
            sex: str | None = None
            location: str | None = None

        class _FakeSemanticCanonicalize:
            def __call__(self, *, candidates_json: str):
                payload = json.loads(candidates_json)
                assert {item["field"] for item in payload} == {"sex", "location"}
                return SimpleNamespace(
                    canonicalized_json=json.dumps(
                        {
                            "sex": {
                                "canonical_value": "Female",
                                "confidence": 0.99,
                                "reasoning": "F is the abbreviation for female.",
                            },
                            "location": {
                                "canonical_value": "Right upper lobe",
                                "confidence": 0.95,
                                "reasoning": "RUL expands to Right upper lobe.",
                            },
                        }
                    )
                )

        field_evidence = {
            "sex": {
                "excerpt": "F",
                "reasoning": "The source token F denotes female sex.",
            },
            "location": {
                "excerpt": "RUL",
                "reasoning": "Abbreviated location in the report.",
            },
        }
        model = Report(sex="Female", location="Right upper lobe")
        updated, diag = _apply_semantic_canonicalization(
            model_instance=model,
            schema_class=Report,
            field_evidence=field_evidence,
            semantic_canonicalize=_FakeSemanticCanonicalize(),
        )

        assert updated.sex == "Female"
        assert updated.location == "Right upper lobe"
        assert diag["classified_count"] == 2
        assert field_evidence["sex"]["canonicalization"]["method"] == "semantic_classifier"
        assert field_evidence["location"]["canonicalization"]["from"] == "RUL"
        assert field_evidence["location"]["canonicalization"]["to"] == "Right upper lobe"

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

    def test_repair_skips_fields_marked_absent_in_source(self):
        from mosaicx.pipelines.extraction import _repair_failed_critical_fields_with_refine

        class Report(BaseModel):
            date_of_birth: str

        base = Report(date_of_birth="")
        repaired, diag = _repair_failed_critical_fields_with_refine(
            model_instance=base,
            schema_class=Report,
            source_text="Patient demographics available, DOB not listed.",
            field_evidence={
                "date_of_birth": {
                    "excerpt": "",
                    "reasoning": "Date of birth not provided in the document.",
                }
            },
        )

        assert repaired is base
        assert diag["triggered"] is False
        assert diag["reason"] == "supported_absence_only"
        assert diag["supported_absent_fields"] == ["date_of_birth"]


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

    def test_long_doc_uses_full_text_default(self, monkeypatch):
        """Documents over planner_min_chars should preserve full text deterministically."""
        from mosaicx.pipelines.extraction import _plan_extraction_document_text

        from mosaicx.config import MosaicxConfig

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(planner_min_chars=100),
        )

        # Long document (over 100 chars threshold)
        doc = "Findings:\n" + "Multilevel disc disease. " * 20
        assert len(doc) > 100

        planned_text, diag = _plan_extraction_document_text(
            document_text=doc,
            schema_name="SpineV1",
        )

        assert diag["planner"] == "full_text_default"
        assert diag["react_used"] is False
        assert diag["fallback_reason"] == "react_removed"
        assert all(r["strategy"] == "heavy_extract" for r in diag["routes"])
        assert all(r["reason"] == "full_text_default" for r in diag["routes"])
        assert diag["planned_chars"] >= diag["original_chars"]


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


class TestOutlinesJsonObjectFallback:
    """Outlines recovery falls back to json_object when endpoint rejects json_schema."""

    def test_falls_back_to_json_object_on_400(self, monkeypatch):
        """When Outlines raises a 400 about json_schema, fall back to json_object."""
        from mosaicx.pipelines.extraction import _recover_schema_instance_with_outlines

        class Report(BaseModel):
            summary: str

        # Outlines generator raises a 400 mentioning json_schema
        class _FakeGenerator:
            def __call__(self, prompt, temperature=0.0, max_tokens=2200):
                raise Exception(
                    "Error code: 400 - Input should be 'text' or 'json_object', "
                    "input was 'json_schema'"
                )

        fake_outlines = SimpleNamespace(
            from_openai=lambda client, model_name: SimpleNamespace(),
            Generator=lambda model, schema: _FakeGenerator(),
            json_schema=lambda cls: "fake_schema",
        )

        # Direct OpenAI client returns valid JSON via json_object fallback
        class _FakeChoice:
            def __init__(self):
                self.message = SimpleNamespace(content='{"summary": "rescued via json_object"}')

        class _FakeResponse:
            def __init__(self):
                self.choices = [_FakeChoice()]

        class _FakeCompletions:
            def create(self, **kwargs):
                # Verify json_object mode is used
                assert kwargs.get("response_format") == {"type": "json_object"}
                return _FakeResponse()

        class _FakeClient:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=_FakeCompletions())

        fake_openai = SimpleNamespace(OpenAI=_FakeClient)

        monkeypatch.setitem(__import__("sys").modules, "outlines", fake_outlines)
        monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)

        from mosaicx.config import MosaicxConfig

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(outlines_timeout=5),
        )

        result = _recover_schema_instance_with_outlines(
            document_text="test document with findings",
            schema_class=Report,
        )

        assert result is not None
        assert result.summary == "rescued via json_object"


class TestThinkParameter:
    """DocumentExtractor respects the think parameter."""

    def test_default_think_is_standard(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        from pydantic import BaseModel

        class SimpleReport(BaseModel):
            summary: str

        extractor = DocumentExtractor(output_schema=SimpleReport)
        assert extractor._think == "fast"

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


class TestThinkFastMode:
    """think=fast skips ChainOfThought, uses Predict fallback only."""

    def test_fast_mode_uses_predict_not_cot(self, monkeypatch):
        """Fast mode: Outlines fails -> dspy.Predict -> coerce. No ChainOfThought."""
        from pydantic import BaseModel
        from mosaicx.pipelines.extraction import DocumentExtractor

        class SimpleReport(BaseModel):
            summary: str
            category: str

        extractor = DocumentExtractor(output_schema=SimpleReport, think="fast")

        # Track whether extract_custom (ChainOfThought) is called
        cot_called = False

        def _tracking_cot(**kwargs):
            nonlocal cot_called
            cot_called = True
            raise RuntimeError("Should not be called in fast mode")

        # Mock Outlines to fail (so we test the Predict fallback path)
        monkeypatch.setattr(
            "mosaicx.pipelines.extraction._recover_schema_instance_with_outlines",
            lambda **kwargs: None,
        )

        # Mock the JSON fallback (Predict) to return valid data
        class _FallbackPred:
            extracted_json = '{"summary": "test summary", "category": "radiology"}'

        monkeypatch.setattr(
            extractor,
            "extract_json_fallback",
            lambda **kwargs: _FallbackPred(),
        )

        # Patch extract_custom to track calls
        monkeypatch.setattr(extractor, "extract_custom", _tracking_cot)

        result = extractor.forward("test document text")
        assert result.extracted.summary == "test summary"
        assert result.extracted.category == "radiology"
        # ChainOfThought should NOT have been called in fast mode
        assert not cot_called, "Fast mode should not use ChainOfThought"


class TestThinkDeepMode:
    """think=deep uses chunked extraction via _deep_extract_chunked."""

    def test_deep_mode_uses_chunked_extraction(self, monkeypatch):
        from pydantic import BaseModel
        from mosaicx.pipelines.extraction import DocumentExtractor

        class SimpleReport(BaseModel):
            summary: str
            category: str

        extractor = DocumentExtractor(output_schema=SimpleReport, think="deep")

        chunked_called = False

        def _mock_deep_chunked(self, document_text, schema, metrics, tracker):
            nonlocal chunked_called
            chunked_called = True
            return SimpleReport(summary="chest pain", category="radiology"), {"chunks": []}

        monkeypatch.setattr(
            type(extractor),
            "_deep_extract_chunked",
            _mock_deep_chunked,
        )

        source_text = "Patient presents with chest pain. Radiology report follows."
        result = extractor.forward(source_text)

        assert chunked_called, "Deep mode should use chunked extraction"
        assert result.extracted.summary == "chest pain"
        assert result.extracted.category == "radiology"

    def test_deep_mode_result_accessible(self, monkeypatch):
        from pydantic import BaseModel
        from mosaicx.pipelines.extraction import DocumentExtractor

        class SimpleReport(BaseModel):
            summary: str
            category: str

        extractor = DocumentExtractor(output_schema=SimpleReport, think="deep")

        def _mock_deep_chunked(self, document_text, schema, metrics, tracker):
            return SimpleReport(summary="headache", category="neurology"), {"chunks": []}

        monkeypatch.setattr(
            type(extractor),
            "_deep_extract_chunked",
            _mock_deep_chunked,
        )

        result = extractor.forward("Patient reports headache. Neurology consult requested.")
        assert result.extracted.summary == "headache"

    def test_deep_mode_works_with_chunked_extraction(self, monkeypatch):
        from pydantic import BaseModel
        from mosaicx.pipelines.extraction import DocumentExtractor

        class SimpleReport(BaseModel):
            summary: str

        extractor = DocumentExtractor(output_schema=SimpleReport, think="deep")

        def _mock_deep_chunked(self, document_text, schema, metrics, tracker):
            return SimpleReport(summary="findings"), {"chunks": []}

        monkeypatch.setattr(
            type(extractor),
            "_deep_extract_chunked",
            _mock_deep_chunked,
        )

        result = extractor.forward("findings noted")
        assert result.extracted.summary == "findings"


class TestThinkRefineControl:
    """think level controls whether dspy.Refine is used for repair."""

    def test_repair_function_accepts_think_parameter(self):
        """_repair_failed_critical_fields_with_refine should accept think param."""
        import inspect
        from mosaicx.pipelines.extraction import _repair_failed_critical_fields_with_refine
        sig = inspect.signature(_repair_failed_critical_fields_with_refine)
        assert "think" in sig.parameters
        assert sig.parameters["think"].default == "standard"

    def test_deep_mode_enables_refine_regardless_of_config(self, monkeypatch):
        """think=deep should enable Refine even when cfg.use_refine is False."""
        from pydantic import BaseModel
        from mosaicx.pipelines.extraction import _repair_failed_critical_fields_with_refine

        class SimpleReport(BaseModel):
            summary: str

        # Start with a missing field to trigger repair path
        instance = SimpleReport(summary="")

        # Mock config to have use_refine=False
        class _MockConfig:
            use_refine = False
            refine_only_missing = True
            refine_max_fields = 3

        monkeypatch.setattr("mosaicx.config.get_config", lambda: _MockConfig())

        # With think=standard, should respect config (disabled)
        _, diag_std = _repair_failed_critical_fields_with_refine(
            model_instance=instance,
            schema_class=SimpleReport,
            source_text="some text",
            think="standard",
        )
        assert diag_std.get("reason") == "use_refine_disabled"

        # With think=deep, should bypass config and attempt Refine
        # (will fail because dspy.Refine is not available in test env, but
        #  it should NOT be blocked by use_refine_disabled)
        _, diag_deep = _repair_failed_critical_fields_with_refine(
            model_instance=instance,
            schema_class=SimpleReport,
            source_text="some text",
            think="deep",
        )
        assert diag_deep.get("reason") != "use_refine_disabled"

    def test_fast_mode_disables_refine_regardless_of_config(self, monkeypatch):
        """think=fast should disable Refine even when cfg.use_refine is True."""
        from pydantic import BaseModel
        from mosaicx.pipelines.extraction import _repair_failed_critical_fields_with_refine

        class SimpleReport(BaseModel):
            summary: str

        instance = SimpleReport(summary="")

        class _MockConfig:
            use_refine = True
            refine_only_missing = True
            refine_max_fields = 3

        monkeypatch.setattr("mosaicx.config.get_config", lambda: _MockConfig())

        _, diag_fast = _repair_failed_critical_fields_with_refine(
            model_instance=instance,
            schema_class=SimpleReport,
            source_text="some text",
            think="fast",
        )
        assert diag_fast.get("reason") == "use_refine_disabled"


class TestRunReportThink:
    def test_run_report_accepts_think_parameter(self):
        """run_report should accept and forward think parameter."""
        import inspect
        from mosaicx.report import run_report
        sig = inspect.signature(run_report)
        assert "think" in sig.parameters
        assert sig.parameters["think"].default == "auto"
