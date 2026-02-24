"""Tests for the RLM-based audit verification module."""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Tool unit tests (no LLM needed)
# ---------------------------------------------------------------------------


class TestSearchSourceTool:
    """Test the search_source tool factory."""

    def test_finds_exact_match(self):
        from mosaicx.verify.audit import _make_search_source

        source = "Patient has a 2.3 cm spiculated nodule in the right upper lobe."
        search = _make_search_source(source)
        results = search("spiculated nodule")
        assert len(results) >= 1
        assert results[0]["match_position"] >= 0
        assert "spiculated" in results[0]["exact_match"]

    def test_returns_not_found_for_missing(self):
        from mosaicx.verify.audit import _make_search_source

        source = "Patient has a 2.3 cm spiculated nodule."
        search = _make_search_source(source)
        results = search("carcinoma")
        assert len(results) == 1
        assert results[0]["match_position"] == -1
        assert "not found" in results[0]["note"]

    def test_finds_multiple_matches(self):
        from mosaicx.verify.audit import _make_search_source

        source = "nodule in RUL. Another nodule in LLL. Third nodule noted."
        search = _make_search_source(source)
        results = search("nodule")
        assert len(results) == 3

    def test_case_insensitive(self):
        from mosaicx.verify.audit import _make_search_source

        source = "SPICULATED NODULE in the right upper lobe."
        search = _make_search_source(source)
        results = search("spiculated nodule")
        assert len(results) >= 1
        assert results[0]["match_position"] >= 0


class TestChunkedSourceTools:
    """Test chunk-indexed source helpers for long-document audits."""

    def test_chunk_source_text_covers_tail_content(self):
        from mosaicx.verify.audit import _chunk_source_text

        source = "A" * 7500 + " TARGET_AT_END"
        chunks = _chunk_source_text(source, chunk_chars=1200, overlap_chars=120)
        assert len(chunks) >= 6
        assert any("TARGET_AT_END" in c["text"] for c in chunks)
        assert chunks[-1]["end"] == len(source)

    def test_search_source_chunks_finds_end_of_long_document(self):
        from mosaicx.verify.audit import _chunk_source_text, _make_search_source_chunks

        source = "B" * 8200 + " diagnosis: prostate carcinoma"
        chunks = _chunk_source_text(source, chunk_chars=1300, overlap_chars=140)
        search_chunks = _make_search_source_chunks(chunks)
        hits = search_chunks("prostate carcinoma", top_k=3)
        assert hits
        assert hits[0]["chunk_id"] >= 0
        assert "carcinoma" in hits[0]["snippet"].lower()

    def test_search_source_chunks_handles_paraphrase_term_match(self):
        from mosaicx.verify.audit import _chunk_source_text, _make_search_source_chunks

        source = (
            "C" * 6400
            + " Impression: right external iliac lymph node increased to 16 mm short-axis."
        )
        chunks = _chunk_source_text(source, chunk_chars=1100, overlap_chars=120)
        search_chunks = _make_search_source_chunks(chunks)
        hits = search_chunks("lesion enlarged 16 mm", top_k=3)
        assert hits
        assert int(hits[0].get("score", 0)) > 0
        assert "16 mm" in hits[0]["snippet"].lower()

    def test_source_manifest_reports_chunk_count(self):
        from mosaicx.verify.audit import _build_source_manifest, _chunk_source_text

        source = "C" * 5000 + " finding near end"
        chunks = _chunk_source_text(source, chunk_chars=900, overlap_chars=100)
        manifest = _build_source_manifest(source, chunks)
        assert "chunks=" in manifest
        assert "chunk_id=" in manifest


class TestGetFieldTool:
    """Test the get_field tool factory."""

    def test_simple_field(self):
        from mosaicx.verify.audit import _make_get_field

        extraction = {"diagnosis": "carcinoma", "stage": "IIA"}
        get_field = _make_get_field(extraction)
        assert json.loads(get_field("diagnosis")) == "carcinoma"

    def test_nested_field(self):
        from mosaicx.verify.audit import _make_get_field

        extraction = {"findings": [{"measurement": {"value": 2.3, "unit": "cm"}}]}
        get_field = _make_get_field(extraction)
        assert json.loads(get_field("findings[0].measurement.value")) == 2.3

    def test_missing_field_returns_error(self):
        from mosaicx.verify.audit import _make_get_field

        extraction = {"diagnosis": "carcinoma"}
        get_field = _make_get_field(extraction)
        result = get_field("nonexistent")
        assert "Error" in result

    def test_array_index(self):
        from mosaicx.verify.audit import _make_get_field

        extraction = {"items": ["a", "b", "c"]}
        get_field = _make_get_field(extraction)
        assert json.loads(get_field("items[1]")) == "b"


class TestListFieldsTool:
    """Test the list_fields tool factory."""

    def test_flat_extraction(self):
        from mosaicx.verify.audit import _make_list_fields

        extraction = {"diagnosis": "carcinoma", "stage": "IIA"}
        list_fields = _make_list_fields(extraction)
        fields = list_fields()
        assert len(fields) == 2
        paths = {f["path"] for f in fields}
        assert "diagnosis" in paths
        assert "stage" in paths

    def test_nested_extraction(self):
        from mosaicx.verify.audit import _make_list_fields

        extraction = {
            "findings": [
                {"anatomy": "RUL", "measurement": {"value": 2.3, "unit": "cm"}}
            ]
        }
        list_fields = _make_list_fields(extraction)
        fields = list_fields()
        paths = {f["path"] for f in fields}
        assert "findings[0].anatomy" in paths
        assert "findings[0].measurement.value" in paths
        assert "findings[0].measurement.unit" in paths

    def test_skips_none_values(self):
        from mosaicx.verify.audit import _make_list_fields

        extraction = {"a": "value", "b": None, "c": "other"}
        list_fields = _make_list_fields(extraction)
        fields = list_fields()
        paths = {f["path"] for f in fields}
        assert "a" in paths
        assert "c" in paths
        assert "b" not in paths


class TestSearchNumbersTool:
    """Test the search_numbers tool factory."""

    def test_finds_measurements(self):
        from mosaicx.verify.audit import _make_search_numbers

        source = "Patient has a 2.3 cm nodule and 5 mm ground-glass opacity."
        search = _make_search_numbers(source)
        results = search()
        values = {r["value"] for r in results}
        assert "2.3" in values
        assert "5" in values

    def test_finds_units(self):
        from mosaicx.verify.audit import _make_search_numbers

        source = "Nodule measures 2.3 cm. Weight is 70 kg."
        search = _make_search_numbers(source)
        results = search()
        units = {f"{r['value']}{r['unit']}" for r in results}
        assert "2.3cm" in units
        assert "70kg" in units

    def test_deduplicates(self):
        from mosaicx.verify.audit import _make_search_numbers

        source = "5 mm nodule. Another 5 mm opacity."
        search = _make_search_numbers(source)
        results = search()
        values_with_units = [f"{r['value']}{r['unit']}" for r in results]
        assert values_with_units.count("5mm") == 1


# ---------------------------------------------------------------------------
# Audit report parsing tests
# ---------------------------------------------------------------------------


class TestParseAuditReport:
    """Test _parse_audit_report with various LLM output formats."""

    def test_well_formed_report(self):
        from mosaicx.verify.audit import _parse_audit_report

        report = json.dumps({
            "field_verdicts": [
                {
                    "field_path": "findings[0].measurement.value",
                    "status": "verified",
                    "source_value": "2.3",
                    "detail": "Found 2.3 cm in source text",
                },
                {
                    "field_path": "findings[0].severity",
                    "status": "mismatch",
                    "source_value": "moderate",
                    "detail": "Source says 'moderate' but extraction says 'severe'",
                },
            ],
            "omissions": ["Patient history of smoking not captured"],
            "summary": "One mismatch found in severity field",
        })

        issues, verdicts = _parse_audit_report(report)

        assert len(verdicts) == 2
        assert verdicts[0].status == "verified"
        assert verdicts[1].status == "mismatch"
        assert verdicts[1].severity == "critical"

        # Should have mismatch issue + omission issue
        assert any(i.type == "audit_mismatch" for i in issues)
        assert any(i.type == "omission" for i in issues)

    def test_markdown_fenced_report(self):
        from mosaicx.verify.audit import _parse_audit_report

        report = "```json\n" + json.dumps({
            "field_verdicts": [
                {"field_path": "stage", "status": "verified", "detail": "OK"},
            ],
            "omissions": [],
            "summary": "All verified",
        }) + "\n```"

        issues, verdicts = _parse_audit_report(report)
        assert len(verdicts) == 1
        assert verdicts[0].status == "verified"

    def test_python_literal_report_is_parsed(self):
        from mosaicx.verify.audit import _parse_audit_report

        report = (
            "{'field_verdicts': [{'field_path': 'claim', 'status': 'verified', "
            "'detail': 'found in source'}], 'omissions': []}"
        )
        issues, verdicts = _parse_audit_report(report)
        assert len(issues) == 0
        assert len(verdicts) == 1
        assert verdicts[0].status == "verified"

    def test_bare_list_report_is_wrapped_as_field_verdicts(self):
        from mosaicx.verify.audit import _parse_audit_report

        report = "[{'field_path': 'claim', 'status': 'mismatch', 'detail': 'not found'}]"
        issues, verdicts = _parse_audit_report(report)
        assert len(verdicts) == 1
        assert verdicts[0].status == "mismatch"
        assert any(i.type == "audit_mismatch" for i in issues)

    def test_source_value_dict_is_normalized_to_text(self):
        from mosaicx.verify.audit import _parse_audit_report

        report = json.dumps({
            "field_verdicts": [
                {
                    "field_path": "claim",
                    "status": "verified",
                    "source_value": {
                        "match_position": 33,
                        "snippet": "FINDINGS: 2.3 cm spiculated nodule in the right upper lobe.",
                        "exact_match": "2.3 cm spiculated nodule",
                    },
                    "detail": "Found exact phrase",
                }
            ],
            "omissions": [],
        })

        issues, verdicts = _parse_audit_report(report)
        assert len(issues) == 0
        assert len(verdicts) == 1
        assert verdicts[0].source_value == "2.3 cm spiculated nodule"

    def test_non_json_returns_empty(self):
        from mosaicx.verify.audit import _parse_audit_report

        issues, verdicts = _parse_audit_report("This is not JSON at all")
        assert issues == []
        assert verdicts == []

    def test_status_normalization(self):
        from mosaicx.verify.audit import _parse_audit_report

        report = json.dumps({
            "field_verdicts": [
                {"field_path": "a", "status": "correct"},
                {"field_path": "b", "status": "incorrect"},
                {"field_path": "c", "status": "not_found"},
                {"field_path": "d", "status": "something_else"},
            ],
            "omissions": [],
        })

        issues, verdicts = _parse_audit_report(report)
        assert verdicts[0].status == "verified"
        assert verdicts[1].status == "mismatch"
        assert verdicts[2].status == "unsupported"
        assert verdicts[3].status == "not_checked"

    def test_structured_evidence_metadata_is_preserved(self):
        from mosaicx.verify.audit import _parse_audit_report

        report = json.dumps({
            "field_verdicts": [
                {
                    "field_path": "claim",
                    "status": "mismatch",
                    "claimed_value": "BP 120/82",
                    "source_value": "BP 128/82",
                    "detail": "Claimed value does not match source",
                    "evidence": {
                        "excerpt": "Vitals: BP 128/82 measured at triage.",
                        "chunk_id": 11,
                        "start": 3812,
                        "end": 3860,
                        "score": 9.0,
                        "evidence_type": "text_chunk",
                        "source": "sample_patient_vitals.pdf",
                    },
                }
            ],
            "omissions": [],
        })

        issues, verdicts = _parse_audit_report(report)
        assert len(issues) == 1
        assert len(verdicts) == 1
        fv = verdicts[0]
        assert fv.evidence_source == "sample_patient_vitals.pdf"
        assert fv.evidence_type == "text_chunk"
        assert fv.evidence_chunk_id == 11
        assert fv.evidence_start == 3812
        assert fv.evidence_end == 3860
        assert fv.evidence_score == 9.0


# ---------------------------------------------------------------------------
# Outlines recovery wiring tests
# ---------------------------------------------------------------------------


class TestOutlinesRecovery:
    """Verify Outlines fallback is used when DSPy RLM serialization fails."""

    def test_claim_audit_uses_outlines_recovery_on_rlm_failure(self, monkeypatch):
        from mosaicx.verify.audit import run_claim_audit

        class _FailingRLM:
            def __init__(self, *_args, **_kwargs):
                pass

            def __call__(self, **_kwargs):
                raise RuntimeError("Adapter JSONAdapter failed to parse LM response")

        monkeypatch.setattr("dspy.RLM", _FailingRLM)
        monkeypatch.setattr(
            "mosaicx.verify.audit._recover_claim_audit_with_outlines",
            lambda **_kwargs: {
                "field_verdicts": [
                    {
                        "field_path": "claim",
                        "status": "mismatch",
                        "source_value": "128/82",
                        "detail": "Claim BP 120/82 conflicts with source BP 128/82",
                        "evidence": {
                            "excerpt": "Blood Pressure 128/82 mmHg",
                            "chunk_id": 0,
                            "start": 0,
                            "end": 24,
                            "score": 8.0,
                            "source": "source_document",
                            "evidence_type": "text_chunk",
                        },
                    }
                ],
                "summary": "Claim contradicted",
            },
        )

        issues, verdicts = run_claim_audit(
            claim="patient BP is 120/82",
            source_text="Patient vital signs: Blood Pressure 128/82 mmHg.",
        )

        assert verdicts
        assert verdicts[0].status == "mismatch"
        assert verdicts[0].source_value == "128/82"
        assert any(i.type == "audit_structured_recovery" for i in issues)

    def test_extraction_audit_uses_outlines_recovery_on_rlm_failure(self, monkeypatch):
        from mosaicx.verify.audit import run_audit

        class _FailingRLM:
            def __init__(self, *_args, **_kwargs):
                pass

            def __call__(self, **_kwargs):
                raise RuntimeError("LM response cannot be serialized to JSON")

        monkeypatch.setattr("dspy.RLM", _FailingRLM)
        monkeypatch.setattr(
            "mosaicx.verify.audit._recover_extraction_audit_with_outlines",
            lambda **_kwargs: {
                "field_verdicts": [
                    {
                        "field_path": "findings[0].measurement.value",
                        "status": "verified",
                        "source_value": "2.3",
                        "detail": "2.3 cm present in source",
                        "evidence": {
                            "excerpt": "2.3 cm spiculated nodule",
                            "chunk_id": 0,
                            "start": 10,
                            "end": 32,
                            "score": 7.0,
                            "source": "source_document",
                            "evidence_type": "text_chunk",
                        },
                    }
                ],
                "summary": "Recovered audit",
            },
        )

        issues, verdicts = run_audit(
            source_text="Findings: 2.3 cm spiculated nodule in the right upper lobe.",
            extraction={"findings": [{"measurement": {"value": 2.3, "unit": "cm"}}]},
        )

        assert verdicts
        assert verdicts[0].status == "verified"
        assert any(i.type == "audit_structured_recovery" for i in issues)

    def test_claim_audit_skips_outlines_when_lm_unconfigured(self, monkeypatch):
        from mosaicx.verify.audit import run_claim_audit

        class _FailingRLM:
            def __init__(self, *_args, **_kwargs):
                pass

            def __call__(self, **_kwargs):
                raise RuntimeError(
                    "No LM is loaded. Please configure the LM using dspy.configure(lm=dspy.LM(...))"
                )

        called = {"recovery": 0}

        def _recovery_stub(**_kwargs):
            called["recovery"] += 1
            return {"field_verdicts": []}

        monkeypatch.setattr("dspy.RLM", _FailingRLM)
        monkeypatch.setattr(
            "mosaicx.verify.audit._recover_claim_audit_with_outlines",
            _recovery_stub,
        )

        with pytest.raises(RuntimeError, match="No LM is loaded"):
            run_claim_audit(
                claim="patient BP is 120/82",
                source_text="Patient vital signs: Blood Pressure 128/82 mmHg.",
            )
        assert called["recovery"] == 0

    def test_extraction_audit_skips_outlines_when_lm_unconfigured(self, monkeypatch):
        from mosaicx.verify.audit import run_audit

        class _FailingRLM:
            def __init__(self, *_args, **_kwargs):
                pass

            def __call__(self, **_kwargs):
                raise RuntimeError(
                    "No LM is loaded. Please configure the LM using dspy.configure(lm=dspy.LM(...))"
                )

        called = {"recovery": 0}

        def _recovery_stub(**_kwargs):
            called["recovery"] += 1
            return {"field_verdicts": []}

        monkeypatch.setattr("dspy.RLM", _FailingRLM)
        monkeypatch.setattr(
            "mosaicx.verify.audit._recover_extraction_audit_with_outlines",
            _recovery_stub,
        )

        with pytest.raises(RuntimeError, match="No LM is loaded"):
            run_audit(
                source_text="Findings: 2.3 cm spiculated nodule in the right upper lobe.",
                extraction={"findings": [{"measurement": {"value": 2.3, "unit": "cm"}}]},
            )
        assert called["recovery"] == 0


# ---------------------------------------------------------------------------
# Engine integration: verify thorough uses RLM
# ---------------------------------------------------------------------------


class TestVerifyEngineAuditWiring:
    """Test that the verify engine correctly routes thorough to RLM audit."""

    def test_thorough_extraction_calls_audit(self, monkeypatch):
        """Thorough extraction verify should call run_audit, not run_spot_check."""
        from mosaicx.verify import engine
        from mosaicx.verify.models import FieldVerdict, Issue, VerificationReport

        calls = {"audit": 0, "spot_check": 0}

        def mock_run_audit(source_text, extraction):
            calls["audit"] += 1
            return (
                [],
                [FieldVerdict(status="verified", claimed_value="2.3")],
            )

        def mock_run_spot_check(source_text, extraction, paths):
            calls["spot_check"] += 1
            return (
                [],
                [FieldVerdict(status="verified", claimed_value="2.3")],
            )

        def mock_verify_deterministic(extraction, source_text):
            return VerificationReport(
                verdict="verified",
                confidence=0.9,
                level="deterministic",
                issues=[],
            )

        def mock_select_high_risk_fields(extraction):
            return ["findings[0].measurement.value"]

        monkeypatch.setattr("mosaicx.verify.engine._enhance_with_audit.__module__", engine.__name__)
        monkeypatch.setattr("mosaicx.verify.audit.run_audit", mock_run_audit)
        monkeypatch.setattr("mosaicx.verify.spot_check.run_spot_check", mock_run_spot_check)
        monkeypatch.setattr("mosaicx.verify.spot_check.select_high_risk_fields", mock_select_high_risk_fields)
        monkeypatch.setattr("mosaicx.verify.deterministic.verify_deterministic", mock_verify_deterministic)

        extraction = {"findings": [{"measurement": {"value": 2.3}}]}
        result = engine.verify(
            extraction=extraction,
            source_text="2.3 cm nodule",
            level="thorough",
        )

        assert calls["audit"] == 1
        assert result.level == "audit"

    def test_audit_module_has_no_5000_char_truncation(self):
        """Long-doc audit should not hard-truncate source context at 5000 chars."""
        import inspect

        from mosaicx.verify import audit

        source = inspect.getsource(audit.run_audit) + inspect.getsource(audit.run_claim_audit)
        assert "[:5000]" not in source

    def test_thorough_claim_calls_claim_audit(self, monkeypatch):
        """Thorough claim verify should call run_claim_audit."""
        from mosaicx.verify import engine
        from mosaicx.verify.models import FieldVerdict

        calls = {"claim_audit": 0}

        def mock_run_claim_audit(claim, source_text):
            calls["claim_audit"] += 1
            return (
                [],
                [FieldVerdict(status="verified", claimed_value=claim)],
            )

        monkeypatch.setattr("mosaicx.verify.audit.run_claim_audit", mock_run_claim_audit)

        result = engine.verify(
            claim="2.3 cm nodule",
            source_text="Patient has a 2.3 cm spiculated nodule.",
            level="thorough",
        )

        assert calls["claim_audit"] == 1
        assert result.level == "audit"

    def test_standard_claim_does_not_call_audit(self, monkeypatch):
        """Standard claim verify should NOT call run_claim_audit."""
        from mosaicx.verify import engine
        from mosaicx.verify.models import FieldVerdict, VerificationReport

        calls = {"claim_audit": 0}

        def mock_run_claim_audit(claim, source_text):
            calls["claim_audit"] += 1
            return ([], [])

        def mock_verify_claim_with_llm(claim, source_text):
            return VerificationReport(
                verdict="verified",
                confidence=0.9,
                level="spot_check",
                field_verdicts=[FieldVerdict(status="verified", claimed_value=claim)],
            )

        monkeypatch.setattr("mosaicx.verify.audit.run_claim_audit", mock_run_claim_audit)
        monkeypatch.setattr("mosaicx.verify.spot_check.verify_claim_with_llm", mock_verify_claim_with_llm)

        result = engine.verify(
            claim="2.3 cm nodule",
            source_text="Patient has a 2.3 cm spiculated nodule.",
            level="standard",
        )

        assert calls["claim_audit"] == 0
        assert result.level == "spot_check"

    def test_audit_failure_falls_back_to_standard(self, monkeypatch):
        """If RLM audit fails, should fall back to standard (spot-check)."""
        from mosaicx.verify import engine
        from mosaicx.verify.models import FieldVerdict, VerificationReport

        def mock_run_audit(source_text, extraction):
            raise RuntimeError("RLM not available")

        def mock_run_spot_check(source_text, extraction, paths):
            return (
                [],
                [FieldVerdict(status="verified", claimed_value="2.3")],
            )

        def mock_verify_deterministic(extraction, source_text):
            return VerificationReport(
                verdict="verified",
                confidence=0.9,
                level="deterministic",
                issues=[],
            )

        def mock_select_high_risk_fields(extraction):
            return ["findings[0].measurement.value"]

        monkeypatch.setattr("mosaicx.verify.audit.run_audit", mock_run_audit)
        monkeypatch.setattr("mosaicx.verify.spot_check.run_spot_check", mock_run_spot_check)
        monkeypatch.setattr("mosaicx.verify.spot_check.select_high_risk_fields", mock_select_high_risk_fields)
        monkeypatch.setattr("mosaicx.verify.deterministic.verify_deterministic", mock_verify_deterministic)

        extraction = {"findings": [{"measurement": {"value": 2.3}}]}
        result = engine.verify(
            extraction=extraction,
            source_text="2.3 cm nodule",
            level="thorough",
        )

        # Should still succeed (fall back to spot_check level)
        assert result.level in ("spot_check", "deterministic")
        assert any("RLM" in i.detail or "rlm" in i.type for i in result.issues)

    def test_claim_audit_failure_falls_back_to_standard(self, monkeypatch):
        """If RLM claim audit fails, thorough should fall back to standard."""
        from mosaicx.verify import engine
        from mosaicx.verify.models import FieldVerdict, VerificationReport

        def mock_run_claim_audit(claim, source_text):
            raise RuntimeError("RLM not available")

        def mock_verify_claim_with_llm(claim, source_text):
            return VerificationReport(
                verdict="verified",
                confidence=0.85,
                level="spot_check",
                field_verdicts=[FieldVerdict(status="verified", claimed_value=claim)],
            )

        monkeypatch.setattr("mosaicx.verify.audit.run_claim_audit", mock_run_claim_audit)
        monkeypatch.setattr("mosaicx.verify.spot_check.verify_claim_with_llm", mock_verify_claim_with_llm)

        result = engine.verify(
            claim="2.3 cm nodule",
            source_text="Patient has a 2.3 cm nodule.",
            level="thorough",
        )

        # Should fall back to spot_check
        assert result.level == "spot_check"
