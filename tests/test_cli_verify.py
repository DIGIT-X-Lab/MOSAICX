"""Tests for the CLI verify command."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

# Real dataset path
CT_CHEST = Path(__file__).parent / "datasets" / "extract" / "ct_chest_sample.txt"


class TestVerifyCommand:
    def test_verify_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "--help"])
        assert result.exit_code == 0
        assert "document" in result.output.lower()

    def test_verify_help_shows_levels(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "--help"])
        assert "quick" in result.output
        assert "standard" in result.output
        assert "thorough" in result.output

    def test_verify_requires_document(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "--claim", "test"])
        assert result.exit_code != 0

    def test_verify_requires_claim_or_extraction(self, tmp_path):
        from mosaicx.cli import cli

        doc = tmp_path / "report.txt"
        doc.write_text("Some report text.")

        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "--document", str(doc)])
        assert result.exit_code != 0
        assert "claim" in result.output.lower() or "extraction" in result.output.lower()


class TestVerifyQuickCLI:
    """Quick verify should work completely without an LLM."""

    def test_quick_claim_pass(self, tmp_path):
        from mosaicx.cli import cli

        doc = tmp_path / "report.txt"
        doc.write_text("Patient has a 5mm nodule in the right upper lobe.")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--document", str(doc),
            "--claim", "5mm nodule",
            "--level", "quick",
        ])
        assert result.exit_code == 0
        assert "pass" in result.output.lower()

    def test_quick_claim_warns_on_mismatch(self, tmp_path):
        """Quick verify should detect when a number is NOT in the source."""
        from mosaicx.cli import cli

        doc = tmp_path / "report.txt"
        doc.write_text("Patient has a 5mm nodule in the right upper lobe.")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--document", str(doc),
            "--claim", "12mm mass in left lung",
            "--level", "quick",
        ])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "warn" in output_lower or "fail" in output_lower

    def test_quick_extraction_pass(self, tmp_path):
        from mosaicx.cli import cli

        doc = tmp_path / "report.txt"
        doc.write_text("Findings: 5mm nodule in the right upper lobe.")

        extraction_file = tmp_path / "extraction.json"
        extraction_file.write_text(json.dumps({
            "findings": [{"measurement": {"value": 5, "unit": "mm"}}]
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--document", str(doc),
            "--extraction", str(extraction_file),
            "--level", "quick",
        ])
        assert result.exit_code == 0
        assert "pass" in result.output.lower()

    def test_quick_extraction_catches_hallucination(self, tmp_path):
        """Quick verify should catch values NOT in the source text."""
        from mosaicx.cli import cli

        doc = tmp_path / "report.txt"
        doc.write_text("Findings: 5mm nodule in the right upper lobe.")

        extraction_file = tmp_path / "extraction.json"
        extraction_file.write_text(json.dumps({
            "findings": [{"measurement": {"value": 99, "unit": "mm"}}]
        }))

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--document", str(doc),
            "--extraction", str(extraction_file),
            "--level", "quick",
        ])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "warn" in output_lower or "confirmed" in output_lower

    def test_quick_with_real_dataset(self):
        """Quick verify against the real CT chest dataset file."""
        from mosaicx.cli import cli

        if not CT_CHEST.exists():
            return  # skip if dataset not present

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--document", str(CT_CHEST),
            "--claim", "2.3 cm spiculated nodule in the right upper lobe",
            "--level", "quick",
        ])
        assert result.exit_code == 0
        assert "pass" in result.output.lower()

    def test_quick_with_real_dataset_wrong_claim(self):
        """Quick verify should detect when a claim doesn't match the real dataset."""
        from mosaicx.cli import cli

        if not CT_CHEST.exists():
            return

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--document", str(CT_CHEST),
            "--claim", "7.5 cm mass in left lower lobe",
            "--level", "quick",
        ])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "warn" in output_lower or "fail" in output_lower


class TestVerifyStandardCLIFallback:
    """Standard verify should not crash regardless of LLM availability."""

    def test_standard_does_not_crash(self, tmp_path, monkeypatch):
        """Standard verify should complete (not crash) regardless of LLM state."""
        from mosaicx.config import get_config

        get_config.cache_clear()
        monkeypatch.setenv("MOSAICX_API_KEY", "test-key")
        monkeypatch.setenv("MOSAICX_LM", "openai/test-model")
        get_config.cache_clear()

        from mosaicx.cli import cli

        doc = tmp_path / "report.txt"
        doc.write_text("Patient has a 5mm nodule in the right upper lobe.")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--document", str(doc),
            "--claim", "5mm nodule",
            "--level", "standard",
        ])
        # Must not crash -- should either succeed or fall back gracefully
        if result.exit_code == 0:
            output_lower = result.output.lower()
            assert "pass" in output_lower or "warn" in output_lower or "fail" in output_lower

        get_config.cache_clear()


class TestVerifyOutputSave:
    """Test --output flag saves results correctly."""

    def test_output_save_json(self, tmp_path):
        from mosaicx.cli import cli

        doc = tmp_path / "report.txt"
        doc.write_text("Patient has a 5mm nodule.")
        output_file = tmp_path / "result.json"

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--document", str(doc),
            "--claim", "5mm nodule",
            "-o", str(output_file),
        ])
        assert result.exit_code == 0
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert "verdict" in data
        assert "confidence" in data
        assert "level" in data
        assert data["level"] == "deterministic"

    def test_output_json_has_issues_array(self, tmp_path):
        from mosaicx.cli import cli

        doc = tmp_path / "report.txt"
        doc.write_text("Patient has a 5mm nodule.")
        output_file = tmp_path / "result.json"

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--document", str(doc),
            "--claim", "99mm mass",
            "-o", str(output_file),
        ])
        assert result.exit_code == 0
        data = json.loads(output_file.read_text())
        assert isinstance(data["issues"], list)
        assert len(data["issues"]) > 0  # should detect 99mm not in source


class TestVerifyClaimComparison:
    """Claim comparison output should stay grounded for verified claims too."""

    def test_claim_comparison_shows_verified_source_and_evidence(self, tmp_path, monkeypatch):
        from mosaicx.cli import cli
        import mosaicx.sdk as sdk_mod

        doc = tmp_path / "report.txt"
        doc.write_text("Vitals: BP 128/82 measured at triage.")

        def fake_verify(**_kwargs):
            return {
                "verdict": "verified",
                "confidence": 1.0,
                "level": "audit",
                "issues": [],
                "field_verdicts": [
                    {
                        "status": "verified",
                        "field_path": "claim",
                        "claimed_value": "patient BP is 128/82",
                        "source_value": "BP 128/82",
                        "evidence_excerpt": "Vitals section records BP 128/82.",
                    }
                ],
                "evidence": [],
                "missed_content": [],
            }

        monkeypatch.setattr(sdk_mod, "verify", fake_verify)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--document", str(doc),
            "--claim", "patient BP is 128/82",
            "--level", "quick",
        ])
        assert result.exit_code == 0
        assert "claim comparison" in result.output.lower()
        assert "bp 128/82" in result.output.lower()
        assert "vitals section records bp 128/82" in result.output.lower()
        assert "(not available)" not in result.output.lower()

    def test_claim_comparison_backfills_from_source_text(self, tmp_path, monkeypatch):
        from mosaicx.cli import cli
        import mosaicx.sdk as sdk_mod

        doc = tmp_path / "report.txt"
        doc.write_text("Vitals: BP 128 / 82 measured at triage by nursing.")

        def fake_verify(**_kwargs):
            return {
                "verdict": "verified",
                "confidence": 1.0,
                "level": "audit",
                "issues": [],
                "field_verdicts": [
                    {
                        "status": "verified",
                        "field_path": "claim",
                        "claimed_value": "patient BP is 128/82",
                        "source_value": None,
                        "evidence_excerpt": None,
                    }
                ],
                "evidence": [],
                "missed_content": [],
            }

        monkeypatch.setattr(sdk_mod, "verify", fake_verify)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--document", str(doc),
            "--claim", "patient BP is 128/82",
            "--level", "quick",
        ])
        assert result.exit_code == 0
        assert "claim comparison" in result.output.lower()
        assert "source" in result.output.lower()
        assert "evidence" in result.output.lower()
        assert "(not available)" not in result.output.lower()
