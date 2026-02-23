"""Behavioral tests for the CLI query command."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner


class TestQueryCommandRegistration:
    def test_query_help(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])
        assert result.exit_code == 0
        assert "--document" in result.output
        assert "--question" in result.output
        assert "--output" in result.output

    def test_query_help_shows_examples(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])
        assert "csv" in result.output.lower() or "json" in result.output.lower()


class TestQueryValidation:
    def test_query_requires_document(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["query"])
        assert result.exit_code != 0
        assert "document" in result.output.lower()

    def test_query_rejects_nonexistent_document(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--document", "/nonexistent/file.txt"])
        assert result.exit_code != 0


class TestQuerySessionSetup:
    """Query should load documents and show catalog without needing LLM."""

    def test_query_loads_text_document(self, tmp_path):
        """Without --question, query should show catalog and exit (no LLM needed)."""
        from mosaicx.cli import cli

        doc = tmp_path / "report.txt"
        doc.write_text("Patient has 5mm nodule in the RUL.")

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--document", str(doc)])
        assert result.exit_code == 0
        assert "report.txt" in result.output
        assert "1 source" in result.output.lower()

    def test_query_loads_csv_document(self, tmp_path):
        from mosaicx.cli import cli

        doc = tmp_path / "patients.csv"
        doc.write_text("id,name,age\n1,Alice,30\n2,Bob,25\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--document", str(doc)])
        assert result.exit_code == 0
        assert "patients.csv" in result.output

    def test_query_loads_json_document(self, tmp_path):
        from mosaicx.cli import cli

        doc = tmp_path / "data.json"
        doc.write_text('{"key": "value"}')

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--document", str(doc)])
        assert result.exit_code == 0
        assert "data.json" in result.output

    def test_query_loads_multiple_documents(self, tmp_path):
        from mosaicx.cli import cli

        doc1 = tmp_path / "report.txt"
        doc1.write_text("Normal exam.")
        doc2 = tmp_path / "data.csv"
        doc2.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "query",
            "--document", str(doc1),
            "--document", str(doc2),
        ])
        assert result.exit_code == 0
        assert "2 source" in result.output.lower()

    def test_query_with_real_dataset(self):
        """Query should load the real CT chest sample."""
        from mosaicx.cli import cli

        ct_file = Path(__file__).parent / "datasets" / "extract" / "ct_chest_sample.txt"
        if not ct_file.exists():
            return  # skip if not present

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--document", str(ct_file)])
        assert result.exit_code == 0
        assert "ct_chest_sample.txt" in result.output


class TestQueryQuestionMode:
    def test_query_question_renders_answer_and_evidence(self, tmp_path, monkeypatch):
        from mosaicx.cli import cli
        from mosaicx.query.loaders import SourceMeta
        import mosaicx.cli as cli_mod
        import mosaicx.sdk as sdk_mod

        doc = tmp_path / "report.txt"
        doc.write_text("Patient has a 5mm nodule in the right upper lobe.")

        class _FakeSession:
            catalog = [SourceMeta(
                name="report.txt",
                format="txt",
                source_type="document",
                size=64,
                preview="Patient has a 5mm nodule.",
            )]

            def ask_structured(self, question: str, *, max_iterations: int = 20, top_k_citations: int = 3):
                return {
                    "question": question,
                    "answer": "There is a 5mm nodule.",
                    "citations": [{
                        "source": "report.txt",
                        "snippet": "Patient has a 5mm nodule in the right upper lobe.",
                        "score": 3,
                    }],
                    "confidence": 0.9,
                    "sources_consulted": ["report.txt"],
                    "turn_index": 1,
                }

            def close(self):
                return None

        monkeypatch.setattr(cli_mod, "_check_api_key", lambda: None)
        monkeypatch.setattr(cli_mod, "_configure_dspy", lambda: None)
        monkeypatch.setattr(sdk_mod, "query", lambda sources: _FakeSession())

        runner = CliRunner()
        result = runner.invoke(cli, [
            "query",
            "--document", str(doc),
            "--question", "What is the nodule size?",
        ])
        assert result.exit_code == 0
        assert "There is a 5mm nodule." in result.output
        assert "Evidence" in result.output
        assert "report.txt" in result.output

    def test_query_question_output_saves_turns_with_citations(self, tmp_path, monkeypatch):
        from mosaicx.cli import cli
        from mosaicx.query.loaders import SourceMeta
        import mosaicx.cli as cli_mod
        import mosaicx.sdk as sdk_mod

        doc = tmp_path / "report.txt"
        doc.write_text("Patient has a 5mm nodule in the right upper lobe.")
        out = tmp_path / "answer.json"

        class _FakeSession:
            catalog = [SourceMeta(
                name="report.txt",
                format="txt",
                source_type="document",
                size=64,
                preview="Patient has a 5mm nodule.",
            )]

            def ask_structured(self, question: str, *, max_iterations: int = 20, top_k_citations: int = 3):
                return {
                    "question": question,
                    "answer": "5mm nodule",
                    "citations": [{
                        "source": "report.txt",
                        "snippet": "5mm nodule",
                        "score": 2,
                    }],
                    "confidence": 0.8,
                    "sources_consulted": ["report.txt"],
                    "turn_index": 1,
                }

            def close(self):
                return None

        monkeypatch.setattr(cli_mod, "_check_api_key", lambda: None)
        monkeypatch.setattr(cli_mod, "_configure_dspy", lambda: None)
        monkeypatch.setattr(sdk_mod, "query", lambda sources: _FakeSession())

        runner = CliRunner()
        result = runner.invoke(cli, [
            "query",
            "--document", str(doc),
            "--question", "What is the finding?",
            "--output", str(out),
        ])
        assert result.exit_code == 0
        assert out.exists()
        payload = json.loads(out.read_text())
        assert "turns" in payload
        assert isinstance(payload["turns"], list)
        assert payload["turns"][0]["citations"][0]["source"] == "report.txt"
