# tests/test_public_api.py
"""Tests for the mosaicx top-level public API functions.

Verifies that the SDK convenience functions (extract, summarize,
generate_schema, deidentify) are importable, callable, and have the
expected signatures.  Also tests the regex-only deidentify path which
requires no LLM, and the new SDK-only functions (list_schemas,
list_modes, batch_extract, evaluate).
"""

import inspect

import pytest


class TestPublicAPI:
    """Verify the public API functions are importable and callable."""

    def test_extract_function(self):
        from mosaicx import extract

        assert callable(extract)

    def test_summarize_function(self):
        from mosaicx import summarize

        assert callable(summarize)

    def test_generate_schema_function(self):
        from mosaicx import generate_schema

        assert callable(generate_schema)

    def test_deidentify_function(self):
        from mosaicx import deidentify

        assert callable(deidentify)

    def test_list_schemas_function(self):
        from mosaicx import list_schemas

        assert callable(list_schemas)

    def test_list_modes_function(self):
        from mosaicx import list_modes

        assert callable(list_modes)

    def test_batch_extract_function(self):
        from mosaicx import batch_extract

        assert callable(batch_extract)

    def test_evaluate_function(self):
        from mosaicx import evaluate

        assert callable(evaluate)

    def test_all_exports(self):
        """All SDK functions should appear in __all__."""
        import mosaicx

        for name in (
            "extract", "summarize", "generate_schema", "deidentify",
            "list_schemas", "list_modes", "list_templates", "evaluate",
            "batch_extract", "process_file", "process_files",
        ):
            assert name in mosaicx.__all__, f"{name!r} missing from __all__"

    def test_file_based_wrappers_exist(self):
        """File-based wrappers should still be accessible."""
        import mosaicx

        assert callable(mosaicx.extract_file)
        assert callable(mosaicx.summarize_files)


class TestSDKSignatures:
    """Verify each SDK function has the documented parameters."""

    def test_extract_signature(self):
        from mosaicx import extract

        sig = inspect.signature(extract)
        params = list(sig.parameters.keys())
        assert "text" in params
        assert "document" in params
        assert "mode" in params
        assert "template" in params
        assert "optimized" in params
        assert sig.parameters["text"].default is None
        assert sig.parameters["document"].default is None
        assert sig.parameters["mode"].default == "auto"
        assert sig.parameters["template"].default is None
        assert sig.parameters["optimized"].default is None

    def test_summarize_signature(self):
        from mosaicx import summarize

        sig = inspect.signature(summarize)
        params = list(sig.parameters.keys())
        assert "reports" in params
        assert "patient_id" in params
        assert "optimized" in params
        assert sig.parameters["patient_id"].default == "unknown"

    def test_generate_schema_signature(self):
        from mosaicx import generate_schema

        sig = inspect.signature(generate_schema)
        params = list(sig.parameters.keys())
        assert "description" in params
        assert "name" in params
        assert "example_text" in params
        assert sig.parameters["name"].default is None
        assert sig.parameters["example_text"].default is None

    def test_deidentify_signature(self):
        from mosaicx import deidentify

        sig = inspect.signature(deidentify)
        params = list(sig.parameters.keys())
        assert "text" in params
        assert "mode" in params
        assert sig.parameters["mode"].default == "remove"

    def test_evaluate_signature(self):
        from mosaicx import evaluate

        sig = inspect.signature(evaluate)
        params = list(sig.parameters.keys())
        assert "pipeline" in params
        assert "testset_path" in params
        assert "optimized" in params

    def test_batch_extract_signature(self):
        from mosaicx import batch_extract

        sig = inspect.signature(batch_extract)
        params = list(sig.parameters.keys())
        assert "texts" in params
        assert "mode" in params
        assert "template" in params

    def test_list_templates_signature(self):
        from mosaicx import list_templates

        sig = inspect.signature(list_templates)
        # No required params
        for param in sig.parameters.values():
            assert param.default is not inspect.Parameter.empty or param.kind in (
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
            )


class TestFileBasedSignatures:
    """Verify file-based wrappers retain the original signatures."""

    def test_extract_file_signature(self):
        from mosaicx import extract_file

        sig = inspect.signature(extract_file)
        params = list(sig.parameters.keys())
        assert "document_path" in params
        assert "schema" in params
        assert "mode" in params
        assert "template" in params

    def test_summarize_files_signature(self):
        from mosaicx import summarize_files

        sig = inspect.signature(summarize_files)
        params = list(sig.parameters.keys())
        assert "document_paths" in params


class TestDeidentifyRegex:
    """Test the regex-only path of deidentify (no LLM needed).

    The SDK deidentify() returns a dict with 'redacted_text'.
    """

    def test_ssn_scrubbed(self):
        from mosaicx import deidentify

        result = deidentify("Patient SSN 123-45-6789", mode="regex")
        assert "123-45-6789" not in result["redacted_text"]
        assert "[REDACTED]" in result["redacted_text"]

    def test_phone_scrubbed(self):
        from mosaicx import deidentify

        result = deidentify("Call 555-123-4567 for info", mode="regex")
        assert "555-123-4567" not in result["redacted_text"]
        assert "[REDACTED]" in result["redacted_text"]

    def test_email_scrubbed(self):
        from mosaicx import deidentify

        result = deidentify("Contact: john.doe@hospital.com", mode="regex")
        assert "john.doe@hospital.com" not in result["redacted_text"]
        assert "[REDACTED]" in result["redacted_text"]

    def test_mrn_scrubbed(self):
        from mosaicx import deidentify

        result = deidentify("MRN: 12345678", mode="regex")
        assert "12345678" not in result["redacted_text"]
        assert "[REDACTED]" in result["redacted_text"]

    def test_no_phi_unchanged(self):
        from mosaicx import deidentify

        clean = "Normal chest radiograph. No acute findings."
        result = deidentify(clean, mode="regex")
        assert result["redacted_text"] == clean

    def test_regex_returns_dict(self):
        from mosaicx import deidentify

        result = deidentify("Some text 123-45-6789", mode="regex")
        assert isinstance(result, dict)
        assert "redacted_text" in result
        assert isinstance(result["redacted_text"], str)

    def test_invalid_mode_raises(self):
        from mosaicx import deidentify

        with pytest.raises(ValueError, match="Unknown deidentify mode"):
            deidentify("test", mode="invalid_mode")


class TestListSchemas:
    """Test list_schemas without needing an LLM."""

    def test_returns_list(self):
        from mosaicx import list_schemas

        result = list_schemas()
        assert isinstance(result, list)


class TestListModes:
    """Test list_modes without needing an LLM."""

    def test_returns_list_of_dicts(self):
        from mosaicx import list_modes

        result = list_modes()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)
            assert "name" in item
            assert "description" in item


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


class TestListTemplates:
    """Test sdk.list_templates() -- no LLM needed."""

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


class TestExtractDocument:
    """Test extract() with the document parameter (file path input)."""

    def test_extract_with_document_path(self, tmp_path, monkeypatch):
        """extract(document=path) should load the file and extract."""
        from mosaicx import sdk
        from mosaicx.sdk import extract

        txt = tmp_path / "report.txt"
        txt.write_text("Normal chest radiograph. No acute findings.")

        monkeypatch.setattr(sdk, "_ensure_configured", lambda: None)

        # Mock DocumentExtractor to avoid needing LLM
        class FakeExtractor:
            def __init__(self, **kw):
                pass

            def __call__(self, **kw):
                class R:
                    extracted = {"summary": kw["document_text"][:20]}
                return R()

        monkeypatch.setattr(
            "mosaicx.pipelines.extraction.DocumentExtractor", FakeExtractor
        )

        result = extract(document=txt)
        assert "extracted" in result
        assert "_document" in result
        assert result["_document"]["format"] == "txt"

    def test_extract_text_still_works(self, monkeypatch):
        """extract(text) should still work as before (no _document key)."""
        from mosaicx import sdk
        from mosaicx.sdk import extract

        monkeypatch.setattr(sdk, "_ensure_configured", lambda: None)

        class FakeExtractor:
            def __init__(self, **kw):
                pass

            def __call__(self, **kw):
                class R:
                    extracted = {"summary": "ok"}
                return R()

        monkeypatch.setattr(
            "mosaicx.pipelines.extraction.DocumentExtractor", FakeExtractor
        )

        result = extract("Some text here")
        assert "extracted" in result
        assert "_document" not in result

    def test_both_text_and_document_raises(self):
        """Providing both text and document should raise ValueError."""
        from mosaicx.sdk import extract

        with pytest.raises(ValueError, match="not both"):
            extract("some text", document="/some/path.pdf")

    def test_neither_text_nor_document_raises(self):
        """Providing neither text nor document should raise ValueError."""
        from mosaicx.sdk import extract

        with pytest.raises(ValueError, match="text or document"):
            extract()

    def test_nonexistent_document_raises(self, tmp_path):
        """Nonexistent document path should raise FileNotFoundError."""
        from mosaicx.sdk import extract

        with pytest.raises(FileNotFoundError):
            extract(document=tmp_path / "nonexistent.pdf")


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
        """Process a .txt file -- no OCR, no LLM (mock extract)."""
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
