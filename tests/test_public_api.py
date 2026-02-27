# tests/test_public_api.py
"""Tests for the mosaicx top-level public API functions.

Verifies that the SDK convenience functions (extract, summarize,
generate_schema, deidentify) are importable, callable, and have the
expected signatures.  Also tests the utility functions (list_schemas,
list_modes, evaluate).
"""

import inspect

import pytest


@pytest.mark.unit
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

    def test_evaluate_function(self):
        from mosaicx import evaluate

        assert callable(evaluate)

    def test_all_exports(self):
        """All SDK functions should appear in __all__."""
        import mosaicx

        for name in (
            "extract", "summarize", "generate_schema", "deidentify",
            "list_schemas", "list_modes", "list_templates", "evaluate",
            "health",
        ):
            assert name in mosaicx.__all__, f"{name!r} missing from __all__"


@pytest.mark.unit
class TestSDKSignatures:
    """Verify each SDK function has the documented parameters."""

    def test_extract_signature(self):
        from mosaicx import extract

        sig = inspect.signature(extract)
        params = list(sig.parameters.keys())
        assert "text" in params
        assert "documents" in params
        assert "filename" in params
        assert "mode" in params
        assert "template" in params
        assert "score" in params
        assert "optimized" in params
        assert "workers" in params
        assert "on_progress" in params
        assert sig.parameters["text"].default is None
        assert sig.parameters["documents"].default is None
        assert sig.parameters["filename"].default is None
        assert sig.parameters["mode"].default == "auto"
        assert sig.parameters["template"].default is None
        assert sig.parameters["score"].default is False
        assert sig.parameters["optimized"].default is None
        assert sig.parameters["workers"].default == 1
        assert sig.parameters["on_progress"].default is None

    def test_summarize_signature(self):
        from mosaicx import summarize

        sig = inspect.signature(summarize)
        params = list(sig.parameters.keys())
        assert "reports" in params
        assert "documents" in params
        assert "patient_id" in params
        assert "optimized" in params
        assert sig.parameters["reports"].default is None
        assert sig.parameters["documents"].default is None
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
        assert "documents" in params
        assert "filename" in params
        assert "mode" in params
        assert "workers" in params
        assert "on_progress" in params
        assert sig.parameters["text"].default is None
        assert sig.parameters["documents"].default is None
        assert sig.parameters["filename"].default is None
        assert sig.parameters["mode"].default == "remove"
        assert sig.parameters["workers"].default == 1
        assert sig.parameters["on_progress"].default is None

    def test_evaluate_signature(self):
        from mosaicx import evaluate

        sig = inspect.signature(evaluate)
        params = list(sig.parameters.keys())
        assert "pipeline" in params
        assert "testset_path" in params
        assert "optimized" in params

    def test_list_templates_signature(self):
        from mosaicx import list_templates

        sig = inspect.signature(list_templates)
        # No required params
        for param in sig.parameters.values():
            assert param.default is not inspect.Parameter.empty or param.kind in (
                inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
            )


@pytest.mark.unit
class TestDeidentifyValidation:
    def test_invalid_mode_raises(self):
        from mosaicx import deidentify

        with pytest.raises(ValueError, match="Unknown deidentify mode"):
            deidentify("test", mode="invalid_mode")


@pytest.mark.unit
class TestListSchemas:
    """Test list_schemas without needing an LLM."""

    def test_returns_list(self):
        from mosaicx import list_schemas

        result = list_schemas()
        assert isinstance(result, list)


@pytest.mark.unit
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


@pytest.mark.unit
class TestHealth:
    """Test sdk.health() -- no LLM needed."""

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


@pytest.mark.unit
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


@pytest.mark.unit
class TestExtractDocuments:
    """Test extract() with documents parameter."""

    def test_text_and_documents_mutually_exclusive(self):
        """Providing both text and documents raises ValueError."""
        import mosaicx

        with pytest.raises(ValueError, match="not both"):
            mosaicx.extract(text="some text", documents="file.pdf")

    def test_neither_text_nor_documents_raises(self):
        """Providing neither text nor documents raises ValueError."""
        import mosaicx

        with pytest.raises(ValueError):
            mosaicx.extract()

    def test_documents_accepts_string_path(self):
        """documents= accepts a string file path."""
        import mosaicx

        sig = inspect.signature(mosaicx.extract)
        param = sig.parameters["documents"]
        assert param.default is None

    def test_documents_accepts_bytes_with_filename(self):
        """documents= accepts bytes when filename is provided."""
        import mosaicx

        sig = inspect.signature(mosaicx.extract)
        assert "filename" in sig.parameters

    def test_workers_parameter_exists(self):
        """workers parameter exists with default 1."""
        import mosaicx

        sig = inspect.signature(mosaicx.extract)
        assert sig.parameters["workers"].default == 1

    def test_on_progress_parameter_exists(self):
        """on_progress callback parameter exists."""
        import mosaicx

        sig = inspect.signature(mosaicx.extract)
        assert "on_progress" in sig.parameters
        assert sig.parameters["on_progress"].default is None

    def test_score_parameter_exists(self):
        """score parameter exists (replaces report())."""
        import mosaicx

        sig = inspect.signature(mosaicx.extract)
        assert sig.parameters["score"].default is False


@pytest.mark.unit
class TestDeidentifyDocuments:
    """Test deidentify() with documents parameter."""

    def test_text_and_documents_mutually_exclusive(self):
        import mosaicx

        with pytest.raises(ValueError, match="not both"):
            mosaicx.deidentify(text="some text", documents="file.pdf")

    def test_neither_text_nor_documents_raises(self):
        import mosaicx

        with pytest.raises(ValueError):
            mosaicx.deidentify()

    def test_workers_parameter_exists(self):
        import mosaicx

        sig = inspect.signature(mosaicx.deidentify)
        assert sig.parameters["workers"].default == 1

    def test_on_progress_parameter_exists(self):
        import mosaicx

        sig = inspect.signature(mosaicx.deidentify)
        assert "on_progress" in sig.parameters


@pytest.mark.unit
class TestSummarizeDocuments:
    """Test summarize() with documents parameter."""

    def test_reports_and_documents_mutually_exclusive(self):
        import mosaicx

        with pytest.raises(ValueError, match="not both"):
            mosaicx.summarize(reports=["text"], documents="file.pdf")

    def test_neither_reports_nor_documents_raises(self):
        import mosaicx

        with pytest.raises(ValueError):
            mosaicx.summarize()

    def test_documents_parameter_exists(self):
        import mosaicx

        sig = inspect.signature(mosaicx.summarize)
        assert "documents" in sig.parameters


@pytest.mark.unit
class TestRemovedFunctions:
    """Verify removed functions are no longer exported."""

    def test_batch_extract_removed(self):
        import mosaicx

        assert not hasattr(mosaicx, "batch_extract") or "batch_extract" not in mosaicx.__all__

    def test_process_file_removed(self):
        import mosaicx

        assert not hasattr(mosaicx, "process_file") or "process_file" not in mosaicx.__all__

    def test_process_files_removed(self):
        import mosaicx

        assert not hasattr(mosaicx, "process_files") or "process_files" not in mosaicx.__all__
