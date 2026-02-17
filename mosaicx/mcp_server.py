# mosaicx/mcp_server.py
"""
MOSAICX MCP Server -- Model Context Protocol interface for AI agents.

Exposes MOSAICX pipelines as MCP tools so that Claude and other AI agents
can extract data from documents, generate schemas, de-identify text, etc.
without shelling out to the CLI.

Run via:
    python -m mosaicx.mcp_server
    mosaicx mcp serve
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP dependency guard
# ---------------------------------------------------------------------------

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "ERROR: The 'mcp' package is required for the MOSAICX MCP server.\n"
        "Install it with:\n\n"
        "    pip install 'mosaicx[mcp]'\n"
        "    # or directly:\n"
        "    pip install 'mcp[cli]>=1.0.0'\n",
        file=sys.stderr,
    )
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP("mosaicx")

# ---------------------------------------------------------------------------
# DSPy configuration (run once, cached)
# ---------------------------------------------------------------------------

_dspy_configured = False


def _ensure_dspy() -> None:
    """Configure DSPy with the LM from MosaicxConfig.

    Called lazily on the first tool invocation that needs the LLM.
    Subsequent calls are no-ops.
    """
    global _dspy_configured
    if _dspy_configured:
        return

    from .config import get_config

    cfg = get_config()
    if not cfg.api_key:
        raise RuntimeError(
            "No API key configured. Set MOSAICX_API_KEY or add api_key to your config."
        )

    try:
        import dspy
    except ImportError:
        raise RuntimeError(
            "DSPy is required for MOSAICX pipelines. Install with: pip install dspy"
        )

    from .metrics import TokenTracker, make_harmony_lm, set_tracker

    lm = make_harmony_lm(cfg.lm, api_key=cfg.api_key, api_base=cfg.api_base, temperature=cfg.lm_temperature)
    dspy.configure(lm=lm)

    tracker = TokenTracker()
    set_tracker(tracker)
    dspy.settings.usage_tracker = tracker
    dspy.settings.track_usage = True

    _dspy_configured = True
    logger.info("DSPy configured with model: %s", cfg.lm)


def _json_result(data: Any) -> str:
    """Serialize *data* to a JSON string, handling Pydantic models and other types."""
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def extract_document(
    document_text: str,
    mode: str = "auto",
    schema_name: str | None = None,
) -> str:
    """Extract structured data from a medical document.

    Supports three extraction paths:
    - auto (default): The LLM infers an appropriate schema from the document and extracts into it.
    - mode-based: Use a registered extraction mode (e.g. "radiology", "pathology") for domain-specific multi-step extraction.
    - schema-based: Use a previously saved schema by name for targeted extraction.

    Args:
        document_text: Full text of the clinical document to extract from.
        mode: Extraction strategy -- "auto", "radiology", or "pathology". Ignored if schema_name is provided.
        schema_name: Name of a saved schema (from ~/.mosaicx/schemas/). If provided, extraction uses this schema.

    Returns:
        JSON string containing the extracted structured data.
    """
    try:
        _ensure_dspy()

        if schema_name is not None:
            # Schema-based extraction
            from .config import get_config
            from .pipelines.extraction import extract_with_schema

            cfg = get_config()
            extracted = extract_with_schema(document_text, schema_name, cfg.schema_dir)
            return _json_result({"extracted": extracted})

        if mode and mode != "auto":
            # Mode-based extraction (radiology, pathology, etc.)
            import mosaicx.pipelines.radiology  # noqa: F401
            import mosaicx.pipelines.pathology  # noqa: F401
            from .pipelines.extraction import extract_with_mode

            output_data, metrics = extract_with_mode(document_text, mode)
            result: dict[str, Any] = dict(output_data)
            if metrics is not None:
                result["_metrics"] = {
                    "total_duration_s": metrics.total_duration_s,
                    "total_tokens": metrics.total_tokens,
                }
            return _json_result(result)

        # Auto mode: LLM infers schema
        from .pipelines.extraction import DocumentExtractor

        extractor = DocumentExtractor()
        prediction = extractor(document_text=document_text)
        output: dict[str, Any] = {}
        if hasattr(prediction, "extracted"):
            val = prediction.extracted
            output["extracted"] = val.model_dump() if hasattr(val, "model_dump") else val
        if hasattr(prediction, "inferred_schema"):
            output["inferred_schema"] = prediction.inferred_schema.model_dump()
        return _json_result(output)

    except Exception as exc:
        logger.exception("extract_document failed")
        return _json_result({"error": str(exc)})


@mcp.tool()
def deidentify_text(
    text: str,
    mode: str = "remove",
) -> str:
    """Remove Protected Health Information (PHI) from clinical text.

    Uses a two-layer approach:
    1. LLM-based redaction identifies context-dependent PHI (names, addresses, etc.).
    2. Regex safety net catches format-based PHI (SSNs, phone numbers, MRNs, emails).

    Args:
        text: The clinical text to de-identify.
        mode: De-identification strategy -- "remove" (replace with [REDACTED]), "pseudonymize" (replace with fake values), or "dateshift" (shift dates by a consistent offset).

    Returns:
        JSON string with "redacted_text" containing the de-identified text.
    """
    try:
        _ensure_dspy()

        from .pipelines.deidentifier import Deidentifier

        deid = Deidentifier()
        result = deid(document_text=text, mode=mode)

        return _json_result({
            "redacted_text": result.redacted_text,
            "mode": mode,
        })

    except Exception as exc:
        logger.exception("deidentify_text failed")
        return _json_result({"error": str(exc)})


@mcp.tool()
def generate_schema(
    description: str,
    name: str | None = None,
) -> str:
    """Generate a Pydantic schema from a natural-language description.

    The LLM will create a structured schema specification based on the
    description. The schema can then be used for targeted extraction.

    Args:
        description: Natural-language description of the document type to create a schema for (e.g. "chest CT radiology report with findings and impressions").
        name: Optional class name for the generated schema. If not provided, the LLM chooses an appropriate name.

    Returns:
        JSON string containing the generated SchemaSpec with class_name, description, and fields.
    """
    try:
        _ensure_dspy()

        from .pipelines.schema_gen import SchemaGenerator, save_schema

        generator = SchemaGenerator()
        result = generator(description=description)

        if name:
            result.schema_spec.class_name = name

        # Save the schema to the default schema directory
        from .config import get_config

        cfg = get_config()
        saved_path = save_schema(result.schema_spec, schema_dir=cfg.schema_dir)

        spec_data = result.schema_spec.model_dump()
        spec_data["_saved_to"] = str(saved_path)
        return _json_result(spec_data)

    except Exception as exc:
        logger.exception("generate_schema failed")
        return _json_result({"error": str(exc)})


@mcp.tool()
def list_schemas() -> str:
    """List all saved extraction schemas.

    Returns the schemas stored in ~/.mosaicx/schemas/ with their
    class names, field counts, and descriptions.

    Returns:
        JSON string containing a list of schema summaries.
    """
    try:
        from .config import get_config
        from .pipelines.schema_gen import list_schemas as _list_schemas

        cfg = get_config()
        specs = _list_schemas(cfg.schema_dir)

        schemas = []
        for spec in specs:
            schemas.append({
                "class_name": spec.class_name,
                "description": spec.description,
                "field_count": len(spec.fields),
                "fields": [
                    {"name": f.name, "type": f.type, "required": f.required}
                    for f in spec.fields
                ],
            })

        return _json_result({
            "schema_dir": str(cfg.schema_dir),
            "count": len(schemas),
            "schemas": schemas,
        })

    except Exception as exc:
        logger.exception("list_schemas failed")
        return _json_result({"error": str(exc)})


@mcp.tool()
def list_modes() -> str:
    """List available extraction modes.

    Extraction modes are specialized multi-step pipelines for specific
    document domains (e.g. radiology reports, pathology reports).

    Returns:
        JSON string containing mode names and descriptions.
    """
    try:
        # Trigger eager registration of mode metadata
        import mosaicx.pipelines.radiology  # noqa: F401
        import mosaicx.pipelines.pathology  # noqa: F401
        from .pipelines.modes import list_modes as _list_modes

        modes = [
            {"name": name, "description": desc}
            for name, desc in _list_modes()
        ]

        return _json_result({
            "count": len(modes),
            "modes": modes,
        })

    except Exception as exc:
        logger.exception("list_modes failed")
        return _json_result({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
