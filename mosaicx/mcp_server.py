# mosaicx/mcp_server.py
"""
MOSAICX MCP Server -- Model Context Protocol interface for AI agents.

Exposes MOSAICX pipelines as MCP tools so that Claude and other AI agents
can extract data from documents, generate templates, de-identify text, etc.
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
    template: str | None = None,
) -> str:
    """Extract structured data from a medical document.

    Supports three extraction paths:
    - auto (default): The LLM infers an appropriate structure from the document and extracts into it.
    - mode-based: Use a registered extraction mode (e.g. "radiology", "pathology") for domain-specific multi-step extraction.
    - template-based: Use a template name for targeted extraction.

    Args:
        document_text: Full text of the clinical document to extract from.
        mode: Extraction strategy -- "auto", "radiology", or "pathology". Ignored if template is provided.
        template: Template name (built-in or user-created) or YAML file path. If provided, extraction uses this template.

    Returns:
        JSON string containing the extracted structured data.
    """
    try:
        _ensure_dspy()

        if template is not None:
            # Template-based extraction
            from .report import detect_mode, resolve_template

            template_model, tpl_name = resolve_template(template=template)
            effective_mode = detect_mode(tpl_name)

            if effective_mode is not None and template_model is None:
                # Mode pipeline
                import mosaicx.pipelines.pathology  # noqa: F401
                import mosaicx.pipelines.radiology  # noqa: F401
                from .pipelines.extraction import extract_with_mode

                output_data, metrics = extract_with_mode(document_text, effective_mode)
                result: dict[str, Any] = dict(output_data)
                if metrics is not None:
                    result["_metrics"] = {
                        "total_duration_s": metrics.total_duration_s,
                        "total_tokens": metrics.total_tokens,
                    }
                return _json_result(result)
            elif template_model is not None:
                from .pipelines.extraction import DocumentExtractor

                extractor = DocumentExtractor(output_schema=template_model)
                prediction = extractor(document_text=document_text)
                output: dict[str, Any] = {}
                if hasattr(prediction, "extracted"):
                    val = prediction.extracted
                    output["extracted"] = val.model_dump() if hasattr(val, "model_dump") else val
                return _json_result(output)
            else:
                return _json_result({"error": f"Template {template!r} resolved but produced no extraction template."})

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
def generate_template(
    description: str,
    name: str | None = None,
    mode: str | None = None,
) -> str:
    """Generate a YAML template from a natural-language description.

    The LLM will create a structured template specification based on the
    description. The template is saved to ~/.mosaicx/templates/ and can
    be used for targeted extraction via the extract_document tool.

    Args:
        description: Natural-language description of the document type to create a template for (e.g. "chest CT radiology report with findings and impressions").
        name: Optional name for the generated template. If not provided, the LLM chooses an appropriate name.
        mode: Optional pipeline mode to embed in the template (e.g. "radiology", "pathology").

    Returns:
        JSON string containing the generated template with name, description, and fields.
    """
    try:
        _ensure_dspy()

        from .pipelines.schema_gen import SchemaGenerator
        from .schemas.template_compiler import schema_spec_to_template_yaml

        generator = SchemaGenerator()
        result = generator(description=description)

        if name:
            result.schema_spec.class_name = name

        # Save as YAML template
        from .config import get_config

        cfg = get_config()
        yaml_str = schema_spec_to_template_yaml(result.schema_spec, mode=mode)
        cfg.templates_dir.mkdir(parents=True, exist_ok=True)
        dest = cfg.templates_dir / f"{result.schema_spec.class_name}.yaml"
        dest.write_text(yaml_str, encoding="utf-8")

        spec_data = result.schema_spec.model_dump()
        spec_data["_saved_to"] = str(dest)
        return _json_result(spec_data)

    except Exception as exc:
        logger.exception("generate_template failed")
        return _json_result({"error": str(exc)})


@mcp.tool()
def list_templates() -> str:
    """List available extraction templates.

    Returns both built-in templates and user-created templates
    from ~/.mosaicx/templates/.

    Returns:
        JSON string containing template summaries.
    """
    try:
        from .config import get_config
        from .schemas.radreport.registry import list_templates as _list_builtin

        cfg = get_config()
        templates = []

        # Built-in templates
        for tpl in _list_builtin():
            templates.append({
                "name": tpl.name,
                "description": tpl.description,
                "mode": tpl.mode,
                "source": "built-in",
            })

        # User templates
        if cfg.templates_dir.is_dir():
            from .schemas.template_compiler import parse_template

            for f in sorted(cfg.templates_dir.glob("*.yaml")) + sorted(cfg.templates_dir.glob("*.yml")):
                try:
                    meta = parse_template(f.read_text(encoding="utf-8"))
                    templates.append({
                        "name": f.stem,
                        "description": meta.description or "",
                        "mode": meta.mode,
                        "source": "user",
                    })
                except Exception:
                    templates.append({
                        "name": f.stem,
                        "description": "(invalid YAML)",
                        "mode": None,
                        "source": "user",
                    })

        return _json_result({
            "count": len(templates),
            "templates": templates,
        })

    except Exception as exc:
        logger.exception("list_templates failed")
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
