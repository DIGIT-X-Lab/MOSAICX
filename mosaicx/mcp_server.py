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
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP dependency guard
# ---------------------------------------------------------------------------

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    logger.warning(
        "mcp package not installed; using lightweight fallback FastMCP. "
        "Install extras with: pip install 'mosaicx[mcp]'"
    )

    class _FallbackToolManager:
        def __init__(self) -> None:
            self._tools: list[Any] = []

        def register(self, func: Any) -> None:
            from types import SimpleNamespace

            self._tools.append(SimpleNamespace(name=getattr(func, "__name__", "unknown"), fn=func))

        def list_tools(self) -> list[Any]:
            return list(self._tools)

    class FastMCP:  # type: ignore[override]
        def __init__(self, name: str) -> None:
            self.name = name
            self._tool_manager = _FallbackToolManager()

        def tool(self):
            def _decorator(func):
                self._tool_manager.register(func)
                return func

            return _decorator

        def run(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "MCP runtime unavailable because dependency 'mcp' is not installed. "
                "Install with: pip install 'mosaicx[mcp]'"
            )

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
    from .runtime_env import configure_dspy_lm

    cfg = get_config()
    if not cfg.api_key:
        raise RuntimeError(
            "No API key configured. Set MOSAICX_API_KEY or add api_key to your config."
        )

    from .metrics import TokenTracker, make_harmony_lm, set_tracker

    lm = make_harmony_lm(cfg.lm, api_key=cfg.api_key, api_base=cfg.api_base, temperature=cfg.lm_temperature)
    try:
        dspy, _adapter_name = configure_dspy_lm(
            lm,
            preferred_cache_dir=cfg.home_dir / ".dspy_cache",
        )
    except ImportError as exc:
        raise RuntimeError(
            "DSPy is required for MOSAICX pipelines. Install with: pip install dspy"
        ) from exc

    tracker = TokenTracker()
    set_tracker(tracker)
    dspy.settings.usage_tracker = tracker
    dspy.settings.track_usage = True

    _dspy_configured = True
    logger.info("DSPy configured with model: %s", cfg.lm)


def _json_result(data: Any) -> str:
    """Serialize *data* to a JSON string, handling Pydantic models and other types."""
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
    """Convert PipelineMetrics to a plain dict with per-step breakdown."""
    return {
        "total_duration_s": metrics.total_duration_s,
        "total_tokens": metrics.total_tokens,
        "steps": [
            {
                "name": s.name,
                "duration_s": s.duration_s,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
            }
            for s in metrics.steps
        ],
    }


def _compute_completeness(model_instance: Any, text: str) -> dict[str, Any]:
    """Compute completeness scoring and return as a plain dict."""
    from dataclasses import asdict

    from .evaluation.completeness import compute_report_completeness

    comp = compute_report_completeness(
        model_instance, text, type(model_instance)
    )
    return asdict(comp)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def extract_document(
    document_text: str,
    mode: str = "auto",
    template: str | None = None,
    score: bool = False,
) -> str:
    """Extract structured data from a medical document.

    Supports three extraction paths:
    - auto (default): The LLM infers an appropriate structure from the document.
    - mode-based: Use a registered extraction mode (e.g. "radiology", "pathology") for domain-specific multi-step extraction.
    - template-based: Use a template name for targeted extraction with an optional completeness score.

    Args:
        document_text: Full text of the clinical document to extract from.
        mode: Extraction strategy -- "auto", "radiology", or "pathology". Ignored if template is provided.
        template: Template name (built-in or user-created) or YAML file path. If provided, extraction uses this template.
        score: If true, compute completeness scoring against the template. Only effective with template or mode extraction (not auto).

    Returns:
        JSON string containing the extracted structured data, with optional "_metrics" and "completeness" keys.
    """
    try:
        # Validate mode early before configuring DSPy
        if mode and mode != "auto" and template is None:
            import mosaicx.pipelines.pathology  # noqa: F401
            import mosaicx.pipelines.radiology  # noqa: F401
            from .pipelines.modes import list_modes

            available = list_modes()
            if mode not in available:
                return _json_result({
                    "error": f"Unknown mode {mode!r}. Available: {', '.join(sorted(available))}"
                })

        _ensure_dspy()
        from .pipelines.extraction import apply_extraction_contract

        def _finalize_extract_result(
            payload: dict[str, Any],
            *,
            metrics: Any = None,
        ) -> str:
            if metrics is not None:
                payload["_metrics"] = _metrics_to_dict(metrics)
            apply_extraction_contract(payload, source_text=document_text)
            return _json_result(payload)

        if template is not None:
            # Template-based extraction
            from .report import detect_mode, resolve_template

            template_model, tpl_name = resolve_template(template=template)
            effective_mode = detect_mode(tpl_name)

            if effective_mode is not None and template_model is None:
                # Mode pipeline (built-in template like chest_ct)
                import mosaicx.pipelines.pathology  # noqa: F401
                import mosaicx.pipelines.radiology  # noqa: F401

                if score:
                    from .pipelines.extraction import extract_with_mode_raw
                    from .report import _find_primary_model

                    output_data, metrics, raw_pred = extract_with_mode_raw(
                        document_text, effective_mode
                    )
                    result: dict[str, Any] = dict(output_data)
                    model_instance = _find_primary_model(raw_pred)
                    if model_instance is not None:
                        result["completeness"] = _compute_completeness(
                            model_instance, document_text
                        )
                else:
                    from .pipelines.extraction import extract_with_mode

                    output_data, metrics = extract_with_mode(document_text, effective_mode)
                    result: dict[str, Any] = dict(output_data)

                return _finalize_extract_result(result, metrics=metrics)

            elif template_model is not None:
                from .pipelines.extraction import DocumentExtractor

                extractor = DocumentExtractor(output_schema=template_model)
                prediction = extractor(document_text=document_text)
                output: dict[str, Any] = {}
                planner_diag = (
                    getattr(prediction, "planner", None)
                    or getattr(extractor, "_last_planner", None)
                )
                if hasattr(prediction, "extracted"):
                    val = prediction.extracted
                    if hasattr(val, "model_dump"):
                        output["extracted"] = val.model_dump(mode="json")
                        if score:
                            output["completeness"] = _compute_completeness(
                                val, document_text
                            )
                    else:
                        output["extracted"] = val
                if isinstance(planner_diag, dict):
                    output["_planner"] = planner_diag
                return _finalize_extract_result(
                    output,
                    metrics=getattr(extractor, "_last_metrics", None),
                )
            else:
                return _json_result({
                    "error": f"Template {template!r} resolved but produced no extraction template."
                })

        if mode and mode != "auto":
            # Mode-based extraction (radiology, pathology, etc.)
            import mosaicx.pipelines.pathology  # noqa: F401
            import mosaicx.pipelines.radiology  # noqa: F401

            if score:
                from .pipelines.extraction import extract_with_mode_raw
                from .report import _find_primary_model

                output_data, metrics, raw_pred = extract_with_mode_raw(
                    document_text, mode
                )
                result: dict[str, Any] = dict(output_data)
                model_instance = _find_primary_model(raw_pred)
                if model_instance is not None:
                    result["completeness"] = _compute_completeness(
                        model_instance, document_text
                    )
            else:
                from .pipelines.extraction import extract_with_mode

                output_data, metrics = extract_with_mode(document_text, mode)
                result: dict[str, Any] = dict(output_data)

            return _finalize_extract_result(result, metrics=metrics)

        # Auto mode: LLM infers schema
        if score:
            logger.warning(
                "score=True has no effect in auto mode (no template to score against)."
            )

        from .pipelines.extraction import DocumentExtractor

        extractor = DocumentExtractor()
        prediction = extractor(document_text=document_text)
        output: dict[str, Any] = {}
        planner_diag = (
            getattr(prediction, "planner", None)
            or getattr(extractor, "_last_planner", None)
        )
        if hasattr(prediction, "extracted"):
            val = prediction.extracted
            output["extracted"] = val.model_dump(mode="json") if hasattr(val, "model_dump") else val
        if hasattr(prediction, "inferred_schema"):
            output["inferred_schema"] = prediction.inferred_schema.model_dump(mode="json")
        if isinstance(planner_diag, dict):
            output["_planner"] = planner_diag
        return _finalize_extract_result(
            output,
            metrics=getattr(extractor, "_last_metrics", None),
        )

    except Exception as exc:
        logger.exception("extract_document failed")
        return _json_result({"error": str(exc)})


@mcp.tool()
def deidentify_text(
    text: str,
    mode: str = "remove",
    regex_only: bool = False,
) -> str:
    """Remove Protected Health Information (PHI) from clinical text.

    Uses a two-layer approach by default:
    1. LLM-based redaction identifies context-dependent PHI (names, addresses, etc.).
    2. Regex safety net catches format-based PHI (SSNs, phone numbers, MRNs, emails).

    Set regex_only=true to skip the LLM and use only regex pattern matching (faster, no API key needed).

    Args:
        text: The clinical text to de-identify.
        mode: De-identification strategy -- "remove" (replace with [REDACTED]), "pseudonymize" (replace with fake values), or "dateshift" (shift dates by a consistent offset).
        regex_only: If true, skip LLM and use only regex-based scrubbing. Faster but less comprehensive.

    Returns:
        JSON string with "redacted_text" containing the de-identified text.
    """
    try:
        from .pipelines.deidentifier import regex_scrub_phi

        if regex_only:
            return _json_result({
                "redacted_text": regex_scrub_phi(text),
                "mode": "regex",
            })

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
def summarize_reports(
    reports: list[str],
    patient_id: str = "unknown",
) -> str:
    """Summarize multiple clinical reports into a patient timeline.

    Takes a list of report texts and produces a narrative summary with
    a structured timeline of clinical events.

    Args:
        reports: List of clinical report texts to summarize.
        patient_id: Patient identifier for the summary.

    Returns:
        JSON string with "narrative" (summary text) and "events" (list of timeline events).
    """
    try:
        if not reports:
            return _json_result({"error": "No reports provided to summarize."})

        _ensure_dspy()

        from .pipelines.summarizer import ReportSummarizer

        summarizer = ReportSummarizer()
        result = summarizer(reports=reports, patient_id=patient_id)

        return _json_result({
            "events": [e.model_dump() for e in result.events],
            "narrative": result.narrative,
        })

    except Exception as exc:
        logger.exception("summarize_reports failed")
        return _json_result({"error": str(exc)})


@mcp.tool()
def generate_template(
    description: str,
    name: str | None = None,
    mode: str | None = None,
    document_text: str | None = None,
) -> str:
    """Generate a YAML template from a natural-language description.

    The LLM will create a structured template specification based on the
    description. Optionally, a sample document can be provided to help the
    LLM infer richer field types. The template is saved to ~/.mosaicx/templates/
    and can be used for targeted extraction via the extract_document tool.

    Args:
        description: Natural-language description of the document type to create a template for (e.g. "chest CT radiology report with findings and impressions").
        name: Optional name for the generated template. If not provided, the LLM chooses an appropriate name.
        mode: Optional pipeline mode to embed in the template (e.g. "radiology", "pathology").
        document_text: Optional sample document text. When provided, the LLM uses it to infer field types and structure.

    Returns:
        JSON string containing the generated template with name, fields, and save location.
    """
    try:
        _ensure_dspy()

        from .pipelines.schema_gen import SchemaGenerator
        from .schemas.template_compiler import schema_spec_to_template_yaml

        generator = SchemaGenerator()
        result = generator(
            description=description,
            document_text=document_text or "",
        )

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
        spec_data["_yaml"] = yaml_str
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
        JSON string containing template summaries with name, description, mode, and source.
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
        import mosaicx.pipelines.pathology  # noqa: F401
        import mosaicx.pipelines.radiology  # noqa: F401
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
# Query session store
# ---------------------------------------------------------------------------

_sessions: dict[str, Any] = {}  # session_id -> QuerySession


# ---------------------------------------------------------------------------
# Query MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def query_start(
    source_texts: dict[str, str] | None = None,
    sources: list[str] | None = None,
    template: str | None = None,
) -> str:
    """Create a new query session from in-memory text sources.

    Accepts a dict mapping source names to their text content and creates
    a stateful session for conversational Q&A. The session is stored
    server-side and identified by a unique session_id.

    Args:
        source_texts: Optional mapping of source names to text content
            (e.g. {"report.txt": "Patient presents..."}).
        sources: Optional list of file-system sources to load into the query
            session.
        template: Optional template hint for downstream query workflows.

    Returns:
        JSON string with "session_id" (str) and "catalog" (list of source metadata dicts).
    """
    try:
        if not source_texts and not sources:
            return _json_result({"error": "Provide source_texts or sources."})

        from .query.session import QuerySession

        session = QuerySession(sources=sources, template=template)

        if source_texts:
            for name, text in source_texts.items():
                session.add_text_source(name, text)

        session_id = str(uuid.uuid4())
        _sessions[session_id] = session

        return _json_result({
            "session_id": session_id,
            "catalog": [m.model_dump() for m in session.catalog],
        })

    except Exception as exc:
        logger.exception("query_start failed")
        return _json_result({"error": str(exc)})


@mcp.tool()
def query_ask(session_id: str, question: str) -> str:
    """Ask a question on an existing query session.

    Uses the session's loaded documents and conversation history to answer
    the question via the RLM-powered query engine.

    Args:
        session_id: The session identifier returned by query_start.
        question: Natural language question about the loaded documents.

    Returns:
        JSON string with "answer" (str), "citations" (list), confidence/fallback
        metadata, and "session_id" (str).
    """
    try:
        if session_id not in _sessions:
            return _json_result({
                "error": f"Unknown session_id: {session_id!r}. Start a session first with query_start."
            })

        session = _sessions[session_id]

        _ensure_dspy()

        from .query.engine import QueryEngine

        engine = QueryEngine(session=session)
        payload = engine.ask_structured(question)
        payload["session_id"] = session_id
        return _json_result(payload)

    except Exception as exc:
        logger.exception("query_ask failed")
        return _json_result({"error": str(exc)})


@mcp.tool()
def query_close(session_id: str) -> str:
    """Close and remove a query session.

    Releases all resources associated with the session and removes it
    from the server's session store.

    Args:
        session_id: The session identifier returned by query_start.

    Returns:
        JSON string with "status" ("closed") and "session_id" (str).
    """
    try:
        if session_id not in _sessions:
            return _json_result({
                "error": f"Unknown session_id: {session_id!r}. No active session with that ID."
            })

        session = _sessions.pop(session_id)
        session.close()

        return _json_result({
            "status": "closed",
            "session_id": session_id,
        })

    except Exception as exc:
        logger.exception("query_close failed")
        return _json_result({"error": str(exc)})


# ---------------------------------------------------------------------------
# Verify MCP Tool
# ---------------------------------------------------------------------------


@mcp.tool()
def verify_output(
    source_text: str,
    extraction: str | None = None,
    claim: str | None = None,
    level: str = "quick",
) -> str:
    """Verify an extraction or claim against the original source document.

    Checks whether extracted fields or a textual claim are supported by the
    source text.  Three verification levels are available:
    - "quick" (default): deterministic checks only, no LLM call (< 1s).
    - "standard": deterministic + LLM spot-check (3-10s).
    - "thorough": deterministic + spot-check + full RLM audit (30-90s).

    Exactly one of ``extraction`` or ``claim`` must be provided.

    Args:
        source_text: The original document text to verify against.
        extraction: JSON string of the structured extraction dict to verify. Mutually exclusive with claim.
        claim: A single textual claim to verify. Mutually exclusive with extraction.
        level: Verification depth -- "quick", "standard", or "thorough".

    Returns:
        JSON string with verdict, confidence, level, issues, and field_verdicts.
    """
    try:
        extraction_dict: dict | None = None
        if extraction is not None:
            try:
                extraction_dict = json.loads(extraction)
            except (json.JSONDecodeError, TypeError) as exc:
                return _json_result({"error": f"Invalid extraction JSON: {exc}"})

        # Standard and thorough levels need DSPy for LLM-based verification.
        # If DSPy setup fails, the engine will fall back to deterministic.
        if level in ("standard", "thorough"):
            try:
                _ensure_dspy()
            except Exception:
                logger.warning("DSPy setup failed for %s verify; engine will fall back", level)

        from .verify.engine import verify

        report = verify(
            extraction=extraction_dict,
            claim=claim,
            source_text=source_text,
            level=level,
        )

        return _json_result(report.to_dict())

    except Exception as exc:
        logger.exception("verify_output failed")
        return _json_result({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
