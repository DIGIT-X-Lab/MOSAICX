# mosaicx/pipelines/extraction.py
"""Document extraction pipeline.

Provides schema-first extraction from medical documents. Two modes:

**Auto mode** (no output_schema):
    1. LLM infers a SchemaSpec from the document text.
    2. SchemaSpec is compiled into a Pydantic model.
    3. Single-step extraction into the inferred model.

**Schema mode** (output_schema provided):
    Single-step ChainOfThought extraction into the provided Pydantic model.

Convenience functions:
    - extract_with_schema(): Load a saved schema by name and extract.
    - extract_with_mode(): Run a registered extraction mode pipeline.
"""
import json
import logging
import os
import re
import types
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Convenience functions (no DSPy dependency at import time)
# ---------------------------------------------------------------------------


def extract_with_schema(document_text: str, schema_name: str, schema_dir: Path) -> dict[str, Any]:
    """Load a saved schema by name and extract into it.

    Args:
        document_text: Full text of the document.
        schema_name: Name of the schema (without .json extension).
        schema_dir: Directory where schemas are stored.

    Returns:
        Dict with the extracted data matching the schema fields.
    """
    from .schema_gen import load_schema, compile_schema

    spec = load_schema(schema_name, schema_dir)
    model = compile_schema(spec)
    # Access DocumentExtractor via module __getattr__ (it's lazily built)
    import sys
    mod = sys.modules[__name__]
    extractor = mod.DocumentExtractor(output_schema=model)
    result = extractor(document_text=document_text)
    val = result.extracted
    return val.model_dump() if hasattr(val, "model_dump") else val


def _prediction_to_dict(prediction: Any) -> dict[str, Any]:
    """Convert a dspy.Prediction to a plain dict, serialising Pydantic models."""
    output: dict[str, Any] = {}
    for key in prediction.keys():
        val = getattr(prediction, key)
        if hasattr(val, "model_dump"):
            output[key] = val.model_dump()
        elif isinstance(val, list):
            output[key] = [v.model_dump() if hasattr(v, "model_dump") else v for v in val]
        else:
            output[key] = val
    return output


def _is_basemodel_type(annotation: Any) -> bool:
    """Return True when *annotation* is a Pydantic BaseModel subclass."""
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _map_trinary_label(value: Any, *, allow_unknown: bool) -> float:
    """Map flexible label-like values into a CheXpert-style trinary score."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        num = float(value)
        if num > 0:
            return 1.0
        if num < 0 and allow_unknown:
            return -1.0
        return 0.0

    text = str(value or "").strip().lower()
    if not text:
        return -1.0 if allow_unknown else 0.0

    if any(
        token in text
        for token in (
            "uncertain",
            "indeterminate",
            "equivocal",
            "possible",
            "cannot exclude",
        )
    ):
        return -1.0 if allow_unknown else 1.0

    if any(
        token in text
        for token in (
            "no ",
            "without",
            "negative for",
            "absent",
            "not seen",
            "none",
        )
    ):
        return 0.0

    return 1.0


def _coerce_literal_value(value: Any, annotation: Any) -> Any:
    """Coerce value for ``typing.Literal`` annotations."""
    options = tuple(get_args(annotation))
    if not options:
        return value

    unwrapped = value.get("value") if isinstance(value, dict) and "value" in value else value
    option_scalars = [o for o in options if isinstance(o, (int, float))]

    if option_scalars:
        allow_unknown = any(float(o) < 0 for o in option_scalars)
        mapped = _map_trinary_label(unwrapped, allow_unknown=allow_unknown)
        for option in option_scalars:
            if float(option) == mapped:
                return option
        if isinstance(unwrapped, (int, float)):
            return float(unwrapped)

    if isinstance(unwrapped, str):
        probe = unwrapped.strip().lower()
        for option in options:
            if isinstance(option, str) and option.lower() == probe:
                return option
            if str(option).strip().lower() == probe:
                return option

    for option in options:
        if unwrapped == option:
            return option

    return unwrapped


def _is_nullish_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    probe = value.strip().lower()
    return probe in {
        "",
        "none",
        "null",
        "nil",
        "n/a",
        "na",
        "not available",
    }


_ABSENCE_ENUM_TOKENS = {
    "none",
    "not present",
    "absent",
    "not applicable",
    "n/a",
    "na",
    "missing",
    "unknown",
}


def _normalize_spinal_level_text(value: str) -> str:
    """Normalize compressed spinal level ranges (for example, C2-3 -> C2-C3)."""

    text = (
        str(value)
        .replace("‐", "-")
        .replace("‑", "-")
        .replace("‒", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )

    def _expand_letter_range(match: re.Match[str]) -> str:
        letter = match.group(1).upper()
        left = match.group(2)
        right = match.group(3)
        return f"{letter}{left}-{letter}{right}"

    return re.sub(
        r"\b([A-Za-z])\s*(\d+)\s*-\s*(\d+)\b",
        _expand_letter_range,
        text,
    )


def _absence_enum_value_for_annotation(annotation: Any) -> Any | None:
    """Return explicit absence enum value for Optional[Enum], if declared."""
    origin = get_origin(annotation)
    if origin not in (Union, types.UnionType):
        return None

    options = list(get_args(annotation))
    if not any(opt is type(None) for opt in options):
        return None

    for option in options:
        if option is type(None):
            continue
        if isinstance(option, type) and issubclass(option, Enum):
            for member in option:
                key = " ".join(str(member.value).strip().lower().split())
                if key in _ABSENCE_ENUM_TOKENS:
                    return member.value
    return None


def _coerce_value_for_annotation(value: Any, annotation: Any) -> Any:
    """Recursively coerce a value to better fit the target annotation."""
    if annotation in (Any, object) or annotation is None:
        return value
    if value is None:
        absence_enum = _absence_enum_value_for_annotation(annotation)
        if absence_enum is not None:
            return absence_enum
        return None

    origin = get_origin(annotation)

    if origin in (Union, types.UnionType):
        if _is_nullish_string(value) and any(a is type(None) for a in get_args(annotation)):
            absence_enum = _absence_enum_value_for_annotation(annotation)
            if absence_enum is not None:
                return absence_enum
            return None
        options = [a for a in get_args(annotation) if a is not type(None)]
        for option in options:
            try:
                coerced = _coerce_value_for_annotation(value, option)
                if _is_basemodel_type(option):
                    option.model_validate(coerced)
                return coerced
            except Exception:
                continue
        return value

    if origin is Literal:
        return _coerce_literal_value(value, annotation)

    if _is_basemodel_type(annotation):
        model_cls: type[BaseModel] = annotation
        if isinstance(value, model_cls):
            return value

        if isinstance(value, dict):
            payload: dict[str, Any] = {}
            for field_name, field_info in model_cls.model_fields.items():
                if field_name in value:
                    payload[field_name] = _coerce_value_for_annotation(
                        value[field_name], field_info.annotation
                    )
            if payload:
                return payload

        if "value" in model_cls.model_fields:
            value_field = model_cls.model_fields["value"]
            payload = {
                "value": _coerce_value_for_annotation(value, value_field.annotation),
            }
            if isinstance(value, str) and "supporting_text" in model_cls.model_fields:
                payload["supporting_text"] = value
            return payload

        return value

    if origin in (list, tuple, set):
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        if _is_nullish_string(value):
            seq = []
        else:
            seq = value if isinstance(value, (list, tuple, set)) else [value]
        coerced = [_coerce_value_for_annotation(item, item_type) for item in seq]
        if origin is tuple:
            return tuple(coerced)
        if origin is set:
            return set(coerced)
        return coerced

    if origin is dict:
        key_type, val_type = get_args(annotation) if get_args(annotation) else (Any, Any)
        if _is_nullish_string(value):
            return {}
        if not isinstance(value, dict):
            return value
        return {
            _coerce_value_for_annotation(k, key_type): _coerce_value_for_annotation(v, val_type)
            for k, v in value.items()
        }

    if isinstance(annotation, type):
        if annotation is bool:
            if isinstance(value, str):
                probe = value.strip().lower()
                if probe in {"true", "yes", "y", "1", "positive"}:
                    return True
                if probe in {"false", "no", "n", "0", "negative"} or _is_nullish_string(value):
                    return False
            return bool(value)
        if annotation in (int, float):
            raw = value.get("value") if isinstance(value, dict) and "value" in value else value
            if isinstance(raw, bool):
                return 1.0 if annotation is float else int(raw)
            try:
                return annotation(raw)
            except Exception:
                return raw
        if annotation is str:
            if isinstance(value, (dict, list)):
                return json.dumps(value, default=str, ensure_ascii=False)
            return _normalize_spinal_level_text(str(value))

    return value


def _coerce_payload_to_schema(payload: Any, schema_class: type[BaseModel]) -> dict[str, Any]:
    """Coerce extracted payload into a dict better aligned with *schema_class*."""
    if isinstance(payload, schema_class):
        return payload.model_dump()
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected dict payload for {schema_class.__name__}, got {type(payload).__name__}"
        )

    out: dict[str, Any] = {}
    for field_name, field_info in schema_class.model_fields.items():
        if field_name not in payload:
            continue
        out[field_name] = _coerce_value_for_annotation(
            payload[field_name], field_info.annotation
        )
    return out


def _recover_schema_instance_from_raw(
    raw_output: str, schema_class: type[BaseModel]
) -> BaseModel:
    """Parse JSON-like model output and coerce/validate into *schema_class*."""
    from mosaicx.verify.parse_utils import parse_json_like

    parsed = parse_json_like(raw_output or "")
    if not isinstance(parsed, dict):
        preview = " ".join(str(raw_output or "").split())
        if len(preview) > 220:
            preview = preview[:217] + "..."
        raise ValueError(
            f"Model output is not valid JSON object for {schema_class.__name__}: {preview or '<empty>'}"
        )

    coerced = _coerce_payload_to_schema(parsed, schema_class)
    return schema_class.model_validate(coerced)


def _normalize_local_api_base(base_url: str) -> str:
    return (
        str(base_url or "")
        .replace("://localhost", "://127.0.0.1")
        .replace("://[::1]", "://127.0.0.1")
    )


def _normalize_model_name_for_openai_compatible(model_name: str) -> str:
    model_name = str(model_name or "").strip()
    if "/" in model_name:
        provider, rest = model_name.split("/", 1)
        if provider.strip().lower() in {"openai", "ollama"} and rest.strip():
            return rest.strip()
    return model_name


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return _is_nullish_string(value)
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _extract_grounding_snippet(*, source_text: str, value: Any) -> tuple[bool | None, str | None]:
    """Return (grounded, snippet) for scalar values when directly found in source text."""
    if value is None:
        return None, None

    if isinstance(value, bool):
        return None, None

    if isinstance(value, (int, float)):
        needle = str(value)
    elif isinstance(value, str):
        needle = value.strip()
        if not needle:
            return None, None
    else:
        return None, None

    haystack = str(source_text or "")
    if not haystack:
        return None, None

    idx = haystack.lower().find(needle.lower())
    if idx < 0:
        return False, None

    start = max(0, idx - 80)
    end = min(len(haystack), idx + len(needle) + 80)
    return True, " ".join(haystack[start:end].split())


def apply_extraction_contract(
    output_data: dict[str, Any],
    *,
    source_text: str,
    critical_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Attach canonical extraction contract to *output_data* in ``_extraction_contract``.

    Contract fields (per critical field):
      - ``value``
      - ``evidence``
      - ``grounded``
      - ``confidence``
      - ``status`` in {``supported``, ``needs_review``, ``insufficient_evidence``}
    """
    if not isinstance(output_data, dict):
        return output_data

    target: Any
    if isinstance(output_data.get("extracted"), dict):
        target = output_data["extracted"]
    else:
        target = output_data

    if not isinstance(target, dict):
        return output_data

    inferred = [
        key for key in target.keys()
        if isinstance(key, str) and not key.startswith("_")
    ]
    fields = critical_fields if critical_fields is not None else inferred

    field_results: list[dict[str, Any]] = []
    counts = {
        "supported": 0,
        "needs_review": 0,
        "insufficient_evidence": 0,
    }

    for field in fields:
        value = target.get(field)
        if _is_missing_value(value):
            status = "insufficient_evidence"
            grounded: bool | None = False
            confidence = 0.0
            evidence = None
        else:
            grounded, evidence = _extract_grounding_snippet(
                source_text=source_text,
                value=value,
            )
            if grounded is True:
                status = "supported"
                confidence = 0.9
            else:
                status = "needs_review"
                confidence = 0.5

        counts[status] += 1
        field_results.append(
            {
                "field": field,
                "value": value,
                "evidence": evidence,
                "grounded": grounded,
                "confidence": confidence,
                "status": status,
            }
        )

    output_data["_extraction_contract"] = {
        "version": "1.0",
        "critical_fields": fields,
        "field_results": field_results,
        "summary": counts,
    }
    return output_data


_SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(clinical information|clinical history|history|indication|comparison|technique|"
    r"procedure information|findings?|impression|diagnosis|assessment|plan|recommendation)s?\s*:?\s*$",
    flags=re.IGNORECASE,
)

_SECTION_HEADER_INLINE_PATTERN = re.compile(
    r"^\s*(clinical information|clinical history|history|indication|comparison|technique|"
    r"procedure information|findings?|impression|diagnosis|assessment|plan|recommendation)s?\s*:\s+",
    flags=re.IGNORECASE,
)

_ALLOWED_ROUTE_STRATEGIES = {"deterministic", "constrained_extract", "heavy_extract", "repair"}


def _normalize_section_name(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return cleaned or "section"


def _split_document_sections(document_text: str, *, max_sections: int = 10) -> list[dict[str, str]]:
    """Split a document into lightweight named sections for route planning."""
    text = str(document_text or "")
    if not text.strip():
        return [{"name": "document", "title": "Document", "text": ""}]

    sections: list[dict[str, str]] = []
    current_title = "Document"
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_lines, current_title
        block = "\n".join(current_lines).strip()
        if block:
            sections.append(
                {
                    "name": _normalize_section_name(current_title),
                    "title": current_title.strip() or "Document",
                    "text": block,
                }
            )
        current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        is_header = bool(_SECTION_HEADER_PATTERN.match(stripped))
        if is_header:
            _flush()
            header = stripped.rstrip(":").strip()
            current_title = header.title() if header else "Section"
            continue
        current_lines.append(line)

    _flush()

    if not sections:
        return [{"name": "document", "title": "Document", "text": text.strip()}]

    if len(sections) <= max_sections:
        return sections

    head = sections[: max_sections - 1]
    tail = sections[max_sections - 1 :]
    merged_tail = "\n\n".join(
        f"{entry['title']}\n{entry['text']}".strip()
        for entry in tail
        if str(entry.get("text", "")).strip()
    ).strip()
    head.append(
        {
            "name": "remaining_sections",
            "title": "Remaining Sections",
            "text": merged_tail,
        }
    )
    return head


def _section_complexity_hint(text: str) -> str:
    """Assign an easy/moderate/hard complexity hint for planner context."""
    sample = str(text or "")
    token_count = len(sample.split())
    digit_count = sum(ch.isdigit() for ch in sample)
    has_units = bool(
        re.search(r"\b\d+(\.\d+)?\s*(mm|cm|kg|mg|ml|%|x10\^?\d+)\b", sample, flags=re.IGNORECASE)
    )
    has_table_like = bool(re.search(r"\b[A-Za-z_]+\s*\|\s*[A-Za-z_]+", sample))

    if token_count <= 70 and digit_count <= 20 and not has_units and not has_table_like:
        return "easy"
    if token_count >= 260 or digit_count >= 80 or has_units or has_table_like:
        return "hard"
    return "moderate"


def _default_strategy_for_hint(hint: str) -> str:
    hint_norm = str(hint or "").strip().lower()
    if hint_norm == "easy":
        return "deterministic"
    if hint_norm == "moderate":
        return "constrained_extract"
    return "heavy_extract"


def _normalize_route_strategy(value: Any, *, default: str) -> str:
    probe = " ".join(str(value or "").strip().lower().split())
    aliases = {
        "deterministic": "deterministic",
        "light": "deterministic",
        "lightweight": "deterministic",
        "constrained_extract": "constrained_extract",
        "constrained": "constrained_extract",
        "json": "constrained_extract",
        "heavy_extract": "heavy_extract",
        "heavy": "heavy_extract",
        "full": "heavy_extract",
        "repair": "repair",
        "retry": "repair",
        "refine": "repair",
    }
    mapped = aliases.get(probe, "")
    if mapped in _ALLOWED_ROUTE_STRATEGIES:
        return mapped
    return default


def _plan_section_routes_with_react(
    *,
    schema_name: str,
    sections: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Plan section-level extraction routes using DSPy ReAct when available."""
    fallback_routes = [
        {
            "section": sec["title"],
            "name": sec["name"],
            "complexity": _section_complexity_hint(sec.get("text", "")),
            "strategy": _default_strategy_for_hint(_section_complexity_hint(sec.get("text", ""))),
            "reason": "deterministic fallback",
        }
        for sec in sections
    ]

    try:
        from mosaicx.runtime_env import import_dspy

        dspy = import_dspy()
    except Exception as exc:
        return fallback_routes, {
            "planner": "deterministic_fallback",
            "react_used": False,
            "fallback_reason": f"dspy_import_failed: {type(exc).__name__}",
        }

    if getattr(dspy.settings, "lm", None) is None:
        return fallback_routes, {
            "planner": "deterministic_fallback",
            "react_used": False,
            "fallback_reason": "lm_not_configured",
        }

    section_map = {sec["name"]: sec for sec in sections}
    section_descriptors = [
        {
            "name": sec["name"],
            "title": sec["title"],
            "chars": len(sec.get("text", "")),
            "complexity": _section_complexity_hint(sec.get("text", "")),
        }
        for sec in sections
    ]

    def list_sections() -> list[dict[str, Any]]:
        """List section descriptors with complexity hints."""
        return section_descriptors

    def read_section(name: str) -> str:
        """Read a section preview by section name."""
        sec = section_map.get(str(name or "").strip())
        if sec is None:
            return ""
        body = str(sec.get("text", ""))
        return body[:1800]

    class _RouteSig(dspy.Signature):
        schema_name: str = dspy.InputField(desc="Target schema class name")
        section_name: str = dspy.InputField(desc="Section identifier")
        section_preview: str = dspy.InputField(desc="Preview of section text")
        complexity_hint: str = dspy.InputField(desc="easy|moderate|hard")
        strategy: str = dspy.OutputField(
            desc="deterministic|constrained_extract|heavy_extract|repair"
        )
        reason: str = dspy.OutputField(desc="Short route justification")

    tools = [
        dspy.Tool(list_sections, name="list_sections", desc="List available extraction sections."),
        dspy.Tool(read_section, name="read_section", desc="Read section preview by name."),
    ]

    try:
        react = dspy.ReAct(_RouteSig, tools=tools, max_iters=4)
    except Exception as exc:
        return fallback_routes, {
            "planner": "deterministic_fallback",
            "react_used": False,
            "fallback_reason": f"react_init_failed: {type(exc).__name__}",
        }

    routes: list[dict[str, Any]] = []
    for sec in sections:
        section_text = str(sec.get("text", ""))
        hint = _section_complexity_hint(section_text)
        default_strategy = _default_strategy_for_hint(hint)
        try:
            pred = react(
                schema_name=schema_name,
                section_name=sec["name"],
                section_preview=section_text[:1200],
                complexity_hint=hint,
            )
            strategy = _normalize_route_strategy(
                getattr(pred, "strategy", ""),
                default=default_strategy,
            )
            reason = str(getattr(pred, "reason", "") or "").strip() or "react_planner"
        except Exception as exc:
            strategy = default_strategy
            reason = f"react_error:{type(exc).__name__}"

        routes.append(
            {
                "section": sec["title"],
                "name": sec["name"],
                "complexity": hint,
                "strategy": strategy,
                "reason": reason,
            }
        )

    return routes, {
        "planner": "react",
        "react_used": True,
        "fallback_reason": None,
    }


def _compose_routed_document_text(
    *,
    original_text: str,
    sections: list[dict[str, str]],
    routes: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Compose planned extraction context from section routes."""
    route_map = {
        str(route.get("name") or ""): str(route.get("strategy") or "heavy_extract")
        for route in routes
    }
    parts: list[str] = []
    summary_counts = {k: 0 for k in _ALLOWED_ROUTE_STRATEGIES}

    for sec in sections:
        body = str(sec.get("text", ""))
        if not body.strip():
            continue
        strategy = _normalize_route_strategy(
            route_map.get(sec["name"], "heavy_extract"),
            default="heavy_extract",
        )
        summary_counts[strategy] += 1

        if strategy == "deterministic":
            chunk = body[:500]
        elif strategy == "constrained_extract":
            chunk = body[:2000]
        else:
            chunk = body

        inline_match = _SECTION_HEADER_INLINE_PATTERN.match(chunk)
        if inline_match:
            chunk = chunk[inline_match.end() :]

        rendered = f"{sec['title']}:\n{chunk.strip()}".strip()
        if rendered:
            parts.append(rendered)

    planned = "\n\n".join(parts).strip()
    if not planned:
        planned = str(original_text or "")

    original_chars = len(str(original_text or ""))
    planned_chars = len(planned)
    compression_ratio = float(planned_chars / original_chars) if original_chars else 1.0

    return planned, {
        "strategy_counts": summary_counts,
        "original_chars": original_chars,
        "planned_chars": planned_chars,
        "compression_ratio": compression_ratio,
    }


def _plan_extraction_document_text(
    *,
    document_text: str,
    schema_name: str,
) -> tuple[str, dict[str, Any]]:
    """Plan extraction context using section routes, preferring DSPy ReAct."""
    sections = _split_document_sections(document_text)
    routes, planner_meta = _plan_section_routes_with_react(
        schema_name=schema_name,
        sections=sections,
    )
    planned_text, plan_stats = _compose_routed_document_text(
        original_text=document_text,
        sections=sections,
        routes=routes,
    )
    diagnostics = {
        **planner_meta,
        **plan_stats,
        "sections": [
            {
                "name": sec["name"],
                "title": sec["title"],
                "chars": len(sec.get("text", "")),
            }
            for sec in sections
        ],
        "routes": routes,
    }
    return planned_text, diagnostics


def _recover_schema_instance_with_outlines(
    *,
    document_text: str,
    schema_class: type[BaseModel],
    error_hint: str = "",
) -> BaseModel | None:
    """Recover a schema-mode extraction with Outlines constrained generation."""
    try:
        import openai
        import outlines

        from mosaicx.config import get_config
        from mosaicx.verify.parse_utils import parse_json_like

        cfg = get_config()
        base_url = _normalize_local_api_base(str(cfg.api_base or "http://127.0.0.1:8000/v1"))
        model_name = _normalize_model_name_for_openai_compatible(str(cfg.lm or ""))
        if not model_name:
            model_name = "mlx-community/gpt-oss-120b-4bit"

        client = openai.OpenAI(base_url=base_url, api_key=(cfg.api_key or "ollama"))
        model = outlines.from_openai(client, model_name=model_name)
        generator = outlines.Generator(model, outlines.json_schema(schema_class))
        prompt = (
            "Extract structured medical data from the document.\n"
            "Return only a JSON object matching the target schema.\n"
            "Rules:\n"
            "- Use null for unknown optional fields.\n"
            "- For optional lists, use [] or null, never the string \"None\".\n"
            "- Keep values grounded in document text.\n"
            f"Prior extraction error hint: {error_hint[:1200]}\n\n"
            f"Document text:\n{document_text[:30000]}"
        )
        raw = generator(prompt, temperature=0.0, max_tokens=2200)
        parsed = parse_json_like(raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False))
        if not isinstance(parsed, dict):
            return None
        coerced = _coerce_payload_to_schema(parsed, schema_class)
        return schema_class.model_validate(coerced)
    except Exception as exc:
        logger.warning("Outlines extraction recovery failed: %s", exc)
        return None


def _coerce_extracted_to_model_instance(
    *,
    extracted: Any,
    schema_class: type[BaseModel],
) -> BaseModel:
    """Coerce an arbitrary extraction payload into a validated schema instance."""
    if isinstance(extracted, schema_class):
        dumped = extracted.model_dump()
        if isinstance(dumped, dict):
            coerced = _coerce_payload_to_schema(dumped, schema_class)
            return schema_class.model_validate(coerced)
        return schema_class.model_validate(dumped)
    if isinstance(extracted, dict):
        coerced = _coerce_payload_to_schema(extracted, schema_class)
        return schema_class.model_validate(coerced)
    if hasattr(extracted, "model_dump"):
        dumped = extracted.model_dump()  # type: ignore[attr-defined]
        if isinstance(dumped, dict):
            coerced = _coerce_payload_to_schema(dumped, schema_class)
            return schema_class.model_validate(coerced)
        return schema_class.model_validate(dumped)
    return schema_class.model_validate(extracted)


def _flatten_scalar_values(payload: Any, *, prefix: str = "", out: list[tuple[str, Any]] | None = None) -> list[tuple[str, Any]]:
    if out is None:
        out = []
    if len(out) >= 256:
        return out
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_name = str(key)
            child = f"{prefix}.{key_name}" if prefix else key_name
            _flatten_scalar_values(value, prefix=child, out=out)
        return out
    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            child = f"{prefix}[{idx}]"
            _flatten_scalar_values(value, prefix=child, out=out)
            if len(out) >= 256:
                break
        return out
    out.append((prefix or "value", payload))
    return out


def _score_extraction_candidate(
    *,
    extracted: Any,
    schema_class: type[BaseModel],
    source_text: str,
) -> tuple[float, dict[str, float]]:
    """Compute deterministic reward components for candidate selection."""
    schema_compliance = 0.0
    evidence_overlap = 0.0
    critical_completeness = 0.0
    contradiction_penalty = 0.0
    null_overuse_penalty = 0.0

    try:
        model_instance = _coerce_extracted_to_model_instance(
            extracted=extracted,
            schema_class=schema_class,
        )
        normalized = model_instance.model_dump()
        schema_compliance = 1.0
    except Exception:
        return 0.0, {
            "schema_compliance": 0.0,
            "evidence_overlap": 0.0,
            "critical_completeness": 0.0,
            "contradiction_penalty": 1.0,
            "null_overuse_penalty": 1.0,
        }

    source_lower = str(source_text or "").lower()
    scalars = _flatten_scalar_values(normalized)
    observed: list[str] = []
    contradictions = 0
    for _, value in scalars:
        if _is_missing_value(value):
            continue
        if isinstance(value, bool):
            continue
        token = str(value).strip()
        if len(token) < 2:
            continue
        observed.append(token)
        probe = token.lower()
        if probe in source_lower:
            evidence_overlap += 1.0
        if f"no {probe}" in source_lower or f"without {probe}" in source_lower:
            contradictions += 1

    if observed:
        evidence_overlap = evidence_overlap / float(len(observed))
        contradiction_penalty = min(1.0, contradictions / float(len(observed)))
    else:
        evidence_overlap = 0.0
        contradiction_penalty = 0.0

    top_level_fields = list(schema_class.model_fields.keys())
    required_fields = [
        name for name, field in schema_class.model_fields.items()
        if bool(getattr(field, "is_required", lambda: False)())
    ]
    targets = required_fields or top_level_fields
    if targets:
        hit = 0
        for field_name in targets:
            if field_name in normalized and not _is_missing_value(normalized.get(field_name)):
                hit += 1
        critical_completeness = hit / float(len(targets))

    if top_level_fields:
        missing_top = sum(
            1 for field_name in top_level_fields
            if _is_missing_value(normalized.get(field_name))
        )
        missing_ratio = missing_top / float(len(top_level_fields))
        null_overuse_penalty = max(0.0, missing_ratio - 0.5)

    score = (
        0.35 * schema_compliance
        + 0.25 * evidence_overlap
        + 0.25 * critical_completeness
        + 0.15 * (1.0 - contradiction_penalty)
        - 0.25 * null_overuse_penalty
    )
    score = max(0.0, min(1.0, score))
    return score, {
        "schema_compliance": schema_compliance,
        "evidence_overlap": evidence_overlap,
        "critical_completeness": critical_completeness,
        "contradiction_penalty": contradiction_penalty,
        "null_overuse_penalty": null_overuse_penalty,
    }


def _planner_routes_uncertain(planner_diag: dict[str, Any] | None) -> bool:
    if not isinstance(planner_diag, dict):
        return False
    routes = planner_diag.get("routes")
    if not isinstance(routes, list):
        return False
    for route in routes:
        if not isinstance(route, dict):
            continue
        strategy = str(route.get("strategy") or "").strip().lower()
        if strategy in {"heavy_extract", "repair"}:
            return True
    return False


def _try_bestofn_for_uncertain_sections(
    *,
    document_text: str,
    schema_class: type[BaseModel],
    typed_extract: Any,
    planner_diag: dict[str, Any] | None,
) -> tuple[BaseModel | None, dict[str, Any]]:
    info: dict[str, Any] = {
        "triggered": False,
        "used": False,
        "score": None,
        "components": None,
        "reason": None,
    }

    if not _planner_routes_uncertain(planner_diag):
        info["reason"] = "no_uncertain_sections"
        return None, info

    info["triggered"] = True
    try:
        from mosaicx.runtime_env import import_dspy

        dspy = import_dspy()
    except Exception as exc:
        info["reason"] = f"dspy_import_failed:{type(exc).__name__}"
        return None, info

    if getattr(dspy.settings, "lm", None) is None:
        info["reason"] = "lm_not_configured"
        return None, info
    if not hasattr(dspy, "BestOfN"):
        info["reason"] = "bestofn_unavailable"
        return None, info

    class _TypedModule:
        def __call__(self, **kwargs: Any) -> Any:
            return typed_extract(**kwargs)

    def reward_fn(_args: dict[str, Any], pred: Any) -> float:
        extracted = getattr(pred, "extracted", pred)
        score, _components = _score_extraction_candidate(
            extracted=extracted,
            schema_class=schema_class,
            source_text=document_text,
        )
        return score

    n = max(2, int(os.environ.get("MOSAICX_EXTRACT_BESTOFN_N", "3") or "3"))
    threshold = float(os.environ.get("MOSAICX_EXTRACT_BESTOFN_THRESHOLD", "0.0") or "0.0")
    try:
        best = dspy.BestOfN(
            module=_TypedModule(),
            N=n,
            reward_fn=reward_fn,
            threshold=threshold,
        )
        pred = best(document_text=document_text)
        extracted = getattr(pred, "extracted", pred)
        model = _coerce_extracted_to_model_instance(
            extracted=extracted,
            schema_class=schema_class,
        )
        score, components = _score_extraction_candidate(
            extracted=model,
            schema_class=schema_class,
            source_text=document_text,
        )
        info["used"] = True
        info["score"] = score
        info["components"] = components
        info["reason"] = "bestofn_selected"
        return model, info
    except Exception as exc:
        info["reason"] = f"bestofn_failed:{type(exc).__name__}"
        return None, info


def _extract_schema_with_structured_chain(
    *,
    document_text: str,
    schema_class: type[BaseModel],
    typed_extract: Any,
    json_extract: Any | None = None,
    planner_diag: dict[str, Any] | None = None,
) -> tuple[BaseModel, dict[str, Any]]:
    """Run deterministic structured extraction fallback chain with diagnostics."""
    attempts: list[dict[str, Any]] = []
    bestofn_info: dict[str, Any] = {
        "triggered": False,
        "used": False,
        "score": None,
        "components": None,
        "reason": "not_evaluated",
    }

    def _record(step: str, ok: bool, error: Exception | None = None) -> None:
        row: dict[str, Any] = {"step": step, "ok": bool(ok)}
        if error is not None:
            row["error"] = f"{type(error).__name__}: {error}"
        attempts.append(row)

    selected_path = ""
    last_exc: Exception | None = None

    dspy = None
    has_lm = False
    try:
        from mosaicx.runtime_env import import_dspy

        dspy = import_dspy()
        has_lm = getattr(dspy.settings, "lm", None) is not None
    except Exception as exc:
        _record("dspy_import", False, exc)
        dspy = None
        has_lm = False

    if has_lm:
        outlines_primary = _recover_schema_instance_with_outlines(
            document_text=document_text,
            schema_class=schema_class,
            error_hint="primary_outlines",
        )
        if outlines_primary is not None:
            bestofn_info["reason"] = "skipped_outlines_primary_succeeded"
            _record("outlines_primary", True)
            selected_path = "outlines_primary"
            return outlines_primary, {
                "selected_path": selected_path,
                "fallback_used": False,
                "attempts": attempts,
                "bestofn": bestofn_info,
            }
        _record("outlines_primary", False, ValueError("outlines_primary_unavailable"))
    else:
        _record("outlines_primary", False, ValueError("lm_not_configured"))

    bestofn_model, bestofn_info = _try_bestofn_for_uncertain_sections(
        document_text=document_text,
        schema_class=schema_class,
        typed_extract=typed_extract,
        planner_diag=planner_diag,
    )
    if bestofn_model is not None:
        _record("bestofn_uncertain", True)
        selected_path = "bestofn_uncertain"
        return bestofn_model, {
            "selected_path": selected_path,
            "fallback_used": True,
            "attempts": attempts,
            "bestofn": bestofn_info,
        }
    if bestofn_info.get("triggered"):
        _record("bestofn_uncertain", False, ValueError(str(bestofn_info.get("reason") or "failed")))

    try:
        pred = typed_extract(document_text=document_text)
        extracted = getattr(pred, "extracted", pred)
        model_instance = _coerce_extracted_to_model_instance(
            extracted=extracted,
            schema_class=schema_class,
        )
        _record("dspy_typed_direct", True)
        selected_path = "dspy_typed_direct"
        return model_instance, {
            "selected_path": selected_path,
            "fallback_used": True,
            "attempts": attempts,
            "bestofn": bestofn_info,
        }
    except Exception as exc:
        last_exc = exc
        _record("dspy_typed_direct", False, exc)

    for step_name, adapter_name in (
        ("dspy_json_adapter", "JSONAdapter"),
        ("dspy_two_step_adapter", "TwoStepAdapter"),
    ):
        if dspy is None:
            _record(step_name, False, ValueError("dspy_unavailable"))
            continue
        if not has_lm:
            _record(step_name, False, ValueError("lm_not_configured"))
            continue

        adapter_cls = getattr(dspy, adapter_name, None)
        if adapter_cls is None:
            _record(step_name, False, ValueError(f"{adapter_name}_missing"))
            continue

        try:
            with dspy.context(adapter=adapter_cls()):
                pred = typed_extract(document_text=document_text)
            extracted = getattr(pred, "extracted", pred)
            model_instance = _coerce_extracted_to_model_instance(
                extracted=extracted,
                schema_class=schema_class,
            )
            _record(step_name, True)
            selected_path = step_name
            return model_instance, {
                "selected_path": selected_path,
                "fallback_used": True,
                "attempts": attempts,
                "bestofn": bestofn_info,
            }
        except Exception as exc:
            last_exc = exc
            _record(step_name, False, exc)

    if json_extract is not None:
        try:
            fallback_pred = json_extract(document_text=document_text)
            model_instance = _recover_schema_instance_from_raw(
                getattr(fallback_pred, "extracted_json", ""),
                schema_class,
            )
            _record("existing_json_fallback", True)
            selected_path = "existing_json_fallback"
            return model_instance, {
                "selected_path": selected_path,
                "fallback_used": True,
                "attempts": attempts,
                "bestofn": bestofn_info,
            }
        except Exception as exc:
            last_exc = exc
            _record("existing_json_fallback", False, exc)

    if has_lm:
        outlines_rescue = _recover_schema_instance_with_outlines(
            document_text=document_text,
            schema_class=schema_class,
            error_hint="; ".join(
                str(a.get("error", ""))
                for a in attempts
                if not a.get("ok")
            )[:1500],
        )
        if outlines_rescue is not None:
            _record("existing_outlines_rescue", True)
            selected_path = "existing_outlines_rescue"
            return outlines_rescue, {
                "selected_path": selected_path,
                "fallback_used": True,
                "attempts": attempts,
                "bestofn": bestofn_info,
            }
        _record("existing_outlines_rescue", False, ValueError("outlines_rescue_unavailable"))
    else:
        _record("existing_outlines_rescue", False, ValueError("lm_not_configured"))
    if last_exc is not None:
        raise last_exc
    raise ValueError("Structured extraction chain failed without recoverable path.")


def extract_with_mode(document_text: str, mode_name: str) -> tuple[dict[str, Any], "PipelineMetrics | None"]:
    """Run a registered extraction mode pipeline.

    Args:
        document_text: Full text of the document.
        mode_name: Name of the registered mode (e.g., "radiology", "pathology").

    Returns:
        Tuple of (output dict, PipelineMetrics or None).
    """
    from .modes import get_mode

    mode_cls = get_mode(mode_name)
    pipeline = mode_cls()
    result = pipeline(report_text=document_text)
    output = _prediction_to_dict(result)
    metrics = getattr(pipeline, "_last_metrics", None)
    return output, metrics


def extract_with_mode_raw(document_text: str, mode_name: str) -> tuple[dict[str, Any], "PipelineMetrics | None", Any]:
    """Run a registered extraction mode pipeline, returning the raw prediction.

    Like :func:`extract_with_mode` but also returns the raw ``dspy.Prediction``
    so callers can inspect the Pydantic model instances it contains (e.g. for
    completeness scoring).

    Returns:
        Tuple of (output dict, PipelineMetrics or None, raw dspy.Prediction).
    """
    from .modes import get_mode

    mode_cls = get_mode(mode_name)
    pipeline = mode_cls()
    result = pipeline(report_text=document_text)
    output = _prediction_to_dict(result)
    metrics = getattr(pipeline, "_last_metrics", None)
    return output, metrics, result


# ---------------------------------------------------------------------------
# DSPy Signatures & Module (lazy)
# ---------------------------------------------------------------------------


def _build_dspy_classes():
    """Lazily define and return DSPy signatures and the DocumentExtractor module."""
    import dspy

    from .schema_gen import SchemaSpec, compile_schema

    class InferSchemaFromDocument(dspy.Signature):
        """Infer a structured schema from a document's content.
        Analyze the document and determine what fields should be extracted."""

        document_text: str = dspy.InputField(
            desc="First portion of the document text to analyze"
        )
        schema_spec: SchemaSpec = dspy.OutputField(
            desc="Inferred schema specification describing what to extract"
        )

    class DocumentExtractor(dspy.Module):
        """DSPy Module for document extraction.

        Two modes:
        - **Auto mode** (no output_schema): infers schema from doc, then extracts.
        - **Schema mode** (output_schema provided): extracts directly into schema.
        """

        def __init__(self, output_schema: type[BaseModel] | None = None) -> None:
            super().__init__()
            self._output_schema = output_schema

            if output_schema is not None:
                # Schema mode: single extraction step
                custom_sig = dspy.Signature(
                    "document_text -> extracted",
                    instructions=(
                        f"Extract structured data matching the "
                        f"{output_schema.__name__} schema from the document."
                    ),
                ).with_updated_fields(
                    "document_text",
                    desc="Full text of the document",
                    type_=str,
                ).with_updated_fields(
                    "extracted",
                    desc=f"Extracted {output_schema.__name__} data",
                    type_=output_schema,
                )
                self.extract_custom = dspy.ChainOfThought(custom_sig)
                fallback_sig = dspy.Signature(
                    "document_text -> extracted_json",
                    instructions=(
                        f"Extract structured data matching the {output_schema.__name__} schema "
                        "from the document. Return ONLY a JSON object with no markdown and no commentary."
                    ),
                ).with_updated_fields(
                    "document_text",
                    desc="Full text of the document",
                    type_=str,
                ).with_updated_fields(
                    "extracted_json",
                    desc="Strict JSON object as text",
                    type_=str,
                )
                self.extract_json_fallback = dspy.Predict(fallback_sig)
            else:
                # Auto mode: infer schema then extract
                self.infer_schema = dspy.ChainOfThought(InferSchemaFromDocument)
                # extract_custom is created dynamically in forward()

        def forward(self, document_text: str) -> dspy.Prediction:
            """Run the extraction pipeline."""
            from mosaicx.metrics import PipelineMetrics, get_tracker, track_step

            metrics = PipelineMetrics()
            tracker = get_tracker()
            planner_diag: dict[str, Any] = {
                "planner": "deterministic_fallback",
                "react_used": False,
                "fallback_reason": "planner_not_run",
                "original_chars": len(str(document_text or "")),
                "planned_chars": len(str(document_text or "")),
                "compression_ratio": 1.0,
                "strategy_counts": {k: 0 for k in _ALLOWED_ROUTE_STRATEGIES},
                "sections": [],
                "routes": [],
                "full_text_rescue_used": False,
            }

            if hasattr(self, "extract_custom") and not hasattr(self, "infer_schema"):
                # Schema mode: single step
                schema = self._output_schema
                assert schema is not None

                planned_text = document_text
                with track_step(metrics, "Plan extraction", tracker):
                    try:
                        planned_text, planner_diag = _plan_extraction_document_text(
                            document_text=document_text,
                            schema_name=schema.__name__,
                        )
                    except Exception as plan_exc:
                        planner_diag = {
                            **planner_diag,
                            "fallback_reason": f"planner_error:{type(plan_exc).__name__}",
                        }
                        planned_text = document_text

                with track_step(metrics, "Extract", tracker):
                    try:
                        model_instance, chain_diag = _extract_schema_with_structured_chain(
                            document_text=planned_text,
                            schema_class=schema,
                            typed_extract=lambda *, document_text: self.extract_custom(
                                document_text=document_text
                            ),
                            json_extract=lambda *, document_text: self.extract_json_fallback(
                                document_text=document_text
                            ),
                            planner_diag=planner_diag,
                        )
                    except Exception:
                        if planned_text != document_text:
                            planner_diag["full_text_rescue_used"] = True
                            model_instance, chain_diag = _extract_schema_with_structured_chain(
                                document_text=document_text,
                                schema_class=schema,
                                typed_extract=lambda *, document_text: self.extract_custom(
                                    document_text=document_text
                                ),
                                json_extract=lambda *, document_text: self.extract_json_fallback(
                                    document_text=document_text
                                ),
                                planner_diag=planner_diag,
                            )
                        else:
                            raise
                planner_diag["planned_text_used"] = planned_text != document_text
                planner_diag["structured_chain"] = chain_diag.get("attempts", [])
                planner_diag["selected_structured_path"] = chain_diag.get("selected_path")
                planner_diag["structured_fallback_used"] = bool(
                    chain_diag.get("fallback_used", False)
                )
                planner_diag["bestofn"] = chain_diag.get("bestofn", {})
                self._last_metrics = metrics
                self._last_planner = planner_diag
                return dspy.Prediction(extracted=model_instance, planner=planner_diag)

            # Auto mode: infer schema, compile, extract
            with track_step(metrics, "Infer schema", tracker):
                infer_result = self.infer_schema(
                    document_text=document_text
                )
            spec: SchemaSpec = infer_result.schema_spec
            model = compile_schema(spec)

            # Build dynamic extraction signature
            extract_sig = dspy.Signature(
                "document_text -> extracted",
                instructions=(
                    f"Extract structured data matching the "
                    f"{model.__name__} schema from the document."
                ),
            ).with_updated_fields(
                "document_text",
                desc="Full text of the document",
                type_=str,
            ).with_updated_fields(
                "extracted",
                desc=f"Extracted {model.__name__} data",
                type_=model,
            )
            extract_step = dspy.ChainOfThought(extract_sig)
            extract_json_sig = dspy.Signature(
                "document_text -> extracted_json",
                instructions=(
                    f"Extract structured data matching the {model.__name__} schema "
                    "from the document. Return ONLY a JSON object with no markdown and no commentary."
                ),
            ).with_updated_fields(
                "document_text",
                desc="Full text of the document",
                type_=str,
            ).with_updated_fields(
                "extracted_json",
                desc="Strict JSON object as text",
                type_=str,
            )
            extract_json_step = dspy.Predict(extract_json_sig)
            planned_text = document_text
            with track_step(metrics, "Plan extraction", tracker):
                try:
                    planned_text, planner_diag = _plan_extraction_document_text(
                        document_text=document_text,
                        schema_name=model.__name__,
                    )
                except Exception as plan_exc:
                    planner_diag = {
                        **planner_diag,
                        "fallback_reason": f"planner_error:{type(plan_exc).__name__}",
                    }
                    planned_text = document_text
            with track_step(metrics, "Extract", tracker):
                try:
                    model_instance, chain_diag = _extract_schema_with_structured_chain(
                        document_text=planned_text,
                        schema_class=model,
                        typed_extract=lambda *, document_text: extract_step(
                            document_text=document_text
                        ),
                        json_extract=lambda *, document_text: extract_json_step(
                            document_text=document_text
                        ),
                        planner_diag=planner_diag,
                    )
                except Exception:
                    if planned_text != document_text:
                        planner_diag["full_text_rescue_used"] = True
                        model_instance, chain_diag = _extract_schema_with_structured_chain(
                            document_text=document_text,
                            schema_class=model,
                            typed_extract=lambda *, document_text: extract_step(
                                document_text=document_text
                            ),
                            json_extract=lambda *, document_text: extract_json_step(
                                document_text=document_text
                            ),
                            planner_diag=planner_diag,
                        )
                    else:
                        raise
            planner_diag["planned_text_used"] = planned_text != document_text
            planner_diag["structured_chain"] = chain_diag.get("attempts", [])
            planner_diag["selected_structured_path"] = chain_diag.get("selected_path")
            planner_diag["structured_fallback_used"] = bool(
                chain_diag.get("fallback_used", False)
            )
            planner_diag["bestofn"] = chain_diag.get("bestofn", {})

            self._last_metrics = metrics
            self._last_planner = planner_diag

            return dspy.Prediction(
                extracted=model_instance,
                inferred_schema=spec,
                planner=planner_diag,
            )

    return {
        "InferSchemaFromDocument": InferSchemaFromDocument,
        "DocumentExtractor": DocumentExtractor,
    }


# Cache for lazily-built DSPy classes
_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({
    "InferSchemaFromDocument",
    "DocumentExtractor",
})


def __getattr__(name: str):
    """Module-level __getattr__ for lazy loading of DSPy classes."""
    global _dspy_classes

    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
