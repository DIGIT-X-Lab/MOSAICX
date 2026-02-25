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

            if hasattr(self, "extract_custom") and not hasattr(self, "infer_schema"):
                # Schema mode: single step
                schema = self._output_schema
                assert schema is not None

                with track_step(metrics, "Extract", tracker):
                    try:
                        result = self.extract_custom(document_text=document_text)
                        extracted = getattr(result, "extracted", result)
                        if isinstance(extracted, schema):
                            extracted_payload: Any = extracted.model_dump()
                        elif isinstance(extracted, dict):
                            extracted_payload = extracted
                        elif hasattr(extracted, "model_dump"):
                            extracted_payload = extracted.model_dump()
                        else:
                            extracted_payload = extracted

                        if isinstance(extracted_payload, dict):
                            coerced = _coerce_payload_to_schema(extracted_payload, schema)
                            model_instance = schema.model_validate(coerced)
                        else:
                            model_instance = schema.model_validate(extracted_payload)
                    except Exception as primary_exc:
                        try:
                            fallback = self.extract_json_fallback(document_text=document_text)
                            model_instance = _recover_schema_instance_from_raw(
                                getattr(fallback, "extracted_json", ""),
                                schema,
                            )
                        except Exception as fallback_exc:
                            recovered = _recover_schema_instance_with_outlines(
                                document_text=document_text,
                                schema_class=schema,
                                error_hint=f"primary={primary_exc}; fallback={fallback_exc}",
                            )
                            if recovered is None:
                                raise fallback_exc from primary_exc
                            model_instance = recovered
                self._last_metrics = metrics
                return dspy.Prediction(extracted=model_instance)

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
            with track_step(metrics, "Extract", tracker):
                result = extract_step(document_text=document_text)

            self._last_metrics = metrics

            return dspy.Prediction(
                extracted=result.extracted,
                inferred_schema=spec,
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
