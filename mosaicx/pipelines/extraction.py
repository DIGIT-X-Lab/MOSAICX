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
from difflib import SequenceMatcher
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, Field

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
    return val.model_dump(mode="json") if hasattr(val, "model_dump") else val


def _prediction_to_dict(prediction: Any) -> dict[str, Any]:
    """Convert a dspy.Prediction to a plain dict, serialising Pydantic models."""
    output: dict[str, Any] = {}
    for key in prediction.keys():
        val = getattr(prediction, key)
        if hasattr(val, "model_dump"):
            output[key] = val.model_dump(mode="json")
        elif isinstance(val, list):
            output[key] = [v.model_dump(mode="json") if hasattr(v, "model_dump") else v for v in val]
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
        # For required primitive types, return the zero/empty value so that
        # schema_class.model_validate() doesn't reject None when the source
        # document genuinely doesn't contain the value.  "" is the canonical
        # missing sentinel for str; 0/0.0/False signal absent numeric fields.
        _PRIMITIVE_ZEROS: dict[Any, Any] = {str: "", int: 0, float: 0.0, bool: False}
        if annotation in _PRIMITIVE_ZEROS:
            return _PRIMITIVE_ZEROS[annotation]
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
        return payload.model_dump(mode="json")
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

    # Normalize whitespace so PDF \r\n line breaks don't break substring matching
    needle_norm = " ".join(needle.split())
    haystack_norm = " ".join(haystack.split())

    idx = haystack_norm.lower().find(needle_norm.lower())
    if idx < 0:
        if not isinstance(value, str):
            return False, None
        needle_tokens = {
            tok
            for tok in re.findall(r"[a-z0-9]+", needle_norm.lower())
            if len(tok) >= 3
        }
        if len(needle_tokens) < 4:
            return False, None

        segments = [
            seg.strip()
            for seg in re.split(r"(?<=[.!?])\s+|\n+", haystack_norm)
            if seg and seg.strip()
        ]
        if not segments:
            segments = [haystack_norm]

        best_ratio = 0.0
        best_snippet = None
        for i in range(len(segments)):
            for width in (1, 2, 3):
                chunk = " ".join(segments[i : i + width]).strip()
                if not chunk:
                    continue
                chunk_tokens = {
                    tok
                    for tok in re.findall(r"[a-z0-9]+", chunk.lower())
                    if len(tok) >= 3
                }
                if not chunk_tokens:
                    continue
                overlap = len(needle_tokens & chunk_tokens) / float(len(needle_tokens))
                if overlap > best_ratio:
                    best_ratio = overlap
                    best_snippet = chunk

        if best_snippet is None or best_ratio < 0.45:
            return False, None

        if len(best_snippet) > 420:
            best_snippet = best_snippet[:417].rstrip() + "..."

        # High-overlap fuzzy match counts as grounded despite punctuation/linebreak/OCR drift.
        if best_ratio >= 0.85:
            return True, best_snippet
        return False, best_snippet

    start = max(0, idx - 80)
    end = min(len(haystack_norm), idx + len(needle_norm) + 80)
    return True, haystack_norm[start:end]


_DATE_FIELD_HINTS = {
    "date",
    "dob",
    "birth",
    "birthday",
    "admission",
    "discharge",
    "study_date",
    "exam_date",
}
_RANGE_FIELD_HINTS = {
    "range",
    "interval",
    "window",
    "duration",
}
_UNIT_FIELD_HINTS = {
    "height",
    "weight",
    "bmi",
    "blood_glucose",
    "uptake_time",
    "size",
    "diameter",
    "volume",
    "dose",
    "activity",
    "suv",
    "hu",
    "pressure",
    "bp",
}
_NUMERIC_FIELD_HINTS = {
    "age",
    "score",
    "count",
    "number",
    "size",
    "diameter",
    "volume",
    "weight",
    "height",
    "bmi",
    "suv",
    "hu",
}
_CRITICAL_FIELD_HINTS = {
    "impression",
    "finding",
    "findings",
    "diagnosis",
    "assessment",
    "procedure",
    "exam_date",
    "date_of_birth",
    "dob",
    "patient_name",
}
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
_RANGE_PATTERN = re.compile(
    r"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:-|to|–|—)\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z%/]+)?\s*$",
    flags=re.IGNORECASE,
)
_VALUE_UNIT_PATTERN = re.compile(
    r"^\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z%/]+)?\s*$",
    flags=re.IGNORECASE,
)
_ALLOWED_UNITS = {
    "mm",
    "cm",
    "m",
    "kg",
    "g",
    "mg",
    "ug",
    "lb",
    "lbs",
    "mmhg",
    "s",
    "sec",
    "seconds",
    "min",
    "mins",
    "minute",
    "minutes",
    "h",
    "hr",
    "hrs",
    "hour",
    "hours",
    "ml",
    "l",
    "%",
    "mg/dl",
    "g/dl",
}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%m/%d/%y",
    "%d/%m/%y",
    "%b %d %Y",
    "%B %d %Y",
)


def _field_name_tokens(field_name: str) -> set[str]:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(field_name or "").strip().lower())
    tokens = {tok for tok in normalized.split("_") if tok}
    if normalized:
        tokens.add(normalized)
    return tokens


def _is_critical_field_name(field_name: str) -> bool:
    tokens = _field_name_tokens(field_name)
    if tokens & _CRITICAL_FIELD_HINTS:
        return True
    joined = "_".join(sorted(tokens))
    return joined in _CRITICAL_FIELD_HINTS


def _format_compact_number(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _normalize_unit_token(unit: str) -> str:
    token = str(unit or "").strip().lower()
    aliases = {
        "millimeter": "mm",
        "millimeters": "mm",
        "centimeter": "cm",
        "centimeters": "cm",
        "milliliter": "ml",
        "milliliters": "ml",
        "liter": "l",
        "liters": "l",
        "kgs": "kg",
        "grams": "g",
        "milligrams": "mg",
        "micrograms": "ug",
        "mm hg": "mmhg",
    }
    return aliases.get(token, token)


def _looks_like_date_field(field_name: str) -> bool:
    tokens = _field_name_tokens(field_name)
    if "date" in tokens:
        return True
    return bool(tokens & _DATE_FIELD_HINTS)


def _looks_like_range_field(field_name: str) -> bool:
    tokens = _field_name_tokens(field_name)
    return bool(tokens & _RANGE_FIELD_HINTS)


def _looks_like_unit_field(field_name: str) -> bool:
    tokens = _field_name_tokens(field_name)
    return bool(tokens & _UNIT_FIELD_HINTS)


def _looks_like_numeric_field(field_name: str) -> bool:
    tokens = _field_name_tokens(field_name)
    return bool(tokens & _NUMERIC_FIELD_HINTS)


def _try_parse_date_value(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()

    text = str(value or "").strip()
    if not text:
        return None

    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_match:
        return iso_match.group(1)

    cleaned = text.replace(",", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.date().isoformat()
        except Exception:
            continue
    return None


def _try_parse_numeric_range(text: str) -> tuple[float, float, str | None] | None:
    match = _RANGE_PATTERN.match(str(text or ""))
    if not match:
        return None
    low = float(match.group(1))
    high = float(match.group(2))
    unit = match.group(3)
    normalized_unit = _normalize_unit_token(unit) if unit else None
    return low, high, normalized_unit


def _try_parse_numeric_unit(text: str) -> tuple[float, str | None] | None:
    match = _VALUE_UNIT_PATTERN.match(str(text or ""))
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    normalized_unit = _normalize_unit_token(unit) if unit else None
    return value, normalized_unit


def _validate_field_semantics(field_name: str, value: Any) -> dict[str, Any]:
    """Deterministically validate/normalize field semantics."""
    tokens = _field_name_tokens(field_name)
    critical = _is_critical_field_name(field_name)
    if _is_missing_value(value):
        return {
            "valid": False,
            "kind": "null",
            "critical": critical,
            "reason": "missing_value",
            "normalized_value": None,
        }

    if isinstance(value, (bool, int, float)):
        return {
            "valid": True,
            "kind": "scalar",
            "critical": critical,
            "reason": None,
            "normalized_value": value,
        }

    if isinstance(value, (list, tuple, set, dict)):
        return {
            "valid": True,
            "kind": "container",
            "critical": critical,
            "reason": None,
            "normalized_value": value,
        }

    raw = str(value).strip()
    if _is_nullish_string(raw):
        return {
            "valid": False,
            "kind": "null",
            "critical": critical,
            "reason": "nullish_string",
            "normalized_value": None,
        }

    if ("bp" in tokens or "pressure" in tokens) and re.match(
        r"^\s*\d{2,3}\s*/\s*\d{2,3}\s*$",
        raw,
    ):
        normalized_bp = "/".join(part.strip() for part in raw.split("/", 1))
        return {
            "valid": True,
            "kind": "blood_pressure",
            "critical": critical,
            "reason": None,
            "normalized_value": normalized_bp,
        }

    if _looks_like_date_field(field_name):
        parsed = _try_parse_date_value(raw)
        if parsed is None:
            return {
                "valid": False,
                "kind": "date",
                "critical": critical,
                "reason": "invalid_date_format",
                "normalized_value": raw,
            }
        return {
            "valid": True,
            "kind": "date",
            "critical": critical,
            "reason": None,
            "normalized_value": parsed,
        }

    range_candidate = _try_parse_numeric_range(raw)
    if _looks_like_range_field(field_name) or range_candidate is not None:
        if range_candidate is None:
            return {
                "valid": False,
                "kind": "range",
                "critical": critical,
                "reason": "invalid_range_format",
                "normalized_value": raw,
            }
        low, high, unit = range_candidate
        if low > high:
            return {
                "valid": False,
                "kind": "range",
                "critical": critical,
                "reason": "invalid_range_order",
                "normalized_value": raw,
            }
        normalized = f"{_format_compact_number(low)}-{_format_compact_number(high)}"
        if unit:
            normalized = f"{normalized} {unit}"
        return {
            "valid": True,
            "kind": "range",
            "critical": critical,
            "reason": None,
            "normalized_value": normalized,
        }

    numeric_unit = _try_parse_numeric_unit(raw)
    if _looks_like_unit_field(field_name):
        if numeric_unit is None:
            return {
                "valid": False,
                "kind": "unit",
                "critical": critical,
                "reason": "invalid_numeric_unit",
                "normalized_value": raw,
            }
        numeric, unit = numeric_unit
        if unit and unit not in _ALLOWED_UNITS:
            return {
                "valid": False,
                "kind": "unit",
                "critical": critical,
                "reason": "unknown_unit",
                "normalized_value": raw,
            }
        normalized = _format_compact_number(numeric)
        if unit:
            normalized = f"{normalized} {unit}"
        return {
            "valid": True,
            "kind": "unit",
            "critical": critical,
            "reason": None,
            "normalized_value": normalized,
        }

    if _looks_like_numeric_field(field_name):
        if _NUMBER_PATTERN.search(raw) is None:
            return {
                "valid": False,
                "kind": "numeric",
                "critical": critical,
                "reason": "missing_numeric_content",
                "normalized_value": raw,
            }

    return {
        "valid": True,
        "kind": "text",
        "critical": critical,
        "reason": None,
        "normalized_value": raw,
    }


def _annotation_allows_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return any(opt is type(None) for opt in get_args(annotation))
    return False


def _coerce_enum_values(
    payload: dict[str, Any],
    schema_class: type[BaseModel],
) -> tuple[dict[str, Any], list[str]]:
    """Fuzzy-match enum field values to their closest valid option.

    Walks the schema recursively and for each enum/Literal field, if the
    current value isn't a valid option, attempts substring and token
    matching to find the closest one.  Returns the updated payload and
    a list of coerced field names.
    """
    import enum as _enum
    from typing import get_args, get_origin

    coerced: list[str] = []

    def _get_enum_values(annotation: Any) -> list[str] | None:
        """Extract allowed string values from an enum or Literal annotation."""
        origin = get_origin(annotation)
        # Unwrap Optional
        if origin in (Union, types.UnionType):
            for arg in get_args(annotation):
                if arg is type(None):
                    continue
                vals = _get_enum_values(arg)
                if vals is not None:
                    return vals
            return None
        # Literal["a", "b", ...]
        if origin is Literal:
            return [str(v) for v in get_args(annotation)]
        # Python enum
        if isinstance(annotation, type) and issubclass(annotation, _enum.Enum):
            return [str(m.value) for m in annotation]
        return None

    def _fuzzy_match(value: str, options: list[str]) -> str | None:
        """Find the best matching option for a value."""
        val_lower = value.strip().lower()
        # Exact match (case-insensitive)
        for opt in options:
            if opt.lower() == val_lower:
                return opt
        # Substring: value contains an option or option contains value
        for opt in options:
            opt_lower = opt.lower()
            if opt_lower in val_lower or val_lower in opt_lower:
                return opt
        # Token overlap: split both into words, find best overlap
        val_tokens = set(val_lower.split())
        best_opt, best_score = None, 0
        for opt in options:
            opt_tokens = set(opt.lower().split())
            overlap = len(val_tokens & opt_tokens)
            if overlap > best_score:
                best_score = overlap
                best_opt = opt
        if best_score > 0:
            return best_opt
        return None

    def _coerce_dict(d: dict[str, Any], model: type[BaseModel], prefix: str = "") -> None:
        for field_name, field_info in model.model_fields.items():
            if field_name not in d:
                continue
            value = d[field_name]
            annotation = field_info.annotation
            path = f"{prefix}.{field_name}" if prefix else field_name

            # Check if this field has enum values
            enum_vals = _get_enum_values(annotation)
            if enum_vals is not None and isinstance(value, str):
                if value not in enum_vals:
                    matched = _fuzzy_match(value, enum_vals)
                    if matched is not None:
                        d[field_name] = matched
                        coerced.append(f"{path}: {value!r} -> {matched!r}")
                continue

            # Recurse into nested objects
            origin = get_origin(annotation)
            if origin in (Union, types.UnionType):
                inner_args = [a for a in get_args(annotation) if a is not type(None)]
                if inner_args:
                    annotation = inner_args[0]
                    origin = get_origin(annotation)

            if isinstance(value, dict) and isinstance(annotation, type) and issubclass(annotation, BaseModel):
                _coerce_dict(value, annotation, path)
            elif isinstance(value, list) and origin is list:
                item_args = get_args(annotation)
                if item_args:
                    item_type = item_args[0]
                    if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                _coerce_dict(item, item_type, f"{path}.{i}")

            # Coerce float-as-string to int where schema expects int
            if isinstance(value, (float, str)) and annotation is int:
                try:
                    d[field_name] = int(round(float(value)))
                    if str(value) != str(d[field_name]):
                        coerced.append(f"{path}: {value!r} -> {d[field_name]!r}")
                except (ValueError, TypeError):
                    pass

    updated = dict(payload)
    _coerce_dict(updated, schema_class)
    return updated, coerced


def _apply_deterministic_semantic_validation(
    *,
    model_instance: BaseModel,
    schema_class: type[BaseModel],
) -> tuple[BaseModel, dict[str, Any]]:
    """Normalize model payload with deterministic semantic validators."""
    # First coerce enum values to closest valid option
    raw_payload = model_instance.model_dump(mode="json")
    payload, coerced_fields = _coerce_enum_values(raw_payload, schema_class)
    if coerced_fields:
        try:
            model_instance = schema_class.model_validate(payload)
        except Exception:
            payload = raw_payload  # revert if coercion broke something
    updated = dict(payload)
    issues: list[dict[str, Any]] = []
    changed_fields: list[str] = []
    invalid_fields: list[str] = []

    for field_name, field_info in schema_class.model_fields.items():
        if field_name not in updated:
            continue
        current = updated.get(field_name)
        verdict = _validate_field_semantics(field_name, current)
        normalized = verdict.get("normalized_value", current)
        valid = bool(verdict.get("valid", True))
        if valid:
            if normalized != current:
                changed_fields.append(field_name)
            updated[field_name] = normalized
            continue

        invalid_fields.append(field_name)
        allows_none = _annotation_allows_none(field_info.annotation)
        downgraded = None if allows_none else current
        updated[field_name] = downgraded
        issues.append(
            {
                "field": field_name,
                "kind": verdict.get("kind"),
                "reason": verdict.get("reason"),
                "critical": bool(verdict.get("critical", False)),
                "original_value": current,
                "normalized_value": downgraded,
                "severity": "error" if bool(verdict.get("critical", False)) else "warning",
            }
        )

    try:
        coerced = _coerce_payload_to_schema(updated, schema_class)
        normalized_model = schema_class.model_validate(coerced)
    except Exception as exc:
        return model_instance, {
            "applied": False,
            "reason": f"validation_model_rebuild_failed:{type(exc).__name__}",
            "issues": issues,
            "changed_fields": changed_fields,
            "invalid_fields": invalid_fields,
        }

    return normalized_model, {
        "applied": True,
        "reason": "semantic_validation_applied",
        "issues": issues,
        "changed_fields": changed_fields,
        "invalid_fields": invalid_fields,
    }


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
    missing_required: list[str] = []
    counts = {
        "supported": 0,
        "needs_review": 0,
        "insufficient_evidence": 0,
    }
    validation_issues: list[dict[str, Any]] = []

    for field in fields:
        original_value = target.get(field)
        validation = _validate_field_semantics(field, original_value)
        value = validation.get("normalized_value", original_value)
        target[field] = value

        if not validation.get("valid", True):
            reason = str(validation.get("reason") or "invalid_value")
            is_critical = bool(validation.get("critical"))
            is_missing = reason in {"missing_value", "nullish_string"}
            if is_missing:
                status = "insufficient_evidence"
                if field not in missing_required:
                    missing_required.append(field)
            else:
                status = "insufficient_evidence" if is_critical else "needs_review"
            grounded = False
            confidence = 0.0 if status == "insufficient_evidence" else 0.25
            evidence = None
            validation_issues.append(
                {
                    "field": field,
                    "kind": validation.get("kind"),
                    "reason": reason,
                    "critical": is_critical,
                    "original_value": original_value,
                    "normalized_value": value,
                    "severity": "error" if is_critical else "warning",
                }
            )
        elif _is_missing_value(value):
            status = "insufficient_evidence"
            grounded: bool | None = False
            confidence = 0.0
            evidence = None
            if field not in missing_required:
                missing_required.append(field)
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
                "validation": {
                    "valid": bool(validation.get("valid", True)),
                    "kind": validation.get("kind"),
                    "reason": validation.get("reason"),
                    "critical": bool(validation.get("critical", False)),
                },
            }
        )

    output_data["_extraction_contract"] = {
        "version": "1.0",
        "critical_fields": fields,
        "required_field_count": len(fields),
        "present_required_count": max(0, len(fields) - len(missing_required)),
        "missing_required": missing_required,
        "field_results": field_results,
        "summary": counts,
        "validation_issues": validation_issues,
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

_LABELLED_LINE_PATTERN = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9()/%&,\-'\s]{1,96}?)\s*:\s*(.*)$"
)

_MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "without",
}

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


def _tokenize_for_match(text: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if token and token not in _MATCH_STOPWORDS
    }
    return tokens


def _field_to_label_match_score(field_name: str, label: str) -> float:
    field_tokens = _tokenize_for_match(field_name)
    label_tokens = _tokenize_for_match(label)
    if not field_tokens or not label_tokens:
        return 0.0

    overlap = len(field_tokens & label_tokens)
    if overlap <= 0:
        return 0.0

    containment = overlap / float(len(field_tokens))
    union = len(field_tokens | label_tokens)
    jaccard = overlap / float(union) if union else 0.0

    normalized_field = "_".join(sorted(field_tokens))
    normalized_label = "_".join(sorted(label_tokens))
    similarity = SequenceMatcher(None, normalized_field, normalized_label).ratio()

    score = max((0.70 * containment) + (0.30 * jaccard), similarity * 0.80)
    if normalized_field == normalized_label:
        score = 1.0
    elif normalized_field in normalized_label or normalized_label in normalized_field:
        score = max(score, 0.92)
    return max(0.0, min(1.0, score))


def _extract_labeled_blocks(document_text: str, *, max_blocks: int = 256) -> list[dict[str, str]]:
    text = str(document_text or "")
    if not text.strip():
        return []

    blocks: list[dict[str, str]] = []
    current_label: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_label, current_lines
        if not current_label:
            current_lines = []
            return
        merged = " ".join(chunk.strip() for chunk in current_lines if chunk.strip()).strip()
        if merged and not _is_nullish_string(merged):
            blocks.append({"label": current_label, "text": merged})
        current_label = None
        current_lines = []

    for raw_line in text.splitlines():
        line = str(raw_line or "")
        stripped = line.strip()

        match = _LABELLED_LINE_PATTERN.match(stripped)
        if match:
            _flush()
            label = str(match.group(1) or "").strip()
            if not label:
                continue
            current_label = label
            initial_value = str(match.group(2) or "").strip()
            if initial_value:
                current_lines.append(initial_value)
            if len(blocks) >= max_blocks:
                break
            continue

        if current_label is None:
            continue

        if stripped:
            current_lines.append(stripped)

    _flush()
    return blocks[:max_blocks]


def _deterministic_backfill_for_field(
    *,
    source_text: str,
    field_name: str,
) -> tuple[str | None, dict[str, Any]]:
    diag: dict[str, Any] = {
        "method": None,
        "score": 0.0,
        "label": None,
    }

    text = str(source_text or "")
    if not text.strip():
        return None, diag

    candidates: list[tuple[float, str, str, str]] = []
    for block in _extract_labeled_blocks(text):
        label = str(block.get("label") or "")
        value = str(block.get("text") or "").strip()
        if not value:
            continue
        score = _field_to_label_match_score(field_name, label)
        if score >= 0.72:
            candidates.append((score, value, label, "label_block"))

    for section in _split_document_sections(text, max_sections=64):
        title = str(section.get("title") or "").strip()
        body = str(section.get("text") or "").strip()
        if not title or not body:
            continue
        score = _field_to_label_match_score(field_name, title)
        if score >= 0.84:
            candidates.append((score, body, title, "section_block"))

    if not candidates:
        return None, diag

    score, value, label, method = max(candidates, key=lambda row: row[0])
    if _is_nullish_string(value):
        return None, diag

    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        return None, diag

    diag["method"] = method
    diag["score"] = float(score)
    diag["label"] = label
    return cleaned, diag


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

    from mosaicx.config import get_config
    min_chars = get_config().planner_min_chars
    if len(document_text) < min_chars:
        # Deterministic routing for short documents — skip ReAct LLM calls
        routes = [
            {
                "section": sec["title"],
                "name": sec["name"],
                "complexity": _section_complexity_hint(sec.get("text", "")),
                "strategy": _default_strategy_for_hint(
                    _section_complexity_hint(sec.get("text", ""))
                ),
                "reason": "short_doc_bypass",
            }
            for sec in sections
        ]
        planner_meta = {
            "planner": "short_doc_bypass",
            "react_used": False,
            "fallback_reason": None,
        }
    else:
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
        import concurrent.futures

        timeout = cfg.outlines_timeout
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(generator, prompt, temperature=0.0, max_tokens=2200)
            raw = future.result(timeout=timeout)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        # Report estimated token usage — Outlines bypasses DSPy's tracker.
        raw_str = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        try:
            from mosaicx.metrics import get_tracker

            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(raw_str) // 4)
            get_tracker().add_usage("outlines", {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            })
        except Exception:
            pass

        parsed = parse_json_like(raw_str)
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
        dumped = extracted.model_dump(mode="json")
        if isinstance(dumped, dict):
            coerced = _coerce_payload_to_schema(dumped, schema_class)
            return schema_class.model_validate(coerced)
        return schema_class.model_validate(dumped)
    if isinstance(extracted, dict):
        coerced = _coerce_payload_to_schema(extracted, schema_class)
        return schema_class.model_validate(coerced)
    if hasattr(extracted, "model_dump"):
        dumped = extracted.model_dump(mode="json")  # type: ignore[attr-defined]
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
        normalized = model_instance.model_dump(mode="json")
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


def _adjudicate_conflicting_candidates(
    *,
    candidates: list[dict[str, Any]],
    source_text: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Adjudicate conflicting candidate outputs, preferring MCC when available."""
    diag: dict[str, Any] = {
        "triggered": bool(candidates),
        "conflict_detected": False,
        "method": "score",
        "chosen_path": None,
        "rationale": "",
        "candidates": [],
    }
    if not candidates:
        return None, diag

    for cand in candidates:
        diag["candidates"].append(
            {
                "path": cand.get("path"),
                "score": cand.get("score"),
                "components": cand.get("components"),
            }
        )

    dumps = {
        json.dumps(c.get("model").model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        for c in candidates
        if c.get("model") is not None
    }
    diag["conflict_detected"] = len(dumps) > 1

    chosen = max(candidates, key=lambda c: float(c.get("score") or 0.0))
    diag["chosen_path"] = chosen.get("path")
    diag["rationale"] = "highest_score"

    if diag["conflict_detected"]:
        try:
            from mosaicx.runtime_env import import_dspy

            dspy = import_dspy()
            if getattr(dspy.settings, "lm", None) is not None and hasattr(dspy, "MultiChainComparison"):
                completions = [
                    {
                        "chosen_path": str(c.get("path") or ""),
                        "rationale": (
                            f"score={float(c.get('score') or 0.0):.3f}, "
                            f"evidence_overlap={float((c.get('components') or {}).get('evidence_overlap', 0.0)):.3f}, "
                            f"null_penalty={float((c.get('components') or {}).get('null_overuse_penalty', 0.0)):.3f}"
                        ),
                    }
                    for c in candidates
                ]
                mcc = dspy.MultiChainComparison(
                    "source_text -> chosen_path, rationale",
                    M=max(2, len(completions)),
                    temperature=0.2,
                )
                pred = mcc(
                    completions=completions,
                    source_text=str(source_text or "")[:4000],
                )
                mcc_path = str(getattr(pred, "chosen_path", "") or "").strip()
                if mcc_path:
                    for cand in candidates:
                        if str(cand.get("path") or "") == mcc_path:
                            chosen = cand
                            diag["method"] = "mcc"
                            diag["chosen_path"] = mcc_path
                            diag["rationale"] = str(getattr(pred, "rationale", "") or "mcc")
                            break
        except Exception:
            pass

    return chosen, diag


def _repair_failed_critical_fields_with_refine(
    *,
    model_instance: BaseModel,
    schema_class: type[BaseModel],
    source_text: str,
    think: str = "standard",
) -> tuple[BaseModel, dict[str, Any]]:
    """Repair only failed critical fields using DSPy Refine when available."""
    payload = model_instance.model_dump(mode="json")
    required = [
        name for name, field in schema_class.model_fields.items()
        if bool(getattr(field, "is_required", lambda: False)())
    ]
    critical_fields = required or list(schema_class.model_fields.keys())

    failed_fields: list[str] = []
    missing_fields: list[str] = []
    ungrounded_fields: list[str] = []
    for field_name in critical_fields:
        value = payload.get(field_name)
        grounded, _snippet = _extract_grounding_snippet(source_text=source_text, value=value)
        if _is_missing_value(value):
            failed_fields.append(field_name)
            missing_fields.append(field_name)
            continue
        if grounded is False:
            failed_fields.append(field_name)
            ungrounded_fields.append(field_name)

    diag: dict[str, Any] = {
        "triggered": bool(failed_fields),
        "failed_fields": list(failed_fields),
        "missing_fields": list(missing_fields),
        "ungrounded_fields": list(ungrounded_fields),
        "remaining_failed_fields": list(failed_fields),
        "repaired_fields": [],
        "skipped_fields": [],
        "reason": None,
    }
    if not failed_fields:
        diag["reason"] = "no_failed_fields"
        return model_instance, diag

    updated = model_instance
    updated_payload = payload

    # Fast deterministic backfill from section/label blocks before any extra LLM calls.
    remaining_failed_fields: list[str] = []
    for field_name in failed_fields:
        before = updated_payload.get(field_name)
        backfilled_value, backfill_diag = _deterministic_backfill_for_field(
            source_text=source_text,
            field_name=field_name,
        )
        if backfilled_value is None:
            remaining_failed_fields.append(field_name)
            continue

        try:
            trial_payload = dict(updated_payload)
            trial_payload[field_name] = backfilled_value
            coerced = _coerce_payload_to_schema(trial_payload, schema_class)
            repaired_model = schema_class.model_validate(coerced)
            repaired_payload = repaired_model.model_dump()
            after = repaired_payload.get(field_name)
            if _is_missing_value(after):
                remaining_failed_fields.append(field_name)
                diag["skipped_fields"].append(
                    {
                        "field": field_name,
                        "reason": "deterministic_backfill_still_missing",
                    }
                )
                continue
            updated = repaired_model
            updated_payload = repaired_payload
            diag["repaired_fields"].append(
                {
                    "field": field_name,
                    "before": before,
                    "after": after,
                    "method": str(backfill_diag.get("method") or "deterministic_backfill"),
                    "score": backfill_diag.get("score"),
                    "label": backfill_diag.get("label"),
                }
            )
        except Exception as exc:
            remaining_failed_fields.append(field_name)
            diag["skipped_fields"].append(
                {
                    "field": field_name,
                    "reason": f"deterministic_backfill_error:{type(exc).__name__}",
                }
            )

    diag["remaining_failed_fields"] = list(remaining_failed_fields)
    if not remaining_failed_fields:
        diag["reason"] = "deterministic_backfill_applied"
        return updated, diag

    # think=fast always disables Refine; think=deep always enables it;
    # think=standard defers to config
    if think == "fast":
        diag["reason"] = "use_refine_disabled"
        return updated, diag
    from mosaicx.config import get_config
    cfg = get_config()
    if think != "deep" and not cfg.use_refine:
        diag["reason"] = "use_refine_disabled"
        return updated, diag

    refine_candidates = list(remaining_failed_fields)
    if bool(getattr(cfg, "refine_only_missing", True)):
        refine_candidates = [
            name for name in refine_candidates if _is_missing_value(updated_payload.get(name))
        ]

    max_refine_fields = max(0, int(getattr(cfg, "refine_max_fields", 3) or 0))
    if max_refine_fields > 0 and len(refine_candidates) > max_refine_fields:
        overflow = refine_candidates[max_refine_fields:]
        for field_name in overflow:
            diag["skipped_fields"].append(
                {
                    "field": field_name,
                    "reason": f"refine_field_limit:{max_refine_fields}",
                }
            )
        refine_candidates = refine_candidates[:max_refine_fields]

    if not refine_candidates:
        diag["reason"] = "no_refine_candidates"
        return updated, diag

    try:
        from mosaicx.runtime_env import import_dspy

        dspy = import_dspy()
    except Exception as exc:
        diag["reason"] = f"dspy_import_failed:{type(exc).__name__}"
        return model_instance, diag
    if getattr(dspy.settings, "lm", None) is None:
        diag["reason"] = "lm_not_configured"
        return model_instance, diag
    if not hasattr(dspy, "Refine"):
        diag["reason"] = "refine_unavailable"
        return model_instance, diag

    source_lower = str(source_text or "").lower()

    class _RepairSig(dspy.Signature):
        field_name: str = dspy.InputField(desc="Field to repair")
        field_type: str = dspy.InputField(desc="Target field type")
        current_value: str = dspy.InputField(desc="Current field value")
        source_text: str = dspy.InputField(desc="Source document text")
        repaired_value: str = dspy.OutputField(desc="Improved value for this field only")

    repair_module = dspy.ChainOfThought(_RepairSig)

    def reward_fn(_args: dict[str, Any], pred: Any) -> float:
        candidate = str(getattr(pred, "repaired_value", "") or "").strip()
        if _is_nullish_string(candidate):
            return 0.0
        score = 0.35
        probe = candidate.lower()
        if probe and probe in source_lower:
            score += 0.55
        if probe and (f"no {probe}" in source_lower or f"without {probe}" in source_lower):
            score -= 0.4
        return max(0.0, min(1.0, score))

    refiner = dspy.Refine(
        module=repair_module,
        N=3,
        reward_fn=reward_fn,
        threshold=0.6,
    )

    from mosaicx.verify.parse_utils import parse_json_like

    for field_name in refine_candidates:
        before = updated_payload.get(field_name)
        field = schema_class.model_fields.get(field_name)
        field_type = str(getattr(field, "annotation", "unknown"))
        try:
            pred = refiner(
                field_name=field_name,
                field_type=field_type,
                current_value=json.dumps(before, ensure_ascii=False, default=str),
                source_text=str(source_text or "")[:12000],
            )
            raw = str(getattr(pred, "repaired_value", "") or "").strip()
            if not raw:
                diag["skipped_fields"].append(
                    {"field": field_name, "reason": "empty_repair_output"}
                )
                continue
            if raw.startswith("{") or raw.startswith("["):
                parsed = parse_json_like(raw)
                candidate_value = parsed
            else:
                candidate_value = raw
            trial_payload = dict(updated_payload)
            trial_payload[field_name] = candidate_value
            coerced = _coerce_payload_to_schema(trial_payload, schema_class)
            repaired_model = schema_class.model_validate(coerced)
            repaired_payload = repaired_model.model_dump(mode="json")
            after = repaired_payload.get(field_name)
            if _is_missing_value(after):
                diag["skipped_fields"].append(
                    {"field": field_name, "reason": "repair_still_missing"}
                )
                continue
            updated = repaired_model
            updated_payload = repaired_payload
            diag["repaired_fields"].append(
                {
                    "field": field_name,
                    "before": before,
                    "after": after,
                    "method": "refine",
                }
            )
        except Exception as exc:
            diag["skipped_fields"].append(
                {"field": field_name, "reason": f"refine_error:{type(exc).__name__}"}
            )

    if not diag["repaired_fields"]:
        diag["reason"] = "no_field_repaired"
    else:
        diag["reason"] = "field_repair_applied"
    return updated, diag


def _extract_schema_with_structured_chain(
    *,
    document_text: str,
    schema_class: type[BaseModel],
    typed_extract: Any,
    json_extract: Any | None = None,
    planner_diag: dict[str, Any] | None = None,
    think: str = "standard",
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
    adjudication_info: dict[str, Any] = {
        "triggered": False,
        "conflict_detected": False,
        "method": None,
        "chosen_path": None,
        "rationale": "",
        "candidates": [],
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

    # ── Fast mode: Outlines only, fallback to json_extract (Predict) ──
    if think == "fast":
        if has_lm:
            outlines_result = _recover_schema_instance_with_outlines(
                document_text=document_text,
                schema_class=schema_class,
                error_hint="fast_mode",
            )
            if outlines_result is not None:
                _record("outlines_fast", True)
                return outlines_result, {
                    "selected_path": "outlines_fast",
                    "fallback_used": False,
                    "attempts": attempts,
                    "bestofn": bestofn_info,
                    "adjudication": adjudication_info,
                }
            _record("outlines_fast", False, ValueError("outlines_fast_unavailable"))

        # Predict fallback (no reasoning)
        if json_extract is not None:
            try:
                fallback_pred = json_extract(document_text=document_text)
                raw = getattr(fallback_pred, "extracted_json", "")
                model_instance = _recover_schema_instance_from_raw(raw, schema_class)
                _record("predict_fast", True)
                return model_instance, {
                    "selected_path": "predict_fast",
                    "fallback_used": True,
                    "attempts": attempts,
                    "bestofn": bestofn_info,
                    "adjudication": adjudication_info,
                }
            except Exception as exc:
                _record("predict_fast", False, exc)

        raise ValueError("Fast mode: both Outlines and Predict failed")

    # ── Deep mode: run both Outlines and CoT, score, pick best ──
    if think == "deep":
        candidates: list[tuple[str, BaseModel, float, dict[str, float]]] = []

        # Candidate 1: Outlines (always attempted — try/except handles missing LM)
        try:
            outlines_result = _recover_schema_instance_with_outlines(
                document_text=document_text,
                schema_class=schema_class,
                error_hint="deep_mode_baseline",
            )
            if outlines_result is not None:
                score, components = _score_extraction_candidate(
                    extracted=outlines_result,
                    schema_class=schema_class,
                    source_text=document_text,
                )
                candidates.append(("outlines_deep", outlines_result, score, components))
                _record("outlines_deep", True)
            else:
                _record("outlines_deep", False, ValueError("outlines_deep_unavailable"))
        except Exception as exc:
            _record("outlines_deep", False, exc)

        # Candidate 2: ChainOfThought (always runs in deep mode)
        try:
            cot_pred = typed_extract(document_text=document_text)
            cot_extracted = getattr(cot_pred, "extracted", cot_pred)
            cot_instance = _coerce_extracted_to_model_instance(
                extracted=cot_extracted,
                schema_class=schema_class,
            )
            score, components = _score_extraction_candidate(
                extracted=cot_instance,
                schema_class=schema_class,
                source_text=document_text,
            )
            candidates.append(("cot_deep", cot_instance, score, components))
            _record("cot_deep", True)
        except Exception as exc:
            _record("cot_deep", False, exc)

        if not candidates:
            raise ValueError("Deep mode: both Outlines and ChainOfThought failed")

        # Pick highest-scoring candidate
        candidates.sort(key=lambda c: c[2], reverse=True)
        best_path, best_instance, best_score, best_components = candidates[0]

        return best_instance, {
            "selected_path": best_path,
            "fallback_used": len(candidates) > 1 and best_path != "cot_deep",
            "attempts": attempts,
            "bestofn": bestofn_info,
            "adjudication": {
                "deep_mode": True,
                "candidates": [
                    {"path": c[0], "score": c[2], "components": c[3]}
                    for c in candidates
                ],
                "chosen": best_path,
            },
        }

    if has_lm:
        outlines_primary = _recover_schema_instance_with_outlines(
            document_text=document_text,
            schema_class=schema_class,
            error_hint="primary_outlines",
        )
        if outlines_primary is not None:
            score, components = _score_extraction_candidate(
                extracted=outlines_primary,
                schema_class=schema_class,
                source_text=document_text,
            )
            completeness = components.get("critical_completeness", 0.0)
            if completeness >= 1.0:
                # All required fields present — accept Outlines fast path
                bestofn_info["reason"] = "skipped_outlines_primary_succeeded"
                _record("outlines_primary", True)
                selected_path = "outlines_primary"
                return outlines_primary, {
                    "selected_path": selected_path,
                    "fallback_used": False,
                    "attempts": attempts,
                    "bestofn": bestofn_info,
                    "adjudication": adjudication_info,
                }
            # Outlines succeeded but missed required fields — fall through to DSPy
            _record("outlines_primary", False, ValueError(
                f"outlines_incomplete:completeness={completeness:.2f}"
            ))
        else:
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

    if _planner_routes_uncertain(planner_diag):
        candidate_pool: list[dict[str, Any]] = []

        def _add_candidate(path: str, model: BaseModel) -> None:
            score, components = _score_extraction_candidate(
                extracted=model,
                schema_class=schema_class,
                source_text=document_text,
            )
            candidate_pool.append(
                {
                    "path": path,
                    "model": model,
                    "score": score,
                    "components": components,
                }
            )

        try:
            pred = typed_extract(document_text=document_text)
            model = _coerce_extracted_to_model_instance(
                extracted=getattr(pred, "extracted", pred),
                schema_class=schema_class,
            )
            _record("uncertain_candidate_typed", True)
            _add_candidate("dspy_typed_direct", model)
        except Exception as exc:
            _record("uncertain_candidate_typed", False, exc)

        if dspy is not None and has_lm:
            for step_name, adapter_name in (
                ("uncertain_candidate_json", "JSONAdapter"),
                ("uncertain_candidate_two_step", "TwoStepAdapter"),
            ):
                adapter_cls = getattr(dspy, adapter_name, None)
                if adapter_cls is None:
                    _record(step_name, False, ValueError(f"{adapter_name}_missing"))
                    continue
                try:
                    with dspy.context(adapter=adapter_cls()):
                        pred = typed_extract(document_text=document_text)
                    model = _coerce_extracted_to_model_instance(
                        extracted=getattr(pred, "extracted", pred),
                        schema_class=schema_class,
                    )
                    _record(step_name, True)
                    route_name = "dspy_json_adapter" if "json" in step_name else "dspy_two_step_adapter"
                    _add_candidate(route_name, model)
                except Exception as exc:
                    _record(step_name, False, exc)

        if candidate_pool:
            chosen_candidate, adjudication_info = _adjudicate_conflicting_candidates(
                candidates=candidate_pool,
                source_text=document_text,
            )
            if chosen_candidate is not None:
                _record("mcc_adjudication", True)
                selected_path = f"adjudicated:{chosen_candidate.get('path')}"
                return chosen_candidate["model"], {
                    "selected_path": selected_path,
                    "fallback_used": True,
                    "attempts": attempts,
                    "bestofn": bestofn_info,
                    "adjudication": adjudication_info,
                }

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
            "adjudication": adjudication_info,
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
                "adjudication": adjudication_info,
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
                "adjudication": adjudication_info,
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
                "adjudication": adjudication_info,
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
# Think-mode routing, chunking, deterministic compute, verify & fix
# ---------------------------------------------------------------------------

_THINK_LEVEL_ORDER = {"fast": 0, "standard": 1, "deep": 2}
_THINK_LEVEL_BY_RANK = {0: "fast", 1: "standard", 2: "deep"}

_CALC_KEYWORDS = re.compile(
    r"\b(count|calculate|calculated\s+as|max\(|sum\(|number\s+of|percentage)\b",
    flags=re.IGNORECASE,
)


def _count_schema_fields(schema_class: type[BaseModel], *, _depth: int = 0) -> tuple[int, int]:
    """Recursively count total fields and max nesting depth of list[object] chains.

    Returns (field_count, max_depth).
    """
    if _depth > 20:
        return 0, _depth
    count = 0
    max_d = _depth
    for _name, field_info in schema_class.model_fields.items():
        count += 1
        ann = field_info.annotation
        # Unwrap Optional
        origin = get_origin(ann)
        if origin in (Union, types.UnionType):
            inner = [a for a in get_args(ann) if a is not type(None)]
            ann = inner[0] if inner else ann
            origin = get_origin(ann)

        if origin in (list, tuple, set):
            item_type = get_args(ann)[0] if get_args(ann) else Any
            if _is_basemodel_type(item_type):
                sub_count, sub_depth = _count_schema_fields(item_type, _depth=_depth + 1)
                count += sub_count
                max_d = max(max_d, sub_depth)
        elif _is_basemodel_type(ann):
            sub_count, sub_depth = _count_schema_fields(ann, _depth=_depth + 1)
            count += sub_count
            max_d = max(max_d, sub_depth)
    return count, max_d


def _has_calculated_fields(schema_class: type[BaseModel]) -> bool:
    """Detect if any field description mentions calculated/derived values."""
    for _name, field_info in schema_class.model_fields.items():
        desc = str(field_info.description or "")
        if _CALC_KEYWORDS.search(desc):
            return True
        # Check nested models
        ann = field_info.annotation
        origin = get_origin(ann)
        if origin in (Union, types.UnionType):
            inner = [a for a in get_args(ann) if a is not type(None)]
            ann = inner[0] if inner else ann
            origin = get_origin(ann)
        if origin in (list, tuple, set):
            item_type = get_args(ann)[0] if get_args(ann) else Any
            if _is_basemodel_type(item_type) and _has_calculated_fields(item_type):
                return True
        elif _is_basemodel_type(ann):
            if _has_calculated_fields(ann):
                return True
    return False


def _has_list_object_nesting(schema_class: type[BaseModel]) -> bool:
    """Return True if any field is a list[BaseModel]."""
    for _name, field_info in schema_class.model_fields.items():
        ann = field_info.annotation
        origin = get_origin(ann)
        if origin in (Union, types.UnionType):
            inner = [a for a in get_args(ann) if a is not type(None)]
            ann = inner[0] if inner else ann
            origin = get_origin(ann)
        if origin in (list, tuple, set):
            item_type = get_args(ann)[0] if get_args(ann) else Any
            if _is_basemodel_type(item_type):
                return True
    return False


def _route_think_level(schema_class: type[BaseModel]) -> str:
    """Deterministic router: analyze schema complexity and return think level.

    Rules:
    - fields <= 8 and no list[object] nesting -> "fast"
    - fields <= 20, nesting_depth <= 2, no calculated fields -> "standard"
    - everything else -> "deep"
    """
    field_count, nesting_depth = _count_schema_fields(schema_class)
    has_list_obj = _has_list_object_nesting(schema_class)
    has_calc = _has_calculated_fields(schema_class)

    if field_count <= 8 and not has_list_obj:
        level = "fast"
    elif field_count <= 20 and nesting_depth <= 2 and not has_calc:
        level = "standard"
    else:
        level = "deep"

    logger.info(
        "Think router: fields=%d, depth=%d, list_obj=%s, calc=%s -> %s",
        field_count, nesting_depth, has_list_obj, has_calc, level,
    )
    return level


def _resolve_think_level(user_think: str, schema_class: type[BaseModel] | None) -> str:
    """Resolve effective think level from user choice and template analysis.

    - "auto": let router pick freely
    - "fast": max(fast, template_floor) -- won't go below template minimum
    - "deep": always deep
    """
    if user_think == "deep":
        return "deep"

    if schema_class is None:
        # No schema to analyze -- fall back to user choice or standard
        return "standard" if user_think == "auto" else user_think

    if user_think == "auto":
        return _route_think_level(schema_class)

    # user_think == "fast": enforce floor
    template_floor = _route_think_level(schema_class)
    user_rank = _THINK_LEVEL_ORDER.get(user_think, 0)
    floor_rank = _THINK_LEVEL_ORDER.get(template_floor, 0)
    effective_rank = max(user_rank, floor_rank)
    return _THINK_LEVEL_BY_RANK.get(effective_rank, "standard")


def _chunk_schema(schema_class: type[BaseModel]) -> list[dict[str, Any]]:
    """Walk schema fields and split at list[object] boundaries.

    Returns a list of chunk dicts:
      [{"name": "scalars", "fields": [...], "schema_subset": <PydanticModel>},
       {"name": "field_name", "fields": [...], "schema_subset": <PydanticModel>}, ...]

    Each chunk has a dynamically-created Pydantic sub-model containing only
    its fields, suitable for a focused extraction call.
    """
    from pydantic import create_model as pydantic_create_model

    scalar_fields: list[str] = []
    scalar_field_defs: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []

    for field_name, field_info in schema_class.model_fields.items():
        ann = field_info.annotation
        # Unwrap Optional to check inner type
        inner_ann = ann
        origin = get_origin(inner_ann)
        if origin in (Union, types.UnionType):
            inner = [a for a in get_args(inner_ann) if a is not type(None)]
            inner_ann = inner[0] if inner else inner_ann
            origin = get_origin(inner_ann)

        is_list_of_obj = False
        if origin in (list, tuple, set):
            item_type = get_args(inner_ann)[0] if get_args(inner_ann) else Any
            if _is_basemodel_type(item_type):
                is_list_of_obj = True

        if is_list_of_obj:
            # This field becomes its own chunk
            default = field_info.default if field_info.default is not None else ...
            if not field_info.is_required():
                default = field_info.default
            chunk_field_def = (ann, Field(
                default=default,
                description=field_info.description,
            ))
            chunk_model = pydantic_create_model(
                f"_Chunk_{field_name}",
                **{field_name: chunk_field_def},
            )
            chunks.append({
                "name": field_name,
                "fields": [field_name],
                "schema_subset": chunk_model,
                "original_field_info": {field_name: field_info},
            })
        else:
            scalar_fields.append(field_name)
            default = field_info.default if field_info.default is not None else ...
            if not field_info.is_required():
                default = field_info.default
            scalar_field_defs[field_name] = (ann, Field(
                default=default,
                description=field_info.description,
            ))

    # Build scalar chunk
    if scalar_fields:
        scalar_model = pydantic_create_model(
            "_Chunk_scalars",
            **scalar_field_defs,
        )
        chunks.insert(0, {
            "name": "scalars",
            "fields": list(scalar_fields),
            "schema_subset": scalar_model,
        })

    return chunks


def _compute_derived_fields(
    extracted: BaseModel,
    schema_class: type[BaseModel],
    source_text: str,
) -> BaseModel:
    """Detect calculated fields from descriptions and overwrite with deterministic values.

    Patterns detected:
    - "count all entries in X" / "count entries in X" -> len(extracted.X)
    - "count entries where Y is true" / "number of tumor-positive" -> conditional count
    - "maximum ... percentage" / "max(" -> max of values
    - "calculate" / "calculated as" -> attempt formula parse

    Best-effort: if pattern doesn't match or referenced field doesn't exist, keep LLM value.
    """
    payload = extracted.model_dump(mode="json")
    updated = dict(payload)
    changes: list[str] = []

    # First: recurse into list[object] fields to compute their inner derived fields
    for field_name, field_info in schema_class.model_fields.items():
        ann = field_info.annotation
        # Unwrap Optional
        inner_ann = ann
        origin = get_origin(inner_ann)
        if origin in (Union, types.UnionType):
            inner = [a for a in get_args(inner_ann) if a is not type(None)]
            inner_ann = inner[0] if inner else inner_ann
            origin = get_origin(inner_ann)

        if origin in (list, tuple, set):
            item_type = get_args(inner_ann)[0] if get_args(inner_ann) else Any
            if _is_basemodel_type(item_type) and _has_calculated_fields(item_type):
                items = updated.get(field_name)
                if isinstance(items, list):
                    updated_items = []
                    for item_data in items:
                        if isinstance(item_data, dict):
                            try:
                                item_instance = item_type.model_validate(item_data)
                                computed = _compute_derived_fields(
                                    item_instance, item_type, source_text,
                                )
                                updated_items.append(computed.model_dump(mode="json"))
                                changes.append(f"{field_name}[*]")
                            except Exception:
                                updated_items.append(item_data)
                        else:
                            updated_items.append(item_data)
                    updated[field_name] = updated_items

    # Then: compute derived fields at the current level
    for field_name, field_info in schema_class.model_fields.items():
        desc = str(field_info.description or "").lower()
        if not _CALC_KEYWORDS.search(desc):
            continue

        try:
            new_value = _compute_single_derived_field(
                field_name=field_name,
                description=desc,
                payload=updated,  # Use updated payload so recursive values are available
                schema_class=schema_class,
            )
            if new_value is not None:
                updated[field_name] = new_value
                changes.append(field_name)
        except Exception as exc:
            logger.debug("Derived field %s compute failed: %s", field_name, exc)
            continue

    if changes:
        logger.info("Computed derived fields: %s", changes)
        try:
            coerced = _coerce_payload_to_schema(updated, schema_class)
            return schema_class.model_validate(coerced)
        except Exception as exc:
            logger.warning("Failed to validate after compute: %s", exc)
            return extracted
    return extracted


def _compute_single_derived_field(
    *,
    field_name: str,
    description: str,
    payload: dict[str, Any],
    schema_class: type[BaseModel],
) -> Any | None:
    """Try to deterministically compute a single derived field value.

    Returns the computed value, or None if no pattern matched.
    """
    # Pattern: "count entries ... where Y is true" / "number of ... where Y is true"
    # (Must be checked BEFORE simple count_all to avoid matching "count entries in X where ...")
    count_where_match = re.search(
        r"(?:count\s+entries|number\s+of\s+\w+).+?where\s+(\w+)\s+is\s+true",
        description,
    )
    if count_where_match:
        cond_field = count_where_match.group(1).lower()
        list_ref = _find_referenced_list_field(description, payload, schema_class)
        if isinstance(list_ref, list):
            return sum(
                1 for item in list_ref
                if isinstance(item, dict) and _is_truthy(item.get(cond_field))
            )

    # Pattern: "number of tumor-positive samples" (heuristic: look for list + bool field)
    tumor_positive_match = re.search(
        r"number\s+of\s+tumor.?positive", description
    )
    if tumor_positive_match:
        list_ref = _find_referenced_list_field(description, payload, schema_class)
        if isinstance(list_ref, list):
            return sum(
                1 for item in list_ref
                if isinstance(item, dict) and _is_truthy(item.get("tumor"))
            )

    # Pattern: "count all entries in X" / "count entries in X" (simple count)
    # Also handles "count all entries in Descriptive Name / actual_field_name"
    count_all_match = re.search(
        r"count\s+(?:all\s+)?entries\s+in\s+(.+?)(?:\.|$)", description
    )
    if count_all_match:
        ref_text = count_all_match.group(1).strip()
        # Remove trailing conditional clauses (e.g. "where tumor is true")
        ref_text = re.sub(r"\s+where\s+.*$", "", ref_text).strip()
        if ref_text:
            # Try each word/token in the reference text as a potential field name
            ref_candidates = [tok.strip().lower().replace("-", "_") for tok in re.split(r"[/,]|\s+", ref_text) if tok.strip()]
            for candidate in ref_candidates:
                ref_value = _find_field_by_name(payload, candidate, schema_class)
                if isinstance(ref_value, list):
                    return len(ref_value)
            # Also try the full text as a fuzzy match
            ref_value = _find_field_by_name(payload, ref_text.replace(" ", "_"), schema_class)
            if isinstance(ref_value, list):
                return len(ref_value)

    # Pattern: "maximum ... percentage" / "max("
    max_match = re.search(r"max(?:imum|imaler?)?\s*[\(]?", description)
    if max_match and "percent" in description:
        # Look for tumor percentage calculation pattern
        # Prefer list that has 'tumor' or 'positive' field (for filtering)
        list_ref = None
        tumor_related = "tumor" in description or "positive" in description
        if tumor_related:
            for _key, val in payload.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    if any("tumor" in k or "positive" in k for k in val[0].keys()):
                        list_ref = val
                        break
        if list_ref is None:
            list_ref = _find_referenced_list_field(description, payload, schema_class)

        # Build a lookup for length from other lists (by nr) for cross-list reference
        length_by_nr: dict[int, float] = {}
        for _key, val in payload.items():
            if isinstance(val, list) and val is not list_ref:
                for item in val:
                    if isinstance(item, dict) and "nr" in item:
                        length = item.get("laenge_cm") or item.get("length_cm")
                        if isinstance(length, (int, float)) and length > 0:
                            length_by_nr[int(item["nr"])] = float(length)

        if isinstance(list_ref, list):
            percentages = []
            for item in list_ref:
                if not isinstance(item, dict):
                    continue
                # Only consider tumor-positive items (require explicit field)
                has_tumor_field = "tumor" in item or "positive" in item
                if has_tumor_field:
                    if not _is_truthy(item.get("tumor", item.get("positive", False))):
                        continue
                # Try direct percentage field
                pct = item.get("tumor_prozent") or item.get("tumor_percent")
                if pct is not None and isinstance(pct, (int, float)) and pct > 0:
                    percentages.append(float(pct))
                    continue
                # Try calculate from extent/length (within same item)
                extent = item.get("tumorausdehnung_mm") or item.get("tumor_extent_mm")
                length_val = item.get("laenge_cm") or item.get("length_cm")
                # Cross-reference length from makroskopie_liste if not in this item
                if length_val is None and "nr" in item:
                    length_val = length_by_nr.get(int(item["nr"]))
                if (
                    extent is not None and length_val is not None
                    and isinstance(extent, (int, float))
                    and isinstance(length_val, (int, float))
                    and length_val > 0
                ):
                    # extent in mm, length in cm -> convert
                    length_mm = length_val * 10
                    if length_mm > 0:
                        percentages.append(extent / length_mm * 100)
            if percentages:
                return round(max(percentages))

    return None


def _find_field_by_name(
    payload: dict[str, Any],
    target_name: str,
    schema_class: type[BaseModel],
) -> Any | None:
    """Find a field value in the payload by fuzzy name matching."""
    target_lower = target_name.lower().replace("_", "").replace("-", "")
    for key, value in payload.items():
        key_lower = key.lower().replace("_", "").replace("-", "")
        if key_lower == target_lower or target_lower in key_lower or key_lower in target_lower:
            return value
    return None


def _find_referenced_list_field(
    description: str,
    payload: dict[str, Any],
    schema_class: type[BaseModel],
) -> list | None:
    """Find the list field referenced in a description."""
    # Look for field names mentioned in the description
    desc_lower = description.lower()
    for key, value in payload.items():
        if isinstance(value, list):
            key_lower = key.lower()
            # Direct match
            if key_lower in desc_lower:
                return value
            # Match without underscores (e.g. "begutachtung liste" in desc)
            key_nound = key_lower.replace("_", " ")
            if key_nound in desc_lower:
                return value
            # Match without _liste suffix (e.g. "begutachtung" in desc)
            key_short = key_lower.replace("_liste", "").replace("_list", "")
            if key_short and key_short in desc_lower:
                return value

    # Fallback: find the first list-of-dicts field
    for _key, value in payload.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return None


def _is_truthy(value: Any) -> bool:
    """Check if a value is truthy for condition counting."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "ja", "1", "positive"}
    if isinstance(value, (int, float)):
        return value > 0
    return bool(value)


def _apply_nested_fix(payload: dict[str, Any], field_path: str, value: Any) -> None:
    """Best-effort apply a fix to a nested field path like 'befunde.0.gleason'."""
    parts = re.split(r"\.|(?=\[)", field_path)
    current = payload
    for _i, part in enumerate(parts[:-1]):
        # Handle array index
        idx_match = re.match(r"\[?(\d+)\]?", part)
        if idx_match and isinstance(current, list):
            idx = int(idx_match.group(1))
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return

    # Set the final value
    last = parts[-1]
    idx_match = re.match(r"\[?(\d+)\]?", last)
    if idx_match and isinstance(current, list):
        idx = int(idx_match.group(1))
        if 0 <= idx < len(current):
            current[idx] = value
    elif isinstance(current, dict):
        current[last] = value


def _run_consistency_checks(
    extracted: BaseModel,
    schema_class: type[BaseModel],
) -> list[dict[str, Any]]:
    """Run deterministic consistency checks on the extracted data.

    Checks:
    - List counts match anzahl_* fields
    - Enum values are valid
    - Required fields are non-null
    """
    payload = extracted.model_dump(mode="json")
    issues: list[dict[str, Any]] = []

    for field_name, field_info in schema_class.model_fields.items():
        value = payload.get(field_name)

        # Check required fields are non-null
        if field_info.is_required() and _is_missing_value(value):
            issues.append({
                "field": field_name,
                "issue": "required_field_missing",
                "severity": "error",
            })

        # Check anzahl_* consistency with lists
        if field_name.startswith("anzahl_") and isinstance(value, (int, float)):
            # Find a matching list field
            for other_name, other_value in payload.items():
                if isinstance(other_value, list) and other_name in field_name.replace("anzahl_", ""):
                    if int(value) != len(other_value):
                        issues.append({
                            "field": field_name,
                            "issue": f"count_mismatch: {field_name}={int(value)} but {other_name} has {len(other_value)} entries",
                            "severity": "warning",
                        })

        # Check enum values
        ann = field_info.annotation
        origin = get_origin(ann)
        if origin in (Union, types.UnionType):
            inner = [a for a in get_args(ann) if a is not type(None)]
            ann = inner[0] if inner else ann
            origin = get_origin(ann)
        if origin is Literal and value is not None:
            allowed = get_args(ann)
            if value not in allowed:
                issues.append({
                    "field": field_name,
                    "issue": f"invalid_enum_value: {value!r} not in {allowed}",
                    "severity": "warning",
                })

    return issues


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

    # ── Verification signature (deep mode) ──
    class VerifyExtractionSig(dspy.Signature):
        """Review a structured extraction for correctness against the source."""

        document_text: str = dspy.InputField(
            desc="Original document text"
        )
        extraction_json: str = dspy.InputField(
            desc="JSON of the extracted data to verify"
        )
        schema_description: str = dspy.InputField(
            desc="Description of the target schema and its fields"
        )
        flagged_issues: str = dspy.OutputField(
            desc=(
                "JSON list of issues found. Each item: "
                '{"field": "dot.path", "issue": "what is wrong", '
                '"evidence": "text from document", "suggested_value": "correct value"}. '
                "Empty list [] if everything is correct."
            )
        )

    # ── Fix signature (deep mode, per flagged field) ──
    class FixFieldSig(dspy.Signature):
        """Extract the correct value for a single flagged field."""

        field_name: str = dspy.InputField(desc="Name of the field to fix")
        field_description: str = dspy.InputField(desc="Schema description for this field")
        issue: str = dspy.InputField(desc="What was wrong with the extracted value")
        evidence: str = dspy.InputField(desc="Evidence from the document")
        document_section: str = dspy.InputField(desc="Relevant section of the document")
        corrected_value: str = dspy.OutputField(desc="The correct value for this field")

    class DocumentExtractor(dspy.Module):
        """DSPy Module for document extraction.

        Two modes:
        - **Auto mode** (no output_schema): infers schema from doc, then extracts.
        - **Schema mode** (output_schema provided): extracts directly into schema.

        Think levels: auto, fast, standard, deep.
        """

        def __init__(self, output_schema: type[BaseModel] | None = None, think: str = "auto") -> None:
            super().__init__()
            if think not in ("auto", "fast", "standard", "deep"):
                raise ValueError(
                    f"think must be 'auto', 'fast', 'standard', or 'deep', got {think!r}"
                )
            self._user_think = think
            self._output_schema = output_schema

            # Resolve effective think level (auto-routing happens here)
            self._think = _resolve_think_level(think, output_schema)
            logger.info(
                "DocumentExtractor: user_think=%s, effective_think=%s, schema=%s",
                think, self._think, output_schema.__name__ if output_schema else "auto",
            )

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
                # Deep mode: verification + fix modules
                self.verify_extraction = dspy.ChainOfThought(VerifyExtractionSig)
                self.fix_field = dspy.ChainOfThought(FixFieldSig)
            else:
                # Auto mode: infer schema then extract
                self.infer_schema = dspy.ChainOfThought(InferSchemaFromDocument)
                # extract_custom is created dynamically in forward()
                self.verify_extraction = dspy.ChainOfThought(VerifyExtractionSig)
                self.fix_field = dspy.ChainOfThought(FixFieldSig)

        # ── Internal helpers for deep mode ──

        def _deep_extract_chunked(self, document_text: str, schema: type[BaseModel], metrics, tracker):
            """Deep mode: chunk schema, extract each chunk, assemble."""
            from mosaicx.metrics import track_step

            chunks = _chunk_schema(schema)
            logger.info("Deep mode: %d chunks for %s", len(chunks), schema.__name__)
            assembled: dict[str, Any] = {}
            chunk_diags: list[dict[str, Any]] = []

            for chunk in chunks:
                chunk_name = chunk["name"]
                chunk_schema = chunk["schema_subset"]
                chunk_fields = chunk["fields"]

                with track_step(metrics, f"Extract chunk: {chunk_name}", tracker):
                    # Build a CoT extractor for just this chunk
                    chunk_sig = dspy.Signature(
                        "document_text -> extracted",
                        instructions=(
                            f"Extract ONLY the following fields from the document: "
                            f"{', '.join(chunk_fields)}. "
                            f"Match the {chunk_schema.__name__} schema. "
                            f"Be thorough and extract every entry."
                        ),
                    ).with_updated_fields(
                        "document_text",
                        desc="Full text of the document",
                        type_=str,
                    ).with_updated_fields(
                        "extracted",
                        desc=f"Extracted {chunk_name} data",
                        type_=chunk_schema,
                    )
                    chunk_extractor = dspy.ChainOfThought(chunk_sig)

                    try:
                        pred = chunk_extractor(document_text=document_text)
                        extracted = getattr(pred, "extracted", pred)
                        if hasattr(extracted, "model_dump"):
                            chunk_data = extracted.model_dump(mode="json")
                        elif isinstance(extracted, dict):
                            chunk_data = extracted
                        else:
                            chunk_data = {}
                        assembled.update(chunk_data)
                        chunk_diags.append({
                            "chunk": chunk_name,
                            "fields": chunk_fields,
                            "ok": True,
                        })
                    except Exception as exc:
                        logger.warning("Chunk %s extraction failed: %s", chunk_name, exc)
                        chunk_diags.append({
                            "chunk": chunk_name,
                            "fields": chunk_fields,
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        })

            # Assemble into the full schema
            try:
                coerced = _coerce_payload_to_schema(assembled, schema)
                model_instance = schema.model_validate(coerced)
            except Exception as exc:
                logger.warning("Deep chunk assembly failed: %s, falling back to CoT", exc)
                # Fallback: run standard CoT on the full schema
                pred = self.extract_custom(document_text=document_text)
                extracted = getattr(pred, "extracted", pred)
                model_instance = _coerce_extracted_to_model_instance(
                    extracted=extracted, schema_class=schema,
                )
                chunk_diags.append({"fallback": "full_cot", "reason": str(exc)})

            return model_instance, {"chunks": chunk_diags}

        def _verify_extraction_pass(self, extracted, document_text: str, schema: type[BaseModel], metrics, tracker):
            """Deep mode: verification pass. Returns list of flagged issues."""
            from mosaicx.metrics import track_step
            from mosaicx.verify.parse_utils import parse_json_like

            extraction_json = json.dumps(
                extracted.model_dump(mode="json") if hasattr(extracted, "model_dump") else extracted,
                ensure_ascii=False, indent=2,
            )
            # Build schema description from field info
            schema_desc_parts = [f"Schema: {schema.__name__}"]
            for fname, finfo in schema.model_fields.items():
                desc = finfo.description or ""
                schema_desc_parts.append(f"  {fname}: {desc}")
            schema_description = "\n".join(schema_desc_parts)

            with track_step(metrics, "Verify extraction", tracker):
                try:
                    pred = self.verify_extraction(
                        document_text=document_text[:30000],
                        extraction_json=extraction_json[:15000],
                        schema_description=schema_description[:5000],
                    )
                    raw_issues = str(getattr(pred, "flagged_issues", "[]") or "[]")
                    parsed = parse_json_like(raw_issues)
                    if isinstance(parsed, list):
                        return [
                            item for item in parsed
                            if isinstance(item, dict) and item.get("field")
                        ]
                    return []
                except Exception as exc:
                    logger.warning("Verification pass failed: %s", exc)
                    return []

        def _fix_flagged_fields_pass(self, extracted, flagged_issues, document_text: str, schema: type[BaseModel], metrics, tracker):
            """Deep mode: fix flagged fields with targeted CoT calls."""
            from mosaicx.metrics import track_step
            from mosaicx.verify.parse_utils import parse_json_like

            if not flagged_issues:
                return extracted, {"fixes_attempted": 0, "fixes_applied": 0}

            payload = extracted.model_dump(mode="json") if hasattr(extracted, "model_dump") else dict(extracted)
            updated = dict(payload)
            fixes_applied = 0

            with track_step(metrics, "Fix flagged fields", tracker):
                for issue in flagged_issues[:10]:  # Cap at 10 fixes
                    field_path = str(issue.get("field", ""))
                    issue_text = str(issue.get("issue", ""))
                    evidence = str(issue.get("evidence", ""))
                    suggested = issue.get("suggested_value")

                    # Resolve the top-level field name from dot path
                    top_field = field_path.split(".")[0].split("[")[0]
                    field_info = schema.model_fields.get(top_field)
                    if field_info is None:
                        continue

                    field_desc = str(field_info.description or "")

                    try:
                        pred = self.fix_field(
                            field_name=field_path,
                            field_description=field_desc,
                            issue=issue_text,
                            evidence=evidence,
                            document_section=document_text[:8000],
                        )
                        raw = str(getattr(pred, "corrected_value", "") or "").strip()
                        if not raw:
                            continue

                        # Parse the corrected value
                        if raw.startswith("{") or raw.startswith("["):
                            corrected = parse_json_like(raw)
                        else:
                            corrected = raw

                        # Apply fix to the appropriate field
                        if "." in field_path or "[" in field_path:
                            # Nested field -- use suggested_value if available
                            if suggested is not None:
                                corrected = suggested
                            # For nested paths, try to update the top-level field
                            # This is best-effort for simple cases
                            _apply_nested_fix(updated, field_path, corrected)
                        else:
                            updated[top_field] = corrected
                        fixes_applied += 1
                    except Exception as exc:
                        logger.debug("Fix for %s failed: %s", field_path, exc)

            # Validate the updated payload
            if fixes_applied > 0:
                try:
                    coerced = _coerce_payload_to_schema(updated, schema)
                    result = schema.model_validate(coerced)
                    return result, {
                        "fixes_attempted": len(flagged_issues),
                        "fixes_applied": fixes_applied,
                    }
                except Exception as exc:
                    logger.warning("Fix validation failed: %s, keeping original", exc)

            return extracted, {
                "fixes_attempted": len(flagged_issues),
                "fixes_applied": fixes_applied,
            }

        def _run_schema_mode_pipeline(
            self,
            document_text: str,
            schema: type[BaseModel],
            metrics,
            tracker,
            planner_diag: dict[str, Any],
        ) -> dspy.Prediction:
            """Schema mode extraction with think-level-aware pipeline."""
            from mosaicx.metrics import track_step

            think = self._think
            chain_diag: dict[str, Any] = {}
            verify_diag: dict[str, Any] = {}
            fix_diag: dict[str, Any] = {}

            planned_text = document_text
            with track_step(metrics, "Plan extraction", tracker):
                try:
                    planned_text, planner_diag_update = _plan_extraction_document_text(
                        document_text=document_text,
                        schema_name=schema.__name__,
                    )
                    planner_diag.update(planner_diag_update)
                except Exception as plan_exc:
                    planner_diag["fallback_reason"] = f"planner_error:{type(plan_exc).__name__}"
                    planned_text = document_text

            if think == "deep":
                # ── DEEP PATH ──
                # Pass 1: Chunked extraction
                with track_step(metrics, "Deep extract (chunked)", tracker):
                    model_instance, chunk_diag = self._deep_extract_chunked(
                        document_text=document_text,
                        schema=schema,
                        metrics=metrics,
                        tracker=tracker,
                    )
                chain_diag = {"selected_path": "deep_chunked", "chunks": chunk_diag}

                # Compute: deterministic overwrite of calculated fields
                with track_step(metrics, "Compute derived fields", tracker):
                    model_instance = _compute_derived_fields(
                        model_instance, schema, document_text,
                    )

                # Pass 2: Verify
                flagged = self._verify_extraction_pass(
                    model_instance, document_text, schema, metrics, tracker,
                )
                verify_diag = {"flagged_count": len(flagged), "issues": flagged}

                # Pass 3: Fix
                if flagged:
                    model_instance, fix_diag = self._fix_flagged_fields_pass(
                        model_instance, flagged, document_text, schema, metrics, tracker,
                    )
                    # Recompute derived fields after fixes
                    with track_step(metrics, "Recompute after fix", tracker):
                        model_instance = _compute_derived_fields(
                            model_instance, schema, document_text,
                        )

            else:
                # ── FAST / STANDARD PATH ──
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
                            think=think,
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
                                think=think,
                            )
                        else:
                            raise

                # Compute: runs at ALL levels
                with track_step(metrics, "Compute derived fields", tracker):
                    model_instance = _compute_derived_fields(
                        model_instance, schema, document_text,
                    )

                # Standard mode: also run consistency checks
                if think == "standard":
                    with track_step(metrics, "Consistency checks", tracker):
                        consistency_issues = _run_consistency_checks(model_instance, schema)
                        if consistency_issues:
                            logger.info(
                                "Consistency issues: %s",
                                [i.get("field") for i in consistency_issues],
                            )

            # Repair + semantic validation (all levels)
            model_instance, repair_diag = _repair_failed_critical_fields_with_refine(
                model_instance=model_instance,
                schema_class=schema,
                source_text=document_text,
                think=think,
            )
            model_instance, semantic_validation_diag = _apply_deterministic_semantic_validation(
                model_instance=model_instance,
                schema_class=schema,
            )

            # Assemble diagnostics
            planner_diag["planned_text_used"] = planned_text != document_text
            planner_diag["structured_chain"] = chain_diag.get("attempts", chain_diag.get("chunks", []))
            planner_diag["selected_structured_path"] = chain_diag.get("selected_path", f"{think}_path")
            planner_diag["structured_fallback_used"] = bool(chain_diag.get("fallback_used", False))
            planner_diag["bestofn"] = chain_diag.get("bestofn", {})
            planner_diag["adjudication"] = chain_diag.get("adjudication", {})
            planner_diag["repair"] = repair_diag
            planner_diag["deterministic_validation"] = semantic_validation_diag
            planner_diag["think_level"] = think
            planner_diag["user_think"] = self._user_think
            if verify_diag:
                planner_diag["verification"] = verify_diag
            if fix_diag:
                planner_diag["fix"] = fix_diag

            self._last_metrics = metrics
            self._last_planner = planner_diag
            return dspy.Prediction(extracted=model_instance, planner=planner_diag)

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
                # Schema mode
                schema = self._output_schema
                assert schema is not None
                return self._run_schema_mode_pipeline(
                    document_text, schema, metrics, tracker, planner_diag,
                )

            # Auto mode: infer schema, compile, extract
            with track_step(metrics, "Infer schema", tracker):
                infer_result = self.infer_schema(
                    document_text=document_text
                )
            spec: SchemaSpec = infer_result.schema_spec
            model = compile_schema(spec)

            # Re-resolve think level now that we have a schema
            self._think = _resolve_think_level(self._user_think, model)

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
            self.extract_custom = dspy.ChainOfThought(extract_sig)
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
            self.extract_json_fallback = dspy.Predict(extract_json_sig)

            # Store schema and run the unified pipeline
            self._output_schema = model
            result = self._run_schema_mode_pipeline(
                document_text, model, metrics, tracker, planner_diag,
            )
            # Add inferred schema to prediction
            return dspy.Prediction(
                extracted=result.extracted,
                inferred_schema=spec,
                planner=result.planner,
            )

    return {
        "InferSchemaFromDocument": InferSchemaFromDocument,
        "VerifyExtractionSig": VerifyExtractionSig,
        "FixFieldSig": FixFieldSig,
        "DocumentExtractor": DocumentExtractor,
    }


# Cache for lazily-built DSPy classes
_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({
    "InferSchemaFromDocument",
    "VerifyExtractionSig",
    "FixFieldSig",
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
