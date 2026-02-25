# mosaicx/pipelines/schema_gen.py
"""Schema generator pipeline.

Replaces the old schema/builder.py which used exec() to run LLM-generated
Python code. This new approach is safe: the LLM outputs a SchemaSpec JSON,
then we use pydantic.create_model() to build the class programmatically.

Key components:
    - FieldSpec / SchemaSpec: Pydantic models describing a schema declaratively.
    - compile_schema(): Converts a SchemaSpec into a real Pydantic BaseModel.
    - GenerateSchemaSpec: DSPy Signature for LLM-driven schema generation.
    - SchemaGenerator: DSPy Module wrapping the full pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Declarative schema specification models
# ---------------------------------------------------------------------------

class FieldSpec(BaseModel):
    """Specification for a single field in a generated schema."""

    name: str = Field(..., description="Field name (valid Python identifier)")
    type: str = Field(
        ...,
        description=(
            "Type string: 'str', 'int', 'float', 'bool', "
            "'list[X]', or 'enum'"
        ),
    )
    description: str = Field("", description="Human-readable field description")
    required: bool = Field(True, description="Whether the field is required")
    enum_values: Optional[List[str]] = Field(
        None,
        description="Allowed values when type is 'enum'",
    )
    fields: Optional[List["FieldSpec"]] = Field(
        None,
        description="Nested fields when type is 'object' or 'list[object]'",
    )


FieldSpec.model_rebuild()


class SchemaSpec(BaseModel):
    """Full specification for a dynamically generated Pydantic model."""

    class_name: str = Field(
        ..., description="Name for the generated Pydantic model class"
    )
    description: str = Field("", description="Docstring for the model")
    fields: List[FieldSpec] = Field(
        default_factory=list,
        description="List of field specifications",
    )


# ---------------------------------------------------------------------------
# Type resolution
# ---------------------------------------------------------------------------

_SIMPLE_TYPE_MAP: dict[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "object": dict,
    "dict": dict,
}

_LIST_RE = re.compile(r"^list\[(\w+)\]$", re.IGNORECASE)
_NON_IDENTIFIER_RE = re.compile(r"[^0-9a-zA-Z_]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")

_TYPE_ALIASES: dict[str, str] = {
    "string": "str",
    "text": "str",
    "integer": "int",
    "number": "float",
    "double": "float",
    "boolean": "bool",
    "dict": "object",
    "array": "list[str]",
}


def _build_nested_model(spec: FieldSpec, *, parent_name: str = "") -> type[BaseModel]:
    """Build a nested Pydantic model for object-like fields."""
    nested_fields = spec.fields or []
    field_defs: dict[str, tuple[Any, Any]] = {}

    for nested in nested_fields:
        nested_type = _resolve_type(nested, parent_name=f"{parent_name}_{spec.name}")
        if nested.required:
            field_defs[nested.name] = (
                nested_type,
                Field(..., description=nested.description or None),
            )
        else:
            field_defs[nested.name] = (
                Optional[nested_type],
                Field(default=None, description=nested.description or None),
            )

    model_name = f"{parent_name}_{spec.name}".strip("_") or "NestedObject"
    model_name = "".join(part.capitalize() for part in model_name.split("_"))
    return create_model(model_name, **field_defs)  # type: ignore[call-overload]


def _resolve_type(spec: FieldSpec, *, parent_name: str = "") -> type:
    """Map a FieldSpec's type string to an actual Python type.

    Supports:
        - Simple scalars: str, int, float, bool (plus aliases)
        - Generic lists:  list[str], list[int], etc.
        - Enums:          enum (requires spec.enum_values)

    Returns the resolved Python type.
    """
    type_str = spec.type.strip().lower()

    # --- object / dict ---
    if type_str in {"object", "dict"}:
        if spec.fields:
            return _build_nested_model(spec, parent_name=parent_name)
        return dict

    # --- simple scalars ---
    if type_str in _SIMPLE_TYPE_MAP:
        return _SIMPLE_TYPE_MAP[type_str]

    # --- list[X] ---
    m = _LIST_RE.match(type_str)
    if m:
        inner = m.group(1).lower()
        if inner == "object":
            if spec.fields:
                nested_item_spec = FieldSpec(
                    name=f"{spec.name}_item",
                    type="object",
                    fields=spec.fields,
                )
                inner_type = _build_nested_model(
                    nested_item_spec,
                    parent_name=parent_name,
                )
            else:
                inner_type = dict
            return list[inner_type]  # type: ignore[valid-type]
        inner_type = _SIMPLE_TYPE_MAP.get(inner)
        if inner_type is None:
            raise ValueError(
                f"Unsupported inner list type: {inner!r} "
                f"(supported: {list(_SIMPLE_TYPE_MAP)})"
            )
        return list[inner_type]  # type: ignore[valid-type]

    # --- enum ---
    if type_str == "enum":
        if not spec.enum_values:
            raise ValueError(
                f"Field {spec.name!r} has type 'enum' but no enum_values"
            )
        # Use (str, Enum) so enum members compare equal to plain strings.
        return Enum(  # type: ignore[return-value]
            f"{spec.name}_enum",
            {v: v for v in spec.enum_values},
            type=str,
        )

    raise ValueError(
        f"Unsupported type string: {spec.type!r} "
        f"(supported: {list(_SIMPLE_TYPE_MAP)} + list[X] + enum)"
    )


def _to_snake_case(raw: str, *, default: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return default
    text = _NON_IDENTIFIER_RE.sub("_", text)
    text = _MULTI_UNDERSCORE_RE.sub("_", text).strip("_").lower()
    if not text:
        return default
    if text[0].isdigit():
        text = f"f_{text}"
    return text


def _to_pascal_case(raw: str, *, default: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return default
    text = _NON_IDENTIFIER_RE.sub(" ", text)
    parts = [p for p in text.split() if p]
    if not parts:
        return default
    out = "".join(part[:1].upper() + part[1:] for part in parts)
    if out[0].isdigit():
        out = f"Schema{out}"
    return out


def _dedupe_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    i = 2
    while f"{name}_{i}" in used:
        i += 1
    deduped = f"{name}_{i}"
    used.add(deduped)
    return deduped


def _normalize_type_string(type_str: str, *, has_fields: bool, has_enum_values: bool) -> str:
    raw = " ".join(str(type_str or "").strip().lower().split())
    if not raw:
        if has_enum_values:
            return "enum"
        if has_fields:
            return "object"
        return "str"

    raw = _TYPE_ALIASES.get(raw, raw)
    raw = re.sub(r"\s+", "", raw)
    raw = raw.replace("array[", "list[")

    if raw == "enum":
        return "enum" if has_enum_values else "str"
    if raw in {"object", "dict"}:
        return "object"
    if raw in {"list", "array"}:
        return "list[str]"
    if raw in _SIMPLE_TYPE_MAP:
        return raw

    m = re.match(r"^(list)\[(.+)\]$", raw)
    if m:
        inner = _TYPE_ALIASES.get(m.group(2), m.group(2))
        if inner in {"dict", "object"}:
            return "list[object]"
        if inner in _SIMPLE_TYPE_MAP:
            return f"list[{inner}]"
        return "list[str]"

    if has_enum_values:
        return "enum"
    if has_fields:
        return "object"
    return "str"


def _normalize_enum_values(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        sv = str(v or "").strip()
        if not sv:
            continue
        key = sv.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sv)
    return out or None


def _normalize_field_spec(field: FieldSpec, *, index: int, used_names: set[str]) -> FieldSpec:
    base_name = _to_snake_case(field.name, default=f"field_{index}")
    name = _dedupe_name(base_name, used_names)

    nested_fields = list(field.fields or [])
    normalized_nested: list[FieldSpec] | None = None
    if nested_fields:
        nested_used: set[str] = set()
        normalized_nested = [
            _normalize_field_spec(nf, index=i + 1, used_names=nested_used)
            for i, nf in enumerate(nested_fields)
        ]

    enum_values = _normalize_enum_values(field.enum_values)
    normalized_type = _normalize_type_string(
        field.type,
        has_fields=bool(normalized_nested),
        has_enum_values=bool(enum_values),
    )

    if normalized_type != "enum":
        enum_values = None
    if normalized_type not in {"object", "list[object]"}:
        normalized_nested = None

    desc = " ".join(str(field.description or "").split())
    return FieldSpec(
        name=name,
        type=normalized_type,
        description=desc,
        required=bool(field.required),
        enum_values=enum_values,
        fields=normalized_nested,
    )


def normalize_schema_spec(spec: SchemaSpec, *, default_class_name: str = "GeneratedSchema") -> SchemaSpec:
    """Normalize class/field names and type aliases into a compile-safe SchemaSpec."""
    normalized_class_name = _to_pascal_case(spec.class_name, default=default_class_name)
    used_names: set[str] = set()
    normalized_fields = [
        _normalize_field_spec(field, index=i + 1, used_names=used_names)
        for i, field in enumerate(spec.fields)
    ]
    return SchemaSpec(
        class_name=normalized_class_name,
        description=" ".join(str(spec.description or "").split()),
        fields=normalized_fields,
    )


def validate_schema_spec(spec: SchemaSpec) -> list[str]:
    """Return structural validation issues for a SchemaSpec."""
    issues: list[str] = []
    if not spec.class_name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", spec.class_name):
        issues.append(f"Invalid class_name: {spec.class_name!r}")
    if not spec.fields:
        issues.append("Schema has no fields.")

    def _walk_fields(fields: list[FieldSpec], *, prefix: str = "") -> None:
        seen: set[str] = set()
        for field in fields:
            field_path = f"{prefix}.{field.name}".strip(".")
            if not field.name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", field.name):
                issues.append(f"Invalid field name at {field_path or '<root>'}")
            if field.name in seen:
                issues.append(f"Duplicate field name: {field_path}")
            seen.add(field.name)

            normalized_type = _normalize_type_string(
                field.type,
                has_fields=bool(field.fields),
                has_enum_values=bool(field.enum_values),
            )
            if field.type != normalized_type:
                issues.append(
                    f"Non-canonical type '{field.type}' at {field_path}; expected '{normalized_type}'."
                )
            if normalized_type == "enum" and not (field.enum_values or []):
                issues.append(f"Enum field missing enum_values: {field_path}")

            if normalized_type in {"object", "list[object]"} and field.fields:
                _walk_fields(list(field.fields), prefix=field_path)

    _walk_fields(list(spec.fields))

    # Deduplicate while preserving order for readable logs.
    deduped: list[str] = []
    seen_issues: set[str] = set()
    for issue in issues:
        if issue in seen_issues:
            continue
        seen_issues.add(issue)
        deduped.append(issue)
    return deduped


# ---------------------------------------------------------------------------
# Schema compilation
# ---------------------------------------------------------------------------

def compile_schema(spec: SchemaSpec) -> type[BaseModel]:
    """Convert a SchemaSpec into a concrete Pydantic BaseModel class.

    Uses ``pydantic.create_model()`` -- no exec() or code generation.

    Args:
        spec: The declarative schema specification.

    Returns:
        A new Pydantic BaseModel subclass with the specified fields.
    """
    field_definitions: dict = {}

    for field in spec.fields:
        py_type = _resolve_type(field, parent_name=spec.class_name)

        # Enhance description for enum fields with allowed values
        desc = field.description or ""
        if field.type.strip().lower() == "enum" and field.enum_values:
            values_str = ", ".join(field.enum_values)
            desc = f"{desc} (allowed: {values_str})" if desc else f"One of: {values_str}"

        if field.required:
            field_definitions[field.name] = (
                py_type,
                Field(..., description=desc or None),
            )
        else:
            field_definitions[field.name] = (
                Optional[py_type],
                Field(default=None, description=desc or None),
            )

    model = create_model(
        spec.class_name,
        **field_definitions,
    )
    model.__doc__ = spec.description
    return model


# ---------------------------------------------------------------------------
# Schema storage
# ---------------------------------------------------------------------------


def save_schema(
    spec: SchemaSpec,
    schema_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Save a SchemaSpec as a JSON file.

    Args:
        spec: The schema specification to save.
        schema_dir: Directory to save into (uses {class_name}.json).
        output_path: Explicit file path (overrides schema_dir).

    Returns:
        Path to the saved file.
    """
    if output_path is not None:
        dest = output_path
    elif schema_dir is not None:
        schema_dir.mkdir(parents=True, exist_ok=True)
        _archive_before_save(spec, schema_dir)
        dest = schema_dir / f"{spec.class_name}.json"
    else:
        raise ValueError("Provide schema_dir or output_path")

    dest.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return dest


def load_schema(name: str, schema_dir: Path) -> SchemaSpec:
    """Load a SchemaSpec by name from a directory.

    Args:
        name: Schema class name (without .json extension).
        schema_dir: Directory to search.

    Returns:
        The loaded SchemaSpec.

    Raises:
        FileNotFoundError: If the schema file doesn't exist.
    """
    path = schema_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    return SchemaSpec.model_validate_json(path.read_text(encoding="utf-8"))


def list_schemas(schema_dir: Path) -> list[SchemaSpec]:
    """List all saved schemas in a directory.

    Returns:
        List of SchemaSpec objects, sorted by class_name.
    """
    if not schema_dir.exists():
        return []
    specs = []
    for f in sorted(schema_dir.glob("*.json")):
        try:
            specs.append(SchemaSpec.model_validate_json(f.read_text(encoding="utf-8")))
        except Exception:
            continue  # skip malformed files
    return specs


# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------


class VersionInfo(NamedTuple):
    """Metadata for a single archived schema version."""
    version: int
    path: Path
    modified: datetime
    field_count: int


def _history_dir(schema_dir: Path) -> Path:
    """Return (and create) the .history/ subdirectory inside *schema_dir*."""
    d = schema_dir / ".history"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _archive_before_save(spec: SchemaSpec, schema_dir: Path) -> None:
    """Copy the current schema file into .history/ before overwriting it."""
    current = schema_dir / f"{spec.class_name}.json"
    if not current.exists():
        return
    hdir = _history_dir(schema_dir)
    existing = sorted(hdir.glob(f"{spec.class_name}_v*.json"))
    next_version = len(existing) + 1
    dest = hdir / f"{spec.class_name}_v{next_version}.json"
    shutil.copy2(current, dest)


def list_versions(name: str, schema_dir: Path) -> list[VersionInfo]:
    """List all archived versions of a schema, sorted by version number."""
    hdir = schema_dir / ".history"
    if not hdir.exists():
        return []
    results: list[VersionInfo] = []
    for p in sorted(hdir.glob(f"{name}_v*.json")):
        m = re.match(rf"^{re.escape(name)}_v(\d+)\.json$", p.name)
        if not m:
            continue
        version = int(m.group(1))
        try:
            spec = SchemaSpec.model_validate_json(p.read_text(encoding="utf-8"))
            field_count = len(spec.fields)
        except Exception:
            field_count = 0
        results.append(VersionInfo(
            version=version,
            path=p,
            modified=datetime.fromtimestamp(p.stat().st_mtime),
            field_count=field_count,
        ))
    return results


def load_version(name: str, version: int, schema_dir: Path) -> SchemaSpec:
    """Load a specific archived version of a schema."""
    path = schema_dir / ".history" / f"{name}_v{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"Version {version} not found for schema {name!r}")
    return SchemaSpec.model_validate_json(path.read_text(encoding="utf-8"))


def revert_schema(name: str, version: int, schema_dir: Path) -> Path:
    """Revert a schema to a previous version.

    Archives the current version first, then copies the target version
    to become the current schema file.
    """
    current = schema_dir / f"{name}.json"
    if not current.exists():
        raise FileNotFoundError(f"Schema {name!r} not found in {schema_dir}")
    target = schema_dir / ".history" / f"{name}_v{version}.json"
    if not target.exists():
        raise FileNotFoundError(f"Version {version} not found for schema {name!r}")

    # Archive current before overwriting
    current_spec = SchemaSpec.model_validate_json(current.read_text(encoding="utf-8"))
    _archive_before_save(current_spec, schema_dir)

    # Copy target version to current
    shutil.copy2(target, current)
    return current


def diff_schemas(
    old: SchemaSpec, new: SchemaSpec
) -> tuple[list[str], list[str], list[str]]:
    """Compare two SchemaSpecs and return (added, removed, modified) field names."""
    old_fields = {f.name: f for f in old.fields}
    new_fields = {f.name: f for f in new.fields}

    added = [n for n in new_fields if n not in old_fields]
    removed = [n for n in old_fields if n not in new_fields]

    modified: list[str] = []
    for n in old_fields:
        if n in new_fields:
            of, nf = old_fields[n], new_fields[n]
            if of.type != nf.type or of.required != nf.required or of.description != nf.description:
                modified.append(n)

    return added, removed, modified


# ---------------------------------------------------------------------------
# DSPy Signature & Module
# ---------------------------------------------------------------------------
# DSPy and its transitive dependencies (litellm) can cause import issues
# in some environments (e.g., circular imports inside litellm).  We defer
# the definition of DSPy-dependent classes so that the pure-Python parts
# of this module (FieldSpec, SchemaSpec, compile_schema) remain importable
# even when dspy is not installed or broken at import time.

# ---------------------------------------------------------------------------
# Field manipulation helpers (for CLI-flag refinement)
# ---------------------------------------------------------------------------


def add_field(spec: SchemaSpec, name: str, type_str: str, description: str = "", required: bool = True) -> SchemaSpec:
    """Return a new SchemaSpec with an additional field."""
    new_field = FieldSpec(name=name, type=type_str, description=description, required=required)
    return spec.model_copy(update={"fields": [*spec.fields, new_field]})


def remove_field(spec: SchemaSpec, name: str) -> SchemaSpec:
    """Return a new SchemaSpec with the named field removed."""
    new_fields = [f for f in spec.fields if f.name != name]
    if len(new_fields) == len(spec.fields):
        raise ValueError(f"Field {name!r} not found in schema")
    return spec.model_copy(update={"fields": new_fields})


def rename_field(spec: SchemaSpec, old_name: str, new_name: str) -> SchemaSpec:
    """Return a new SchemaSpec with a field renamed."""
    found = False
    new_fields = []
    for f in spec.fields:
        if f.name == old_name:
            new_fields.append(f.model_copy(update={"name": new_name}))
            found = True
        else:
            new_fields.append(f)
    if not found:
        raise ValueError(f"Field {old_name!r} not found in schema")
    return spec.model_copy(update={"fields": new_fields})


def _schema_quality_score(spec: SchemaSpec) -> float:
    """Heuristic structural quality score used by BestOfN selection."""
    issues = validate_schema_spec(spec)
    total = max(len(spec.fields), 1)
    typed_fields = sum(1 for f in spec.fields if f.type not in {"str", "string"})
    enum_fields = sum(1 for f in spec.fields if f.type == "enum" and (f.enum_values or []))
    required_fields = sum(1 for f in spec.fields if f.required)

    score = 0.0
    try:
        compile_schema(spec)
        score += 0.45
    except Exception:
        score += 0.0

    score += min(0.2, (len(spec.fields) / 12.0) * 0.2)
    score += min(0.15, (typed_fields / total) * 0.15)
    score += min(0.1, (required_fields / total) * 0.1)
    score += min(0.1, (enum_fields / total) * 0.1)

    if issues:
        score *= 0.65

    return max(0.0, min(1.0, score))


def _coerce_schema_spec(value: Any) -> SchemaSpec:
    if isinstance(value, SchemaSpec):
        return value
    if isinstance(value, dict):
        return SchemaSpec.model_validate(value)
    if isinstance(value, str):
        return SchemaSpec.model_validate_json(value)
    raise TypeError(f"Unsupported schema payload type: {type(value).__name__}")


def _is_blank_payload_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        probe = value.strip().lower()
        return probe in {"", "none", "null", "n/a", "na"}
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _sample_probe_value_for_field(field: FieldSpec, *, depth: int = 0) -> Any:
    """Build a deterministic sample value for runtime probe text."""
    if depth >= 3:
        return "value"

    t = _normalize_type_string(
        field.type,
        has_fields=bool(field.fields),
        has_enum_values=bool(field.enum_values),
    )
    if t == "enum":
        return (field.enum_values or ["unknown"])[0]
    if t == "bool":
        return True
    if t == "int":
        return 1
    if t == "float":
        return 1.0
    if t == "object":
        nested = {}
        for nested_field in field.fields or []:
            nested[nested_field.name] = _sample_probe_value_for_field(
                nested_field,
                depth=depth + 1,
            )
        return nested
    if t.startswith("list["):
        inner = t[5:-1].strip().lower()
        if inner == "object":
            first_item = {}
            for nested_field in field.fields or []:
                first_item[nested_field.name] = _sample_probe_value_for_field(
                    nested_field,
                    depth=depth + 1,
                )
            return [first_item or {"item": "value"}]
        if inner == "int":
            return [1, 2]
        if inner == "float":
            return [1.0, 2.0]
        if inner == "bool":
            return [True, False]
        return ["value_a", "value_b"]
    return "value"


def _build_synthetic_runtime_probe_text(
    spec: SchemaSpec,
    *,
    description: str = "",
    example_text: str = "",
) -> str:
    """Create synthetic source text so describe-only generation can be runtime-gated."""
    payload = {}
    for field in spec.fields:
        payload[field.name] = _sample_probe_value_for_field(field)

    blocks: list[str] = [
        "Synthetic document for schema runtime validation.",
        "This payload provides explicit key-value evidence for all schema fields.",
        f"Schema class: {spec.class_name}",
    ]
    if str(description or "").strip():
        blocks.append(f"Schema description: {str(description).strip()[:1200]}")
    if str(example_text or "").strip():
        blocks.append(f"Example context: {str(example_text).strip()[:2000]}")
    blocks.append("Structured sample payload:")
    blocks.append(json.dumps(payload, indent=2, ensure_ascii=False))
    return "\n\n".join(blocks)


def _runtime_validate_schema(
    spec: SchemaSpec,
    *,
    document_text: str,
    missing_required_threshold: float = 0.5,
) -> tuple[bool, list[str]]:
    """Dry-run extraction to validate runtime safety of generated schema."""
    if not str(document_text or "").strip():
        return True, []

    issues: list[str] = []
    try:
        model = compile_schema(spec)
    except Exception as exc:
        return False, [f"runtime_compile_error: {exc}"]

    try:
        from .extraction import DocumentExtractor

        extractor = DocumentExtractor(output_schema=model)
        result = extractor(document_text=str(document_text)[:30000])
        extracted = getattr(result, "extracted", result)
        if hasattr(extracted, "model_dump"):
            payload = extracted.model_dump()
        elif isinstance(extracted, dict):
            payload = extracted
        else:
            payload = {}

        required = [f.name for f in spec.fields if f.required]
        if required:
            missing = [name for name in required if _is_blank_payload_value(payload.get(name))]
            ratio = len(missing) / max(len(required), 1)
            if ratio > max(0.0, min(1.0, float(missing_required_threshold))):
                issues.append(
                    "runtime_missing_required_fields: "
                    + ", ".join(missing[:10])
                )
        return len(issues) == 0, issues
    except Exception as exc:
        return False, [f"runtime_extract_error: {str(exc)[:500]}"]


def _is_structured_parse_failure(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    markers = (
        "jsonadapter",
        "adapterparseerror",
        "cannot be serialized to a json object",
        "failed to parse lm response",
    )
    return any(marker in msg for marker in markers)


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


def _recover_schema_with_outlines(
    *,
    description: str,
    example_text: str,
    document_text: str,
    error_hint: str,
) -> SchemaSpec | None:
    """Recover SchemaSpec via Outlines when DSPy JSON parsing fails."""
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
        generator = outlines.Generator(model, outlines.json_schema(SchemaSpec))
        prompt = (
            "Generate a strict medical extraction SchemaSpec JSON object.\n"
            "Rules:\n"
            "- class_name must be PascalCase.\n"
            "- field names must be snake_case identifiers.\n"
            "- Allowed types: str, int, float, bool, object, list[X], enum.\n"
            "- enum requires enum_values.\n"
            "- list[object]/object may include nested fields.\n"
            f"Prior parser failure hint: {error_hint}\n\n"
            f"Description:\n{description[:5000]}\n\n"
            f"Example text:\n{example_text[:10000]}\n\n"
            f"Document text:\n{document_text[:20000]}"
        )
        raw = generator(prompt, temperature=0.0, max_tokens=1800)
        parsed = parse_json_like(raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False))
        if parsed is None:
            return None
        return SchemaSpec.model_validate(parsed)
    except Exception as exc:
        logger.warning("Outlines schema recovery failed: %s", exc)
        return None


def _build_dspy_classes():
    """Lazily define and return DSPy signature/module classes.

    Called on first access via module-level __getattr__.
    """
    import dspy  # noqa: F811  — intentional lazy import

    class GenerateSchemaSpec(dspy.Signature):
        """Given a description of a medical document and optionally example text
        or actual document content, generate a structured SchemaSpec that
        captures all relevant fields.

        Choose the most specific type for each field:
        - Use 'enum' with enum_values for categorical fields (modality, severity, laterality).
        - Use 'bool' for yes/no or present/absent fields.
        - Use 'list[str]' for multi-value fields (findings, diagnoses).
        - Use 'list[object]' for repeating structured groups and provide nested
          field definitions using the "fields" attribute.
        - Use 'int' or 'float' for numeric measurements.
        - Use 'str' only for genuinely free-text content (impressions, narratives).

        Use snake_case for field names and PascalCase for class_name."""

        description: str = dspy.InputField(
            desc="Natural-language description of the document type to structure",
            default="",
        )
        example_text: str = dspy.InputField(
            desc="Optional example document text for grounding",
            default="",
        )
        document_text: str = dspy.InputField(
            desc="Optional full document text to infer schema structure from",
            default="",
        )
        schema_spec: SchemaSpec = dspy.OutputField(
            desc="The generated schema specification as a SchemaSpec JSON object"
        )

    class RepairSchemaSpec(dspy.Signature):
        """Repair an invalid SchemaSpec using explicit validation and compile errors."""

        description: str = dspy.InputField(
            desc="Original description used for schema generation",
            default="",
        )
        example_text: str = dspy.InputField(
            desc="Optional example document text used for grounding",
            default="",
        )
        document_text: str = dspy.InputField(
            desc="Optional full document text used for grounding",
            default="",
        )
        invalid_schema: str = dspy.InputField(
            desc="Current invalid or weak schema as JSON",
        )
        validation_issues: str = dspy.InputField(
            desc="Line-separated validation issues and compile errors to fix",
        )
        repaired_schema: SchemaSpec = dspy.OutputField(
            desc="A corrected SchemaSpec that resolves the listed issues",
        )

    class SchemaGenerator(dspy.Module):
        """Generate a Pydantic model from a text description.

        Uses ``dspy.ChainOfThought(GenerateSchemaSpec)`` so that prompts
        are compiled by DSPy and can be optimised with GEPA / MIPROv2.
        """

        def __init__(self) -> None:
            super().__init__()
            base_generate = dspy.ChainOfThought(GenerateSchemaSpec)
            self.generate = base_generate
            if hasattr(dspy, "BestOfN"):
                def reward_fn(_args: dict[str, Any], pred: Any) -> float:
                    try:
                        candidate = normalize_schema_spec(
                            _coerce_schema_spec(getattr(pred, "schema_spec", pred))
                        )
                    except Exception:
                        return 0.0
                    return _schema_quality_score(candidate)

                try:
                    self.generate = dspy.BestOfN(
                        module=base_generate,
                        N=3,
                        reward_fn=reward_fn,
                        threshold=0.72,
                    )
                except Exception:
                    self.generate = base_generate

            self.repair = dspy.ChainOfThought(RepairSchemaSpec)

        def forward(
            self,
            description: str = "",
            example_text: str = "",
            document_text: str = "",
            max_repairs: int = 2,
            runtime_dryrun: bool = False,
            runtime_missing_required_threshold: float = 0.5,
        ) -> dspy.Prediction:
            """Run the schema generation pipeline."""
            from mosaicx.metrics import PipelineMetrics, get_tracker, track_step

            metrics = PipelineMetrics()
            tracker = get_tracker()
            generated: Any = None
            parse_error: Exception | None = None
            spec: SchemaSpec | None = None

            with track_step(metrics, "Generate schema", tracker):
                try:
                    generated = self.generate(
                        description=description,
                        example_text=example_text,
                        document_text=document_text,
                    )
                    spec = normalize_schema_spec(
                        _coerce_schema_spec(getattr(generated, "schema_spec", generated))
                    )
                except Exception as exc:
                    parse_error = exc

            if spec is None and parse_error is not None and _is_structured_parse_failure(parse_error):
                with track_step(metrics, "Schema recovery (Outlines)", tracker):
                    recovered = _recover_schema_with_outlines(
                        description=description,
                        example_text=example_text,
                        document_text=document_text,
                        error_hint=str(parse_error),
                    )
                    if recovered is not None:
                        spec = normalize_schema_spec(recovered)

            if spec is None and parse_error is not None:
                raise parse_error
            if spec is None:
                raise ValueError("Schema generation returned no schema specification.")

            structural_issues = validate_schema_spec(spec)
            compile_error: str | None = None
            compiled: type[BaseModel] | None = None
            try:
                compiled = compile_schema(spec)
            except Exception as exc:
                compile_error = str(exc)
                structural_issues.append(f"compile_error: {compile_error}")

            runtime_issues: list[str] = []
            runtime_enabled = bool(runtime_dryrun)
            runtime_source_text = str(document_text or "").strip()
            if runtime_enabled and compiled is not None:
                runtime_probe_text = runtime_source_text or _build_synthetic_runtime_probe_text(
                    spec,
                    description=description,
                    example_text=example_text,
                )
                runtime_ok, runtime_issues = _runtime_validate_schema(
                    spec,
                    document_text=runtime_probe_text,
                    missing_required_threshold=runtime_missing_required_threshold,
                )
                if not runtime_ok and runtime_issues:
                    logger.info(
                        "Schema runtime dry-run flagged issues: %s",
                        "; ".join(runtime_issues),
                    )

            repairs_remaining = max(0, int(max_repairs))
            while (structural_issues or compiled is None or runtime_issues) and repairs_remaining > 0:
                repairs_remaining -= 1
                with track_step(metrics, "Repair schema", tracker):
                    repair_result = self.repair(
                        description=description,
                        example_text=example_text,
                        document_text=document_text,
                        invalid_schema=spec.model_dump_json(indent=2),
                        validation_issues="\n".join((structural_issues + runtime_issues)[:20]) or "compile failed",
                    )
                spec = normalize_schema_spec(
                    _coerce_schema_spec(getattr(repair_result, "repaired_schema", repair_result))
                )
                structural_issues = validate_schema_spec(spec)
                compile_error = None
                try:
                    compiled = compile_schema(spec)
                except Exception as exc:
                    compile_error = str(exc)
                    compiled = None
                    structural_issues.append(f"compile_error: {compile_error}")

                runtime_issues = []
                if runtime_enabled and compiled is not None:
                    runtime_probe_text = runtime_source_text or _build_synthetic_runtime_probe_text(
                        spec,
                        description=description,
                        example_text=example_text,
                    )
                    runtime_ok, runtime_issues = _runtime_validate_schema(
                        spec,
                        document_text=runtime_probe_text,
                        missing_required_threshold=runtime_missing_required_threshold,
                    )
                    if not runtime_ok and runtime_issues:
                        logger.info(
                            "Schema runtime dry-run flagged issues after repair: %s",
                            "; ".join(runtime_issues),
                        )

            if compiled is None or structural_issues or runtime_issues:
                all_issues = structural_issues + runtime_issues
                joined = "; ".join(all_issues[:8]) if all_issues else (compile_error or "unknown error")
                raise ValueError(f"Schema generation failed after repair attempts: {joined}")

            self._last_metrics = metrics
            return dspy.Prediction(
                schema_spec=spec,
                compiled_model=compiled,
                schema_issues=structural_issues + runtime_issues,
                runtime_dryrun_used=runtime_enabled,
            )

    class RefineSchemaSpec(dspy.Signature):
        """Refine an existing schema based on user instructions.
        Preserve existing fields unless the instruction says to change them."""

        current_schema: str = dspy.InputField(
            desc="Current SchemaSpec as JSON"
        )
        instruction: str = dspy.InputField(
            desc="User instruction describing how to refine the schema"
        )
        refined_schema: SchemaSpec = dspy.OutputField(
            desc="The updated SchemaSpec incorporating the requested changes"
        )

    class SchemaRefiner(dspy.Module):
        """Refine an existing schema based on user instructions.

        Uses ``dspy.ChainOfThought(RefineSchemaSpec)`` so that prompts
        are compiled by DSPy and can be optimised with GEPA / MIPROv2.
        """

        def __init__(self) -> None:
            super().__init__()
            self.refine = dspy.ChainOfThought(RefineSchemaSpec)

        def forward(self, current_schema: str, instruction: str) -> dspy.Prediction:
            from mosaicx.metrics import PipelineMetrics, get_tracker, track_step

            metrics = PipelineMetrics()
            tracker = get_tracker()

            with track_step(metrics, "Refine schema", tracker):
                result = self.refine(
                    current_schema=current_schema,
                    instruction=instruction,
                )

            spec: SchemaSpec = result.refined_schema
            compiled = compile_schema(spec)
            self._last_metrics = metrics
            return dspy.Prediction(
                schema_spec=spec,
                compiled_model=compiled,
            )

    return GenerateSchemaSpec, SchemaGenerator, RefineSchemaSpec, SchemaRefiner


# Cache for lazily-built DSPy classes
_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({
    "GenerateSchemaSpec", "SchemaGenerator",
    "RefineSchemaSpec", "SchemaRefiner",
})


def __getattr__(name: str):
    """Module-level __getattr__ for lazy loading of DSPy classes."""
    global _dspy_classes

    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            gen_sig, gen_mod, ref_sig, ref_mod = _build_dspy_classes()
            _dspy_classes = {
                "GenerateSchemaSpec": gen_sig,
                "SchemaGenerator": gen_mod,
                "RefineSchemaSpec": ref_sig,
                "SchemaRefiner": ref_mod,
            }
        return _dspy_classes[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
