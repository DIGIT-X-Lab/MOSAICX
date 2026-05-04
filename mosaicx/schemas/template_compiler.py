"""YAML template compiler — converts YAML template definitions into Pydantic models.

This module powers the ``mosaicx extract --template my_custom.yaml`` command by
parsing user-defined YAML templates and producing validated Pydantic models at
runtime.  No ``exec()`` is used; all dynamic model creation goes through
:func:`pydantic.create_model`.

Public API
----------
- :func:`parse_template` — parse raw YAML into a :class:`TemplateMeta` descriptor.
- :func:`compile_template` — compile a YAML string into a Pydantic ``BaseModel`` subclass.
- :func:`compile_template_file` — same, but reads from a file path.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import re
from typing import Any, List, Optional, Type

import yaml
from pydantic import BaseModel, Field, create_model

__all__ = [
    "SectionSpec",
    "TemplateMeta",
    "parse_template",
    "compile_template",
    "compile_template_file",
    "schema_spec_to_template_yaml",
]

# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


class FieldSpec(BaseModel):
    """Descriptor for a single field inside an ``object`` item.

    ``name`` is optional because list ``item`` descriptors describe the
    element type without being a named field themselves.
    """

    name: Optional[str] = None
    type: str  # str | int | float | bool | enum | object
    required: bool = True
    description: Optional[str] = None
    values: Optional[List[str]] = None  # for enum type
    value_labels: Optional[dict[str, str]] = None  # optional enum value labels
    metadata: Optional[dict[str, Any]] = None  # optional source/catalog metadata
    fields: Optional[List["FieldSpec"]] = None  # for nested object type
    item: Optional["FieldSpec"] = None  # for list type


class SectionSpec(BaseModel):
    """Descriptor for one top-level section in a template."""

    name: str
    type: str  # str | int | float | bool | enum | list | object
    required: bool = True
    description: Optional[str] = None
    values: Optional[List[str]] = None  # for enum type
    value_labels: Optional[dict[str, str]] = None  # optional enum value labels
    metadata: Optional[dict[str, Any]] = None  # optional source/catalog metadata
    item: Optional[FieldSpec] = None  # for list type (describes each element)
    fields: Optional[List[FieldSpec]] = None  # for object type


class TemplateMeta(BaseModel):
    """Parsed metadata for an entire template."""

    name: str
    description: Optional[str] = None
    radreport_id: Optional[str] = None
    mode: Optional[str] = None  # pipeline mode (e.g. "radiology", "pathology")
    sections: List[SectionSpec]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_template(yaml_content: str) -> TemplateMeta:
    """Parse a YAML template string into a :class:`TemplateMeta` descriptor.

    Parameters
    ----------
    yaml_content:
        Raw YAML text conforming to the MOSAICX template schema.

    Returns
    -------
    TemplateMeta
        Validated template metadata with all sections resolved.
    """
    data = yaml.safe_load(yaml_content)
    meta = TemplateMeta(**data)
    _normalize_optional_enum_absence_values(meta)
    return meta


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


def _has_absence_enum(values: list[str]) -> bool:
    for value in values:
        if " ".join(str(value or "").strip().lower().split()) in _ABSENCE_ENUM_TOKENS:
            return True
    return False


def _absence_label_for(values: list[str]) -> str:
    alpha_values = [v for v in values if re.search(r"[A-Za-z]", str(v))]
    if alpha_values and all(str(v)[:1].isupper() for v in alpha_values):
        return "None"
    return "none"


def _normalize_optional_enum_absence_values(meta: TemplateMeta) -> None:
    """Ensure optional enum fields can encode explicit absence.

    Legacy/user templates sometimes define optional enums without any neutral
    value (for example only `Mild|Moderate|Severe`). In such cases extraction
    tends to collapse "no finding" into null. Adding an explicit absence token
    preserves medically meaningful negatives (for example `none`).
    """

    def _visit_field(field: FieldSpec | SectionSpec) -> None:
        if field.type == "enum":
            values = list(field.values or [])
            if values and (not field.required) and (not _has_absence_enum(values)):
                values.append(_absence_label_for(values))
                field.values = values
            return

        if field.type == "object" and field.fields:
            for child in field.fields:
                _visit_field(child)
            return

        if field.type == "list" and field.item is not None:
            _visit_field(field.item)

    for section in meta.sections:
        _visit_field(section)


def _description_with_values(spec: FieldSpec | SectionSpec) -> str:
    """Build a field description that exposes enum constraints to extraction."""
    desc = spec.description or ""
    if spec.type == "enum" and spec.values:
        labels = spec.value_labels or {}
        values_str = ", ".join(
            f"{value}={labels[value]}" if value in labels else str(value)
            for value in spec.values
        )
        return f"{desc} (allowed: {values_str})" if desc else f"One of: {values_str}"
    return desc


# ---------------------------------------------------------------------------
# Type compilation helpers
# ---------------------------------------------------------------------------

# Simple scalar type map
_SCALAR_TYPES: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def _build_enum(name: str, values: list[str]) -> type:
    """Create a string ``Enum`` subclass with the given values.

    Each enum member's *name* and *value* are set to the original string so
    that Pydantic serialises enums as their string value.
    """
    return Enum(name, {v: v for v in values}, type=str)  # type: ignore[misc]


def _build_nested_model(
    model_name: str,
    fields: list[FieldSpec],
) -> type[BaseModel]:
    """Recursively build a Pydantic model from a list of :class:`FieldSpec`."""
    field_definitions: dict[str, Any] = {}

    for fspec in fields:
        assert fspec.name is not None, "Fields inside an object must have a name."
        py_type = _resolve_type(fspec, parent_name=model_name)
        desc = _description_with_values(fspec)
        if fspec.required:
            field_definitions[fspec.name] = (py_type, Field(..., description=desc or None))
        else:
            field_definitions[fspec.name] = (Optional[py_type], Field(None, description=desc or None))

    return create_model(model_name, **field_definitions)  # type: ignore[call-overload]


def _resolve_type(
    spec: FieldSpec | SectionSpec,
    *,
    parent_name: str = "",
) -> type:
    """Map a spec's ``type`` string to a concrete Python / Pydantic type."""
    type_str = spec.type

    # Scalar types
    if type_str in _SCALAR_TYPES:
        return _SCALAR_TYPES[type_str]

    # Enum
    if type_str == "enum":
        if not spec.values:
            raise ValueError(
                f"Enum field '{getattr(spec, 'name', '?')}' must provide 'values'."
            )
        spec_name = getattr(spec, "name", None) or "Item"
        enum_name = f"{parent_name}_{spec_name}".lstrip("_").title().replace("_", "")
        return _build_enum(enum_name, spec.values)

    # Object (inline nested model)
    if type_str == "object":
        if not spec.fields:
            raise ValueError(
                f"Object field '{getattr(spec, 'name', '?')}' must provide 'fields'."
            )
        spec_name = getattr(spec, "name", None) or "Item"
        nested_name = f"{parent_name}_{spec_name}".lstrip("_").title().replace("_", "")
        return _build_nested_model(nested_name, spec.fields)

    # List
    if type_str == "list":
        if spec.item is None:
            # Default to list[str] when no item spec is provided
            return list[str]  # type: ignore[return-value]
        item_type = _resolve_type(spec.item, parent_name=parent_name)
        return list[item_type]  # type: ignore[valid-type]

    raise ValueError(f"Unsupported type '{type_str}' in field '{getattr(spec, 'name', '?')}'.")


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_template(yaml_content: str) -> type[BaseModel]:
    """Compile a YAML template string into a Pydantic ``BaseModel`` subclass.

    Parameters
    ----------
    yaml_content:
        Raw YAML text conforming to the MOSAICX template schema.

    Returns
    -------
    type[BaseModel]
        A dynamically created Pydantic model class whose ``__name__`` matches
        the template's ``name`` field and whose fields correspond to the
        template's ``sections``.
    """
    meta = parse_template(yaml_content)
    field_definitions: dict[str, Any] = {}

    for section in meta.sections:
        py_type = _resolve_type(section, parent_name=meta.name)
        # Build description — for enum fields, include allowed values
        desc = _description_with_values(section)
        if section.required:
            field_definitions[section.name] = (py_type, Field(..., description=desc or None))
        else:
            field_definitions[section.name] = (Optional[py_type], Field(None, description=desc or None))

    return create_model(meta.name, **field_definitions)  # type: ignore[call-overload]


def compile_template_file(path: str | Path) -> type[BaseModel]:
    """Read a YAML template file and compile it into a Pydantic model.

    Parameters
    ----------
    path:
        Filesystem path to a ``.yaml`` / ``.yml`` template file.

    Returns
    -------
    type[BaseModel]
        Compiled Pydantic model class.
    """
    content = Path(path).read_text(encoding="utf-8")
    return compile_template(content)


def _generate_objective(meta: "TemplateMeta", cache_path: Path) -> str:
    """Generate extraction objective from field descriptions via one LLM call. Cache to disk."""
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip()

    # Collect all field descriptions that contain rules
    descriptions = []

    def _collect(sections, prefix=""):
        for s in sections:
            desc = _description_with_values(s)
            if desc:
                name = f"{prefix}{s.name}" if prefix else s.name
                descriptions.append(f"{name}: {desc}")
            if s.fields:
                _collect(s.fields, prefix=f"{s.name}.")

    _collect(meta.sections)

    prompt = (
        "Below are field descriptions from a medical data extraction template.\n"
        "Write a single concise paragraph (2-3 sentences max) that summarizes "
        "the key extraction rules and format requirements. "
        "Use shorthand like 'TUR=Biopsie' or 'Gleason=X+Y=Z'. "
        "Only include rules that constrain values or formats — skip plain descriptions. "
        "This will be used as the extraction objective for an LLM.\n\n"
        "Field descriptions:\n"
        + "\n".join(f"- {d}" for d in descriptions)
    )

    try:
        import dspy
        result = dspy.settings.lm(prompt=prompt, max_tokens=200, temperature=0.0)
        if isinstance(result, list) and result:
            objective = result[0] if isinstance(result[0], str) else str(result[0])
        else:
            objective = str(result)
        objective = objective.strip()
    except Exception:
        # Fallback: extract rules deterministically
        objective = _generate_objective_deterministic(descriptions)

    cache_path.write_text(objective, encoding="utf-8")
    return objective


def _generate_objective_deterministic(descriptions: list[str]) -> str:
    """Fallback: extract rules from descriptions using keyword matching."""
    rule_keywords = ["must be", "always", "use ", "not ", "format", "count", "exactly", "highest", "overall"]
    rules = []
    for desc in descriptions:
        lower = desc.lower()
        if any(kw in lower for kw in rule_keywords):
            # Extract the rule part after the colon
            parts = desc.split(": ", 1)
            if len(parts) > 1:
                rules.append(parts[1].strip())
    if rules:
        return "Extract structured data. " + " ".join(rules[:10])
    return "Extract structured data from the document. Cite exact excerpts for each field."


def compile_template_file_inline(path: str | Path) -> type[BaseModel]:
    """Compile YAML template with inline evidence: each leaf field becomes {value, excerpt}.

    Auto-generates extraction objective from field descriptions (cached to disk).
    """
    from typing import Any as _Any

    path = Path(path)
    content = path.read_text(encoding="utf-8")
    meta = parse_template(content)

    # Generate and cache objective
    cache_path = path.parent / f".{path.stem}.objective.cache"
    objective = _generate_objective(meta, cache_path)

    def _make_evidence_field(name: str, desc: str) -> type[BaseModel]:
        return create_model(
            name.title().replace("_", "") + "F",
            value=(Optional[_Any], Field(default=None, description=desc)),
            excerpt=(Optional[str], Field(default=None, description="Exact quote from report")),
        )

    def _resolve_inline(section: "SectionSpec", parent_name: str = "") -> tuple:
        desc = _description_with_values(section)
        if section.type == "object" and section.fields:
            nested: dict[str, Any] = {}
            for sub in section.fields:
                nested[sub.name] = _resolve_inline(sub, section.name)
            nm = create_model(section.name.title().replace("_", ""), **nested)
            if section.required:
                return (nm, Field(description=desc))
            return (Optional[nm], Field(default=None, description=desc))
        ev_model = _make_evidence_field(section.name, desc)
        if section.required:
            return (ev_model, Field(description=desc))
        return (Optional[ev_model], Field(default=None, description=desc))

    fields: dict[str, Any] = {}
    for section in meta.sections:
        fields[section.name] = _resolve_inline(section)

    model = create_model(meta.name + "Inline", **fields)
    # Attach objective to model so DocumentExtractor can read it
    model.__extraction_objective__ = objective
    return model


# ---------------------------------------------------------------------------
# SchemaSpec → YAML conversion
# ---------------------------------------------------------------------------


def schema_spec_to_template_yaml(
    spec: Any,
    *,
    mode: str | None = None,
) -> str:
    """Convert a SchemaSpec (from ``schema_gen``) into YAML template format.

    This bridges the two schema representations:

    - **SchemaSpec** (``schema_gen.py``): flat fields with type strings like
      ``"list[str]"``, ``"enum"``.
    - **YAML template**: nested structure with ``type: list`` + ``item:``
      blocks, ``type: enum`` + ``values:`` blocks.

    Parameters
    ----------
    spec:
        A ``SchemaSpec`` instance (from ``mosaicx.pipelines.schema_gen``).
    mode:
        Optional pipeline mode to embed (e.g. ``"radiology"``).

    Returns
    -------
    str
        YAML string conforming to the MOSAICX template format.
    """
    import re

    list_re = re.compile(r"^list\[(\w+)\]$", re.IGNORECASE)

    def _field_spec_to_dict(field: Any) -> dict[str, Any]:
        """Recursively convert schema_gen.FieldSpec into template FieldSpec dict."""
        field_type = field.type.strip().lower()
        alias_map = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
        }
        normalized_type = alias_map.get(field_type, field_type)
        payload: dict[str, Any] = {"name": field.name, "type": normalized_type}

        if getattr(field, "description", ""):
            payload["description"] = field.description
        if not getattr(field, "required", True):
            payload["required"] = False

        if normalized_type == "enum":
            enum_values = getattr(field, "enum_values", None) or getattr(field, "values", None)
            if enum_values:
                payload["values"] = list(enum_values)
            value_labels = getattr(field, "value_labels", None)
            if value_labels:
                payload["value_labels"] = dict(value_labels)
        metadata = getattr(field, "metadata", None)
        if metadata:
            payload["metadata"] = dict(metadata)

        if normalized_type == "object":
            nested = getattr(field, "fields", None)
            if nested:
                payload["fields"] = [_field_spec_to_dict(sub) for sub in nested]

        list_match = list_re.match(field_type)
        if list_match:
            inner = alias_map.get(list_match.group(1).lower(), list_match.group(1).lower())
            payload["type"] = "list"
            if inner == "object":
                nested = getattr(field, "fields", None)
                if nested:
                    payload["item"] = {
                        "type": "object",
                        "fields": [_field_spec_to_dict(sub) for sub in nested],
                    }
                else:
                    # Safe fallback: avoid emitting invalid object item with no fields.
                    payload["item"] = {"type": "str"}
            else:
                payload["item"] = {"type": inner}

        return payload

    sections: list[dict[str, Any]] = []

    for field in spec.fields:
        section: dict[str, Any] = {"name": field.name}
        type_str = field.type.strip().lower()

        # Parse list[X] types
        m = list_re.match(type_str)
        if m:
            section["type"] = "list"
            inner = m.group(1).lower()
            # Normalise aliases
            inner_map = {
                "string": "str", "integer": "int",
                "number": "float", "boolean": "bool",
            }
            inner = inner_map.get(inner, inner)
            if inner == "object":
                nested = getattr(field, "fields", None)
                if nested:
                    section["item"] = {
                        "type": "object",
                        "fields": [_field_spec_to_dict(sub) for sub in nested],
                    }
                else:
                    # Safe fallback: avoid generating an invalid template.
                    section["item"] = {"type": "str"}
            else:
                section["item"] = {"type": inner}
        elif type_str == "enum":
            section["type"] = "enum"
            if field.enum_values:
                section["values"] = list(field.enum_values)
            value_labels = getattr(field, "value_labels", None)
            if value_labels:
                section["value_labels"] = dict(value_labels)
        elif type_str == "object":
            nested = getattr(field, "fields", None)
            if nested:
                section["type"] = "object"
                section["fields"] = [_field_spec_to_dict(sub) for sub in nested]
            else:
                # Safe fallback: avoid invalid object definitions with missing fields.
                section["type"] = "str"
        else:
            # Normalise scalar aliases
            alias_map = {
                "string": "str", "integer": "int",
                "number": "float", "boolean": "bool",
            }
            section["type"] = alias_map.get(type_str, type_str)

        if not field.required:
            section["required"] = False
        if field.description:
            section["description"] = field.description
        metadata = getattr(field, "metadata", None)
        if metadata:
            section["metadata"] = dict(metadata)

        sections.append(section)

    template: dict[str, Any] = {"name": spec.class_name}
    if spec.description:
        template["description"] = spec.description
    if mode:
        template["mode"] = mode
    template["sections"] = sections

    return yaml.dump(
        template,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
