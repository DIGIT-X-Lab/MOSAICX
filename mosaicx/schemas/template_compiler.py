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
    fields: Optional[List["FieldSpec"]] = None  # for nested object type
    item: Optional["FieldSpec"] = None  # for list type


class SectionSpec(BaseModel):
    """Descriptor for one top-level section in a template."""

    name: str
    type: str  # str | int | float | bool | enum | list | object
    required: bool = True
    description: Optional[str] = None
    values: Optional[List[str]] = None  # for enum type
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
    return TemplateMeta(**data)


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
        desc = fspec.description or ""
        if fspec.type == "enum" and fspec.values:
            values_str = ", ".join(fspec.values)
            desc = f"{desc} (allowed: {values_str})" if desc else f"One of: {values_str}"
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
        desc = section.description or ""
        if section.type == "enum" and section.values:
            values_str = ", ".join(section.values)
            desc = f"{desc} (allowed: {values_str})" if desc else f"One of: {values_str}"
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

    sections: list[dict[str, Any]] = []
    list_re = re.compile(r"^list\[(\w+)\]$", re.IGNORECASE)

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
            section["item"] = {"type": inner}
        elif type_str == "enum":
            section["type"] = "enum"
            if field.enum_values:
                section["values"] = list(field.enum_values)
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
