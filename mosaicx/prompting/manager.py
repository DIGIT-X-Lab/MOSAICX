from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type
import textwrap

from pydantic import BaseModel

from ..constants import (
    PACKAGE_PROMPT_TEMPLATES_DIR,
    PROMPT_BASE_FILENAME,
    PROMPT_NOTES_FILENAME,
    PROMPT_OPTIMIZED_FILENAME,
)


class PromptVariant(str, Enum):
    """Available prompt variants stored on disk."""

    BASE = "base"
    OPTIMIZED = "optimized"


class PromptPreference(str, Enum):
    """Caller preference when resolving prompts."""

    AUTO = "auto"
    BASE = PromptVariant.BASE.value
    OPTIMIZED = PromptVariant.OPTIMIZED.value


@dataclass(slots=True)
class PromptArtifact:
    """Resolved prompt metadata used by the extraction pipeline."""

    content: str
    path: Path
    variant: PromptVariant
    schema_hash: str
    prompt_dir: Path


def synthesise_base_prompt(schema_class: Type[BaseModel]) -> str:
    """
    Generate a schema-conditioned base prompt that can be persisted for reuse.

    The prompt encodes:
      - A high-level description derived from the model docstring.
      - Per-field guidance including type hints, enums, and constraints.
      - The full JSON Schema for strict validation.
      - A canonical response skeleton initialised with nulls/empty containers.
    """

    schema_json = schema_class.model_json_schema()
    schema_str = json.dumps(schema_json, indent=2, ensure_ascii=False)
    class_name = schema_class.__name__
    module_name = schema_class.__module__
    class_doc = inspect.getdoc(schema_class) or "No class-level documentation supplied."

    field_guidance = _build_field_guidance(schema_class, schema_json)
    response_template = json.dumps(
        _build_default_payload(schema_json, schema_json),
        indent=2,
        ensure_ascii=False,
    )

    lines = [
        "You are an extraction specialist converting radiology reports into a JSON payload",
        f"that strictly conforms to the `{class_name}` schema defined in `{module_name}`.",
        "Work only with evidence stated in the provided report—never hallucinate data.",
        "",
        "Extraction directives",
        "---------------------",
        "• Read the entire report (history, technique, findings, impression) before deciding.",
        "• Apply consistent label semantics across the record:",
        "      1.0  → finding explicitly present",
        "      0.0  → finding explicitly negated",
        "     -1.0  → mention is uncertain or qualified",
        "      null → finding not mentioned; use null for optional keys with no evidence.",
        "• When statements conflict, prioritise definitive diagnostic language (typically the "
        "Impression section). If ambiguity remains, prefer -1.0 and cite the conflicting phrases.",
        "• Confidence scores must be floats between 0.0 and 1.0 reflecting how certain the "
        "assignment is; use null only when the label is null.",
        "• supporting_text should contain the minimal quote(s) that justify the label.",
        "• Return ONLY valid JSON—no markdown, no commentary, no extra keys.",
        "",
        "Schema overview",
        "----------------",
        textwrap.indent(class_doc.strip(), prefix="  "),
        "",
        "Field guidance",
        "--------------",
        field_guidance,
        "",
        "Response skeleton (strict key order):",
        response_template,
        "",
        "Full JSON Schema (source of truth):",
        schema_str,
        "",
        "Use this prompt as the starting point for optimisation passes (e.g., DSPy CoT, GEPRO).",
    ]

    return "\n".join(lines).strip() + "\n"


def resolve_prompt_for_schema(
    schema_class: Type[BaseModel],
    *,
    preference: PromptPreference | str = PromptPreference.AUTO,
    allow_synthesis: bool = True,
) -> PromptArtifact:
    """
    Resolve the prompt content for ``schema_class`` honouring the caller preference.

    Order of resolution:
      1. Explicit preference for OPTIMIZED if the file exists.
      2. Fallback to BASE (auto-generated or previously curated).
      3. If base prompt missing and ``allow_synthesis`` is True, synthesise and persist it.
    """

    pref = PromptPreference(preference)
    schema_json = schema_class.model_json_schema()
    schema_hash = _hash_schema(schema_json)
    prompt_dir = _ensure_prompt_dir(schema_class, schema_hash)
    notes_path = prompt_dir / PROMPT_NOTES_FILENAME
    notes = _load_notes(notes_path)

    base_path = prompt_dir / PROMPT_BASE_FILENAME
    optimised_path = prompt_dir / PROMPT_OPTIMIZED_FILENAME

    base_content: Optional[str] = None
    if base_path.exists():
        base_content = base_path.read_text(encoding="utf-8")

    if (not base_content or notes.get("schema_hash") != schema_hash) and allow_synthesis:
        base_content = synthesise_base_prompt(schema_class)
        base_path.write_text(base_content, encoding="utf-8")
        notes["base_prompt"] = {
            "path": str(base_path),
            "updated_at": _timestamp(),
        }

    if base_content is None:
        raise FileNotFoundError(
            f"Base prompt missing for {schema_class.__name__}; "
            f"expected at {base_path}"
        )

    chosen_variant = PromptVariant.BASE
    chosen_path = base_path
    chosen_content = base_content

    if pref in {PromptPreference.OPTIMIZED, PromptPreference.AUTO} and optimised_path.exists():
        chosen_variant = PromptVariant.OPTIMIZED
        chosen_path = optimised_path
        chosen_content = optimised_path.read_text(encoding="utf-8")
        notes.setdefault("optimized_prompt", {})
        notes["optimized_prompt"].update(
            {"path": str(optimised_path), "last_used": _timestamp()}
        )
    else:
        notes.setdefault("base_prompt", {})
        notes["base_prompt"].update({"path": str(base_path), "last_used": _timestamp()})

    notes.update(
        {
            "schema_module": schema_class.__module__,
            "schema_class": schema_class.__name__,
            "schema_hash": schema_hash,
            "prompt_dir": str(prompt_dir),
            "last_resolved": _timestamp(),
            "preferred_variant": pref.value,
            "resolved_variant": chosen_variant.value,
        }
    )
    _save_notes(notes_path, notes)

    return PromptArtifact(
        content=chosen_content,
        path=chosen_path,
        variant=chosen_variant,
        schema_hash=schema_hash,
        prompt_dir=prompt_dir,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_prompt_dir(schema_class: Type[BaseModel], schema_hash: str) -> Path:
    slug = _schema_slug(schema_class, schema_hash)
    prompt_dir = PACKAGE_PROMPT_TEMPLATES_DIR / slug
    prompt_dir.mkdir(parents=True, exist_ok=True)
    return prompt_dir


def _schema_slug(schema_class: Type[BaseModel], schema_hash: str) -> str:
    module = getattr(schema_class, "__module__", "") or ""
    module = module.split(".")[-1]
    if not module or module == "schema_module":
        module = schema_class.__name__
    module = module.lower()
    return f"{module}__{schema_class.__name__.lower()}_{schema_hash[:8]}"


def _hash_schema(schema_json: Dict[str, Any]) -> str:
    payload = json.dumps(schema_json, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_notes(notes_path: Path) -> Dict[str, Any]:
    if notes_path.exists():
        try:
            return json.loads(notes_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_notes(notes_path: Path, payload: Dict[str, Any]) -> None:
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _json_pointer_get(doc: Dict[str, Any], pointer: str) -> Dict[str, Any]:
    if not pointer or pointer == "#":
        return doc
    if not pointer.startswith("#/"):
        raise KeyError(f"Unsupported $ref pointer: {pointer}")
    parts = pointer[2:].split("/")
    cur: Any = doc
    for part in parts:
        part = part.replace("~1", "/").replace("~0", "~")
        cur = cur[part]
    return cur


def _deref(schema: Dict[str, Any], root: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        try:
            return _json_pointer_get(root, ref)
        except Exception:
            return schema
    return schema


def _build_field_guidance(
    schema_class: Type[BaseModel],
    schema_json: Dict[str, Any],
) -> str:
    props = schema_json.get("properties", {}) or {}
    if not props:
        return "  - No field metadata available."

    lines = []
    for name, field in schema_class.model_fields.items():
        spec = props.get(name)
        if not spec:
            continue
        deref = _deref(spec, schema_json)
        type_summary = _summarise_type(deref, schema_json)
        description = deref.get("description") or field.description or ""
        piece = f"- {name}: {type_summary}"
        if description:
            piece += f". {description.strip()}"
        lines.append(piece)
    return textwrap.indent("\n".join(lines), prefix="  ")


def _summarise_type(schema: Dict[str, Any], root: Dict[str, Any], *, depth: int = 0) -> str:
    schema = _deref(schema, root)
    type_info = schema.get("type")
    if isinstance(type_info, list):
        filtered = [t for t in type_info if t != "null"]
        type_info = filtered[0] if filtered else "null"

    if type_info == "object":
        props = schema.get("properties", {}) or {}
        if not props:
            return "object"
        child_bits = []
        for key, value in props.items():
            child = _deref(value, root)
            child_type = child.get("type")
            if isinstance(child_type, list):
                child_type = "/".join(t for t in child_type if t != "null")
            constraints = _summarise_constraints(child)
            enum = child.get("enum")
            descriptor = f"{key} ({child_type or 'any'})"
            if enum:
                enum_vals = ", ".join(map(str, enum))
                descriptor += f" enum[{enum_vals}]"
            if constraints:
                descriptor += f" {constraints}"
            child_bits.append(descriptor)
        return "object with keys: " + "; ".join(child_bits)

    if type_info == "array":
        items = schema.get("items", {})
        described = _summarise_type(items, root, depth=depth + 1)
        return f"array of {described}"

    enum = schema.get("enum")
    constraints = _summarise_constraints(schema)
    desc = type_info or "any"
    if enum:
        enum_vals = ", ".join(map(str, enum))
        desc += f" with enum[{enum_vals}]"
    if constraints:
        desc += f" {constraints}"
    return desc


def _summarise_constraints(schema: Dict[str, Any]) -> str:
    bits = []
    if "minimum" in schema:
        bits.append(f"min={schema['minimum']}")
    if "maximum" in schema:
        bits.append(f"max={schema['maximum']}")
    if "pattern" in schema:
        bits.append(f"pattern={schema['pattern']}")
    if "format" in schema:
        bits.append(f"format={schema['format']}")
    return "(" + ", ".join(bits) + ")" if bits else ""


def _build_default_payload(schema: Dict[str, Any], root: Dict[str, Any]) -> Any:
    schema = _deref(schema, root)
    type_info = schema.get("type")
    if isinstance(type_info, list):
        type_info = [t for t in type_info if t != "null"]
        type_info = type_info[0] if type_info else None

    if type_info == "object":
        props = schema.get("properties", {}) or {}
        return {
            key: _build_default_payload(value, root)
            for key, value in props.items()
        }
    if type_info == "array":
        return []
    return None


def build_example_template(
    schema_class: Type[BaseModel],
    *,
    text_placeholder: str = "<replace with source text>",
) -> Dict[str, Any]:
    """
    Return a minimal example record for DSPy training datasets based on ``schema_class``.

    The result conforms to the expected ``{"text": ..., "json_output": ...}`` shape,
    with ``json_output`` pre-populated with ``null``/empty defaults that mirror the schema.
    """

    schema_json = schema_class.model_json_schema()
    template = _build_default_payload(schema_json, schema_json)
    return {"text": text_placeholder, "json_output": template}
