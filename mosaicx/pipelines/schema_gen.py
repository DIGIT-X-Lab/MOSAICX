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

import re
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, NamedTuple, Optional

from pydantic import BaseModel, Field, create_model


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
}

_LIST_RE = re.compile(r"^list\[(\w+)\]$", re.IGNORECASE)


def _resolve_type(spec: FieldSpec) -> type:
    """Map a FieldSpec's type string to an actual Python type.

    Supports:
        - Simple scalars: str, int, float, bool (plus aliases)
        - Generic lists:  list[str], list[int], etc.
        - Enums:          enum (requires spec.enum_values)

    Returns the resolved Python type.
    """
    type_str = spec.type.strip().lower()

    # --- simple scalars ---
    if type_str in _SIMPLE_TYPE_MAP:
        return _SIMPLE_TYPE_MAP[type_str]

    # --- list[X] ---
    m = _LIST_RE.match(type_str)
    if m:
        inner = m.group(1).lower()
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
        py_type = _resolve_type(field)

        if field.required:
            # Required field: (type, Field(...))
            field_definitions[field.name] = (
                py_type,
                Field(..., description=field.description),
            )
        else:
            # Optional field: (Optional[type], Field(default=None))
            field_definitions[field.name] = (
                Optional[py_type],
                Field(default=None, description=field.description),
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


def _build_dspy_classes():
    """Lazily define and return DSPy signature/module classes.

    Called on first access via module-level __getattr__.
    """
    import dspy  # noqa: F811  — intentional lazy import

    class GenerateSchemaSpec(dspy.Signature):
        """Given a description of a medical document and optionally example text,
        generate a structured SchemaSpec that captures all relevant fields."""

        description: str = dspy.InputField(
            desc="Natural-language description of the document type to structure"
        )
        example_text: str = dspy.InputField(
            desc="Optional example document text for grounding",
            default="",
        )
        schema_spec: SchemaSpec = dspy.OutputField(
            desc="The generated schema specification as a SchemaSpec JSON object"
        )

    class SchemaGenerator(dspy.Module):
        """DSPy Module that generates a Pydantic model from a text description.

        Pipeline:
            1. Use ChainOfThought to produce a SchemaSpec from the description.
            2. Compile the SchemaSpec into a real Pydantic BaseModel.
        """

        def __init__(self) -> None:
            super().__init__()
            self.generate = dspy.ChainOfThought(GenerateSchemaSpec)

        def forward(
            self,
            description: str,
            example_text: str = "",
        ) -> dspy.Prediction:
            """Run the schema generation pipeline."""
            from mosaicx.metrics import PipelineMetrics, get_tracker, track_step

            metrics = PipelineMetrics()
            tracker = get_tracker()

            with track_step(metrics, "Generate schema", tracker):
                result = self.generate(
                    description=description,
                    example_text=example_text,
                )
            spec: SchemaSpec = result.schema_spec
            compiled = compile_schema(spec)

            self._last_metrics = metrics

            return dspy.Prediction(
                schema_spec=spec,
                compiled_model=compiled,
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
        """DSPy Module that refines an existing schema based on instructions."""

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
