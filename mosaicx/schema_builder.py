"""
MOSAICX Schema Builder - Natural Language to Pydantic Model Generation

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Overview:
---------
This module provides intelligent schema generation capabilities for the MOSAICX
application, enabling automatic creation of Pydantic models from natural language
descriptions using local Large Language Models (LLMs). It serves as the core
engine for transforming unstructured medical report descriptions into structured
data schemas.

Core Functionality:
------------------
• Natural language processing for schema specification extraction
• Integration with Ollama for local LLM-based schema induction
• Pydantic model compilation from schema specifications
• Support for complex data types: primitives, arrays, objects, enums
• Field validation with constraints (regex, min/max, units)
• Python code generation for standalone Pydantic classes
• Rich terminal interface for interactive schema development

Architecture:
------------
The module follows a pipeline architecture:
1. SchemaSpec definition (JSON-serializable schema specification)
2. LLM-based schema induction from natural language
3. Runtime Pydantic model compilation
4. Code generation for persistent model classes

Usage Examples:
--------------
Generate schema from natural language:
    >>> python mosaicx/schema_builder.py --desc "Patient demographics with age, gender, and diagnosis"
    
Interactive schema building:
    >>> from mosaicx.schema_builder import induce_schemaspec_with_ollama
    >>> spec = induce_schemaspec_with_ollama("llama3", "Blood test results")

Dependencies:
------------
External Libraries:
    • ollama (^0.3.0): Local LLM client for schema generation
    • pydantic (^2.0.0): Data validation and model compilation
    • rich (^13.0.0): Advanced terminal formatting and progress display
    • cfonts (^1.0.0): ASCII art text generation for branding

Standard Library:
    • typing: Type hint support and runtime type creation
    • json: Schema serialization and parsing
    • pathlib: File system operations for model persistence

Module Metadata:
---------------
Author:        Lalith Kumar Shiyam Sundar, PhD
Email:         Lalith.shiyam@med.uni-muenchen.de  
Institution:   DIGIT-X Lab, LMU Radiology | LMU University Hospital
License:       AGPL-3.0 (GNU Affero General Public License v3.0)
Version:       1.0.0
Created:       2025-09-18
Last Modified: 2025-09-18

Copyright Notice:
----------------
© 2025 DIGIT-X Lab, LMU Radiology | LMU University Hospital
This software is distributed under the AGPL-3.0 license.
See LICENSE file for full terms and conditions.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import ollama

# Suppress HTTP request logging for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    conint,
    confloat,
    constr,
    create_model,
)

# === Pretty CLI ===
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON as RichJSON
from rich.syntax import Syntax
from rich.traceback import install as rich_traceback
from cfonts import render  # python-cfonts

# Import constants
from .constants import SCHEMA_GENERATION_SYSTEM_PROMPT, MOSAICX_COLORS

console = Console()
rich_traceback(show_locals=False)

# =============================================================================
# 1) SchemaSpec (what the LLM must output)
# =============================================================================

Primitive = Literal["string", "integer", "number", "boolean", "date", "datetime"]


class Constraint(BaseModel):
    regex: Optional[str] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    units: Optional[str] = None  # e.g., "%", "ng/mL"


class EnumSpec(BaseModel):
    name: str
    values: List[str]


class FieldSpec(BaseModel):
    name: str
    type: Primitive | Literal["array", "object"]
    description: Optional[str] = None
    required: bool = False
    enum: Optional[str] = None  # reference to EnumSpec.name
    constraints: Optional[Constraint] = None
    items: Optional["FieldSpec"] = None  # for arrays
    properties: Optional[List["FieldSpec"]] = None  # for objects


FieldSpec.model_rebuild()


class SchemaSpec(BaseModel):
    name: str
    version: str = Field(default="1.0.0")
    description: Optional[str] = None
    enums: List[EnumSpec] = Field(default_factory=list)
    fields: List[FieldSpec]


# =============================================================================
# 2) Compile SchemaSpec -> Pydantic Model (no runtime code exec)
# =============================================================================

def _literal_from_enum(values: List[str]):
    from typing import Literal as _Lit
    return _Lit[tuple(values)]  # type: ignore


def _pyd_primitive(fs: FieldSpec):
    c = fs.constraints or Constraint()
    t = fs.type
    if fs.enum:
        # handled at caller level
        raise RuntimeError("Internal: enum should be handled at caller level")
    if t == "string":
        return constr(pattern=c.regex) if c.regex else str
    if t == "integer":
        return conint(
            ge=c.minimum if c.minimum is not None else None,
            le=c.maximum if c.maximum is not None else None,
        )
    if t == "number":
        return confloat(
            ge=c.minimum if c.minimum is not None else None,
            le=c.maximum if c.maximum is not None else None,
        )
    if t == "boolean":
        return bool
    if t in ("date", "datetime"):
        # keep ISO strings; convert to date/datetime later if preferred
        return str
    raise ValueError(f"Unsupported primitive: {t}")


def _compile_field(fs: FieldSpec, enum_map: Dict[str, List[str]]):
    # Enums take precedence: enforce via Literal
    if fs.enum:
        values = enum_map.get(fs.enum)
        if not values:
            raise ValueError(f"Field '{fs.name}' references unknown enum '{fs.enum}'")
        return _literal_from_enum(values)

    if fs.type in ("string", "integer", "number", "boolean", "date", "datetime"):
        return _pyd_primitive(fs)

    if fs.type == "array":
        assert fs.items, f"Array field '{fs.name}' must have 'items'"
        inner = _compile_field(fs.items, enum_map)
        from typing import List as _List
        return _List[inner]  # type: ignore

    if fs.type == "object":
        assert fs.properties and len(fs.properties) > 0, \
            f"Object field '{fs.name}' must have 'properties'"
        fields: Dict[str, tuple] = {}
        for prop in fs.properties:
            ann = _compile_field(prop, enum_map)
            default = ... if prop.required else None
            fields[prop.name] = (ann, Field(default, description=prop.description))
        Model = create_model(fs.name.capitalize() or "AnonObject", **fields)  # type: ignore
        Model.__doc__ = fs.description or fs.name
        return Model

    raise ValueError(f"Unsupported field type: {fs.type}")


def _to_camel(name: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def compile_schema_to_model(spec: SchemaSpec) -> type[BaseModel]:
    enum_map = {e.name: e.values for e in spec.enums}
    fields: Dict[str, tuple] = {}
    for f in spec.fields:
        ann = _compile_field(f, enum_map)
        default = ... if f.required else None
        desc = f.description or ""
        if f.constraints and f.constraints.units:
            desc = (desc + f" [units=" + f.constraints.units + "]").strip()
        fields[f.name] = (ann, Field(default, description=desc))
    Model = create_model(_to_camel(spec.name), **fields)  # type: ignore
    Model.__doc__ = spec.description or spec.name
    return Model


# =============================================================================
# 3) Ollama robust call + schema induction with repair
# =============================================================================

def _extract_first_json_blob(text: str) -> Optional[str]:
    """Find the first top-level JSON object/array (tolerates fenced blocks)."""
    text = text.strip()
    if not text:
        return None
    # Try fenced ```json blocks first
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    # Generic scan for first {...} or [...]
    stack, start = [], -1
    for i, ch in enumerate(text):
        if ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                continue
            if not stack and start != -1:
                return text[start : i + 1].strip()
    return None


def call_ollama_robust(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    try_format_json: bool = True,
    debug: bool = False,
) -> str:
    """
    Try (1) chat+format=json, (2) chat (no format), (3) generate+format=json.
    Return raw assistant text (may still include prose; upstream salvages JSON).
    """
    client = ollama.Client()

    def _chat(format_json: bool) -> Optional[str]:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0},
        }
        if format_json:
            kwargs["format"] = "json"
        try:
            resp = client.chat(**kwargs)
            if "error" in resp:
                raise RuntimeError(f"Ollama chat error: {resp['error']}")
            msg = resp.get("message", {})
            return (msg or {}).get("content") or resp.get("content") or ""
        except Exception as e:
            if debug:
                console.print(f"[yellow][DEBUG chat exception][/]: {e!r}")
            return None

    def _generate(format_json: bool) -> Optional[str]:
        prompt = system_prompt + "\n\n" + user_prompt
        kwargs = {"model": model, "prompt": prompt, "options": {"temperature": 0}}
        if format_json:
            kwargs["format"] = "json"
        try:
            resp = client.generate(**kwargs)
            if "error" in resp:
                raise RuntimeError(f"Ollama generate error: {resp['error']}")
            return resp.get("response") or ""
        except Exception as e:
            if debug:
                console.print(f"[yellow][DEBUG generate exception][/]: {e!r}")
            return None

    # Attempt 1: chat + format=json
    if try_format_json:
        r = _chat(format_json=True)
        if r:
            return r
    # Attempt 2: chat (no format)
    r = _chat(format_json=False)
    if r:
        return r
    # Attempt 3: generate + format=json
    r = _generate(format_json=True)
    if r:
        return r
    raise RuntimeError("Ollama returned no content via chat/generate (check model name and server logs).")


def induce_schemaspec_with_ollama(
    model: str,
    nl_description: str,
    examples: Optional[List[str]] = None,
    max_retries: int = 2,
    debug: bool = False,
) -> SchemaSpec:
    """Ask Ollama to produce a SchemaSpec in JSON; validate; on failure, send repair hints."""
    examples = examples or []
    user = f"Natural-language description:\n{nl_description.strip()}\n"
    if examples:
        user += "\nOptional deidentified examples:\n"
        for ex in examples:
            user += "\n---\n" + ex.strip() + "\n---\n"

    base_prompt = user
    last_error = None

    for attempt in range(max_retries + 1):
        raw = call_ollama_robust(
            model,
            SCHEMA_GENERATION_SYSTEM_PROMPT,
            base_prompt if attempt == 0 else base_prompt + "\nThe previous JSON was invalid or empty. FIX STRICTLY.",
            try_format_json=True,
            debug=debug,
        )

        candidate = raw.strip()
        if not candidate.startswith("{") and not candidate.startswith("["):
            candidate = _extract_first_json_blob(candidate) or candidate

        try:
            data = json.loads(candidate)
            return SchemaSpec.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            if debug:
                console.print(Panel.fit(last_error, title="DEBUG validation/JSON error", style="yellow"))

    raise RuntimeError(
        f"Failed to induce a valid SchemaSpec after {max_retries+1} attempts.\n"
        f"Last error:\n{last_error}\n"
        f"Tip: update Ollama, try a stricter model (e.g., qwen2:7b-instruct), or run with --debug."
    )


# =============================================================================
# 4) Codegen for a Pydantic class (.py) & “cool” Rich views
# =============================================================================

def _py_type_from_field(fs: FieldSpec, enum_map: Dict[str, List[str]]) -> str:
    if fs.enum:
        values = enum_map.get(fs.enum, [])
        return "Literal[" + ", ".join([repr(v) for v in values]) + "]"
    t = fs.type
    if t == "string":
        return "str"
    if t == "integer":
        return "int"
    if t == "number":
        return "float"
    if t == "boolean":
        return "bool"
    if t in ("date", "datetime"):
        return "str"  # keep as ISO strings for portability
    if t == "array":
        assert fs.items
        return f"List[{_py_type_from_field(fs.items, enum_map)}]"
    if t == "object":
        # inline nested objects as Dict[str, Any] (simple codegen)
        return "Dict[str, Any]"
    raise ValueError(f"Unsupported field type: {t}")


def generate_model_py(spec: SchemaSpec) -> str:
    enum_map = {e.name: e.values for e in spec.enums}
    class_name = _to_camel(spec.name)
    lines = [
        "from __future__ import annotations",
        "from typing import Any, Dict, List, Optional, Literal",
        "from pydantic import BaseModel, Field",
        "",
        f"class {class_name}(BaseModel):",
        f"    \"\"\"{spec.description or spec.name}\"\"\"",
    ]
    for f in spec.fields:
        ann = _py_type_from_field(f, enum_map)
        default = "..." if f.required else "None"
        desc = (f.description or "").replace('"', '\\"')
        if f.constraints and f.constraints.units:
            desc = (desc + f" [units={f.constraints.units}]").strip()
        lines.append(f'    {f.name}: {ann} = Field({default}, description="{desc}")')
    lines.append("")
    return "\n".join(lines)


def fields_table(spec: SchemaSpec) -> Table:
    """Create a beautiful table showing schema fields with Dracula colors."""
    t = Table(
        show_lines=False,
        border_style=MOSAICX_COLORS["secondary"],
        header_style=f"bold {MOSAICX_COLORS['primary']}",
        min_width=80,  # Set a consistent minimum width
        expand=False   # Don't expand to full console width
    )
    
    # Add columns with Dracula colors
    t.add_column("Field", style=MOSAICX_COLORS["info"], no_wrap=True)
    t.add_column("Type", style=MOSAICX_COLORS["accent"])
    t.add_column("Required", style=MOSAICX_COLORS["warning"], justify="center")
    t.add_column("Enum", style=MOSAICX_COLORS["success"])
    t.add_column("Constraints", style=MOSAICX_COLORS["secondary"])

    enum_map = {e.name: e.values for e in spec.enums}

    def type_repr(fs: FieldSpec) -> str:
        if fs.enum:
            return f"enum:{fs.enum}"
        if fs.type == "array":
            inner = type_repr(fs.items) if fs.items else "?"
            return f"array[{inner}]"
        if fs.type == "object":
            if not fs.properties:
                return "object"
            props = ", ".join(p.name for p in fs.properties[:3])
            more = "..." if len(fs.properties) > 3 else ""
            return f"object({props}{more})"
        return fs.type

    for f in spec.fields:
        enum_vals = ""
        if f.enum:
            enum_vals = ", ".join(enum_map.get(f.enum, []))
        cons = []
        if f.constraints:
            if f.constraints.regex:
                cons.append(f"regex={f.constraints.regex}")
            if f.constraints.minimum is not None:
                cons.append(f"min={f.constraints.minimum}")
            if f.constraints.maximum is not None:
                cons.append(f"max={f.constraints.maximum}")
            if f.constraints.units:
                cons.append(f"units={f.constraints.units}")
        t.add_row(
            f.name,
            type_repr(f),
            "yes" if f.required else "no",
            enum_vals,
            "; ".join(cons),
        )
    return t


# =============================================================================
# 5) CLI
# =============================================================================

def _banner() -> str:
    try:
        out = render("MOSAICX", align="center")
        return out["string"] if isinstance(out, dict) else str(out)
    except Exception:
        return "MOSAICX"


def main():
    ap = argparse.ArgumentParser(description="Induce a Pydantic model from natural language with Ollama.")
    ap.add_argument("--model", default="gpt-oss:120b", help="Ollama model name")
    ap.add_argument("--desc", required=True, help="Natural-language description of fields you want")
    ap.add_argument("--example", action="append", help="Optional example text/report (can pass multiple)")
    ap.add_argument("--save-schema", type=Path, help="Write induced SchemaSpec JSON to this path")
    ap.add_argument("--save-model", type=Path, help="Write generated Pydantic class .py to this path")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logs")
    args = ap.parse_args()

    console.print(_banner(), highlight=False, markup=False)

    # 1) Induce schema
    with console.status(f"[{MOSAICX_COLORS['info']}]Inducing schema...", spinner="dots"):
        spec = induce_schemaspec_with_ollama(args.model, args.desc, args.example or [], debug=args.debug)

    console.print(Panel("Induced SchemaSpec", style="bold cyan"))
    #console.print(RichJSON.from_data(spec.model_dump()))

    if args.save_schema:
        args.save_schema.write_text(json.dumps(spec.model_dump(), indent=2))
        console.print(f"[green]Saved SchemaSpec → {args.save_schema}")

    # 2) Compile to runtime model
    with console.status("[bold cyan]Compiling Pydantic model..."):
        Model = compile_schema_to_model(spec)

    # 3) Show model metadata
    meta = Table.grid()
    meta.add_column(style="cyan", no_wrap=True)
    meta.add_column(style="magenta")
    meta.add_row("Class", Model.__name__)
    meta.add_row("Doc", Model.__doc__ or "")
    console.print(Panel(meta, title="Compiled Pydantic Model", style="bold cyan"))

    # 4) Display fields as a table
    console.print(fields_table(spec))

    # 5) Show generated class code (cool view)
    class_code = generate_model_py(spec)
    console.print(Panel("Generated Pydantic Class (code view)", style="bold cyan"))
    console.print(Syntax(class_code, "python", theme="monokai", line_numbers=True))

    # 6) Optional: write the class to a .py file
    if args.save_model:
        args.save_model.write_text(class_code)
        console.print(f"[green]Saved Pydantic class → {args.save_model}")

if __name__ == "__main__":
    main()