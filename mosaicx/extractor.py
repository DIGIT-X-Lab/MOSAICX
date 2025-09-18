"""
MOSAICX Document Extraction Module — PDF → Structured Data

MOSAICX: Medical cOmputational Suite for Advanced Intelligence for X‑ray analysis

This module extracts structured data from PDFs using:
- **Docling** for text extraction
- **Instructor + Ollama** for schema‑driven, validated outputs (Pydantic)

Pipeline: PDF → Text → LLM + JSON Schema → Pydantic model

Key Features (schema‑agnostic):
- Works with **any** generated Pydantic model (no schema-specific code)
- Instructor JSON‑Schema mode with robust Ollama fallback (`format="json"`)
- Output sanitation, generic type coercion per JSON Schema, strict validation
- Optional JSON save; rich console feedback

Usage:
    >>> from mosaicx.extractor import extract_from_pdf
    >>> result = extract_from_pdf("report.pdf", "PatientRecord")
    >>> print(result)

Dependencies:
    • docling (^2.0.0)
    • instructor (^1.0.0)
    • ollama (^0.3.0)
    • pydantic
    • openai (OpenAI‑compatible client)

Module Metadata:
    Author:        Lalith Kumar Shiyam Sundar, PhD
    Email:         Lalith.shiyam@med.uni-muenchen.de
    Institution:   DIGIT-X Lab, LMU Radiology | LMU University Hospital
    License:       AGPL-3.0 (GNU Affero General Public License v3.0) 
    Version:       1.0.0
    Created:       2025-09-18
    Last Modified: 2025-09-18

Copyright:
    © 2025 DIGIT-X Lab, LMU Radiology | LMU University Hospital
    Distributed under the AGPL-3.0 license. See LICENSE for details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import json
import importlib.util
import sys
import logging
import re

# Optional: native Ollama JSON route; handled gracefully if missing
try:
    import requests  # noqa: F401
except Exception:  # pragma: no cover
    requests = None  # type: ignore

# Suppress noisy logging from Docling and HTTP requests
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("docling.document_converter").setLevel(logging.WARNING)
logging.getLogger("docling.datamodel.base_models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("instructor").setLevel(logging.WARNING)  # Suppress Instructor debug output
logging.getLogger("instructor.retry").setLevel(logging.WARNING)  # Suppress retry logs

from docling.document_converter import DocumentConverter
import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .constants import (
    DEFAULT_LLM_MODEL,
    PACKAGE_SCHEMA_PYD_DIR,  # Removed unused PACKAGE_SCHEMA_JSON_DIR (redundant)
    MOSAICX_COLORS,
)
from .display import styled_message, console


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ExtractionError(Exception):
    """Custom exception for extraction-related errors."""
    pass


# ---------------------------------------------------------------------------
# Schema loader
# ---------------------------------------------------------------------------

def load_schema_model(schema_name: str) -> Type[BaseModel]:
    """
    Load a Pydantic model from the generated schema files.

    Args:
        schema_name: Name of the schema model class

    Returns:
        The Pydantic model class

    Raises:
        ExtractionError: If schema file not found or cannot be loaded
    """
    pyd_dir = Path(PACKAGE_SCHEMA_PYD_DIR)

    matching_files: List[Path] = [
        py_file for py_file in pyd_dir.glob("*.py")
        if schema_name.lower() in py_file.name.lower()
    ]

    if not matching_files:
        raise ExtractionError(
            f"No schema file found for '{schema_name}' in {pyd_dir}. "
            f"Generate a schema first using: mosaicx generate --desc '...'"
        )

    schema_file = max(matching_files, key=lambda f: f.stat().st_mtime)

    try:
        spec = importlib.util.spec_from_file_location("schema_module", schema_file)
        if spec is None or spec.loader is None:  # safety
            raise RuntimeError("Failed to create module spec.")
        module = importlib.util.module_from_spec(spec)
        sys.modules["schema_module"] = module
        spec.loader.exec_module(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseModel)
                and attr.__name__ == schema_name
            ):
                return attr

        raise ExtractionError(f"Schema class '{schema_name}' not found in {schema_file}")

    except Exception as e:
        raise ExtractionError(f"Failed to load schema from {schema_file}: {e}") from e


# ---------------------------------------------------------------------------
# PDF → Text
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extract text from PDF using Docling.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content (Markdown)

    Raises:
        ExtractionError: If PDF cannot be processed
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise ExtractionError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ExtractionError(f"File is not a PDF: {pdf_path}")

    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        text_content = result.document.export_to_markdown()

        if not text_content or not text_content.strip():
            raise ExtractionError(f"No text content extracted from {pdf_path}")

        return text_content

    except Exception as e:
        raise ExtractionError(f"Failed to extract text from PDF {pdf_path}: {e}") from e


# ---------------------------------------------------------------------------
# Schema‑agnostic JSON output hardening & coercion helpers
# ---------------------------------------------------------------------------

# Strip reasoning and fences (e.g., DeepSeek R1 / fenced JSON)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

def _strip_reasoning_and_fences(text: str) -> str:
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_outer_json(text: str) -> str:
    """Return the first well-balanced top-level JSON object/array substring."""
    if not text:
        return text
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return text
    stack: List[str] = []
    for j, ch in enumerate(text[start:], start):
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                continue
            if not stack:
                return text[start : j + 1]
    return text[start:]


# ---- JSON Schema utilities (generic; supports $ref, anyOf/oneOf, formats) ----

def _json_pointer_get(doc: Dict[str, Any], pointer: str) -> Dict[str, Any]:
    # Supports local refs like "#/$defs/Name" or "#/definitions/Name"
    if not pointer or pointer == "#":
        return doc
    if not pointer.startswith("#/"):
        raise KeyError(f"Unsupported $ref pointer: {pointer}")
    parts = pointer[2:].split("/")
    cur: Any = doc
    for p in parts:
        p = p.replace("~1", "/").replace("~0", "~")
        cur = cur[p]
    return cur


def _deref(schema: Dict[str, Any], root: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        try:
            return _json_pointer_get(root, ref)
        except Exception:
            return schema
    return schema


def _is_nullable(schema: Dict[str, Any]) -> bool:
    t = schema.get("type")
    if isinstance(t, list):
        if "null" in t:
            return True
    elif t == "null":
        return True
    for key in ("anyOf", "oneOf"):
        if key in schema:
            for sub in schema[key]:
                if sub.get("type") == "null":
                    return True
    return False


def _types(schema: Dict[str, Any]) -> Optional[List[str]]:
    t = schema.get("type")
    if t is None:
        return None
    return t if isinstance(t, list) else [t]


_num_re = re.compile(r"[-+]?\d+(?:\.\d+)?")
_int_re = re.compile(r"[-+]?\d+")


def _coerce_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "yes", "y", "1", "present"}:
            return True
        if s in {"false", "f", "no", "n", "0", "absent"}:
            return False
        if s in {"", "na", "n/a", "null", "none", "-"}:
            return None
    return None


def _coerce_number(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.replace(",", "")
        m = _num_re.search(s)
        if m:
            try:
                return float(m.group())
            except Exception:
                return None
        if s.strip().lower() in {"", "na", "n/a", "null", "none", "-"}:
            return None
    return None


def _coerce_integer(v: Any) -> Optional[int]:
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.replace(",", "")
        m = _int_re.search(s)
        if m:
            try:
                return int(m.group())
            except Exception:
                return None
        if s.strip().lower() in {"", "na", "n/a", "null", "none", "-"}:
            return None
    return None


def _norm_date(s: str) -> str:
    s2 = s.strip().replace("/", "-")
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s2)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return s


def _norm_datetime(s: str) -> str:
    s2 = s.strip().replace("/", "-")
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?", s2)
    if m:
        y, mo, d, hh, mm, ss = m.groups()
        if ss is None:
            ss = "00"
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}T{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
    return _norm_date(s)


def _coerce_to_schema(value: Any, schema: Dict[str, Any], root: Dict[str, Any]) -> Any:
    """
    Generic, schema‑driven coercion:
    - Supports objects, arrays, enums, numbers/integers/booleans/strings
    - Honors 'format: date|date-time'
    - Handles anyOf/oneOf and local $ref
    - Drops unknown keys when additionalProperties == False
    """
    schema = _deref(schema, root)

    # anyOf / oneOf: try subschemas
    for key in ("anyOf", "oneOf"):
        if key in schema:
            for sub in schema[key]:
                v2 = _coerce_to_schema(value, sub, root)
                stypes = set(_types(_deref(sub, root)) or [])
                if "object" in stypes and isinstance(v2, dict):
                    return v2
                if "array" in stypes and isinstance(v2, list):
                    return v2
                if "string" in stypes and isinstance(v2, str):
                    return v2
                if "integer" in stypes and isinstance(v2, int) and not isinstance(v2, bool):
                    return v2
                if "number" in stypes and isinstance(v2, (int, float)) and not isinstance(v2, bool):
                    return v2
                if "boolean" in stypes and isinstance(v2, bool):
                    return v2
            # fall through

    stypes = set(_types(schema) or [])

    # enums (case-insensitive normalization for strings)
    if "enum" in schema:
        enums = schema["enum"]
        if isinstance(value, str):
            lower_map = {str(e).lower(): e for e in enums}
            v = value.strip()
            if v.lower() in lower_map:
                value = lower_map[v.lower()]
        if value not in enums:
            s = str(value)
            if s in enums:
                value = s

    # object
    if "object" in stypes:
        if isinstance(value, str):
            try:
                candidate = json.loads(value)
                if isinstance(candidate, dict):
                    value = candidate
            except Exception:
                pass
        if isinstance(value, dict):
            props = schema.get("properties", {}) or {}
            for k, sub in props.items():
                if k in value:
                    value[k] = _coerce_to_schema(value[k], sub, root)
            addl = schema.get("additionalProperties", True)
            if addl is False:
                for k in list(value.keys()):
                    if k not in props:
                        value.pop(k, None)
            elif isinstance(addl, dict):
                for k in list(value.keys()):
                    if k not in props:
                        value[k] = _coerce_to_schema(value[k], addl, root)
        return value

    # array
    if "array" in stypes:
        items = schema.get("items", {}) or {}
        if not isinstance(value, list):
            if isinstance(value, str):
                s = value.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        arr = json.loads(s)
                        if isinstance(arr, list):
                            value = arr
                        else:
                            value = [value]
                    except Exception:
                        value = [v for v in [p.strip() for p in s.split(",")] if v]
                else:
                    value = [v for v in [p.strip() for p in s.split(",")] if v]
            else:
                value = [value]
        return [_coerce_to_schema(v, items, root) for v in value]

    # boolean
    if "boolean" in stypes:
        b = _coerce_bool(value)
        return b if b is not None else value

    # integer
    if "integer" in stypes:
        iv = _coerce_integer(value)
        return iv if iv is not None else value

    # number
    if "number" in stypes:
        nv = _coerce_number(value)
        return nv if nv is not None else value

    # string
    if "string" in stypes:
        fmt = schema.get("format")
        if isinstance(value, str):
            s = value
        else:
            s = str(value)
        if fmt == "date":
            return _norm_date(s)
        if fmt == "date-time":
            return _norm_datetime(s)
        if _is_nullable(schema) and s.strip().lower() in {"", "na", "n/a", "null", "none", "-"}:
            return None
        return s

    # no explicit type: return as-is
    return value


def _summarize_schema_for_prompt(schema_json: Dict[str, Any]) -> str:
    """Human-readable summary to steer local models without schema drift."""
    props = schema_json.get("properties", {}) or {}
    required = schema_json.get("required", []) or []
    lines: List[str] = []
    for name, spec in props.items():
        spec = _deref(spec, schema_json)
        t = spec.get("type", "any")
        if isinstance(t, list):
            t = "/".join(t)
        enum = spec.get("enum")
        fmt = spec.get("format")
        piece = f"{name}: type={t}"
        if fmt:
            piece += f", format={fmt}"
        if enum:
            vals = ", ".join(map(str, enum))
            if len(vals) > 120:
                vals = vals[:117] + "..."
            piece += f", enum=[{vals}]"
        lines.append("  - " + piece)
    allowed = ", ".join(props.keys())
    req = ", ".join(required)
    return (
        "Allowed top-level keys: [" + allowed + "]\n"
        "Required keys: [" + req + "]\n"
        "Field hints:\n" + "\n".join(lines) + "\n"
        "For nested objects/arrays, follow the JSON Schema provided below.\n"
    )


def _build_extraction_prompt(text_content: str, schema_json: Dict[str, Any]) -> str:
    summary = _summarize_schema_for_prompt(schema_json)
    schema_str = json.dumps(schema_json, indent=2)
    return (
        "Extract the data as a single JSON object that **strictly** matches the JSON Schema.\n"
        "- Output ONLY valid JSON: no code fences, no commentary, no <think> blocks.\n"
        "- Include all required keys.\n"
        "- Use null for optional keys not present in the text.\n"
        "- Use only the allowed keys; do not invent keys.\n"
        "- Booleans must be true/false; numbers must be numbers; enums must match canonical values (case-insensitive acceptable for input).\n\n"
        + summary +
        "JSON Schema (exact structure):\n"
        f"{schema_str}\n\n"
        "Text to extract from:\n"
        f"{text_content}\n"
    )


# ---------------------------------------------------------------------------
# Text → Structured Data (schema‑agnostic, hardened)
# ---------------------------------------------------------------------------

def extract_structured_data(
    text_content: str,
    schema_class: Type[BaseModel],
    model: str = DEFAULT_LLM_MODEL
) -> BaseModel:
    """
    Schema‑agnostic extraction using Instructor (JSON‑Schema mode) with Ollama fallback.

    Steps:
      1) Try Instructor JSON‑Schema mode (validated return of `schema_class`)
      2) Fallback to Ollama native /api/generate with format='json' (if `requests` available)
      3) Sanitize output, extract outer JSON, coerce to JSON Schema, validate via Pydantic
      4) One-shot auto‑repair by feeding schema + validation errors back to the model
    """
    schema_json = schema_class.model_json_schema()
    prompt = _build_extraction_prompt(text_content, schema_json)

    # --- Attempt 1: Instructor JSON‑Schema ---
    try:
        client = instructor.from_openai(
            OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
            mode=instructor.Mode.JSON_SCHEMA,
        )
        result = client.chat.completions.create(
            model=model,
            response_model=schema_class,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON that matches the schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_retries=1,
            # Some OpenAI‑compat servers honor this; if not, it's ignored or raises, which we catch.
            response_format={"type": "json_object"},  # type: ignore[arg-type]
        )
        return result  # already validated instance of schema_class

    except Exception as instructor_error:
        try:
            # Suppress error messages in console output
            pass
        except Exception:
            pass

    # --- Attempt 2: Ollama native JSON (format='json') ---
    raw: Optional[str] = None
    if requests is not None:
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": "json",                # enforces JSON on Ollama route
                    "options": {"temperature": 0, "top_p": 0.1},
                    "stream": False,
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "")
        except Exception as e:
            try:
                # Suppress error messages in console output
                pass
            except Exception:
                pass

    # --- Attempt 2b: Fallback to chat.completions, forcing JSON if possible ---
    if not raw:
        try:
            client2 = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            comp = client2.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Return ONLY a valid JSON object. No commentary."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},  # type: ignore[arg-type]
            )
            raw = comp.choices[0].message.content or ""
        except Exception as e:
            raise ExtractionError(f"Model calls failed: {e}") from e

    # --- Post-process to JSON ---
    cleaned = _extract_outer_json(_strip_reasoning_and_fences(raw))
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Model returned invalid JSON: {e}\nContent: {raw}") from e

    # --- Generic coercion to schema ---
    coerced = _coerce_to_schema(payload, schema_json, schema_json)

    # --- Validate via Pydantic ---
    try:
        return schema_class(**coerced)
    except ValidationError as ve:
        # Auto-repair once using the model
        try:
            client3 = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            repair = client3.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Return ONLY a valid JSON object that matches the schema exactly."},
                    {
                        "role": "user",
                        "content": (
                            "The JSON below does not validate against the schema.\n\n"
                            f"JSON Schema:\n{json.dumps(schema_json, indent=2)}\n\n"
                            f"Pydantic validation error:\n{ve}\n\n"
                            f"Original JSON:\n{json.dumps(coerced, indent=2)}\n\n"
                            "Fix it and return only the corrected JSON object."
                        ),
                    },
                ],
                response_format={"type": "json_object"},  # type: ignore[arg-type]
            )
            repaired_text = repair.choices[0].message.content or ""
            repaired_text = _extract_outer_json(_strip_reasoning_and_fences(repaired_text))
            repaired_payload = json.loads(repaired_text)
            repaired_payload = _coerce_to_schema(repaired_payload, schema_json, schema_json)
            return schema_class(**repaired_payload)
        except Exception:
            # Surface the original error if repair fails
            raise ExtractionError(
                f"Failed to validate data: {ve}\nPayload: {json.dumps(coerced, indent=2)}"
            ) from ve


# ---------------------------------------------------------------------------
# Orchestration: PDF → Text → Structured Data
# ---------------------------------------------------------------------------

def extract_from_pdf(
    pdf_path: Union[str, Path],
    schema_name: str,
    model: str = DEFAULT_LLM_MODEL,
    save_result: Optional[Union[str, Path]] = None
) -> BaseModel:
    """
    Complete pipeline: PDF → Text → Structured Data.

    Args:
        pdf_path: Path to the PDF file
        schema_name: Name of the schema model class (must exist in generated Pydantic files)
        model: Ollama model name to use
        save_result: Optional path to save extracted JSON result

    Returns:
        Instance of the schema class with extracted data

    Raises:
        ExtractionError: If any step in the pipeline fails
    """
    pdf_path = Path(pdf_path)

    # Step 1: Load schema model
    with console.status(f"[{MOSAICX_COLORS['info']}]Loading schema model...", spinner="dots"):
        schema_class = load_schema_model(schema_name)

    console.print()
    styled_message(f"✨ Schema Model: {schema_class.__name__} ✨", "primary", center=True)
    console.print()

    # Step 2: Extract text from PDF
    with console.status(f"[{MOSAICX_COLORS['accent']}]Reading PDF document...", spinner="dots"):
        text_content = extract_text_from_pdf(pdf_path)

    # Step 3: Structured extraction
    with console.status(f"[{MOSAICX_COLORS['primary']}]Extracting structured data...", spinner="dots"):
        extracted_data = extract_structured_data(text_content, schema_class, model)

    # Step 4: Save result if requested
    if save_result:
        save_path = Path(save_result)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data.model_dump(), f, indent=2, ensure_ascii=False, default=str)

        styled_message(f"💾 Saved result → {save_path.name}", "info", center=True)

    return extracted_data