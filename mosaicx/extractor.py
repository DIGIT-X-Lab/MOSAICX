"""
MOSAICX Document Extraction - Documents to Structured Data

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Structure first. Insight follows.

Author: Lalith Kumar Shiyam Sundar, PhD
Lab: DIGIT-X Lab
Department: Department of Radiology
University: LMU University Hospital | LMU Munich

Overview:
---------
Streamline the transformation of clinical documents into validated Pydantic records
using Docling for text extraction and OpenAI-compatible LLMs for schema-guided
structuring. The module underpins the API and CLI ``extract`` flows and
provides reusable helpers for scriptable pipelines.

Processing Pipeline:
--------------------
1. Convert documents to Markdown using Docling's converter.
2. Invoke Instructor/OpenAI-compatible clients with strict JSON schemas.
3. Coerce LLM output into typed Pydantic models with rich validation feedback.

Highlights:
-----------
- Works with any generated schema module, avoiding schema-specific branching.
- Offers graceful fallbacks when optional dependencies (Instructor, Ollama,
  OpenAI) are unavailable.
- Surfaces coloured status messages via ``mosaicx.display`` to aid CLI users.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Callable
import json
import importlib.util
import sys
import logging
import re

# Optional: native Ollama JSON route; handled gracefully if missing
try:
    import requests  # noqa: F401
except Exception:
    requests = None  # type: ignore

# Suppress noisy logging from Docling and HTTP requests
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("docling.document_converter").setLevel(logging.WARNING)
logging.getLogger("docling.datamodel.base_models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("instructor").setLevel(logging.WARNING)
logging.getLogger("instructor.retry").setLevel(logging.WARNING)

try:
    import instructor  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    instructor = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]
from pydantic import BaseModel, ValidationError

from .constants import (
    DEFAULT_LLM_MODEL,
    MOSAICX_COLORS,
    PACKAGE_SCHEMA_TEMPLATES_PY_DIR,
    USER_SCHEMA_DIR,
)
from .document_loader import DocumentLoadingError
from .text_extraction import TextExtractionError, extract_text_with_fallback
from .schema.registry import get_schema_by_id, get_schema_by_path
from .display import styled_message, console
from .utils import derive_ollama_generate_url, resolve_openai_config


class ExtractionError(Exception):
    """Custom exception for extraction-related errors."""
    pass


def load_schema_model(schema_identifier: str) -> Type[BaseModel]:
    """
    Load a Pydantic model from the generated schema files.

    Args:
        schema_identifier: Can be either:
            - A file path (absolute or relative) to the schema file
            - A schema name for backward compatibility (will do fuzzy search)

    Returns:
        The Pydantic model class

    Raises:
        ExtractionError: If schema file not found or cannot be loaded
    """
    from pathlib import Path

    registry_entry = get_schema_by_id(schema_identifier)
    schema_name: Optional[str] = None

    if registry_entry:
        schema_file = Path(registry_entry["file_path"])
        schema_name = registry_entry.get("class_name")
        if not schema_file.exists():
            raise ExtractionError(f"Schema file not found: {schema_file}")
    else:
        # Check if it's a file path (contains / or \ or ends with .py)
        if (
            "/" in schema_identifier
            or "\\" in schema_identifier
            or schema_identifier.endswith(".py")
            or schema_identifier.startswith("mosaicx/")
        ):
            schema_file = Path(schema_identifier)
            if not schema_file.is_absolute():
                schema_file = Path.cwd() / schema_file

            if not schema_file.exists():
                raise ExtractionError(f"Schema file not found: {schema_file}")

            registered = get_schema_by_path(schema_file)
            if registered:
                schema_name = registered.get("class_name")
        else:
            # Backward compatibility: fuzzy search by schema name
            schema_name = schema_identifier
            search_roots: List[Path] = [
                USER_SCHEMA_DIR,
                Path(PACKAGE_SCHEMA_TEMPLATES_PY_DIR),
            ]
            matching_files: List[Path] = []
            for root in search_roots:
                root = root.expanduser()
                if not root.exists():
                    continue
                matching_files.extend(
                    py_file
                    for py_file in root.glob("*.py")
                    if schema_name.lower() in py_file.name.lower()
                )

            if not matching_files:
                search_hint = (
                    ", ".join(str(root) for root in search_roots if root.exists())
                    or "configured schema directories"
                )
                raise ExtractionError(
                    f"No schema file found for '{schema_identifier}' in {search_hint}. "
                    "Generate a schema first using: mosaicx generate --desc '...'"
                )

            schema_file = max(matching_files, key=lambda f: f.stat().st_mtime)

    try:
        spec = importlib.util.spec_from_file_location("schema_module", schema_file)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to create module spec.")
        module = importlib.util.module_from_spec(spec)
        sys.modules["schema_module"] = module
        spec.loader.exec_module(module)

        def camel_to_snake(name: str) -> str:
            return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

        root_schema_class = getattr(module, "ROOT_SCHEMA_CLASS", None)
        if isinstance(root_schema_class, str) and (
            schema_name is None or schema_name == root_schema_class
        ):
            attr = getattr(module, root_schema_class, None)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseModel)
                and attr is not BaseModel
            ):
                return attr
            raise ExtractionError(
                f"ROOT_SCHEMA_CLASS points to '{root_schema_class}' but no matching BaseModel exists in {schema_file}"
            )

        # Collect candidate models preserving definition order
        candidates: List[Type[BaseModel]] = []
        for attr_name, attr in module.__dict__.items():
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseModel)
                and attr is not BaseModel
            ):
                candidates.append(attr)

        if not candidates:
            raise ExtractionError(f"No BaseModel class found in {schema_file}")

        if schema_name:
            for candidate in candidates:
                if candidate.__name__ == schema_name:
                    return candidate
            raise ExtractionError(f"Schema class '{schema_name}' not found in {schema_file}")

        exports = getattr(module, "__all__", None)
        if isinstance(exports, (list, tuple)):
            for name in exports:
                attr = getattr(module, name, None)
                if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
                    return attr

        stem_lower = schema_file.stem.lower()
        matching = [
            candidate
            for candidate in candidates
            if camel_to_snake(candidate.__name__) in stem_lower
        ]
        if matching:
            return matching[-1]

        # Fallback: prefer the last defined BaseModel (outer schemas typically appear last)
        return candidates[-1]

    except Exception as e:
        raise ExtractionError(f"Failed to load schema from {schema_file}: {e}") from e


def extract_text_from_document(
    document_path: Union[str, Path],
    *,
    return_details: bool = False,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Union[str, LayeredTextResult]:
    """
    Extract text from a supported clinical document using Docling.

    Args:
        document_path: Path to the document file
        return_details: When True, return the full LayeredTextResult
        status_callback: Optional callable invoked when a fallback mode is used

    Returns:
        Extracted text content (Markdown)

    Raises:
        ExtractionError: If the document cannot be processed
    """
    doc_path = Path(document_path)
    try:
        extraction = extract_text_with_fallback(doc_path)
    except (DocumentLoadingError, TextExtractionError) as exc:
        raise ExtractionError(str(exc)) from exc

    if not extraction.markdown or not extraction.markdown.strip():
        raise ExtractionError(f"No text content extracted from {doc_path}")

    if status_callback and extraction.mode != "native":
        status_callback(f"{extraction.mode.upper()} fallback for {doc_path.name}")

    if return_details:
        return extraction
    return extraction.markdown


def extract_text_from_pdf(
    pdf_path: Union[str, Path],
    *,
    return_details: bool = False,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Union[str, LayeredTextResult]:
    """Backward-compatible alias for :func:`extract_text_from_document`."""
    return extract_text_from_document(
        pdf_path,
        return_details=return_details,
        status_callback=status_callback,
    )


# ---------------------------------------------------------------------------
# Helpers to strip chain-of-thought / fences and extract JSON
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

def _strip_reasoning_and_fences(text: str) -> str:
    """Remove <think> blocks and fenced code; return raw text."""
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


# ---------------------------------------------------------------------------
# JSON Schema utilities (generic; supports $ref, anyOf/oneOf, formats)
# ---------------------------------------------------------------------------

def _json_pointer_get(doc: Dict[str, Any], pointer: str) -> Dict[str, Any]:
    """Dereference a JSON pointer (#/$defs/Name or #/definitions/Name)."""
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
    """Dereference $ref within a schema against the root document."""
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

    clinical_instructions = """
You are a careful specialist physician extracting structured data from clinical text.
Read the entire text, understand the clinical meaning, and then fill the JSON schema.

Use your medical knowledge to map clinically equivalent phrases to the same concept.
Do NOT rely only on exact word matches.

Examples of semantic mappings (not exhaustive):
- "Chest tightness", "pressure in the chest", "chest discomfort", "stiffness in the chest",
  especially when related to exertion, stress or effort, COUNT as chest pain.
- "Shortness of breath", "dyspnoea", "breathlessness", "difficulty breathing" are the same concept.
- "Fainting", "passed out", "loss of consciousness" may count as syncope if clinically appropriate.
- "No history of X", "denies X", "X is not present" means that X is absent (set the boolean to false).
- "Unclear", "cannot be assessed", "unknown" means the status is unknown (use null if the schema allows).

If symptoms clearly describe the same clinical entity in different words, treat them as present.
Only use null when the information is truly not mentioned or cannot be inferred clinically.

Be consistent in your clinical interpretation.
If the same description appears in different patients, interpret it the same way and produce the same kind of JSON.
Do not randomly choose between plausible options; prefer the most clinically plausible and conservative interpretation.
"""

    return (
        "Extract the data as a single JSON object that strictly matches the JSON Schema.\n"
        "- Output ONLY valid JSON: no code fences, no commentary, no <think> blocks.\n"
        "- Include all required keys.\n"
        "- Use null for optional keys not present in the text OR when the status is genuinely unknown.\n"
        "- Use only the allowed keys; do not invent keys.\n"
        "- Booleans must be true/false; numbers must be numbers; enums must match canonical values.\n\n"
        + clinical_instructions
        + "\nSchema description:\n"
        + summary
        + "\nJSON Schema (exact structure):\n"
        + f"{schema_str}\n\n"
        + "Clinical text to extract from:\n"
        + f"{text_content}\n"
    )


# ---------------------------------------------------------------------------
# Text → Structured Data (schema‑agnostic, hardened)
# ---------------------------------------------------------------------------

def extract_structured_data(
    text_content: str,
    schema_class: Type[BaseModel],
    model: str = DEFAULT_LLM_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseModel:
    """
    Schema‑agnostic extraction using Instructor (JSON‑Schema mode) with special handling
    for reasoning models like DeepSeek and GPT‑OSS, plus Ollama fallback.

    Steps:
      1) Try Instructor JSON‑Schema mode (for non‑reasoning models)
      2) Reasoning models skip Instructor and go directly to Ollama /api/generate
      3) Fallback to chat.completions (no response_format for reasoning models)
      4) Sanitize output, extract JSON, coerce to schema, validate via Pydantic
      5) One-shot auto‑repair if validation fails
    """
    if instructor is None or OpenAI is None:
        raise ExtractionError(
            "Instructor and openai packages are required for schema-driven extraction. "
            "Install optional dependencies to use this feature."
        )

    schema_json = schema_class.model_json_schema()
    prompt = _build_extraction_prompt(text_content, schema_json)
    resolved_base_url, resolved_api_key = resolve_openai_config(base_url, api_key)
    effective_temperature = max(0.0, temperature)

    # Detect DeepSeek / GPT‑OSS "reasoning" models by name
    model_lower = model.lower()
    is_reasoning_model = any(
        kw in model_lower for kw in ("deepseek", "gpt-oss", "reasoner", "r1")
    )

    # 1) Instructor JSON‑Schema (only for non‑reasoning models)
    if not is_reasoning_model:
        try:
            client = instructor.from_openai(
                OpenAI(base_url=resolved_base_url, api_key=resolved_api_key),
                mode=instructor.Mode.JSON_SCHEMA,
            )
            result = client.chat.completions.create(
                model=model,
                response_model=schema_class,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON that matches the schema."},
                    {"role": "user", "content": prompt},
                ],
                temperature=effective_temperature,
                max_retries=1,
                response_format={"type": "json_object"},  # honored on many models
            )
            return result
        except Exception:
            pass  # Silently skip to fallback

    # 2) Try Ollama native /api/generate
    raw: Optional[str] = None
    generate_url = derive_ollama_generate_url(resolved_base_url)
    if requests is not None and generate_url:
        try:
            # For reasoning models, omit unsupported options like top_p; temperature is ignored for DeepSeek【975898227377524†screenshot】
            options = {"temperature": effective_temperature}
            if not is_reasoning_model:
                options["top_p"] = 0.1
            resp = requests.post(
                generate_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": "json",
                    "options": options,
                    "stream": False,
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "")
        except Exception:
            raw = None

    # 3) Fallback to chat.completions via OpenAI API
    if not raw:
        try:
            client2 = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
            messages = [
                {"role": "system", "content": "Return ONLY a valid JSON object. No commentary."},
                {"role": "user", "content": prompt},
            ]
            if is_reasoning_model:
                # Reasoning models do not support response_format
                comp = client2.chat.completions.create(
                    model=model,
                    temperature=effective_temperature,
                    messages=messages,
                )
            else:
                comp = client2.chat.completions.create(
                    model=model,
                    temperature=effective_temperature,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
            # Some reasoning models may leave content blank and put the JSON elsewhere,
            # but we can still search raw text for JSON.
            raw = comp.choices[0].message.content or ""
            # If the message object has reasoning_content or thinking, merge it
            msg = comp.choices[0].message
            for attr in ("reasoning_content", "thinking"):
                if hasattr(msg, attr):
                    val = getattr(msg, attr)
                    if isinstance(val, str):
                        raw += "\n" + val
        except Exception as e:
            raise ExtractionError(f"Model calls failed: {e}") from e

    # 4) Post-process, coerce, validate
    cleaned = _extract_outer_json(_strip_reasoning_and_fences(raw))
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Model returned invalid JSON: {e}\nContent: {raw}") from e

    coerced = _coerce_to_schema(payload, schema_json, schema_json)
    try:
        return schema_class(**coerced)
    except ValidationError as ve:
        # 5) One-shot auto-repair
        try:
            client3 = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
            repair = client3.chat.completions.create(
                model=model,
                temperature=effective_temperature,
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
                **({} if is_reasoning_model else {"response_format": {"type": "json_object"}}),
            )
            repaired_text = repair.choices[0].message.content or ""
            for attr in ("reasoning_content", "thinking"):
                if hasattr(repair.choices[0].message, attr):
                    val = getattr(repair.choices[0].message, attr)
                    if isinstance(val, str):
                        repaired_text += "\n" + val
            repaired_text = _extract_outer_json(_strip_reasoning_and_fences(repaired_text))
            repaired_payload = json.loads(repaired_text)
            repaired_payload = _coerce_to_schema(repaired_payload, schema_json, schema_json)
            return schema_class(**repaired_payload)
        except Exception:
            raise ExtractionError(
                f"Failed to validate data: {ve}\nPayload: {json.dumps(coerced, indent=2)}"
            ) from ve


def extract_from_pdf(
    pdf_path: Union[str, Path],
    schema_name: Optional[str] = None,
    *,
    schema_file_path: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    save_result: Optional[Union[str, Path]] = None,
) -> BaseModel:
    """
    Complete pipeline: Document → Text → Structured Data.

    Args:
        pdf_path: Path to the document file
        schema_name: Optional schema identifier (ID, filename, or file path)
        schema_file_path: Optional explicit schema file path (deprecated alias)
        model: Model identifier for the OpenAI-compatible endpoint
        base_url: Optional custom base URL for the OpenAI-compatible endpoint
        api_key: Optional API key for the endpoint
        temperature: Sampling temperature forwarded to the LLM calls
        save_result: Optional path to save extracted JSON result

    Returns:
        Instance of the schema class with extracted data

    Raises:
        ExtractionError: If any step in the pipeline fails
    """
    pdf_path = Path(pdf_path)
    schema_reference = schema_file_path or schema_name
    if not schema_reference:
        raise ExtractionError("A schema identifier or file path must be provided")
    schema_reference_str = str(schema_reference)
    with console.status(f"[{MOSAICX_COLORS['info']}]Loading schema model...", spinner="dots"):
        schema_class = load_schema_model(schema_reference_str)
    console.print()
    styled_message(f"✨ Schema Model: {schema_class.__name__} ✨", "primary", center=True)
    console.print()
    with console.status(f"[{MOSAICX_COLORS['accent']}]Reading document contents...", spinner="dots"):
        text_content = extract_text_from_document(pdf_path)
    with console.status(f"[{MOSAICX_COLORS['primary']}]Extracting structured data...", spinner="dots"):
        extracted_data = extract_structured_data(
            text_content,
            schema_class,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
    if save_result:
        save_path = Path(save_result)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        styled_message(f"💾 Saved result → {save_path.name}", "info", center=True)
    return extracted_data
