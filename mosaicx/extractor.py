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
from enum import Enum
import inspect

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
try:
    import dspy  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    dspy = None  # type: ignore[assignment]

from .constants import (
    DEFAULT_LLM_MODEL,
    MOSAICX_COLORS,
    PACKAGE_SCHEMA_TEMPLATES_PY_DIR,
    PROMPT_OPTIMIZED_FILENAME,
    USER_SCHEMA_DIR,
)
from .document_loader import DocumentLoadingError
from .text_extraction import TextExtractionError, extract_text_with_fallback
from .schema.registry import get_schema_by_id, get_schema_by_path
from .display import styled_message, console
from .utils import derive_ollama_generate_url, resolve_openai_config
from .prompting import PromptPreference, resolve_prompt_for_schema


class ExtractionStrategy(str, Enum):
    """Selectable extraction backends."""

    CLASSIC = "classic"
    DSPY = "dspy"


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


# ---------------------------------------------------------------------------
# Text → Structured Data (schema‑agnostic, hardened)
# ---------------------------------------------------------------------------

def _extract_structured_data_classic(
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
      1) Try Instructor JSON‑Schema mode (for non‑reasoning models).
      2) Reasoning models skip Instructor and go directly to Ollama /api/generate.
      3) Fallback to chat.completions (no response_format for reasoning models).
      4) Sanitize output, extract JSON, coerce to schema, validate via Pydantic.
      5) One-shot auto‑repair if validation fails.

    Args:
        text_content: Markdown/plain-text report content to parse.
        schema_class: Target Pydantic model.
        model: LLM identifier.
        base_url: Custom OpenAI-compatible endpoint.
        api_key: API credential for the endpoint.
        temperature: Sampling temperature.
    """
    if instructor is None or OpenAI is None:
        raise ExtractionError(
            "Instructor and openai packages are required for schema-driven extraction. "
            "Install optional dependencies to use this feature."
        )

    schema_json = schema_class.model_json_schema()
    resolved_base_url, resolved_api_key = resolve_openai_config(base_url, api_key)
    effective_temperature = max(0.0, temperature)

    schema_str = json.dumps(schema_json, indent=2)
    prompt = (
        "Extract the data as a single JSON object that **strictly** matches the JSON Schema.\n"
        "- Output ONLY valid JSON: no code fences, no commentary, no <think> blocks.\n"
        "- Include all required keys.\n"
        "- Use null for optional keys not present in the text.\n"
        "- Use only the allowed keys; do not invent keys.\n"
        "- Booleans must be true/false; numbers must be numbers; enums must match canonical values (case-insensitive acceptable for input).\n\n"
        "JSON Schema (exact structure):\n"
        f"{schema_str}\n\n"
        "Text to extract from:\n"
        f"{text_content}\n"
    )

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


def _extract_structured_data_dspy(
    text_content: str,
    schema_class: Type[BaseModel],
    model: str = DEFAULT_LLM_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    *,
    prompt_preference: PromptPreference | str = PromptPreference.AUTO.value,
    prompt_path: Optional[Union[str, Path]] = None,
    prompt_text: Optional[str] = None,
    examples_path: Optional[Union[str, Path]] = None,
    optimizer_trials: int = 0,
    max_demos: int = 4,
    store_optimized_prompt: bool = False,
) -> BaseModel:
    if dspy is None:
        raise ExtractionError(
            "DSPy is not installed. Install it with `pip install dspy-ai` to use the DSPy strategy."
        )
    if OpenAI is None:
        raise ExtractionError(
            "The openai package is required for the DSPy strategy. Install optional dependencies."
        )

    artifact = None
    if prompt_text is not None:
        prompt_instructions = prompt_text
    elif prompt_path is not None:
        prompt_path = Path(prompt_path)
        if not prompt_path.exists():
            raise ExtractionError(f"Prompt file not found: {prompt_path}")
        prompt_instructions = prompt_path.read_text(encoding="utf-8")
    else:
        pref = (
            prompt_preference
            if isinstance(prompt_preference, PromptPreference)
            else PromptPreference(prompt_preference)
        )
        artifact = resolve_prompt_for_schema(schema_class, preference=pref)
        prompt_instructions = artifact.content

    resolved_base_url, resolved_api_key = resolve_openai_config(base_url, api_key)
    effective_temperature = max(0.0, temperature)

    llm_kwargs: Dict[str, Any] = {"model": model, "temperature": effective_temperature}
    if resolved_api_key:
        llm_kwargs["api_key"] = resolved_api_key
    if resolved_base_url:
        llm_kwargs["api_base"] = resolved_base_url

    def _make_lm(config: Dict[str, Any]) -> Any:
        try:
            if hasattr(dspy, "OpenAI"):  # type: ignore[attr-defined]
                return dspy.OpenAI(**config)  # type: ignore[attr-defined]
            if hasattr(dspy, "LM"):
                cfg = dict(config)
                cfg_model = cfg.pop("model", model)
                provider = cfg.pop("provider", None)
                if provider is None:
                    api_base = cfg.get("api_base") or cfg.get("base_url")
                    if isinstance(api_base, str):
                        lowered = api_base.lower()
                        if "ollama" in lowered or "11434" in lowered:
                            provider = "ollama"
                    if provider is None and cfg_model and "gpt-oss" in cfg_model:
                        provider = "ollama"
                    if provider is None and cfg_model and cfg_model.startswith("gpt-4"):
                        provider = "openai"
                if provider is not None:
                    if provider == "ollama" and "/" not in cfg_model:
                        cfg_model = f"ollama_chat/{cfg_model}"
                    cfg["provider"] = provider
                return dspy.LM(model=cfg_model, **cfg)  # type: ignore[attr-defined]
        except Exception as exc:
            raise ExtractionError(f"Failed to initialise DSPy OpenAI client: {exc}") from exc
        raise ExtractionError("Installed DSPy version does not expose an OpenAI-compatible LM.")

    lm = _make_lm(llm_kwargs)

    dspy.settings.configure(lm=lm)  # type: ignore[attr-defined]

    class ExtractionSignature(dspy.Signature):  # type: ignore[misc,attr-defined]
        guidance = dspy.InputField(desc="Extraction instructions including schema details.")  # type: ignore[attr-defined]
        report = dspy.InputField(desc="Radiology report text to parse.")  # type: ignore[attr-defined]
        json_output = dspy.OutputField(desc="JSON object matching the schema.")  # type: ignore[attr-defined]

    class ExtractionProgram(dspy.Module):  # type: ignore[misc,attr-defined]
        def __init__(self) -> None:
            super().__init__()
            self.predict = dspy.ChainOfThought(ExtractionSignature)  # type: ignore[attr-defined]

        def forward(self, guidance: str, report: str) -> str:
            completion = self.predict(guidance=guidance, report=report)
            candidate = getattr(completion, "json_output", None)
            if candidate is None or candidate == "":
                for attr in ("response", "text", "output", "completion"):
                    candidate = getattr(completion, attr, None)
                    if candidate:
                        break
            if isinstance(candidate, dict):
                candidate_text = json.dumps(candidate)
            elif candidate is None:
                candidate_text = ""
            else:
                candidate_text = str(candidate)

            candidate_text = candidate_text.strip()
            if candidate_text:
                try:
                    candidate_text = _extract_outer_json(_strip_reasoning_and_fences(candidate_text))
                except Exception:
                    pass

            completion.json_output = candidate_text
            return {"json_output": candidate_text}

    program = ExtractionProgram()
    trainset: List[Any] = []
    if examples_path:
        trainset = _load_dspy_examples(examples_path)
        if not trainset:
            styled_message(f"No DSPy training examples found in {examples_path}", "warning")
        else:
            for demo in trainset:
                setattr(demo, "guidance", prompt_instructions)
        if trainset and optimizer_trials > 0:
            metric = _dspy_exact_json_match_metric
            gepa_cls = getattr(getattr(dspy, "teleprompt", None), "GEPA", None)  # type: ignore[attr-defined]
            if gepa_cls is None:
                raise ExtractionError(
                    "DSPy GEPA optimizer not available. Install a DSPy version that provides dspy.teleprompt.GEPA."
                )

            demo_limit = max(1, min(max_demos, len(trainset)))
            ctor_params = inspect.signature(gepa_cls.__init__).parameters
            ctor_kwargs: Dict[str, Any] = {}
            if "metric" in ctor_params:
                ctor_kwargs["metric"] = metric

            for name in ("max_demos", "max_examples", "max_bootstrapped_demos", "k_examples", "k"):
                if name in ctor_params:
                    ctor_kwargs[name] = demo_limit
                    break

            for name in ("num_iterations", "num_epochs", "epochs", "iterations"):
                if name in ctor_params:
                    ctor_kwargs[name] = optimizer_trials
                    break

            if "reflection_lm" in ctor_params and "reflection_lm" not in ctor_kwargs:
                reflection_kwargs = dict(llm_kwargs)
                reflection_kwargs["temperature"] = max(1.0, reflection_kwargs.get("temperature", 0.0) or 1.0)
                ctor_kwargs["reflection_lm"] = _make_lm(reflection_kwargs)
            elif "reflection_model" in ctor_params and "reflection_model" not in ctor_kwargs:
                ctor_kwargs["reflection_model"] = model

            control_assigned = False
            for name in ("auto", "max_full_evals", "max_metric_calls"):
                if name in ctor_params:
                    if name == "auto":
                        ctor_kwargs[name] = "medium"
                    elif name == "max_full_evals":
                        ctor_kwargs[name] = max(1, optimizer_trials)
                    else:  # max_metric_calls
                        ctor_kwargs[name] = max(1, optimizer_trials * demo_limit)
                    control_assigned = True
                    break
            if not control_assigned and optimizer_trials:
                raise ExtractionError(
                    "DSPy GEPA optimiser requires one of auto/max_full_evals/max_metric_calls arguments."
                )

            try:
                optimizer = gepa_cls(**ctor_kwargs)  # type: ignore[call-arg]
            except TypeError as exc:
                raise ExtractionError(f"DSPy GEPA optimisation failed: {exc}") from exc

            def _prepare_call_kwargs(signature: inspect.Signature) -> Dict[str, Any]:
                prepared: Dict[str, Any] = {}
                if "trainset" in signature.parameters:
                    prepared["trainset"] = trainset
                elif "train_data" in signature.parameters:
                    prepared["train_data"] = trainset
                elif "examples" in signature.parameters:
                    prepared["examples"] = trainset
                else:
                    prepared["trainset"] = trainset

                if "valset" in signature.parameters:
                    prepared["valset"] = trainset
                elif "evalset" in signature.parameters:
                    prepared["evalset"] = trainset
                elif "validation_set" in signature.parameters:
                    prepared["validation_set"] = trainset
                elif "devset" in signature.parameters:
                    prepared["devset"] = trainset
                else:
                    prepared.setdefault("valset", trainset)
                return prepared

            success = False
            last_error: Optional[Exception] = None

            def _try_call(method: Callable[..., Any], signature: inspect.Signature, *, positional_ok: bool) -> Optional[Any]:
                nonlocal last_error
                call_kwargs = _prepare_call_kwargs(signature)
                try:
                    if not positional_ok:
                        result = method(**call_kwargs)
                    else:
                        result = method(program, **call_kwargs)
                    last_error = None
                    return result
                except TypeError as exc:
                    last_error = exc
                    try:
                        result = method(program, trainset, trainset)
                        last_error = None
                        return result
                    except Exception as inner_exc:
                        last_error = inner_exc
                        return None
                except Exception as exc:
                    last_error = exc
                    return None

            compile_method = getattr(optimizer, "compile", None)
            if callable(compile_method):
                signature = inspect.signature(compile_method)
                call_kwargs = _prepare_call_kwargs(signature)
                call_kwargs.setdefault("student", program)
                result = None
                try:
                    result = compile_method(**call_kwargs)
                except Exception as exc:
                    last_error = exc
                else:
                    if result is not None and hasattr(result, "predict"):
                        program = result
                    success = True

            if not success:
                for method_name in ("__call__", "optimize", "step"):
                    method = getattr(optimizer, method_name, None)
                    if not callable(method):
                        continue
                    try:
                        signature = inspect.signature(method)
                    except (TypeError, ValueError) as exc:
                        last_error = exc
                        continue

                    result = _try_call(method, signature, positional_ok="program" not in signature.parameters and "student" not in signature.parameters)
                    if result is not None and hasattr(result, "predict"):
                        program = result
                    if result is not None or last_error is None:
                        success = True
                        break

            if not success:
                if last_error is not None:
                    raise ExtractionError(
                        "DSPy GEPA optimiser does not expose a supported optimisation entry point."
                    ) from last_error
                raise ExtractionError(
                    "DSPy GEPA optimiser does not expose a supported optimisation entry point."
                )

    predict_module = getattr(program, "predict", None)
    if trainset and predict_module is not None and not getattr(predict_module, "demonstrations", None):
        predict_module.demonstrations = trainset[:max_demos]  # type: ignore[attr-defined]
    try:
        raw = program(prompt_instructions, text_content)
    except Exception as exc:
        raise ExtractionError(f"DSPy program execution failed: {exc}") from exc

    if not isinstance(raw, str):
        raw = str(raw)

    schema_json = schema_class.model_json_schema()
    cleaned = _extract_outer_json(_strip_reasoning_and_fences(raw))
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"DSPy model returned invalid JSON: {e}\nContent: {raw}") from e

    coerced = _coerce_to_schema(payload, schema_json, schema_json)
    predict_module = getattr(program, "predict", None)
    if store_optimized_prompt and artifact is not None and predict_module is not None and getattr(predict_module, "demonstrations", None):
        prompt_lines = [prompt_instructions.rstrip()]
        for idx, demo in enumerate(predict_module.demonstrations, start=1):  # type: ignore[attr-defined]
            report_text = getattr(demo, "report", None)
            expected_json = getattr(demo, "json_output", None)
            if not isinstance(report_text, str) or not isinstance(expected_json, str):
                continue
            prompt_lines.append(
                f"\n# Example {idx}\nReport:\n{report_text}\n\nExpected JSON:\n{expected_json}"
            )
        optimized_prompt = "\n".join(prompt_lines).strip() + "\n"
        optimized_path = artifact.prompt_dir / PROMPT_OPTIMIZED_FILENAME
        optimized_path.write_text(optimized_prompt, encoding="utf-8")

    try:
        return schema_class(**coerced)
    except ValidationError as ve:
        raise ExtractionError(f"DSPy validation failed: {ve}") from ve


def _load_dspy_examples(path: Union[str, Path]) -> List[Any]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise ExtractionError(f"DSPy examples file not found: {dataset_path}")
    try:
        if dataset_path.suffix.lower() == ".jsonl":
            data = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"Failed to parse DSPy examples: {exc}") from exc

    if isinstance(data, dict):
        data = data.get("examples", [])
    if not isinstance(data, list):
        raise ExtractionError("DSPy examples must be a list of objects with 'report' and 'json_output'.")

    examples: List[Any] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        report = entry.get("text") or entry.get("report") or entry.get("input") or entry.get("prompt")
        json_output = entry.get("json_output") or entry.get("output")
        if not isinstance(report, str) or not isinstance(json_output, (str, dict)):
            continue
        if isinstance(json_output, dict):
            json_output = json.dumps(json_output)
        example = dspy.Example(  # type: ignore[attr-defined]
            guidance=None,
            report=report,
            json_output=json_output,
        ).with_inputs("guidance", "report")
        examples.append(example)
    return examples


def _dspy_exact_json_match_metric(
    gold: Any,
    pred: Any,
    trace: Any = None,
    pred_name: Optional[str] = None,
    pred_trace: Any = None,
) -> float:
    """
    GEPA-compatible metric that rewards exact JSON matches.

    Parameters follow the GEPA contract: (gold, pred, trace, pred_name, pred_trace).
    We only need gold/pred, but accept the others for compatibility.
    """

    def _extract_json_text(payload: Any) -> Optional[str]:
        if isinstance(payload, str):
            return payload
        text = getattr(payload, "json_output", None)
        if isinstance(text, str):
            return text
        if isinstance(text, dict):
            return json.dumps(text)
        if isinstance(payload, dict):
            inner = payload.get("json_output")
            if isinstance(inner, str):
                return inner
            if isinstance(inner, dict):
                return json.dumps(inner)
        return None

    expected = _extract_json_text(gold)
    actual = _extract_json_text(pred)
    if expected is None or actual is None:
        return 0.0
    try:
        expected_obj = json.loads(expected)
        actual_obj = json.loads(actual)
    except json.JSONDecodeError:
        return 0.0
    return 1.0 if expected_obj == actual_obj else 0.0


def extract_structured_data(
    text_content: str,
    schema_class: Type[BaseModel],
    model: str = DEFAULT_LLM_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    *,
    strategy: ExtractionStrategy | str = ExtractionStrategy.CLASSIC,
    prompt_preference: PromptPreference | str = PromptPreference.AUTO.value,
    prompt_path: Optional[Union[str, Path]] = None,
    prompt_text: Optional[str] = None,
    dspy_examples_path: Optional[Union[str, Path]] = None,
    dspy_optimizer_trials: int = 0,
    dspy_max_demos: int = 4,
    dspy_store_optimized_prompt: bool = False,
) -> BaseModel:
    chosen_strategy = (
        strategy if isinstance(strategy, ExtractionStrategy) else ExtractionStrategy(strategy)
    )

    if chosen_strategy is ExtractionStrategy.CLASSIC:
        pref = (
            prompt_preference
            if isinstance(prompt_preference, PromptPreference)
            else PromptPreference(prompt_preference)
        )
        if (
            prompt_path is not None
            or prompt_text is not None
            or pref is not PromptPreference.AUTO
            or dspy_examples_path is not None
            or dspy_optimizer_trials
            or dspy_store_optimized_prompt
        ):
            raise ExtractionError(
                "DSPy-only options (--prompt-variant, --prompt-path, --dspy-*) require the DSPy strategy."
            )
        return _extract_structured_data_classic(
            text_content=text_content,
            schema_class=schema_class,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )

    return _extract_structured_data_dspy(
        text_content=text_content,
        schema_class=schema_class,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        prompt_preference=prompt_preference,
        prompt_path=prompt_path,
        prompt_text=prompt_text,
        examples_path=dspy_examples_path,
        optimizer_trials=dspy_optimizer_trials,
        max_demos=dspy_max_demos,
        store_optimized_prompt=dspy_store_optimized_prompt,
    )


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
