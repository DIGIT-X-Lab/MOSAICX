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
import time

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
    from json_repair import repair_json  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    repair_json = None  # type: ignore[assignment]

try:
    import outlines  # type: ignore[import-not-found]
    import ollama as ollama_client  # type: ignore[import-not-found]
    OUTLINES_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    outlines = None  # type: ignore[assignment]
    ollama_client = None  # type: ignore[assignment]
    OUTLINES_AVAILABLE = False

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
from .utils.logging import (
    get_logger,
    log_extraction_start,
    log_extraction_method_attempt,
    log_extraction_method_success,
    log_extraction_method_failure,
    log_prompt,
    log_llm_response,
    log_extraction_complete,
    log_schema_info,
    log_text_content,
)

# Module-level logger
_logger = get_logger(__name__)


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
# Pattern to match common LLM reasoning prefixes before JSON
_REASONING_PREFIX_RE = re.compile(
    r"^(?:(?:We need to|Let me|I need to|Let's|Here is|Here's|The JSON|Below is|I will|I'll|"
    r"First,|Now,|Looking at|Based on|According to|To extract|Parsing|Analyzing).*?(?=\{|\[))",
    re.IGNORECASE | re.DOTALL
)

def _strip_reasoning_and_fences(text: str) -> str:
    """Remove <think> blocks, reasoning prefixes, and fenced code; return raw text."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Strip common reasoning prefixes that precede JSON
    text = _REASONING_PREFIX_RE.sub("", text)
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


def _robust_json_parse(raw: str) -> Dict[str, Any]:
    """
    Attempt to parse JSON using multiple strategies, from strict to lenient.
    
    Strategy order:
    1. Standard json.loads (fastest, strictest)
    2. json-repair library (fixes common issues)
    3. Extract balanced JSON substring and retry
    4. Regex-based extraction for common patterns
    """
    if not raw or not raw.strip():
        raise ValueError("Empty input")
    
    # Clean the input first
    cleaned = _extract_outer_json(_strip_reasoning_and_fences(raw))
    
    # Strategy 1: Standard JSON parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Use json-repair if available
    if repair_json is not None:
        try:
            repaired = repair_json(cleaned, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
            elif isinstance(repaired, str):
                return json.loads(repaired)
        except Exception:
            pass
    
    # Strategy 3: Try to find JSON in the original raw text
    # (in case our cleaning was too aggressive)
    try:
        extracted = _extract_outer_json(raw)
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Look for JSON between common delimiters
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # Markdown code block
        r'```\s*([\s\S]*?)\s*```',       # Generic code block
        r'<json>\s*([\s\S]*?)\s*</json>', # XML-style tags
        r'JSON:\s*(\{[\s\S]*\})',         # "JSON:" prefix
        r'Output:\s*(\{[\s\S]*\})',       # "Output:" prefix
        r'Result:\s*(\{[\s\S]*\})',       # "Result:" prefix
    ]
    for pattern in json_patterns:
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            try:
                candidate = m.group(1).strip()
                if repair_json is not None:
                    repaired = repair_json(candidate, return_objects=True)
                    if isinstance(repaired, dict):
                        return repaired
                return json.loads(candidate)
            except Exception:
                continue
    
    # Strategy 5: Last resort - try to repair the raw text directly
    if repair_json is not None:
        try:
            # Find anything that looks like it could be JSON
            for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw):
                candidate = match.group(0)
                try:
                    repaired = repair_json(candidate, return_objects=True)
                    if isinstance(repaired, dict) and repaired:
                        return repaired
                except Exception:
                    continue
        except Exception:
            pass
    
    raise ValueError(f"Could not parse JSON from response")


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


def _extract_with_outlines(
    text_content: str,
    schema_class: Type[BaseModel],
    model: str,
    base_url: Optional[str] = None,
) -> Optional[BaseModel]:
    """
    Use Outlines with Ollama for guaranteed structured JSON output.
    
    Outlines uses grammar-constrained generation to ensure the LLM
    produces valid JSON matching the Pydantic schema on the first try.
    This eliminates the need for JSON repair or retry logic.
    
    Args:
        text_content: The text to extract from
        schema_class: The Pydantic model class for the output schema
        model: The model name (e.g., "gpt-oss:20b")
        base_url: Optional base URL for Ollama (defaults to localhost:11434)
        
    Returns:
        Instance of schema_class if successful, None on failure
    """
    if not OUTLINES_AVAILABLE or outlines is None or ollama_client is None:
        _logger.debug("Outlines not available, skipping")
        return None
    
    start_time = time.time()
    _logger.info("[Step 0] Attempting: Outlines grammar-constrained generation")
    
    try:
        # Parse the Ollama host from the base URL
        host = None
        if base_url:
            # Convert from OpenAI-style URL to Ollama host
            # e.g., "http://localhost:11434/v1" -> "http://localhost:11434"
            parsed = base_url.rstrip("/")
            if parsed.endswith("/v1"):
                host = parsed[:-3]
            else:
                host = parsed
        
        _logger.debug(f"Outlines config: model={model}, host={host or 'default'}")
        
        # Create Ollama client and outlines model
        client = ollama_client.Client(host=host) if host else ollama_client.Client()
        llm = outlines.from_ollama(client, model)  # type: ignore[attr-defined]
        
        # Build the extraction prompt - use the same rich prompt as other methods
        schema_json = schema_class.model_json_schema()
        prompt = _build_extraction_prompt(text_content, schema_json)
        
        # Log the prompt being sent
        log_prompt(_logger, "Outlines Extraction", prompt)
        log_schema_info(_logger, schema_class.__name__, json.dumps(schema_json, indent=2))
        
        # Create generator with Pydantic schema as output_type (new Outlines API)
        generator = outlines.Generator(llm, output_type=schema_class)  # type: ignore[attr-defined]
        
        _logger.debug("Calling Outlines generator...")
        
        # Generate - Outlines guarantees valid JSON matching the schema
        result = generator(prompt)  # type: ignore[no-any-return]
        
        duration = time.time() - start_time
        
        # Handle both cases: Outlines may return Pydantic model or JSON string
        if isinstance(result, schema_class):
            _logger.info(f"✓ SUCCESS: Outlines returned Pydantic model directly ({duration:.2f}s)")
            log_llm_response(_logger, "Outlines Result", json.dumps(result.model_dump(), indent=2, default=str))
            return result
        elif isinstance(result, str):
            _logger.debug(f"Outlines returned string, parsing to Pydantic model")
            log_llm_response(_logger, "Outlines Raw String", result)
            # Parse JSON string into Pydantic model
            parsed = json.loads(result)
            model_instance = schema_class(**parsed)
            _logger.info(f"✓ SUCCESS: Outlines string parsed to model ({duration:.2f}s)")
            return model_instance
        elif isinstance(result, dict):
            _logger.debug(f"Outlines returned dict, converting to Pydantic model")
            log_llm_response(_logger, "Outlines Dict", json.dumps(result, indent=2, default=str))
            model_instance = schema_class(**result)
            _logger.info(f"✓ SUCCESS: Outlines dict converted to model ({duration:.2f}s)")
            return model_instance
        else:
            # Unknown type, let fallbacks handle it
            _logger.warning(f"✗ Outlines returned unexpected type: {type(result)}, falling back...")
            console.print(f"[dim]Outlines returned unexpected type: {type(result)}, falling back...[/dim]")
            return None
            
    except Exception as e:
        duration = time.time() - start_time
        _logger.warning(f"✗ FAILED [Step 0] Outlines: {e} ({duration:.2f}s)")
        _logger.debug(f"Outlines exception details:", exc_info=True)
        # Log for debugging but don't fail - let fallbacks handle it
        console.print(f"[dim]Outlines extraction failed: {e}, falling back...[/dim]")
        return None


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
    Schema‑agnostic extraction using multiple strategies with Outlines as the primary method.

    Extraction Strategy (in order):
      0) **Outlines** (PRIMARY) - Grammar-constrained generation for guaranteed valid JSON
      1) Instructor JSON‑Schema mode (for non‑reasoning models)
      2) Reasoning models skip Instructor and go directly to Ollama /api/generate
      3) Fallback to chat.completions (no response_format for reasoning models)
      4) Sanitize output, extract JSON, coerce to schema, validate via Pydantic
      5) One-shot auto‑repair if validation fails
    """
    extraction_start_time = time.time()
    method_used = None
    
    _logger.info("=" * 70)
    _logger.info("EXTRACTION START")
    _logger.info(f"  Schema: {schema_class.__name__}")
    _logger.info(f"  Model: {model}")
    _logger.info(f"  Temperature: {temperature}")
    _logger.info(f"  Text length: {len(text_content)} chars")
    _logger.info("=" * 70)
    
    if instructor is None or OpenAI is None:
        _logger.error("Missing required dependencies: instructor and/or openai")
        raise ExtractionError(
            "Instructor and openai packages are required for schema-driven extraction. "
            "Install optional dependencies to use this feature."
        )

    schema_json = schema_class.model_json_schema()
    prompt = _build_extraction_prompt(text_content, schema_json)
    resolved_base_url, resolved_api_key = resolve_openai_config(base_url, api_key)
    effective_temperature = max(0.0, temperature)
    
    # Log schema and prompt for debugging
    log_schema_info(_logger, schema_class.__name__, json.dumps(schema_json, indent=2))
    log_text_content(_logger, "input document", text_content)
    log_prompt(_logger, "Extraction Prompt", prompt)
    
    _logger.debug(f"Resolved base URL: {resolved_base_url}")

    # Detect DeepSeek / GPT‑OSS "reasoning" models by name
    model_lower = model.lower()
    is_reasoning_model = any(
        kw in model_lower for kw in ("deepseek", "gpt-oss", "reasoner", "r1")
    )
    _logger.info(f"Model type: {'reasoning model' if is_reasoning_model else 'standard model'}")

    # 0) **PRIMARY**: Try Outlines for grammar-constrained JSON generation
    # This is the most reliable method - guarantees valid JSON matching the schema
    if OUTLINES_AVAILABLE:
        outlines_result = _extract_with_outlines(
            text_content=text_content,
            schema_class=schema_class,
            model=model,
            base_url=resolved_base_url,
        )
        if outlines_result is not None:
            total_duration = time.time() - extraction_start_time
            _logger.info("-" * 70)
            _logger.info("EXTRACTION COMPLETE")
            _logger.info(f"  Method: Outlines (grammar-constrained)")
            _logger.info(f"  Total duration: {total_duration:.2f}s")
            _logger.info(f"  Result fields: {len(outlines_result.model_fields)}")
            _logger.info("-" * 70)
            return outlines_result
    else:
        _logger.debug("Outlines not available, skipping Step 0")

    # 1) Instructor JSON‑Schema (only for non‑reasoning models)
    if not is_reasoning_model:
        step1_start = time.time()
        _logger.info("[Step 1] Attempting: Instructor JSON-Schema mode")
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
            step1_duration = time.time() - step1_start
            total_duration = time.time() - extraction_start_time
            _logger.info(f"✓ SUCCESS: Instructor ({step1_duration:.2f}s)")
            log_llm_response(_logger, "Instructor Result", json.dumps(result.model_dump(), indent=2, default=str))
            _logger.info("-" * 70)
            _logger.info("EXTRACTION COMPLETE")
            _logger.info(f"  Method: Instructor JSON-Schema")
            _logger.info(f"  Total duration: {total_duration:.2f}s")
            _logger.info("-" * 70)
            return result
        except Exception as e:
            step1_duration = time.time() - step1_start
            _logger.warning(f"✗ FAILED [Step 1] Instructor: {e} ({step1_duration:.2f}s)")
            pass  # Continue to fallback
    else:
        _logger.debug("Skipping Instructor (reasoning model detected)")

    # 2) Try Ollama native /api/generate
    raw: Optional[str] = None
    generate_url = derive_ollama_generate_url(resolved_base_url)
    if requests is not None and generate_url:
        step2_start = time.time()
        _logger.info(f"[Step 2] Attempting: Ollama /api/generate ({generate_url})")
        try:
            # For reasoning models, omit unsupported options like top_p; temperature is ignored for DeepSeek【975898227377524†screenshot】
            options = {"temperature": effective_temperature}
            if not is_reasoning_model:
                options["top_p"] = 0.1
            _logger.debug(f"Ollama request options: {options}")
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
            step2_duration = time.time() - step2_start
            _logger.info(f"✓ Ollama /api/generate returned response ({step2_duration:.2f}s, {len(raw or '')} chars)")
            log_llm_response(_logger, "Ollama Raw Response", raw or "")
        except Exception as e:
            step2_duration = time.time() - step2_start
            _logger.warning(f"✗ FAILED [Step 2] Ollama /api/generate: {e} ({step2_duration:.2f}s)")
            raw = None
    else:
        _logger.debug("Skipping Ollama /api/generate (requests not available or no generate URL)")

    # 3) Fallback to chat.completions via OpenAI API
    if not raw:
        step3_start = time.time()
        _logger.info("[Step 3] Attempting: OpenAI chat.completions API")
        try:
            client2 = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
            messages = [
                {"role": "system", "content": "Return ONLY a valid JSON object. No commentary."},
                {"role": "user", "content": prompt},
            ]
            if is_reasoning_model:
                # Reasoning models do not support response_format
                _logger.debug("Using reasoning model mode (no response_format)")
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
                        _logger.debug(f"Merged {attr} content into raw response")
            step3_duration = time.time() - step3_start
            _logger.info(f"✓ OpenAI chat.completions returned response ({step3_duration:.2f}s, {len(raw)} chars)")
            log_llm_response(_logger, "chat.completions Raw Response", raw)
        except Exception as e:
            step3_duration = time.time() - step3_start
            _logger.error(f"✗ FAILED [Step 3] chat.completions: {e} ({step3_duration:.2f}s)")
            raise ExtractionError(f"Model calls failed: {e}") from e

    # 4) Post-process with robust JSON parsing (multiple fallback strategies)
    _logger.info("[Step 4] Attempting: Robust JSON parsing")
    try:
        payload = _robust_json_parse(raw)
        _logger.info(f"✓ JSON parsing succeeded")
        log_llm_response(_logger, "Parsed JSON", json.dumps(payload, indent=2, default=str))
    except (json.JSONDecodeError, ValueError) as e:
        _logger.warning(f"✗ Initial JSON parsing failed: {e}")
        # Retry with a stricter prompt if all parsing strategies fail
        _logger.info("[Step 4b] Attempting: LLM retry with stricter prompt")
        try:
            client_retry = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
            retry_prompt = (
                "Your previous response was not valid JSON. "
                "You MUST output ONLY a valid JSON object. No explanation, no reasoning, no markdown, no code fences.\n\n"
                f"Required JSON Schema:\n{json.dumps(schema_json, indent=2)}\n\n"
                f"Text to extract from:\n{text_content[:3000]}\n\n"
                "CRITICAL: Start your response with {{ and end with }}. Nothing else."
            )
            log_prompt(_logger, "Retry Prompt", retry_prompt)
            retry_resp = client_retry.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You output ONLY raw JSON. No markdown. No explanation. Just the JSON object."},
                    {"role": "user", "content": retry_prompt},
                ],
            )
            retry_raw = retry_resp.choices[0].message.content or ""
            log_llm_response(_logger, "Retry Raw Response", retry_raw)
            payload = _robust_json_parse(retry_raw)
            _logger.info("✓ Retry JSON parsing succeeded")
        except Exception as retry_err:
            _logger.error(f"✗ FAILED [Step 4b] LLM retry: {retry_err}")
            total_duration = time.time() - extraction_start_time
            _logger.info("-" * 70)
            _logger.info("EXTRACTION FAILED")
            _logger.info(f"  Error: Invalid JSON after retry")
            _logger.info(f"  Total duration: {total_duration:.2f}s")
            _logger.info("-" * 70)
            raise ExtractionError(f"Model returned invalid JSON after retry: {e}\nContent: {raw}") from e

    # Coerce to schema
    _logger.debug("Coercing payload to match schema...")
    coerced = _coerce_to_schema(payload, schema_json, schema_json)
    
    # Validate with Pydantic
    _logger.info("[Step 5] Attempting: Pydantic validation")
    try:
        result = schema_class(**coerced)
        total_duration = time.time() - extraction_start_time
        _logger.info(f"✓ SUCCESS: Pydantic validation passed")
        _logger.info("-" * 70)
        _logger.info("EXTRACTION COMPLETE")
        _logger.info(f"  Method: Fallback chain (Ollama/OpenAI → JSON parse → Pydantic)")
        _logger.info(f"  Total duration: {total_duration:.2f}s")
        _logger.info("-" * 70)
        return result
    except ValidationError as ve:
        # 5) One-shot auto-repair
        _logger.warning(f"✗ Pydantic validation failed: {ve}")
        _logger.info("[Step 5b] Attempting: LLM auto-repair")
        try:
            client3 = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
            repair_prompt = (
                "The JSON below does not validate against the schema.\n\n"
                f"JSON Schema:\n{json.dumps(schema_json, indent=2)}\n\n"
                f"Pydantic validation error:\n{ve}\n\n"
                f"Original JSON:\n{json.dumps(coerced, indent=2)}\n\n"
                "Fix it and return only the corrected JSON object."
            )
            log_prompt(_logger, "Repair Prompt", repair_prompt)
            repair = client3.chat.completions.create(
                model=model,
                temperature=effective_temperature,
                messages=[
                    {"role": "system", "content": "Return ONLY a valid JSON object that matches the schema exactly."},
                    {"role": "user", "content": repair_prompt},
                ],
                **({} if is_reasoning_model else {"response_format": {"type": "json_object"}}),
            )
            repaired_text = repair.choices[0].message.content or ""
            for attr in ("reasoning_content", "thinking"):
                if hasattr(repair.choices[0].message, attr):
                    val = getattr(repair.choices[0].message, attr)
                    if isinstance(val, str):
                        repaired_text += "\n" + val
            log_llm_response(_logger, "Repair Raw Response", repaired_text)
            repaired_text = _extract_outer_json(_strip_reasoning_and_fences(repaired_text))
            repaired_payload = json.loads(repaired_text)
            repaired_payload = _coerce_to_schema(repaired_payload, schema_json, schema_json)
            result = schema_class(**repaired_payload)
            total_duration = time.time() - extraction_start_time
            _logger.info(f"✓ SUCCESS: Auto-repair validated")
            _logger.info("-" * 70)
            _logger.info("EXTRACTION COMPLETE")
            _logger.info(f"  Method: LLM auto-repair")
            _logger.info(f"  Total duration: {total_duration:.2f}s")
            _logger.info("-" * 70)
            return result
        except Exception as repair_err:
            total_duration = time.time() - extraction_start_time
            _logger.error(f"✗ FAILED [Step 5b] Auto-repair: {repair_err}")
            _logger.info("-" * 70)
            _logger.info("EXTRACTION FAILED")
            _logger.info(f"  Error: Validation failed after auto-repair")
            _logger.info(f"  Total duration: {total_duration:.2f}s")
            _logger.info("-" * 70)
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
