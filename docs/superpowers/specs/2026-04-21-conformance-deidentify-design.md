# Conformance Modes + Deidentify Output Simplification

**Date:** 2026-04-21
**Status:** Approved
**Scope:** New `mosaicx/conformance/` package, deidentifier output changes, CLI flag, config field

## Problem

1. The deidentify output is verbose and inconsistent with extract's minimal `{value, excerpt}` pattern. It includes internal bookkeeping (`start`, `end`, `method`, `reasoning`, `spans`) that belongs in a postprocessing step, not the core output.
2. There's no way to specify which privacy standard governs de-identification. HIPAA's 18 identifiers are hardcoded in regex patterns and the LLM prompt. Adding GDPR or other standards requires editing core pipeline code.
3. Zenta needs to ship proprietary conformance modes (GDPR, Swiss DPA) without leaking implementation details into the open-source repo.

## Design

### 1. Simplified Deidentify Output

Before:
```json
{
  "redacted_text": "...",
  "mode": "remove",
  "redaction_map": [
    {"original": "John Doe", "replacement": "[REDACTED]", "start": 8, "end": 16,
     "phi_type": "NAME", "method": "llm", "reasoning": "...", "spans": [...]}
  ]
}
```

After:
```json
{
  "conformance": "hipaa",
  "redacted_text": "Patient [REDACTED] presented with...",
  "phi": [
    {"value": "John Doe", "type": "NAME", "excerpt": "Patient John Doe presented with"},
    {"value": "01/15/1990", "type": "DATE", "excerpt": "DOB: 01/15/1990"},
    {"value": "MRN12345678", "type": "MRN", "excerpt": "MRN12345678, admitted"}
  ]
}
```

- Each PHI item follows extract's `{value, excerpt}` pattern, plus `type` for the PHI category
- `conformance` field tells the consumer which standard was applied
- No `start`/`end`, `method`, `reasoning`, `spans`, `replacement` — Sebi's postprocessor handles remapping to source coordinates
- `mode` (remove/pseudonymize/dateshift) is an input parameter, not echoed in output

### 2. Conformance Registry

**File:** `mosaicx/conformance/registry.py`

```python
@dataclass(frozen=True)
class ConformanceSpec:
    name: str                                    # "hipaa"
    description: str                             # "HIPAA Safe Harbor (18 identifiers)"
    phi_categories: list[str]                    # ["NAME", "DATE", "SSN", ...]
    regex_patterns: list[tuple[re.Pattern, str]] # [(compiled_pattern, type_label), ...]
    prompt_fragment: str                         # injected into LLM redaction prompt

_REGISTRY: dict[str, ConformanceSpec] = {}

def register_conformance(spec: ConformanceSpec) -> None:
    """Register a conformance mode. Called at import time."""
    _REGISTRY[spec.name] = spec

def get_conformance(name: str) -> ConformanceSpec:
    """Look up a registered conformance. Raises KeyError with helpful message."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(
            f"Unknown conformance '{name}'. Available: {available}. "
            f"Install additional packages for more conformance modes."
        )
    return _REGISTRY[name]

def list_conformances() -> list[str]:
    """Return names of all registered conformances."""
    return sorted(_REGISTRY)
```

### 3. HIPAA Built-in

**File:** `mosaicx/conformance/hipaa.py`

Contains:
- `HIPAA_PHI_CATEGORIES`: The 18 Safe Harbor identifier types — NAME, DATE, SSN, PHONE, EMAIL, MRN, ADDRESS, FAX, URL, IP_ADDRESS, ACCOUNT_NUMBER, LICENSE_NUMBER, VEHICLE_ID, DEVICE_ID, BIOMETRIC, PHOTO, SOCIAL_MEDIA, OTHER
- `HIPAA_REGEX_PATTERNS`: The existing patterns from `deidentifier.py` (SSN, phone, MRN, email, dates), each paired with its type label
- `HIPAA_PROMPT_FRAGMENT`: LLM instruction text specifying the 18 HIPAA Safe Harbor categories to look for
- Registers itself via `register_conformance(HIPAA_SPEC)` at module level

### 4. Package Init

**File:** `mosaicx/conformance/__init__.py`

```python
from .registry import ConformanceSpec, register_conformance, get_conformance, list_conformances
from . import hipaa  # auto-registers HIPAA on import
```

Importing `mosaicx.conformance` registers HIPAA automatically. Zenta adds its own import that registers GDPR.

### 5. Plugin Interface for External Conformances

External packages (like Zenta) register conformances by:

1. Creating a `ConformanceSpec` with their categories, regex patterns, and prompt fragment
2. Calling `register_conformance(spec)` 
3. Ensuring their module is imported before the CLI runs (e.g., via entry point or explicit import in their wrapper)

**File:** `mosaicx/conformance/README.md` (generic, no Zenta/GDPR specifics)

Documents:
- The `ConformanceSpec` dataclass and what each field does
- How to register a custom conformance
- How to test it
- Example skeleton

### 6. Pipeline Integration

**`Deidentifier.forward()` changes:**

Currently the regex patterns and LLM prompt are hardcoded. After this change:

1. `Deidentifier.__init__()` accepts `conformance: str = "hipaa"`
2. Looks up `ConformanceSpec` via `get_conformance(conformance)`
3. Uses `spec.prompt_fragment` in the `RedactPHI` signature instructions
4. Uses `spec.regex_patterns` in the regex guard layer (replaces hardcoded `PHI_PATTERNS`)
5. The forward method builds the simplified output: `{conformance, redacted_text, phi: [{value, type, excerpt}]}`

The existing `PHI_PATTERNS` and `PHI_PATTERN_TYPES` module-level lists in `deidentifier.py` move into `hipaa.py`.

### 7. CLI

```bash
mosaicx deidentify --document report.pdf                          # defaults to hipaa
mosaicx deidentify --document report.pdf --conformance hipaa      # explicit
mosaicx deidentify --document report.pdf --conformance gdpr       # error if not installed
```

New `--conformance` option on the `deidentify` command. Defaults to `cfg.conformance`.

### 8. Config

New field in `MosaicxConfig`:

```python
conformance: str = "hipaa"  # MOSAICX_CONFORMANCE env var
```

### 9. Zenta Tier

The `zenta` flag and `strip_for_open_source()` function in `source_mapping.py` already exist. For this change:

- Open-source output: the simplified `{conformance, redacted_text, phi}` shape described above
- Zenta output (`MOSAICX_ZENTA=1`): same shape, but may include additional fields from premium conformances (e.g., GDPR-specific metadata)

The gating is on **which conformances are available**, not on the output format. `--conformance gdpr` simply isn't registered unless Zenta is installed.

## File Changes

| File | Change |
|------|--------|
| `mosaicx/conformance/__init__.py` | New: package init, auto-registers HIPAA |
| `mosaicx/conformance/registry.py` | New: `ConformanceSpec`, register/get/list |
| `mosaicx/conformance/hipaa.py` | New: HIPAA categories, regex, prompt, registration |
| `mosaicx/conformance/README.md` | New: generic plugin docs for adding conformances |
| `mosaicx/pipelines/deidentifier.py` | Use `ConformanceSpec` for patterns + prompt; simplified output |
| `mosaicx/config.py` | Add `conformance: str = "hipaa"` |
| `mosaicx/cli.py` | Add `--conformance` option to deidentify command |

## Out of Scope

- GDPR implementation (Zenta's repo)
- Postprocessing/remapping (Sebi's work)
- Provenance enrichment / `_source` block
- Chandra bbox integration
- Extract output changes (already simplified)
