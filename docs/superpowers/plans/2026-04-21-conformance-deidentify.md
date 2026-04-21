# Conformance Modes + Deidentify Output Simplification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable conformance registry (HIPAA built-in, GDPR pluggable from Zenta), simplify deidentify output to `{conformance, redacted_text, phi: [{value, type, excerpt}]}`, and add `--conformance` CLI flag.

**Architecture:** New `mosaicx/conformance/` package with a registry + HIPAA spec. The `Deidentifier` module receives a `ConformanceSpec` and uses its regex patterns + prompt fragment. CLI passes conformance from config/flag. Existing regex patterns and HIPAA categories move from `deidentifier.py` into `conformance/hipaa.py`.

**Tech Stack:** Python dataclasses, DSPy signatures, existing regex patterns

**Spec:** `docs/superpowers/specs/2026-04-21-conformance-deidentify-design.md`

---

### Task 1: Conformance Registry

**Files:**
- Create: `mosaicx/conformance/__init__.py`
- Create: `mosaicx/conformance/registry.py`
- Test: `tests/test_conformance.py`

- [ ] **Step 1: Write failing tests for the registry**

Create `tests/test_conformance.py`:

```python
"""Tests for the conformance registry."""
from __future__ import annotations

import re

import pytest


def test_register_and_get_conformance():
    from mosaicx.conformance.registry import ConformanceSpec, get_conformance, register_conformance

    spec = ConformanceSpec(
        name="test_standard",
        description="Test standard",
        phi_categories=["NAME", "DATE"],
        regex_patterns=[
            (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
        ],
        prompt_fragment="Detect names and dates.",
    )
    register_conformance(spec)
    result = get_conformance("test_standard")
    assert result.name == "test_standard"
    assert result.phi_categories == ["NAME", "DATE"]
    assert len(result.regex_patterns) == 1


def test_get_unknown_conformance_raises():
    from mosaicx.conformance.registry import get_conformance

    with pytest.raises(KeyError, match="Unknown conformance"):
        get_conformance("nonexistent_standard_xyz")


def test_list_conformances():
    from mosaicx.conformance.registry import list_conformances

    result = list_conformances()
    assert isinstance(result, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_conformance.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'mosaicx.conformance'`

- [ ] **Step 3: Implement the registry**

Create `mosaicx/conformance/registry.py`:

```python
"""Conformance registry for de-identification standards.

Each conformance defines the PHI categories, regex patterns, and LLM prompt
fragment for a specific privacy standard (e.g., HIPAA, GDPR).

External packages register conformances by calling ``register_conformance()``
at import time. See ``mosaicx/conformance/README.md`` for the plugin interface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ConformanceSpec:
    """Definition of a de-identification conformance standard."""

    name: str
    description: str
    phi_categories: list[str]
    regex_patterns: list[tuple[re.Pattern, str]]
    prompt_fragment: str


_REGISTRY: dict[str, ConformanceSpec] = {}


def register_conformance(spec: ConformanceSpec) -> None:
    """Register a conformance mode. Called at import time by each standard."""
    _REGISTRY[spec.name] = spec


def get_conformance(name: str) -> ConformanceSpec:
    """Look up a registered conformance by name."""
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

Create `mosaicx/conformance/__init__.py`:

```python
"""Conformance standards for de-identification.

Importing this package auto-registers built-in conformances (HIPAA).
External packages can register additional conformances by calling
``register_conformance()`` with a ``ConformanceSpec``.
"""
from __future__ import annotations

from .registry import ConformanceSpec, get_conformance, list_conformances, register_conformance

__all__ = [
    "ConformanceSpec",
    "get_conformance",
    "list_conformances",
    "register_conformance",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_conformance.py -v`

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mosaicx/conformance/__init__.py mosaicx/conformance/registry.py tests/test_conformance.py
git commit -m "feat: add conformance registry for de-identification standards"
```

---

### Task 2: HIPAA Conformance Spec

**Files:**
- Create: `mosaicx/conformance/hipaa.py`
- Modify: `mosaicx/conformance/__init__.py` (add hipaa import)
- Test: `tests/test_conformance.py` (add HIPAA-specific tests)

- [ ] **Step 1: Write failing tests for HIPAA conformance**

Add to `tests/test_conformance.py`:

```python
def test_hipaa_registered_on_import():
    import mosaicx.conformance  # noqa: F401 — triggers auto-registration
    from mosaicx.conformance.registry import get_conformance

    spec = get_conformance("hipaa")
    assert spec.name == "hipaa"
    assert "NAME" in spec.phi_categories
    assert "DATE" in spec.phi_categories
    assert "SSN" in spec.phi_categories
    assert "MRN" in spec.phi_categories
    assert len(spec.regex_patterns) > 0
    assert len(spec.prompt_fragment) > 0


def test_hipaa_regex_catches_ssn():
    import mosaicx.conformance  # noqa: F401
    from mosaicx.conformance.registry import get_conformance

    spec = get_conformance("hipaa")
    text = "SSN: 123-45-6789"
    matches = []
    for pattern, phi_type in spec.regex_patterns:
        for m in pattern.finditer(text):
            matches.append((m.group(), phi_type))
    assert any(phi_type == "SSN" for _, phi_type in matches)


def test_hipaa_regex_no_false_positive_on_clean_text():
    import mosaicx.conformance  # noqa: F401
    from mosaicx.conformance.registry import get_conformance

    spec = get_conformance("hipaa")
    text = "The lungs are clear. No pleural effusion."
    matches = []
    for pattern, phi_type in spec.regex_patterns:
        for m in pattern.finditer(text):
            matches.append(m.group())
    assert len(matches) == 0


def test_hipaa_in_list_conformances():
    import mosaicx.conformance  # noqa: F401
    from mosaicx.conformance.registry import list_conformances

    assert "hipaa" in list_conformances()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_conformance.py::test_hipaa_registered_on_import -v`

Expected: FAIL with `KeyError: "Unknown conformance 'hipaa'"`

- [ ] **Step 3: Implement HIPAA conformance**

Create `mosaicx/conformance/hipaa.py`:

```python
"""HIPAA Safe Harbor conformance for de-identification.

Defines the 18 HIPAA Safe Harbor identifier categories, regex patterns
for format-based PHI (SSNs, phone numbers, MRNs, emails, dates), and
an LLM prompt fragment specifying what to look for.
"""
from __future__ import annotations

import re

from .registry import ConformanceSpec, register_conformance

HIPAA_PHI_CATEGORIES: list[str] = [
    "NAME",
    "DATE",
    "AGE",
    "ADDRESS",
    "ZIP",
    "PHONE",
    "FAX",
    "EMAIL",
    "URL",
    "SSN",
    "MRN",
    "ID",
    "ACCOUNT",
    "LICENSE",
    "INSURANCE",
    "CERTIFICATE",
    "DEVICE_ID",
    "IP_ADDRESS",
    "BIOMETRIC",
    "PHOTO",
    "OTHER",
]

HIPAA_REGEX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # SSN: 123-45-6789
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    # Phone: (555) 123-4567 or 555-123-4567 or 555.123.4567
    (re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"), "PHONE"),
    # MRN: MRN: 12345678 (case-insensitive)
    (re.compile(r"\bMRN\s*:?\s*\d{6,}\b", re.IGNORECASE), "MRN"),
    # Email: john.doe@hospital.com
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "EMAIL"),
    # US/EU dates with slashes: 1/2/2024 or 01/02/24
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "DATE"),
    # Dot-separated dates: 27.02.2026 or 27.02.26
    (re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"), "DATE"),
    # ISO dates: 2026-02-27
    (re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"), "DATE"),
    # German/English abbreviated month dates: 27.Feb.2026, 03.Mär.2024
    (
        re.compile(
            r"\b\d{1,2}\."
            r"(?:Jan|Feb|Mär|Mar|Apr|Mai|May|Jun|Jul|Aug|Sep|Okt|Oct|Nov|Dez|Dec)\."
            r"\d{2,4}\b",
            re.IGNORECASE,
        ),
        "DATE",
    ),
]

HIPAA_PROMPT_FRAGMENT: str = (
    "You are de-identifying a medical document under HIPAA Safe Harbor rules. "
    "Detect ALL of the following 18 identifier categories: "
    "names, dates (except year), ages over 89, addresses, ZIP codes, "
    "phone numbers, fax numbers, email addresses, URLs, "
    "Social Security numbers, medical record numbers, "
    "health plan beneficiary numbers, account numbers, "
    "license/certificate numbers, vehicle/device identifiers and serial numbers, "
    "IP addresses, biometric identifiers, and full-face photographs. "
    "Also detect names of physicians, institutions, and any other "
    "unique identifying numbers or codes. "
    "Be thorough: it is better to over-detect than to miss PHI."
)

HIPAA_SPEC = ConformanceSpec(
    name="hipaa",
    description="HIPAA Safe Harbor (18 identifiers)",
    phi_categories=HIPAA_PHI_CATEGORIES,
    regex_patterns=HIPAA_REGEX_PATTERNS,
    prompt_fragment=HIPAA_PROMPT_FRAGMENT,
)

register_conformance(HIPAA_SPEC)
```

Update `mosaicx/conformance/__init__.py` — add the HIPAA import at the bottom:

```python
# Auto-register built-in conformances
from . import hipaa as _hipaa  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_conformance.py -v`

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mosaicx/conformance/hipaa.py mosaicx/conformance/__init__.py tests/test_conformance.py
git commit -m "feat: add HIPAA Safe Harbor conformance with regex + prompt"
```

---

### Task 3: Conformance Plugin README

**Files:**
- Create: `mosaicx/conformance/README.md`

- [ ] **Step 1: Write the README**

Create `mosaicx/conformance/README.md`:

```markdown
# Adding Conformance Standards

MOSAICX supports pluggable conformance standards for de-identification.
HIPAA Safe Harbor ships built-in. Additional standards can be registered
by external packages.

## How It Works

Each conformance standard is a `ConformanceSpec` that defines:

- **name** — identifier used in `--conformance` flag (e.g., `"hipaa"`)
- **description** — human-readable label
- **phi_categories** — list of PHI type labels (e.g., `["NAME", "DATE", "SSN"]`)
- **regex_patterns** — list of `(compiled_regex, type_label)` tuples for deterministic PHI detection
- **prompt_fragment** — text injected into the LLM prompt telling it which categories to detect

## Registering a Custom Conformance

```python
# my_package/conformance/my_standard.py

import re
from mosaicx.conformance import ConformanceSpec, register_conformance

MY_SPEC = ConformanceSpec(
    name="my_standard",
    description="My Privacy Standard",
    phi_categories=["NAME", "DATE", "NATIONAL_ID"],
    regex_patterns=[
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "NATIONAL_ID"),
    ],
    prompt_fragment=(
        "Detect all personally identifiable information under My Standard: "
        "names, dates, national ID numbers."
    ),
)

register_conformance(MY_SPEC)
```

Ensure your module is imported before the CLI runs (e.g., in your
package's `__init__.py` or via a wrapper script).

## Testing

```python
from mosaicx.conformance import get_conformance

spec = get_conformance("my_standard")
assert spec.name == "my_standard"

# Test regex patterns
text = "ID: 123-45-6789"
for pattern, phi_type in spec.regex_patterns:
    for m in pattern.finditer(text):
        print(f"Found {phi_type}: {m.group()}")
```

## Built-in Standards

| Standard | File | Description |
|----------|------|-------------|
| `hipaa` | `hipaa.py` | HIPAA Safe Harbor (18 identifiers) |
```

- [ ] **Step 2: Commit**

```bash
git add mosaicx/conformance/README.md
git commit -m "docs: add conformance plugin README"
```

---

### Task 4: Wire Conformance into Deidentifier Pipeline

**Files:**
- Modify: `mosaicx/pipelines/deidentifier.py`
- Test: `tests/test_deidentifier_pipeline.py` (update existing tests)

- [ ] **Step 1: Write failing tests for conformance-aware deidentifier**

Add to `tests/test_deidentifier_pipeline.py`:

```python
class TestConformanceAwareRegex:
    """Test that regex scrubbing uses conformance patterns."""

    def test_scrub_with_hipaa_conformance(self):
        from mosaicx.conformance import get_conformance
        from mosaicx.pipelines.deidentifier import regex_scrub_phi_conformance

        spec = get_conformance("hipaa")
        text = "Patient SSN 123-45-6789 presents with cough."
        scrubbed, mappings = regex_scrub_phi_conformance(text, spec)
        assert "123-45-6789" not in scrubbed
        assert "cough" in scrubbed
        assert any(m["phi_type"] == "SSN" for m in mappings)

    def test_scrub_conformance_clean_text(self):
        from mosaicx.conformance import get_conformance
        from mosaicx.pipelines.deidentifier import regex_scrub_phi_conformance

        spec = get_conformance("hipaa")
        text = "Normal chest radiograph. No acute findings."
        scrubbed, mappings = regex_scrub_phi_conformance(text, spec)
        assert scrubbed == text
        assert len(mappings) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_deidentifier_pipeline.py::TestConformanceAwareRegex -v`

Expected: FAIL with `ImportError: cannot import name 'regex_scrub_phi_conformance'`

- [ ] **Step 3: Add `regex_scrub_phi_conformance` to deidentifier.py**

Add this function after the existing `regex_scrub_phi_with_mappings` function (around line 163) in `mosaicx/pipelines/deidentifier.py`:

```python
def regex_scrub_phi_conformance(
    text: str,
    spec: "ConformanceSpec",
) -> tuple[str, list[dict[str, Any]]]:
    """Run regex PHI scrubbing using patterns from a ConformanceSpec.

    Same logic as ``regex_scrub_phi_with_mappings`` but uses the
    conformance-specific patterns instead of the hardcoded module-level ones.
    """
    raw_matches: list[tuple[int, int, str, str]] = []
    for pattern, phi_type in spec.regex_patterns:
        for m in pattern.finditer(text):
            raw_matches.append((m.start(), m.end(), m.group(), phi_type))

    raw_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    merged: list[tuple[int, int, str, str]] = []
    for start, end, matched_text, phi_type in raw_matches:
        if merged and start < merged[-1][1]:
            continue
        merged.append((start, end, matched_text, phi_type))

    mappings: list[dict[str, Any]] = []
    result = text
    for start, end, matched_text, phi_type in reversed(merged):
        result = result[:start] + _REDACTED + result[end:]
        mappings.append({
            "original": matched_text,
            "replacement": _REDACTED,
            "start": start,
            "end": end,
            "phi_type": phi_type,
            "method": "regex",
        })

    mappings.reverse()
    return result, mappings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_deidentifier_pipeline.py -v`

Expected: All tests PASS (existing + 2 new)

- [ ] **Step 5: Update `Deidentifier.__init__` and `forward` to accept conformance**

In `_build_dspy_classes()` in `mosaicx/pipelines/deidentifier.py`, update the `RedactPHI` signature and `Deidentifier` class:

Replace the `RedactPHI` class (lines 454-490) with:

```python
    class RedactPHI(dspy.Signature):
        """Remove or replace Protected Health Information (PHI) from a
        medical document while preserving clinical content.

        Modes:
            - remove: Replace PHI with [REDACTED].
            - pseudonymize: Replace PHI with realistic fake values.
            - dateshift: Shift all dates by a consistent random offset.
        """

        document_text: str = dspy.InputField(
            desc="Full text of the medical document to de-identify"
        )
        mode: str = dspy.InputField(
            desc="De-identification mode: 'remove', 'pseudonymize', or 'dateshift'",
            default="remove",
        )
        redacted_text: str = dspy.OutputField(
            desc="De-identified text with PHI removed or replaced"
        )
        phi_entities: str = dspy.OutputField(
            desc="JSON list of PHI entities found, each with 'text' (the "
                 "exact original PHI string as it appears in document_text), "
                 "'type' (PHI category), "
                 "'excerpt' (the full line from the document "
                 "containing this PHI). "
                 "Be thorough: detect ALL identifiers including names of "
                 "physicians, institutions, dates, locations, and any "
                 "unique identifying numbers. "
                 'Example: '
                 '[{"text": "Jane Doe", "type": "NAME", '
                 '"excerpt": "Patient Name: Jane Doe"}]'
        )
```

Replace the `Deidentifier` class (lines 494-598) with:

```python
    class Deidentifier(dspy.Module):
        """DSPy Module implementing the belt-and-suspenders de-identifier.

        Accepts a ``conformance`` name to select which privacy standard
        governs detection (regex patterns + LLM prompt). Defaults to HIPAA.
        """

        def __init__(self, conformance: str = "hipaa") -> None:
            super().__init__()
            from mosaicx.conformance import get_conformance

            self._spec = get_conformance(conformance)

            # Build signature with conformance-specific prompt
            sig = RedactPHI
            # Prepend conformance prompt to the phi_entities field description
            original_desc = sig.output_fields["phi_entities"].json_schema_extra.get("desc", "")
            conformance_desc = (
                self._spec.prompt_fragment + " "
                "For each PHI entity, return a JSON object with 'text', 'type' "
                f"(one of: {', '.join(self._spec.phi_categories)}), "
                "and 'excerpt'. " + original_desc
            )
            custom_sig = sig.with_updated_fields(
                "phi_entities", desc=conformance_desc,
            )
            self.redact = dspy.ChainOfThought(custom_sig)

        def forward(
            self, document_text: str, mode: str = "remove"
        ) -> dspy.Prediction:
            """Run the de-identification pipeline.

            Returns a simplified output:
            - ``conformance`` (str) -- which standard was applied.
            - ``redacted_text`` (str) -- the de-identified text.
            - ``phi`` (list[dict]) -- PHI items, each with ``value``,
              ``type``, and ``excerpt``.
            """
            from mosaicx.metrics import PipelineMetrics, get_tracker, track_step

            metrics = PipelineMetrics()
            tracker = get_tracker()

            # Layer 1: LLM-based redaction
            with track_step(metrics, "LLM redaction", tracker):
                llm_result = self.redact(
                    document_text=document_text,
                    mode=mode,
                )
            llm_text: str = llm_result.redacted_text

            # Layer 2: regex safety net (only in "remove" mode)
            if mode == "remove":
                with track_step(metrics, "Regex guard", tracker):
                    scrubbed_text, _ = regex_scrub_phi_conformance(
                        llm_text, self._spec,
                    )
            else:
                scrubbed_text = llm_text

            # Build simplified PHI list
            with track_step(metrics, "PHI mapping", tracker):
                import json as _json
                try:
                    raw_entities = _json.loads(
                        getattr(llm_result, "phi_entities", "[]")
                    )
                    if not isinstance(raw_entities, list):
                        raw_entities = []
                except (ValueError, TypeError):
                    raw_entities = []

                # LLM-detected PHI
                phi: list[dict[str, str]] = []
                seen: set[str] = set()
                for entity in raw_entities:
                    value = entity.get("text", "").strip()
                    if not value or value in seen:
                        continue
                    seen.add(value)
                    phi.append({
                        "value": value,
                        "type": entity.get("type", "OTHER").upper(),
                        "excerpt": entity.get("excerpt", ""),
                    })

                # Regex-detected PHI (on original text, remove mode only)
                if mode == "remove":
                    _, regex_mappings = regex_scrub_phi_conformance(
                        document_text, self._spec,
                    )
                    for m in regex_mappings:
                        value = m["original"]
                        if value not in seen:
                            seen.add(value)
                            phi.append({
                                "value": value,
                                "type": m["phi_type"],
                                "excerpt": "",
                            })

            self._last_metrics = metrics

            return dspy.Prediction(
                conformance=self._spec.name,
                redacted_text=scrubbed_text,
                phi=phi,
            )
```

- [ ] **Step 6: Run all deidentifier tests**

Run: `.venv/bin/python -m pytest tests/test_deidentifier_pipeline.py tests/test_conformance.py -v`

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add mosaicx/pipelines/deidentifier.py tests/test_deidentifier_pipeline.py
git commit -m "feat: wire conformance into Deidentifier pipeline + simplified output"
```

---

### Task 5: Config + CLI Wiring

**Files:**
- Modify: `mosaicx/config.py` (add `conformance` field)
- Modify: `mosaicx/cli.py` (add `--conformance` option, update output handling)

- [ ] **Step 1: Add `conformance` field to config**

In `mosaicx/config.py`, add after the `deidentify_mode` line (around line 83):

```python
    conformance: str = "hipaa"  # MOSAICX_CONFORMANCE — privacy standard for de-identification
```

- [ ] **Step 2: Add `--conformance` option to CLI deidentify command**

In `mosaicx/cli.py`, add a new Click option to the `deidentify` command. Find the existing options block (around line 2064-2081) and add after the `--mode` option:

```python
@click.option(
    "--conformance",
    type=str,
    default=None,
    help="Privacy conformance standard (default: hipaa). Use 'mosaicx deidentify --list-conformances' for available standards.",
)
```

Update the `deidentify` function signature to include `conformance: str | None`:

```python
def deidentify(
    document: Path | None,
    directory: Path | None,
    mode: str,
    conformance: str | None,
    workers: int,
    ...
```

- [ ] **Step 3: Update the deidentify function body**

After `_configure_dspy()` (around line 2149), replace the `Deidentifier()` instantiation:

```python
    # Resolve conformance
    effective_conformance = conformance or get_config().conformance
    from .pipelines.deidentifier import Deidentifier

    try:
        deid = Deidentifier(conformance=effective_conformance)
    except KeyError as exc:
        raise click.ClickException(str(exc))
```

- [ ] **Step 4: Update the output handling to use simplified format**

Replace the JSON output section (around lines 2229-2245). After the spinner completes and result is returned, the output data should use the new shape:

```python
    redacted = result.redacted_text
    phi = getattr(result, "phi", [])
    result_conformance = getattr(result, "conformance", effective_conformance)

    # Build output data in the simplified format
    output_data: dict[str, Any] = {
        "conformance": result_conformance,
        "redacted_text": redacted,
    }
    if phi:
        output_data["phi"] = phi
```

For the `--output` JSON/YAML path, replace the `save_data` construction to use `output_data` directly (no more `redaction_map`, `_source`, `build_source_block`):

```python
        else:
            if not suffix:
                output = output.with_suffix(".json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(output_data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            console.print(theme.ok(f"Saved to {output}"))
```

**Note:** Keep the native format output path (PDF/image/text redaction) working — it still needs a redaction_map for coordinate-based redaction. For native formats, reconstruct the redaction_map from the phi list by calling `_build_redaction_map_from_entities` on the phi items. This is only needed when `--output` points to a native format.

- [ ] **Step 5: Update the CLI display section**

Update the PHI display table (around lines 2260-2280) to use the new `phi` list instead of `redaction_map`:

```python
    # PHI table display
    if phi:
        theme.section("PHI Detected", console, "02")
        t = theme.make_clean_table()
        t.add_column("Type", style=f"bold {theme.CORAL}")
        t.add_column("Value")
        t.add_column("Excerpt", style=theme.MUTED)
        for item in phi:
            t.add_row(
                item.get("type", "OTHER"),
                item.get("value", ""),
                item.get("excerpt", ""),
            )
        from rich.padding import Padding
        console.print(Padding(t, (0, 0, 0, 2)))
```

- [ ] **Step 6: Verify the full CLI flow**

Run: `.venv/bin/python -m mosaicx deidentify --help`

Expected: Shows `--conformance` option in help output.

- [ ] **Step 7: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_conformance.py tests/test_deidentifier_pipeline.py -v`

Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add mosaicx/config.py mosaicx/cli.py
git commit -m "feat: add --conformance CLI flag + simplified deidentify output"
```
