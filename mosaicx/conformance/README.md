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
