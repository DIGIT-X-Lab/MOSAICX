"""
MOSAICX API - Schema Generation Helpers

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
Programmatic façade around the schema synthesis pipeline used by the CLI.
Callers supply natural-language descriptions and receive strongly typed
Pydantic modules ready for persistence or direct execution.  This mirrors the
``mosaicx generate`` command while keeping the interface friendly for tests and
notebooks.

Features:
---------
- Returns structured ``GeneratedSchema`` objects with convenience writers.
- Normalises legacy Pydantic ``regex`` arguments to v2-compatible ``pattern``.
- Delegates filename selection to the registry helpers for predictable naming.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union
import re

from ..constants import (
    DEFAULT_LLM_MODEL,
    USER_SCHEMA_DIR,
    PACKAGE_SCHEMA_TEMPLATES_PY_DIR,
)
from ..schema.builder import synthesize_pydantic_model
from ..schema.registry import get_suggested_filename
from ..utils.logging import get_logger

# Module-level logger
_logger = get_logger(__name__)


def _ensure_root_schema_constant(code: str, class_name: str) -> str:
    """Ensure generated code declares ``ROOT_SCHEMA_CLASS``."""
    if "ROOT_SCHEMA_CLASS" in code:
        return code

    lines = code.splitlines()
    last_import_idx = -1
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("from ") or stripped.startswith("import "):
            last_import_idx = idx
            continue
        if stripped == "":
            continue
        break

    insert_at = last_import_idx + 1 if last_import_idx >= 0 else 0
    before = lines[:insert_at]
    after = lines[insert_at:]

    new_lines = before + [f'ROOT_SCHEMA_CLASS = "{class_name}"']
    if not after or after[0].strip():
        new_lines.append("")
    new_lines.extend(after)

    return "\n".join(new_lines).rstrip() + "\n"


@dataclass(slots=True)
class GeneratedSchema:
    """Container for a generated Pydantic schema."""

    class_name: str
    description: str
    code: str
    suggested_filename: str

    def write(
        self,
        destination: Optional[Path | str] = None,
        *,
        template: bool = False,
    ) -> Path:
        """Persist the schema to disk and return the final path.

        Args:
            destination: Optional explicit location for the generated file.
            template: When ``True``, store the schema in the bundled
                template directory regardless of ``destination``.
        """
        if destination and template:
            raise ValueError("Specify either destination or template=True, not both.")

        if template:
            target = PACKAGE_SCHEMA_TEMPLATES_PY_DIR / self.suggested_filename
        elif destination:
            target = Path(destination).expanduser()
        else:
            target = USER_SCHEMA_DIR / self.suggested_filename

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.code, encoding="utf-8")
        return target.resolve(strict=False)


def generate_schema(
    description: Union[str, Sequence[str]],
    *,
    class_name: str = "GeneratedModel",
    model: str = DEFAULT_LLM_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> GeneratedSchema:
    """Generate a Pydantic schema module from a natural-language prompt."""
    _logger.info(f"API generate_schema: class_name={class_name}, model={model}")

    if isinstance(description, Sequence) and not isinstance(description, (str, bytes)):
        prompt = "\n".join(str(seg) for seg in description)
    else:
        prompt = str(description)
    
    _logger.debug(f"Schema description: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

    code = synthesize_pydantic_model(
        description=prompt,
        class_name=class_name,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )

    # Normalise legacy Pydantic v1 "regex" keyword to the v2 "pattern" name.
    code = re.sub(r"\bregex(?=\s*=)", "pattern", code)
    code = _ensure_root_schema_constant(code, class_name)
    suggested = get_suggested_filename(class_name, prompt)
    
    _logger.info(f"Schema generated: {suggested} ({len(code)} chars)")
    
    return GeneratedSchema(
        class_name=class_name,
        description=description,
        code=code,
        suggested_filename=suggested,
    )


__all__ = ["GeneratedSchema", "generate_schema"]
