"""Schema generation helpers for MOSAICX API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union
import re

from ..constants import DEFAULT_LLM_MODEL, PACKAGE_SCHEMA_PYD_DIR
from ..schema.builder import synthesize_pydantic_model
from ..schema.registry import get_suggested_filename


@dataclass(slots=True)
class GeneratedSchema:
    """Container for a generated Pydantic schema."""

    class_name: str
    description: str
    code: str
    suggested_filename: str

    def write(self, destination: Optional[Path | str] = None) -> Path:
        """Persist the schema to disk and return the final path."""
        target = Path(destination) if destination else Path(PACKAGE_SCHEMA_PYD_DIR) / self.suggested_filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.code)
        return target


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

    if isinstance(description, Sequence) and not isinstance(description, (str, bytes)):
        prompt = "\n".join(str(seg) for seg in description)
    else:
        prompt = str(description)

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
    suggested = get_suggested_filename(class_name, prompt)
    return GeneratedSchema(
        class_name=class_name,
        description=description,
        code=code,
        suggested_filename=suggested,
    )


__all__ = ["GeneratedSchema", "generate_schema"]
