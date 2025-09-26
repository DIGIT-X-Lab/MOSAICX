"""PDF extraction helpers for MOSAICX API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel

from ..constants import DEFAULT_LLM_MODEL
from ..extractor import extract_structured_data, extract_text_from_pdf, load_schema_model


@dataclass(slots=True)
class ExtractionResult:
    """Structured extraction payload produced by :func:`extract_pdf`."""

    record: BaseModel
    schema_path: Path
    pdf_path: Path

    def to_dict(self) -> dict:
        """Return the extracted data as a plain ``dict``."""
        return self.record.model_dump()

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the extracted data to a JSON string."""
        return self.record.model_dump_json(indent=indent)

    def write_json(self, path: Path, *, indent: int = 2) -> Path:
        """Write the extraction result to ``path`` in JSON format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent), encoding="utf-8")
        return path


def extract_pdf(
    pdf_path: Union[Path, str],
    schema_path: Union[Path, str],
    *,
    model: str = DEFAULT_LLM_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> ExtractionResult:
    """Extract structured data from a PDF."""

    pdf_path = Path(pdf_path)
    schema_path = Path(schema_path)

    schema_class = load_schema_model(str(schema_path))
    text_content = extract_text_from_pdf(pdf_path)
    record = extract_structured_data(
        text_content,
        schema_class,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )
    return ExtractionResult(record=record, schema_path=schema_path, pdf_path=pdf_path)


__all__ = ["ExtractionResult", "extract_pdf"]
