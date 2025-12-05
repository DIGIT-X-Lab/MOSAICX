"""
MOSAICX API - Document Extraction Helpers

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
Expose high-level functions that transform clinical documents into validated
Pydantic records, matching the behaviour of the CLI ``extract`` command.
Inputs are resolved to schema classes automatically and the resulting payloads
offer multiple serialisation options for downstream systems.

Highlights:
-----------
- ``ExtractionResult`` dataclass wraps the model instance with handy
  ``to_dict`` / ``to_json`` utilities and filesystem writers.
- Delegates schema loading, PDF parsing, and LLM orchestration to the
  underlying extractor module while keeping the API surface compact.
- Shares default model configuration with the rest of the package via
  ``DEFAULT_LLM_MODEL``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Callable

from pydantic import BaseModel

from ..constants import DEFAULT_LLM_MODEL
from ..extractor import (
    extract_structured_data,
    extract_text_from_document,
    extract_text_from_pdf,
    load_schema_model,
)
from ..utils.logging import get_logger

# Module-level logger
_logger = get_logger(__name__)


@dataclass(slots=True)
class ExtractionResult:
    """Structured extraction payload produced by :func:`extract_document`."""

    record: BaseModel
    schema_path: Path
    document_path: Path

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


def extract_document(
    document_path: Union[Path, str],
    schema_path: Union[Path, str],
    *,
    model: str = DEFAULT_LLM_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    backend: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> ExtractionResult:
    """Extract structured data from a clinical document."""
    doc_path = Path(document_path)
    schema_path = Path(schema_path)
    
    _logger.info(f"API extraction: document={doc_path.name}, schema={schema_path.name}")
    _logger.debug(f"Extraction config: model={model}, temperature={temperature}, backend={backend or 'auto'}")

    _logger.debug("Loading schema model")
    schema_class = load_schema_model(str(schema_path))
    
    _logger.debug("Extracting text from document")
    extraction = extract_text_from_document(
        doc_path,
        return_details=True,
        status_callback=status_callback,
    )
    text_content = extraction.markdown
    _logger.debug(f"Extracted {len(text_content)} chars from document")
    
    _logger.info("Extracting structured data with LLM")
    record = extract_structured_data(
        text_content,
        schema_class,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        backend=backend,
    )
    _logger.info(f"Extraction complete: {type(record).__name__}")
    
    return ExtractionResult(record=record, schema_path=schema_path, document_path=doc_path)


def extract_pdf(
    pdf_path: Union[Path, str],
    schema_path: Union[Path, str],
    *,
    model: str = DEFAULT_LLM_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    backend: Optional[str] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> ExtractionResult:
    """Backward-compatible wrapper for :func:`extract_document`."""
    _logger.debug("extract_pdf called (delegating to extract_document)")

    return extract_document(
        document_path=pdf_path,
        schema_path=schema_path,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        backend=backend,
        status_callback=status_callback,
    )


__all__ = ["ExtractionResult", "extract_document", "extract_pdf"]
