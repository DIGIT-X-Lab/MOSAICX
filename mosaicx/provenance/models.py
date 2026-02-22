"""Data models for field-level provenance tracking."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceSpan(BaseModel):
    """Location of evidence in a source document."""

    page: int | None = None
    line_start: int | None = None
    char_start: int
    char_end: int


class FieldEvidence(BaseModel):
    """Evidence linking an extracted field to its source text."""

    field_path: str = Field(description="Dotted path e.g. 'findings[0].severity'")
    source_excerpt: str = Field(description="Exact text from source document")
    source_location: SourceSpan | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ProvenanceMap(BaseModel):
    """Collection of field evidence for an entire extraction."""

    fields: list[FieldEvidence] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the _provenance dict format keyed by field_path."""
        result: dict[str, Any] = {}
        for ev in self.fields:
            entry: dict[str, Any] = {
                "source_excerpt": ev.source_excerpt,
                "confidence": ev.confidence,
            }
            if ev.source_location is not None:
                entry["source_location"] = ev.source_location.model_dump()
            result[ev.field_path] = entry
        return result
