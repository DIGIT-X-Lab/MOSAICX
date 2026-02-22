from __future__ import annotations

import json


class TestSourceSpan:
    def test_construction(self):
        from mosaicx.provenance.models import SourceSpan

        span = SourceSpan(page=1, line_start=14, char_start=847, char_end=912)
        assert span.page == 1
        assert span.char_start == 847

    def test_serializable(self):
        from mosaicx.provenance.models import SourceSpan

        span = SourceSpan(page=1, line_start=14, char_start=847, char_end=912)
        data = span.model_dump()
        assert json.dumps(data)  # must be JSON-serializable


class TestFieldEvidence:
    def test_construction(self):
        from mosaicx.provenance.models import FieldEvidence, SourceSpan

        ev = FieldEvidence(
            field_path="findings[0].severity",
            source_excerpt="mild bilateral bony neural foraminal narrowing",
            source_location=SourceSpan(page=1, line_start=14, char_start=847, char_end=912),
            confidence=0.95,
        )
        assert ev.field_path == "findings[0].severity"
        assert ev.confidence == 0.95

    def test_optional_location(self):
        from mosaicx.provenance.models import FieldEvidence

        ev = FieldEvidence(
            field_path="exam_type",
            source_excerpt="CT Chest",
            confidence=0.99,
        )
        assert ev.source_location is None


class TestProvenanceMap:
    def test_to_dict(self):
        from mosaicx.provenance.models import FieldEvidence, ProvenanceMap

        pm = ProvenanceMap(fields=[
            FieldEvidence(field_path="findings[0].severity", source_excerpt="mild", confidence=0.95),
            FieldEvidence(field_path="findings[0].anatomy", source_excerpt="C6-7", confidence=0.98),
        ])
        d = pm.to_dict()
        assert "findings[0].severity" in d
        assert d["findings[0].severity"]["source_excerpt"] == "mild"
        assert d["findings[0].severity"]["confidence"] == 0.95
