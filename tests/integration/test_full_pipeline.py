# tests/integration/test_full_pipeline.py
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.mark.integration
class TestExtractVerifyPipeline:
    """Test the extract -> verify pipeline end-to-end (deterministic only)."""

    def test_extract_then_verify_quick(self, tmp_path: Path):
        """Extract from text, then verify the extraction deterministically."""
        from mosaicx.verify.engine import verify

        # Simulate extraction output (no LLM needed)
        extraction = {
            "findings": [
                {"anatomy": "RUL", "measurement": {"value": 5, "unit": "mm"}, "description": "5mm nodule"},
            ],
            "impressions": [{"statement": "Pulmonary nodule", "finding_refs": [0]}],
        }
        source_text = "Findings: 5mm ground glass nodule in the right upper lobe. Impression: Pulmonary nodule."

        report = verify(extraction=extraction, source_text=source_text, level="quick")
        assert report.verdict == "verified"
        assert report.level == "deterministic"
        assert len(report.issues) == 0

    def test_extract_with_mosaicx_envelope(self):
        """Every output should have the _mosaicx envelope."""
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology", template="chest_ct")
        output = {"exam_type": "CT Chest", "_mosaicx": env}
        assert output["_mosaicx"]["pipeline"] == "radiology"
        assert output["_mosaicx"]["version"] is not None
