# tests/test_scoring.py
"""Tests for radiology scoring systems."""

import pytest


class TestScoringEnums:
    def test_lung_rads_values(self):
        from mosaicx.schemas.radreport.scoring import LungRADS
        assert LungRADS.CATEGORY_1.value == "1"
        assert LungRADS.CATEGORY_4B.value == "4B"

    def test_birads_values(self):
        from mosaicx.schemas.radreport.scoring import BIRADS
        assert BIRADS.CATEGORY_0.value == "0"
        assert BIRADS.CATEGORY_6.value == "6"

    def test_tirads_values(self):
        from mosaicx.schemas.radreport.scoring import TIRADS
        assert TIRADS.TR1.value == "TR1"

    def test_deauville_values(self):
        from mosaicx.schemas.radreport.scoring import Deauville
        assert Deauville.SCORE_1.value == "1"
        assert Deauville.SCORE_5.value == "5"

    def test_scoring_descriptions(self):
        from mosaicx.schemas.radreport.scoring import get_scoring_description
        desc = get_scoring_description("Lung-RADS", "4B")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_unknown_scoring_returns_empty(self):
        from mosaicx.schemas.radreport.scoring import get_scoring_description
        desc = get_scoring_description("Nonexistent", "X")
        assert desc == ""
