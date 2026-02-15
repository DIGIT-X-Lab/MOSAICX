# tests/test_radreport_registry.py
"""Tests for RadReport template registry."""

import pytest


class TestTemplateRegistry:
    def test_list_templates(self):
        from mosaicx.schemas.radreport.registry import list_templates
        templates = list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0

    def test_get_template_by_name(self):
        from mosaicx.schemas.radreport.registry import get_template
        t = get_template("generic")
        assert t is not None
        assert t.name == "generic"

    def test_get_template_unknown_returns_none(self):
        from mosaicx.schemas.radreport.registry import get_template
        t = get_template("nonexistent_modality_xyz")
        assert t is None

    def test_template_has_exam_type(self):
        from mosaicx.schemas.radreport.registry import list_templates
        for t in list_templates():
            assert hasattr(t, "exam_type")
            assert isinstance(t.exam_type, str)

    def test_chest_ct_template_exists(self):
        from mosaicx.schemas.radreport.registry import get_template
        t = get_template("chest_ct")
        assert t is not None
        assert t.exam_type == "chest_ct"
        assert t.radreport_id == "RDES3"
