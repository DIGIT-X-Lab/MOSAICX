"""Tests for the extraction mode registry."""
import pytest


class TestModeRegistry:
    def test_register_and_get_mode(self):
        from mosaicx.pipelines.modes import MODES, register_mode, get_mode

        @register_mode("test_mode", "A test mode")
        class TestMode:
            mode_name: str
            mode_description: str

        assert "test_mode" in MODES
        assert get_mode("test_mode") is TestMode

    def test_get_unknown_mode_raises(self):
        from mosaicx.pipelines.modes import get_mode
        with pytest.raises(ValueError, match="Unknown mode"):
            get_mode("nonexistent_mode_xyz")

    def test_list_modes_returns_tuples(self):
        from mosaicx.pipelines.modes import list_modes
        modes = list_modes()
        assert isinstance(modes, list)
        for name, desc in modes:
            assert isinstance(name, str)
            assert isinstance(desc, str)


class TestBuiltInModes:
    def test_radiology_mode_registered(self):
        from mosaicx.pipelines.radiology import RadiologyReportStructurer  # noqa: F401
        from mosaicx.pipelines.modes import MODES
        assert "radiology" in MODES

    def test_pathology_mode_registered(self):
        from mosaicx.pipelines.pathology import PathologyReportStructurer  # noqa: F401
        from mosaicx.pipelines.modes import MODES
        assert "pathology" in MODES

    def test_list_modes_includes_both(self):
        from mosaicx.pipelines.radiology import RadiologyReportStructurer  # noqa: F401
        from mosaicx.pipelines.pathology import PathologyReportStructurer  # noqa: F401
        from mosaicx.pipelines.modes import list_modes
        names = [name for name, _ in list_modes()]
        assert "radiology" in names
        assert "pathology" in names
