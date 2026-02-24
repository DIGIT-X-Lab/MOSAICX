from __future__ import annotations

from enum import Enum


def test_format_value_humanizes_internal_enum_token():
    from mosaicx.cli_display import _format_value

    value = "Mricervicalreportv2FindingsBoneMarrowSignal.normal"
    assert _format_value(value) == "Normal"


def test_format_value_humanizes_enum_instance_value():
    from mosaicx.cli_display import _format_value

    CervicalSignal = Enum(
        "CervicalSignal",
        {"normal": "Mricervicalreportv2FindingsBoneMarrowSignal.normal"},
    )
    assert _format_value(CervicalSignal.normal) == "Normal"


def test_format_value_humanizes_internal_values_inside_lists():
    from mosaicx.cli_display import _format_value

    values = [
        "Mricervicalreportv2FindingsitemDiscLocation.none",
        "Mricervicalreportv2FindingsitemDiscBulgeType.posterior_central",
    ]
    assert _format_value(values) == "None, Posterior central"


def test_format_value_preserves_vertebral_level_hyphen():
    from mosaicx.cli_display import _format_value

    assert _format_value("C3-C4") == "C3-C4"
    assert _format_value("c7-t1") == "C7-T1"


def test_render_list_table_keeps_sparse_level_finding_columns():
    from rich.console import Console

    from mosaicx.cli_display import _render_list_table

    items = [
        {"level": "C2-C3", "disc_bulge_type": None, "disc_protrusion": None},
        {"level": "C3-C4", "disc_bulge_type": "Disc", "disc_protrusion": None},
        {"level": "C4-C5", "disc_bulge_type": None, "disc_protrusion": None},
    ]
    console = Console(record=True, force_terminal=True, width=120)
    _render_list_table("level_findings", items, console)
    out = console.export_text()
    assert "Disc Bulge Type" in out
    assert "C3-C4" in out
