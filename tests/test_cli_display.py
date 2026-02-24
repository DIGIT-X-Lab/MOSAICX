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
