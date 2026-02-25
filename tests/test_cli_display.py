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


def test_render_extracted_data_hides_internal_diagnostic_keys():
    """Underscore-prefixed keys (_planner, _extraction_contract) should not
    appear in CLI display output — they're for the JSON file only."""
    from rich.console import Console

    from mosaicx.cli_display import render_extracted_data

    data = {
        "extracted": {
            "procedure_information": "CT Angiography",
            "impression": "Normal coronary arteries",
        },
        "_planner": {
            "planner": "short_doc_bypass",
            "react_used": False,
        },
        "_extraction_contract": {
            "version": "1.0",
            "field_results": [],
        },
    }
    console = Console(record=True, force_terminal=True, width=120)
    render_extracted_data(data, console)
    out = console.export_text()

    # Extracted fields should be visible (humanized to Title Case)
    assert "CT Angiography" in out
    assert "Normal coronary arteries" in out

    # Internal diagnostics should be hidden
    assert "short_doc_bypass" not in out
    assert "react_used" not in out
    assert "_planner" not in out
    assert "extraction_contract" not in out.lower()


def test_render_extracted_data_still_shows_inferred_schema():
    """Auto mode with extracted + inferred_schema should show both."""
    from rich.console import Console

    from mosaicx.cli_display import render_extracted_data

    data = {
        "extracted": {"summary": "stable"},
        "inferred_schema": {"name": "AutoSchema", "fields": ["summary"]},
        "_planner": {"planner": "react"},
    }
    console = Console(record=True, force_terminal=True, width=120)
    render_extracted_data(data, console)
    out = console.export_text()

    # Values are humanized (title-cased) by the display layer
    assert "Stable" in out
    assert "Auto Schema" in out
    assert "react" not in out


def test_render_verification_shows_verdict_and_confidence():
    """Verification panel should show verdict, confidence, level, and field counts."""
    from rich.console import Console

    from mosaicx.cli_display import render_verification

    verification = {
        "result": "verified",
        "confidence": 0.95,
        "verify_type": "extraction",
        "requested_mode": "thorough",
        "executed_mode": "spot_check",
        "fallback_used": True,
        "fallback_reason": "Requested thorough but executed spot_check",
        "based_on_source": True,
        "support_score": 0.95,
        "field_checks": {"verified": 13, "total": 13},
    }
    console = Console(record=True, force_terminal=True, width=120)
    render_verification(verification, console)
    out = console.export_text()

    assert "verified" in out
    assert "95%" in out
    assert "spot_check" in out
    assert "13/13" in out


def test_render_verification_shows_fallback_notice():
    """When thorough is downgraded, the panel should say so."""
    from rich.console import Console

    from mosaicx.cli_display import render_verification

    verification = {
        "result": "verified",
        "confidence": 0.90,
        "requested_mode": "thorough",
        "executed_mode": "spot_check",
        "fallback_used": True,
        "field_checks": {"verified": 5, "total": 5},
    }
    console = Console(record=True, force_terminal=True, width=120)
    render_verification(verification, console)
    out = console.export_text()

    assert "thorough" in out
    assert "spot_check" in out
    assert "downgraded" in out


def test_render_verification_shows_absent_fields():
    """When template_field_count > checked fields, show absent count and 'present fields'."""
    from rich.console import Console

    from mosaicx.cli_display import render_verification

    verification = {
        "result": "verified",
        "confidence": 0.95,
        "requested_mode": "thorough",
        "executed_mode": "spot_check",
        "fallback_used": True,
        "field_checks": {"verified": 12, "total": 12},
    }
    console = Console(record=True, force_terminal=True, width=120)
    render_verification(verification, console, template_field_count=13)
    out = console.export_text()

    assert "12/12" in out
    assert "present fields verified" in out
    assert "absent" in out
    assert "not in document" in out


def test_render_verification_shows_contradicted():
    """Contradicted verdict should appear in output."""
    from rich.console import Console

    from mosaicx.cli_display import render_verification

    verification = {
        "result": "contradicted",
        "confidence": 0.20,
        "requested_mode": "standard",
        "executed_mode": "spot_check",
        "fallback_used": False,
        "field_checks": {"verified": 1, "total": 4},
    }
    console = Console(record=True, force_terminal=True, width=120)
    render_verification(verification, console)
    out = console.export_text()

    assert "contradicted" in out
    assert "1/4" in out


def test_render_extracted_data_shows_none_fields_as_missing():
    """None and empty-string fields should show the missing badge."""
    from rich.console import Console

    from mosaicx.cli_display import render_extracted_data

    data = {
        "extracted": {
            "procedure_information": "CT Angiography",
            "impression": None,
            "clinical_information": "",
        }
    }
    console = Console(record=True, force_terminal=True, width=120)
    render_extracted_data(data, console)
    out = console.export_text()

    assert "CT Angiography" in out
    assert "Impression" in out        # key still appears
    assert "missing" in out.lower()   # badge text present


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
