"""RadReport template registry — discovery and auto-selection."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TemplateInfo:
    """Metadata for a registered RadReport template."""
    name: str
    exam_type: str
    radreport_id: Optional[str] = None
    description: str = ""


# Built-in templates
_TEMPLATES: list[TemplateInfo] = [
    TemplateInfo(name="generic", exam_type="generic", description="Generic radiology report"),
    TemplateInfo(name="chest_ct", exam_type="chest_ct", radreport_id="RDES3", description="Chest CT report"),
    TemplateInfo(name="chest_xr", exam_type="chest_xr", radreport_id="RDES2", description="Chest X-ray report"),
    TemplateInfo(name="brain_mri", exam_type="brain_mri", radreport_id="RDES28", description="Brain MRI report"),
    TemplateInfo(name="abdomen_ct", exam_type="abdomen_ct", radreport_id="RDES44", description="Abdomen CT report"),
    TemplateInfo(name="mammography", exam_type="mammography", radreport_id="RDES4", description="Mammography report"),
    TemplateInfo(name="thyroid_us", exam_type="thyroid_us", radreport_id="RDES72", description="Thyroid ultrasound report"),
    TemplateInfo(name="lung_ct", exam_type="lung_ct", radreport_id="RDES195", description="Lung CT screening report"),
    TemplateInfo(name="msk_mri", exam_type="msk_mri", description="MSK MRI report"),
    TemplateInfo(name="cardiac_mri", exam_type="cardiac_mri", radreport_id="RDES214", description="Cardiac MRI report"),
    TemplateInfo(name="pet_ct", exam_type="pet_ct", radreport_id="RDES76", description="PET/CT report"),
]

_TEMPLATE_MAP = {t.name: t for t in _TEMPLATES}


def list_templates() -> list[TemplateInfo]:
    """Return all registered templates."""
    return list(_TEMPLATES)


def get_template(name: str) -> Optional[TemplateInfo]:
    """Get a template by name. Returns None if not found."""
    return _TEMPLATE_MAP.get(name)
