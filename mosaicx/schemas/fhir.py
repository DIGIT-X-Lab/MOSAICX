# mosaicx/schemas/fhir.py
"""FHIR R4 DiagnosticReport bundle builder.

Builds a JSON-serializable FHIR R4 Bundle (type ``"collection"``)
containing a ``DiagnosticReport`` and one ``Observation`` per finding.
The output is a plain Python dict ready for ``json.dumps()``.

This module does **not** depend on any FHIR library; it constructs the
resource dicts directly to keep the dependency footprint minimal.

Key function:
    - build_diagnostic_report: Create a FHIR Bundle from structured
      radiology findings and an impression string.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional


def _make_uuid() -> str:
    """Return a new ``urn:uuid:...`` identifier."""
    return f"urn:uuid:{uuid.uuid4()}"


def _build_observation(
    finding: dict[str, Any],
    patient_reference: str,
) -> dict[str, Any]:
    """Build a single FHIR Observation resource from a finding dict.

    Parameters
    ----------
    finding:
        A dict with keys like ``"anatomy"``, ``"observation"``,
        ``"description"``.
    patient_reference:
        The FHIR reference string for the patient
        (e.g. ``"Patient/P001"``).

    Returns
    -------
    dict
        A FHIR R4 Observation resource dict.
    """
    anatomy = finding.get("anatomy", "")
    observation = finding.get("observation", "")
    description = finding.get("description", "")

    code_text = " ".join(part for part in [anatomy, observation] if part).strip()
    if not code_text:
        code_text = "Unspecified finding"

    obs_resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "code": {
            "text": code_text,
        },
        "subject": {"reference": patient_reference},
    }

    if description:
        obs_resource["valueString"] = description

    return obs_resource


def build_diagnostic_report(
    patient_id: str,
    findings: list[dict[str, Any]],
    impression: str,
    procedure_code: Optional[str] = None,
) -> dict[str, Any]:
    """Build a FHIR R4 Bundle containing a DiagnosticReport and Observations.

    Creates a FHIR Bundle of type ``"collection"`` with:

    * One ``DiagnosticReport`` resource with status ``"final"``,
      the impression as ``conclusion``, and (optionally) a LOINC
      ``code`` for the procedure.
    * One ``Observation`` resource per item in *findings*, each
      referenced from the DiagnosticReport.

    All resource IDs are generated using ``uuid.uuid4()``.

    Parameters
    ----------
    patient_id:
        Logical patient identifier (e.g. ``"P001"``).  Used to
        construct ``"Patient/<patient_id>"`` references.
    findings:
        List of finding dicts.  Each dict may contain ``"anatomy"``,
        ``"observation"``, and ``"description"`` keys.
    impression:
        Free-text impression / conclusion for the report.
    procedure_code:
        Optional LOINC code for the imaging procedure
        (e.g. ``"24627-2"`` for CT chest).

    Returns
    -------
    dict
        A JSON-serializable FHIR R4 Bundle dict.
    """
    patient_reference = f"Patient/{patient_id}"
    now = datetime.now(timezone.utc).isoformat()

    # Build Observation resources and collect references
    observation_entries: list[dict[str, Any]] = []
    observation_references: list[dict[str, str]] = []

    for finding in findings:
        obs = _build_observation(finding, patient_reference)
        observation_entries.append({
            "fullUrl": _make_uuid(),
            "resource": obs,
        })
        observation_references.append({
            "reference": f"Observation/{obs['id']}",
        })

    # Build DiagnosticReport
    report_resource: dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": str(uuid.uuid4()),
        "status": "final",
        "subject": {"reference": patient_reference},
        "effectiveDateTime": now,
        "conclusion": impression,
    }

    if procedure_code:
        report_resource["code"] = {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": procedure_code,
                }
            ],
        }
    else:
        report_resource["code"] = {
            "text": "Diagnostic Report",
        }

    if observation_references:
        report_resource["result"] = observation_references

    # Assemble Bundle
    bundle: dict[str, Any] = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "collection",
        "timestamp": now,
        "entry": [
            {
                "fullUrl": _make_uuid(),
                "resource": report_resource,
            },
            *observation_entries,
        ],
    }

    return bundle
