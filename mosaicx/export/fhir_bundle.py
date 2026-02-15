"""FHIR R4 Bundle file export.

Writes one JSON file per extraction result, each containing a FHIR R4
``Bundle`` (type ``"collection"``) with a ``DiagnosticReport`` and one
``Observation`` per finding.

Relies on :func:`mosaicx.schemas.fhir.build_diagnostic_report` for the
actual resource construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mosaicx.schemas.fhir import build_diagnostic_report


def export_fhir_bundles(
    results: list[dict[str, Any]],
    output_dir: Path | str,
) -> list[Path]:
    """Export each result as a FHIR R4 Bundle JSON file.

    Parameters
    ----------
    results:
        List of extraction-result dicts.  Each dict should contain at
        least ``"source_file"``, ``"patient_id"``, ``"findings"``, and
        ``"impression"`` keys.
    output_dir:
        Directory where the JSON files will be written.  Created
        automatically if it does not exist.

    Returns
    -------
    list[Path]
        Paths to the written JSON files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for result in results:
        bundle = build_diagnostic_report(
            patient_id=result.get("patient_id", "unknown"),
            findings=result.get("findings", []),
            impression=result.get("impression", ""),
        )
        source = Path(result.get("source_file", "unknown")).stem
        path = output_dir / f"{source}_fhir.json"
        path.write_text(json.dumps(bundle, indent=2))
        paths.append(path)

    return paths
