"""Narrative report export -- Markdown, optional PDF/DOCX.

Generates human-readable structured reports from extraction results.
The primary format is Markdown; PDF and DOCX converters may be added
in a future release.

Key function:
    - export_markdown: Write a single extraction result as a Markdown file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_markdown(result: dict[str, Any], path: Path | str) -> None:
    """Write a structured Markdown report for a single extraction result.

    Parameters
    ----------
    result:
        An extraction-result dict with keys such as ``"source_file"``,
        ``"exam_type"``, ``"completeness"``, ``"indication"``,
        ``"findings"`` (list of dicts), and ``"impression"``.
    path:
        Destination file path for the Markdown output.
    """
    path = Path(path)

    lines: list[str] = [
        f"# Structured Report: {result.get('source_file', 'Unknown')}",
        f"\n**Exam type:** {result.get('exam_type', 'generic')}",
        f"**Completeness:** {result.get('completeness', 'N/A')}",
    ]

    if result.get("indication"):
        lines.append(f"\n## Indication\n{result['indication']}")

    lines.append("\n## Findings")
    for i, f in enumerate(result.get("findings", []), 1):
        anatomy = f.get("anatomy", "")
        desc = f.get("description", f.get("observation", ""))
        lines.append(f"{i}. **{anatomy}**: {desc}")

    if not result.get("findings"):
        lines.append("No findings extracted.")

    lines.append(f"\n## Impression\n{result.get('impression', 'N/A')}")

    path.write_text("\n".join(lines))
