# mosaicx/schemas/radreport/fetcher.py
"""Fetch and parse RadReport templates from the RSNA API.

The RadReport Template Library (https://radreport.org) publishes structured
radiology reporting templates.  Each template is an HTML document with
``<section>`` elements, ``<textarea>`` default text, and LOINC-coded entries
embedded in an XML ``<script>`` block.

This module:

1. Fetches template JSON from the public REST API
   (``https://api3.rsna.org/radreport/v1/templates/{id}/details``).
2. Parses the embedded HTML deterministically — extracts section names,
   LOINC codes, and default text.
3. Returns a structured context string suitable for feeding into the
   SchemaGenerator LLM for rich type inference.

No authentication is required.

Public API
----------
- :func:`fetch_radreport` — fetch + parse a RadReport template by ID.
- :class:`RadReportSection` — parsed section descriptor.
- :class:`RadReportTemplate` — full parsed template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

RADREPORT_API_BASE = "https://api3.rsna.org/radreport/v1"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RadReportSection:
    """A single section parsed from a RadReport HTML template."""

    id: str
    name: str
    default_text: str = ""
    loinc_code: str | None = None
    loinc_meaning: str | None = None


@dataclass
class RadReportTemplate:
    """A fully parsed RadReport template."""

    template_id: str
    title: str
    description: str = ""
    specialty: list[str] = field(default_factory=list)
    author: str = ""
    sections: list[RadReportSection] = field(default_factory=list)
    raw_html: str = ""

    def to_llm_context(self) -> str:
        """Build a structured context string for the SchemaGenerator LLM.

        This gives the LLM clean, pre-parsed information instead of raw HTML,
        so it can focus on inferring rich types (list, enum, object) from the
        section content.
        """
        lines = [
            f"RadReport Template: {self.title} ({self.template_id})",
        ]
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.specialty:
            lines.append(f"Specialty: {', '.join(self.specialty)}")
        lines.append("")
        lines.append("Sections:")
        lines.append("")

        for s in self.sections:
            loinc_part = f" [LOINC {s.loinc_code}: {s.loinc_meaning}]" if s.loinc_code else ""
            lines.append(f"--- {s.name}{loinc_part} ---")
            if s.default_text.strip():
                # Truncate very long default text to keep context focused
                text = s.default_text.strip()
                if len(text) > 1500:
                    text = text[:1500] + "..."
                lines.append(text)
            else:
                lines.append("(empty — free text)")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_radreport(template_id: str) -> RadReportTemplate:
    """Fetch and parse a RadReport template by ID.

    Parameters
    ----------
    template_id:
        RadReport template ID, e.g. ``"RPT50890"`` or ``"50890"``.
        The ``RPT`` prefix is added automatically if missing.

    Returns
    -------
    RadReportTemplate
        Parsed template with sections, LOINC codes, and default text.

    Raises
    ------
    ValueError
        If the API response is invalid or the template is not found.
    httpx.HTTPError
        If the HTTP request fails.
    """
    import httpx

    # Normalise ID: accept "50890" or "RPT50890"
    tid = template_id.strip()
    if not tid.upper().startswith("RPT"):
        tid = f"RPT{tid}"

    url = f"{RADREPORT_API_BASE}/templates/{tid}/details"
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    if not data.get("SUCCESS"):
        raise ValueError(
            f"RadReport API returned an error for {tid}. "
            f"Check that the template ID is correct."
        )

    info = data["DATA"]
    html = info.get("templateData", "")
    if not html:
        raise ValueError(f"Template {tid} has no HTML content.")

    sections = _parse_html_sections(html)
    loinc_map = _parse_loinc_entries(html)

    # Merge LOINC codes into sections
    for s in sections:
        if s.id in loinc_map:
            s.loinc_code, s.loinc_meaning = loinc_map[s.id]

    specialty_list = [
        sp["name"] for sp in info.get("specialty", []) if "name" in sp
    ]

    return RadReportTemplate(
        template_id=tid,
        title=info.get("title", tid),
        description=info.get("description", ""),
        specialty=specialty_list,
        author=f"{info.get('firstname', '')} {info.get('lastname', '')}".strip(),
        sections=sections,
        raw_html=html,
    )


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------


def _parse_html_sections(html: str) -> list[RadReportSection]:
    """Extract ``<section>`` elements from a RadReport HTML template.

    Each section has:
    - ``id`` attribute (e.g. ``"findings"``)
    - ``data-section-name`` attribute (e.g. ``"Findings"``)
    - A ``<textarea>`` child with optional default text
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: regex-based extraction (less robust but no extra dep)
        return _parse_sections_regex(html)

    soup = BeautifulSoup(html, "html.parser")
    sections: list[RadReportSection] = []

    for sec in soup.find_all("section"):
        sec_id = sec.get("id", "")
        sec_name = sec.get("data-section-name", "")
        if not sec_id:
            continue

        # Fall back to header text if data-section-name is missing
        if not sec_name:
            header = sec.find("header")
            if header:
                sec_name = header.get_text(strip=True)

        # Extract textarea default text
        default_text = ""
        textarea = sec.find("textarea")
        if textarea and textarea.string:
            default_text = textarea.string.strip()

        sections.append(RadReportSection(
            id=sec_id,
            name=sec_name or sec_id,
            default_text=default_text,
        ))

    return sections


def _parse_sections_regex(html: str) -> list[RadReportSection]:
    """Regex fallback for parsing sections when BeautifulSoup is unavailable."""
    sections: list[RadReportSection] = []

    # Match <section id="..." data-section-name="...">
    section_re = re.compile(
        r'<section\s+[^>]*id="([^"]+)"[^>]*data-section-name="([^"]*)"',
        re.IGNORECASE,
    )
    textarea_re = re.compile(
        r'<textarea[^>]*>(.*?)</textarea>',
        re.IGNORECASE | re.DOTALL,
    )

    for m in section_re.finditer(html):
        sec_id = m.group(1)
        sec_name = m.group(2)

        # Find the next textarea after this section tag
        default_text = ""
        ta = textarea_re.search(html, m.end())
        if ta and ta.start() < html.find("</section>", m.end()):
            default_text = ta.group(1).strip()

        sections.append(RadReportSection(
            id=sec_id,
            name=sec_name or sec_id,
            default_text=default_text,
        ))

    return sections


def _parse_loinc_entries(html: str) -> dict[str, tuple[str, str]]:
    """Extract LOINC codes from the ``<script type="text/xml">`` block.

    Returns a mapping from section ID (``origtxt``) to
    ``(loinc_code, loinc_meaning)``.
    """
    loinc_map: dict[str, tuple[str, str]] = {}

    # Find the XML script block
    xml_match = re.search(
        r'<script\s+type="text/xml">(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if not xml_match:
        return loinc_map

    xml_text = xml_match.group(1)

    # Parse entries: <entry origtxt="findings"> ... <code meaning="..." value="..." scheme="LOINC"/>
    entry_re = re.compile(
        r'<entry\s+origtxt="([^"]+)"[^>]*>.*?'
        r'<code\s+meaning="([^"]*)"[^>]*value="([^"]*)"[^>]*scheme="LOINC"',
        re.DOTALL | re.IGNORECASE,
    )

    for m in entry_re.finditer(xml_text):
        section_id = m.group(1)
        meaning = m.group(2)
        code = m.group(3)
        loinc_map[section_id] = (code, meaning)

    return loinc_map
