# mosaicx/summarizer.py
"""
MOSAICX Summariser - Timeline and Narrative Generation

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Structure first. Insight follows.

Author: Lalith Kumar Shiyam Sundar, PhD
Lab: DIGIT-X Lab
Department: Department of Radiology
University: LMU University Hospital | LMU Munich

Overview:
---------
Generate structured patient timelines and concise narrative summaries from one
or more radiology reports. The summariser powers the CLI ``summarize`` command
and can be embedded directly for notebook experimentation or workflow
automation.

Capabilities:
-------------
- Multi-report ingestion with specialty-agnostic timeline synthesis.
- Rich terminal rendering via ``rich`` plus optional PDF export using ReportLab.
- Layered LLM fallbacks (Instructor JSON, raw JSON, deterministic heuristics) to
  ensure a summary is produced even under degraded conditions.

Design Notes:
-------------
- Defaults to OpenAI-compatible endpoints while remaining friendly to local
  Ollama deployments through shared configuration helpers.
- Preserves MOSAICX colour palette for consistent UI theming across outputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any, Dict, Iterable, List, Optional
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field, ValidationError

# Rich rendering
from rich.align import Align
from rich.panel import Panel
from rich.table import Table

# Package theming & UI helpers
from .display import console, styled_message
from .constants import APPLICATION_VERSION, MOSAICX_COLORS, PROJECT_ROOT
from .utils import resolve_openai_config
from .document_loader import DOC_SUFFIXES
from .text_extraction import TextExtractionError, extract_text_with_fallback

try:
    import instructor  # type: ignore[import-not-found]
    from instructor import Mode  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    instructor = None  # type: ignore[assignment]
    Mode = None  # type: ignore[assignment]

try:
    from openai import OpenAI  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

try:
    from reportlab.lib import colors as rl_colors  # type: ignore[import-not-found]
    from reportlab.lib.pagesizes import A4  # type: ignore[import-not-found]
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore[import-not-found]
    from reportlab.lib.units import cm  # type: ignore[import-not-found]
    from reportlab.platypus import (  # type: ignore[import-not-found]
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table as RLTable,
        TableStyle,
        Indenter,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    rl_colors = None  # type: ignore[assignment]
    A4 = None  # type: ignore[assignment]
    ParagraphStyle = None  # type: ignore[assignment]
    getSampleStyleSheet = None  # type: ignore[assignment]
    cm = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    Paragraph = None  # type: ignore[assignment]
    SimpleDocTemplate = None  # type: ignore[assignment]
    Spacer = None  # type: ignore[assignment]
    RLTable = None  # type: ignore[assignment]
    TableStyle = None  # type: ignore[assignment]
    Indenter = None  # type: ignore[assignment]


# =============================================================================
# Models (Pydantic)
# =============================================================================

class PatientHeader(BaseModel):
    patient_id: Optional[str] = Field(default=None, description="Pseudonymous patient ID")
    dob: Optional[str] = Field(default=None, description="Date of birth (ISO)")
    sex: Optional[str] = Field(default=None, description="Patient sex")
    last_updated: Optional[str] = Field(default=None, description="UTC ISO timestamp of summary generation")


class CriticalEvent(BaseModel):
    date: Optional[str] = Field(default=None, description="ISO date")
    source: Optional[str] = Field(default=None, description="Report file or modality")
    note: str = Field(..., description="Critical note (≤ 160 chars)", max_length=160)


class PatientSummary(BaseModel):
    patient: PatientHeader
    timeline: List[CriticalEvent]
    overall: str = Field(..., description="Concise overall summary (5–7 lines ideal)")


# =============================================================================
# Ephemeral “flash once” helper (suppress repeated warnings)
# =============================================================================

_ONCE: set[str] = set()

def _flash_once(key: str, text: str, *, color_key: str = "warning", duration: float = 0.9) -> None:
    """Show a transient, one-time status line that disappears after the context."""
    if key in _ONCE:
        return
    _ONCE.add(key)
    color = MOSAICX_COLORS.get(color_key, MOSAICX_COLORS["secondary"])
    # Ephemeral status that vanishes when the context exits
    with console.status(f"[{color}]{text}[/]", spinner="dots"):
        sleep(duration)


# =============================================================================
# Ingestion & helpers
# =============================================================================

@dataclass
class ReportDoc:
    path: Path
    text: str
    date_hint: Optional[str] = None
    modality_hint: Optional[str] = None


_DATE_PAT = re.compile(
    r"(?P<iso>\d{4}-\d{2}-\d{2})|(?P<eu>\d{2}[./]\d{2}[./]\d{4})|(?P<us>\d{1,2}/\d{1,2}/\d{4})"
)


def _normalize_date(raw: str) -> Optional[str]:
    """Normalize raw dates to YYYY-MM-DD if obvious; else return as-is."""
    if not raw:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    # Try EU or US-ish: dd.mm.yyyy / dd/mm/yyyy / mm/dd/yyyy; prefer EU when ambiguous.
    if "." in raw or "/" in raw:
        norm = raw.replace(".", "/")
        parts = norm.split("/")
        if len(parts) == 3:
            a, b, y = parts
            a_i, b_i = int(a), int(b)
            if len(a) == 4:  # yyyy/mm/dd
                return f"{a}-{b_i:02d}-{int(y):02d}"
            # assume dd/mm/yyyy rather than mm/dd/yyyy in clinical EU setting
            return f"{y}-{b_i:02d}-{a_i:02d}"
    return raw


def _first_date(text: str) -> Optional[str]:
    m = _DATE_PAT.search(text)
    return _normalize_date(m.group(0)) if m else None


def _guess_modality(text: str) -> Optional[str]:
    for key in ("PET/CT", "CT", "MRI", "MR", "PET", "XR", "X-RAY", "ULTRASOUND", "US", "DXA"):
        if re.search(rf"\b{re.escape(key)}\b", text, flags=re.IGNORECASE):
            return key
    return None


def _read_text(path: Path) -> str:
    """Read clinical documents via layered extraction; fall back to plain text."""
    try:
        result = extract_text_with_fallback(path)
        if result.markdown and result.markdown.strip():
            return result.markdown
    except TextExtractionError:
        pass
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def load_reports(paths: List[Path]) -> List[ReportDoc]:
    docs: List[ReportDoc] = []
    for p in paths:
        txt = _read_text(p)
        if not txt or not txt.strip():
            continue
        head = txt[:2000]
        docs.append(
            ReportDoc(
                path=p,
                text=txt,
                date_hint=_first_date(head) or _first_date(p.name),
                modality_hint=_guess_modality(head) or _guess_modality(txt),
            )
        )
    return docs


def _extract_json_block(text: str) -> Optional[str]:
    """Extract the first JSON object from raw LLM output (handles fenced blocks)."""
    if not text:
        return None
    # ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # naive brace slice
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return None


def _pick_impression(text: str) -> Optional[str]:
    """Grab an 'Impression' paragraph; fallback to the first one or two sentences."""
    m = re.search(r"(?is)\bImpression\b[:\n]+(.*?)(?:\n\s*\n|\Z)", text)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
    return " ".join(sents[:2]).strip() if sents else None


def _heuristic_summary(docs: List[ReportDoc], patient_id: Optional[str]) -> PatientSummary:
    """Deterministic safety net: derive a usable summary from text only."""
    timeline: List[CriticalEvent] = []
    notes: List[str] = []
    for d in sorted(docs, key=lambda r: (r.date_hint or "", r.path.name)):
        note = _pick_impression(d.text) or "Key findings summarized from report."
        note = note[:160]  # enforce note bound
        notes.append(note)
        timeline.append(
            CriticalEvent(
                date=d.date_hint,
                source=d.path.name if d.path else (d.modality_hint or "report"),
                note=note,
            )
        )
    overall = re.sub(r"\s+", " ", " ".join(notes)).strip()
    overall = overall[:1200] if overall else "Concise summary compiled from available reports."
    return PatientSummary(
        patient=PatientHeader(
            patient_id=patient_id,
            last_updated=datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        ),
        timeline=timeline,
        overall=overall,
    )


# =============================================================================
# LLM Summarization
# =============================================================================

SUMM_SYSTEM = """
ROLE
You are a radiology-first clinical summarizer (adaptable across specialties). Summarize ONE patient by producing ONLY:
  (1) a CRITICAL timeline of events and
  (2) a CONCISE but COMPLETE overall summary (single paragraph).

OUTPUT CONTRACT
- Return JSON that matches the schema: { patient, timeline[], overall }.
- No extra keys, no markdown, no prose outside JSON.

GLOBAL PRINCIPLES (ADAPTIVE, NOT RIGID)
- Include only decisive, decision-relevant facts explicitly stated in the reports.
- Strive for conciseness: merge multiple findings from the same report into a compact note, prioritize recency and clinical impact, and avoid redundancy.
- Prefer facts with explicit evidence (measurements, clear status terms, modality tags). Do not invent or infer beyond what the text says.

TIMELINE — WHAT COUNTS AS "CRITICAL"
- Explicit status: progression / stable disease / partial response / complete response.
- New or resolved lesions; clear size change of tracked lesions.
- Key quantitative metrics when present (with units):
  • Parenchymal: longest diameter (mm/cm)
  • Lymph nodes: short-axis (mm)
  • PET/NM: SUVmax (state tracer if given)
  • Vascular: diameters (e.g., AAA), stenosis %
  • Cardiac: Agatston score, EF %, wall motion
- Therapy START/STOP/CHANGED if stated in the report body.
- Serious complications/adverse events mentioned in the report.
- If no critical events, include the key negative/neutral finding (e.g., “no acute intracranial abnormality”).

TIMELINE — FORMAT & RULES
- Exactly one timeline entry per report/source (merge multiple facts into the same entry if necessary).
- Each note must be self-contained, concise, ≤160 characters, no “see above” phrasing.
- Prefer ISO dates (YYYY-MM-DD). If unknown, set date = null.
- Set a short source tag (e.g., “CT 2025-09-10”, “MRI 2025-08-01”, “PET/CT”, or short report ID).
- Sort ascending by date; null dates go last (preserve input order among nulls).
- Use exact numbers/units ONLY if present; never fabricate priors or deltas.

ADAPTIVE MODALITY HINTS (use when applicable; otherwise ignore)
- CT/MRI (body): lesion size (longest), nodes (short-axis), enhancement, new/resolved lesions.
- Neuro (CT/MRI brain/spine): acute hemorrhage/infarct, mass effect/midline shift, new/enlarging masses, DWI/ADC trends.
- PET/SPECT/NM: new/vanishing foci, SUVmax trend, tracer (FDG/PSMA), distribution (nodal/visceral/bone).
- Ultrasound: focal lesions with size/location; vascularity/Doppler if reported; organ-specific key findings.
- Radiographs (XR): fracture/dislocation/alignment; consolidation/atelectasis; lines/tubes if emphasized.
- Breast (MG/MRI/US): BI-RADS and the driver finding (size, morphology, location); no management advice.
- Cardiac (CT/MR): Agatston score; coronary stenosis %; EF %, wall motion; valve/device issues if reported.
- Vascular: aneurysm diameters, stenosis %, endoleak presence/type; graft/stent patency if reported.
- Interventional: procedure, target/approach, device(s), immediate outcome, complications.

STYLE & VOCABULARY
- Terse, standard radiology language. Compact trends are preferred (e.g., “LN 12→16 mm — progression”).
- SI units and common abbreviations (mm, cm, ng/mL, SUVmax). Keep consistent within a patient.

STRICT DO-NOTS
- Do NOT recommend tests, management, or follow-up.
- Do NOT suggest differential diagnoses or “next steps”.
- Do NOT extrapolate beyond the provided reports.
- Do NOT include PHI unless a pseudonym was provided.
- Do NOT invent dates, numbers, sources, or priors.

EDGE CASES
- Multiple critical facts in one report: merge them into a concise entry; prioritize recency/impact; avoid redundancy.
- Same-day studies: keep both; disambiguate the source (“CT am”/“CT pm” or short report IDs).
- Cross-modality comparisons: state numeric deltas only if the report itself makes that comparison.
- Qualitative priors (“larger than prior” without numbers): state trend qualitatively without numbers.
- Uncertain dates: set date = null and still include the note.

OVERALL SUMMARY (single narrative paragraph; executive; complete; source-aware)
- For multiple reports that are not related (e.g., different modalities or body parts), cover each briefly. And mention that they are unrelated.
- Cover all critical events from the timeline, in order.
- Retain the temporal sequence of events from the timeline. Do not mix up the order.
- Write a single paragraph that makes the temporal sequence clear even without the timeline.
- After each event/claim, include a bracketed source tag for trace-back (e.g., “… stable disease [Source: CT 2023-01-15] … partial response in liver [Source: MRI 2023-03-10] …”). Make sure each claim is traceable.
- Cover: current status + anatomic distribution; trends (mm/% when present or qualitative when not); functional/biomarker highlights only if mentioned; therapy ON/OFF/CHANGED when stated; material discrepancies/limitations.
- Factual only; no recommendations; no differentials; no interpretations beyond the text.

FINAL CHECKS
- Timeline notes ≤160 chars, sorted; numbers/units match the report text.
- Overall paragraph is concise yet complete and includes bracketed source tags for each claim.
"""



def _instructor_client(base_url: Optional[str], api_key: Optional[str]) -> OpenAI:
    if OpenAI is None or instructor is None or Mode is None:
        raise RuntimeError("Instructor/OpenAI dependencies are not installed.")
    resolved_base_url, resolved_api_key = resolve_openai_config(base_url, api_key)
    client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)
    return instructor.patch(client, mode=Mode.JSON)


def summarize_with_llm(
    docs: List[ReportDoc],
    *,
    patient_id: Optional[str],
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    temperature: float = 0.2,
    max_events_per_report: int = 2,
) -> PatientSummary:
    """
    LLM → PatientSummary with robust fallbacks:
      1) Instructor JSON (response_model=PatientSummary)
      2) Raw JSON (extract JSON block → parse → validate)
      3) Heuristic summary (deterministic)

    Fallback notices are shown at most once per process, and are transient (disappear).
    """
    if OpenAI is None or instructor is None or Mode is None:
        _flash_once(
            "missing_llm_deps",
            "Optional dependencies for LLM summarization not installed. Using heuristic summary.",
            color_key="warning",
            duration=0.6,
        )
    if OpenAI is None or instructor is None or Mode is None:
        return _heuristic_summary(docs, patient_id)

    # Build user content
    parts: List[str] = []
    if patient_id:
        parts.append(f"Patient ID: {patient_id}")
    for i, d in enumerate(sorted(docs, key=lambda r: (r.date_hint or "", r.path.name))):
        parts.append(f"\n--- REPORT {i+1} ---")
        parts.append(f"Source: {d.path.name}")
        if d.date_hint:
            parts.append(f"Date (hint): {d.date_hint}")
        if d.modality_hint:
            parts.append(f"Modality (hint): {d.modality_hint}")
        parts.append("\nContent:\n" + d.text[:6000])  # cap per report for local models

    user = "\n".join(parts) + f"\n\nMax events per report: {max_events_per_report}."
    messages = [
        {"role": "system", "content": SUMM_SYSTEM},
        {"role": "user", "content": user},
    ]

    # 1) Instructor JSON
    try:
        client = _instructor_client(base_url, api_key)
        ps: PatientSummary = client.chat.completions.create(  # type: ignore[assignment]
            model=model,
            temperature=temperature,
            messages=messages,
            response_model=PatientSummary,
            max_retries=2,
        )
        # Fill missing header bits
        ps.patient.last_updated = datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        if patient_id and not ps.patient.patient_id:
            ps.patient.patient_id = patient_id
        return ps
    except Exception:
        _flash_once(
            "fallback_instructor",
            "Model returned invalid/empty structured output. Falling back to raw‑JSON parsing…",
            color_key="warning",
        )

    # 2) Raw JSON fallback
    try:
        json_guard = (
            "Return ONLY a JSON object with keys: patient, timeline, overall. "
            "Schema:\n"
            "{\n"
            '  "patient": {"patient_id": str|null, "dob": str|null, "sex": str|null, "last_updated": str|null},\n'
            '  "timeline": [{"date": str|null, "source": str|null, "note": str}],\n'
            '  "overall": str\n'
            "}\n"
            "No markdown, no prose—only JSON."
        )
        raw_messages = [
            {"role": "system", "content": f"{SUMM_SYSTEM}\n{json_guard}"},
            {"role": "user", "content": user},
        ]
        resolved_base_url, resolved_api_key = resolve_openai_config(base_url, api_key)
        raw_client = OpenAI(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
        )
        resp = raw_client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=raw_messages,
        )
        content = (resp.choices[0].message.content or "").strip()
        js_text = _extract_json_block(content)
        if not js_text:
            raise ValueError("Empty content or no JSON found in model output.")
        data = json.loads(js_text)
        ps = PatientSummary.model_validate(data)
        ps.patient.last_updated = datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        if patient_id and not ps.patient.patient_id:
            ps.patient.patient_id = patient_id
        return ps
    except (ValidationError, ValueError, json.JSONDecodeError):
        _flash_once(
            "fallback_rawjson",
            "Raw‑JSON parsing failed (empty/malformed). Switching to heuristic summary…",
            color_key="warning",
        )
        _flash_once(
            "fallback_tip",
            "Tip: For stricter JSON, try `llama3.1:8b-instruct` or `qwen2.5:7b-instruct` with temperature 0.0–0.2.",
            color_key="muted",
            duration=1.1,
        )

    # 3) Heuristic summary
    return _heuristic_summary(docs, patient_id)


# =============================================================================
# Rendering (Rich) + JSON writer
# =============================================================================

def render_summary_rich(ps: PatientSummary) -> None:
    """Pretty print a summary in the terminal using MOSAICX colors."""
    pid = ps.patient.patient_id or "Unknown"
    header_lines: List[str] = []
    if ps.patient.dob:
        header_lines.append(f"DOB: {ps.patient.dob}")
    if ps.patient.sex:
        header_lines.append(f"Sex: {ps.patient.sex}")
    if ps.patient.last_updated:
        header_lines.append(f"Updated: {ps.patient.last_updated}")
    header_body = "\n".join(header_lines) if header_lines else "No demographics available."

    header_panel = Panel.fit(
        header_body,
        title=f"[bold {MOSAICX_COLORS['primary']}]Patient: {pid}[/bold {MOSAICX_COLORS['primary']}]",
        border_style=MOSAICX_COLORS["accent"],
    )

    table = Table(
        show_lines=False,
        border_style=MOSAICX_COLORS["secondary"],
        header_style=f"bold {MOSAICX_COLORS['primary']}",
        expand=True,
    )
    table.add_column("Date", style=MOSAICX_COLORS["info"], no_wrap=True)
    table.add_column("Source", style=MOSAICX_COLORS["muted"], no_wrap=True)
    table.add_column("Critical Note", style=MOSAICX_COLORS["accent"])

    for ev in sorted(ps.timeline, key=lambda e: (e.date or "", e.source or "")):
        table.add_row(ev.date or "[dim]—[/dim]", (ev.source or "—")[:40], ev.note)

    overall_panel = Panel(
        ps.overall.strip(),
        title=f"[bold {MOSAICX_COLORS['primary']}]Overall Summary[/bold {MOSAICX_COLORS['primary']}]",
        border_style=MOSAICX_COLORS["accent"],
        padding=(1, 2),
    )

    console.print(Align.center(header_panel))
    console.print()
    console.print(Align.center(table))
    console.print()
    console.print(Align.center(overall_panel))


def write_summary_json(ps: PatientSummary, json_path: Path) -> None:
    """Write the PatientSummary to a JSON file (UTF-8, pretty)."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    data = ps.model_dump(mode="json")      # pydantic → plain python types
    text = json.dumps(data, indent=2, ensure_ascii=False)
    json_path.write_text(text + "\n", encoding="utf-8")


def write_summary_pdf(
    ps: PatientSummary,
    pdf_path: Path,
    *,
    title: str = "MOSAICX: Longitudinal Report",
    logo_path: Optional[Path] = None,
    model_name: Optional[str] = None,
) -> None:
    """Render a stylised PDF artifact for the provided :class:`PatientSummary`."""

    required = (
        SimpleDocTemplate,
        Paragraph,
        RLTable,
        TableStyle,
        getSampleStyleSheet,
        A4,
        cm,
        rl_colors,
        Image,
        Indenter,
    )
    if any(dep is None for dep in required):
        raise RuntimeError("PDF export unavailable – install ReportLab (`pip install reportlab`).")

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2 * cm,
    )

    def _sanitize(text: Optional[str]) -> str:
        """Normalize narrative text for PDF rendering (convert uncommon separators)."""
        if not text:
            return ""
        replacements = {
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
            "\u00a0": " ",
        }
        normalised = text
        for src, dest in replacements.items():
            normalised = normalised.replace(src, dest)
        return normalised

    accent_hex = "#1b1b1d"
    muted_hex = "#6a6a6d"
    text_hex = "#1a1a1c"

    primary = rl_colors.HexColor("#0f0f10")  # type: ignore[call-arg]
    accent = rl_colors.HexColor(accent_hex)  # type: ignore[call-arg]
    muted = rl_colors.HexColor(muted_hex)  # type: ignore[call-arg]
    surface_header = rl_colors.HexColor("#ffffff")  # type: ignore[call-arg]
    surface = rl_colors.HexColor("#f6f6f7")  # type: ignore[call-arg]
    surface_alt = rl_colors.HexColor("#ffffff")  # type: ignore[call-arg]
    grid_color = rl_colors.HexColor("#d7d7da")  # type: ignore[call-arg]
    text_dark = rl_colors.HexColor(text_hex)  # type: ignore[call-arg]

    class RoundedTable(RLTable):  # type: ignore[misc]
        """Table with a rounded rectangle silhouette for a softer card aesthetic."""

        def __init__(
            self,
            *args,
            corner_radius: float = 6.0,
            border_color=accent,
            border_width: float = 0.8,
            fill_color=surface_header,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._corner_radius = corner_radius
            self._border_color = border_color
            self._border_width = border_width
            self._fill_color = fill_color

        def draw(self) -> None:  # pragma: no cover - visual rendering
            canvas = self.canv
            canvas.saveState()
            stroke = 0
            if self._border_color is not None and self._border_width > 0:
                canvas.setStrokeColor(self._border_color)
                canvas.setLineWidth(self._border_width)
                stroke = 1
            if self._fill_color is not None:
                canvas.setFillColor(self._fill_color)
                fill = 1
            else:
                fill = 0
            canvas.roundRect(0, 0, self._width, self._height, self._corner_radius, stroke=stroke, fill=fill)
            canvas.restoreState()
            super().draw()

    stylesheet = getSampleStyleSheet()  # type: ignore[operator]
    stylesheet.add(
        ParagraphStyle(
            name="SummaryBody",
            parent=stylesheet["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=text_dark,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummarySection",
            parent=stylesheet["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=18,
            textColor=primary,
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummaryHeaderTitle",
            parent=stylesheet["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=primary,
            alignment=2,  # right
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummaryHeaderSubtitle",
            parent=stylesheet["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=12,
            textColor=muted,
            alignment=2,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummaryHeaderMeta",
            parent=stylesheet["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=muted,
            alignment=2,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummaryHeaderNote",
            parent=stylesheet["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=muted,
            alignment=2,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummaryCardHeading",
            parent=stylesheet["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=text_dark,
            spaceBefore=0,
            spaceAfter=4,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummaryLabel",
            parent=stylesheet["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=accent,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummaryValue",
            parent=stylesheet["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=text_dark,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummaryOverall",
            parent=stylesheet["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=text_dark,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummaryFooter",
            parent=stylesheet["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=muted,
            alignment=1,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummaryTableHeader",
            parent=stylesheet["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=rl_colors.HexColor("#ffffff"),  # type: ignore[call-arg]
            alignment=1,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="SummaryTableCell",
            parent=stylesheet["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=text_dark,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="TimelineDate",
            parent=stylesheet["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=text_dark,
        )
    )
    stylesheet.add(
        ParagraphStyle(
            name="TimelineNote",
            parent=stylesheet["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=text_dark,
        )
    )
    body_style = stylesheet["SummaryBody"]
    section_style = stylesheet["SummarySection"]
    header_title_style = stylesheet["SummaryHeaderTitle"]
    header_meta_style = stylesheet["SummaryHeaderMeta"]
    header_note_style = stylesheet["SummaryHeaderNote"]
    label_style = stylesheet["SummaryLabel"]
    value_style = stylesheet["SummaryValue"]
    overall_style = stylesheet["SummaryOverall"]
    card_heading_style = stylesheet["SummaryCardHeading"]
    footer_style = stylesheet["SummaryFooter"]
    table_header_style = stylesheet["SummaryTableHeader"]
    table_cell_style = stylesheet["SummaryTableCell"]
    timeline_date_style = stylesheet["TimelineDate"]
    timeline_note_style = stylesheet["TimelineNote"]

    section_indent = 0.55 * cm
    content_width = doc.width - section_indent

    def _rounded_heading(label: str) -> RoundedTable:
        """Rounded rectangle heading that introduces a subsection."""
        heading_para = Paragraph(escape(label.upper()), card_heading_style)
        heading_table = RoundedTable(
            [[heading_para]],
            colWidths=[content_width],
            corner_radius=0.3 * cm,
            border_color=grid_color,
            border_width=0.8,
            fill_color=surface_header,
        )
        heading_table.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        return heading_table

    story: List[Any] = []

    logo_candidates: List[Path] = []
    if logo_path is not None:
        logo_candidates.append(Path(logo_path))

    package_logo = PROJECT_ROOT / "assets" / "digitx_logo.png"
    relative_logo = Path("assets") / "digitx_logo.png"
    package_alt = Path(__file__).resolve().parent / "assets" / "digitx_logo.png"

    for candidate in (package_logo, relative_logo, package_alt):
        if candidate not in logo_candidates:
            logo_candidates.append(candidate)

    discovered_logo: Optional[Path] = None
    for candidate in logo_candidates:
        if candidate.exists():
            discovered_logo = candidate
            break

    if discovered_logo is not None:
        logo_flowable = Image(str(discovered_logo))  # type: ignore[call-arg]
        max_width = 4.2 * cm
        if getattr(logo_flowable, "imageWidth", max_width) > max_width:
            scale = max_width / float(logo_flowable.imageWidth)
            logo_flowable.drawWidth = max_width
            logo_flowable.drawHeight = float(logo_flowable.imageHeight) * scale
        logo_flowable.hAlign = "LEFT"
    else:
        missing_paths = ", ".join(str(p) for p in logo_candidates)
        _flash_once(
            "pdf_logo_missing",
            f"PDF header logo not found (checked: {missing_paths}). Using text fallback.",
            color_key="muted",
            duration=0.6,
        )
        logo_flowable = Paragraph("DIGIT-X Lab", section_style)

    generated_on_display = _sanitize(ps.patient.last_updated) if ps.patient.last_updated else "—"
    model_display = _sanitize(model_name) if model_name else "—"

    header_title = Paragraph(escape(title), header_title_style)
    meta_html = "<br/>".join(
        [
            escape(f"Generated by: MOSAICX v{APPLICATION_VERSION}"),
            escape(f"Created on: {generated_on_display}"),
            escape(f"Model: {model_display}"),
            escape("DIGIT-X Lab, LMU Radiology, LMU University Munich"),
        ]
    )
    header_meta = Paragraph(meta_html, header_meta_style)
    header_note = Paragraph(escape("Note: AI generated"), header_note_style)

    header_rows: List[List[Any]] = [
        [logo_flowable, header_title],
        ["", header_meta],
        ["", header_note],
    ]

    header_table = RoundedTable(
        header_rows,
        colWidths=[doc.width * 0.32, doc.width * 0.68],
        corner_radius=0.35 * cm,
        border_color=accent,
        border_width=0.8,
        fill_color=surface_header,
    )
    header_last_row = len(header_rows) - 1
    header_table.setStyle(
        TableStyle(
            [
                ("SPAN", (0, 0), (0, header_last_row)),
                ("VALIGN", (0, 0), (0, header_last_row), "MIDDLE"),
                ("ALIGN", (1, 0), (1, header_last_row), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (1, 1), (1, 1), 6),
                ("BOTTOMPADDING", (1, 1), (1, 1), 4),
                ("TOPPADDING", (1, 2), (1, 2), 2),
                ("BOTTOMPADDING", (1, 2), (1, 2), 2),
            ]
        )
    )
    story.append(header_table)

    accent_bar = RLTable([[""]], colWidths=[doc.width])
    accent_bar.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), surface_alt),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    accent_bar._argH = [0.1 * cm]  # type: ignore[attr-defined]
    story.append(accent_bar)

    story.append(Spacer(1, 0.35 * cm))

    overall_text = _sanitize(ps.overall.strip()) if ps.overall else "Narrative summary unavailable."

    story.append(Indenter(section_indent, 0))

    snapshot_html = []
    snapshot_html.append(
        "<font color='{muted}' size='8'>PATIENT ID</font><br/><font size='11'>{value}</font>".format(
            muted=muted_hex,
            value=escape(_sanitize(ps.patient.patient_id) or "Not provided"),
        )
    )
    snapshot_html.append(
        "<font color='{muted}' size='8'>DATE OF BIRTH</font><br/><font size='11'>{value}</font>".format(
            muted=muted_hex,
            value=escape(_sanitize(ps.patient.dob) or "Not provided"),
        )
    )
    snapshot_html.append(
        "<font color='{muted}' size='8'>SEX</font><br/><font size='11'>{value}</font>".format(
            muted=muted_hex,
            value=escape(_sanitize(ps.patient.sex) or "Not provided"),
        )
    )
    snapshot_html.append(
        "<font color='{muted}' size='8'>CRITICAL EVENTS</font><br/><font size='11'>{value}</font>".format(
            muted=muted_hex,
            value=str(len(ps.timeline)),
        )
    )

    snapshot_cols = [Paragraph(item, value_style) for item in snapshot_html]
    col_width = content_width / max(len(snapshot_cols), 1)
    snapshot_table = RLTable([snapshot_cols], colWidths=[col_width] * len(snapshot_cols))
    snapshot_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), surface_alt),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(snapshot_table)
    story.append(Indenter(-section_indent, 0))

    story.append(Spacer(1, 0.4 * cm))
    story.append(_rounded_heading("Timeline"))
    story.append(Spacer(1, 0.18 * cm))
    story.append(Indenter(section_indent, 0))

    events = sorted(ps.timeline, key=lambda e: (e.date or "", e.source or ""))

    timeline_rows: List[List[Paragraph]] = []
    if events:
        for ev in events:
            date_text = _sanitize(ev.date) or "—"
            source_text = _sanitize(ev.source)
            note_text = _sanitize(ev.note)

            date_html = escape(date_text)
            if source_text:
                date_html += f"<br/><font color='{muted_hex}' size='8'>{escape(source_text)}</font>"
            note_html = escape(note_text).replace("\n", "<br/>")

            timeline_rows.append(
                [
                    Paragraph(date_html, timeline_date_style),
                    Paragraph(note_html, timeline_note_style),
                ]
            )
    else:
        timeline_rows.append(
            [
                Paragraph("—", timeline_date_style),
                Paragraph("No critical events detected.", timeline_note_style),
            ]
        )

    timeline_table = RLTable(
        timeline_rows,
        colWidths=[2.9 * cm, content_width - 2.9 * cm],
    )

    timeline_style: List[tuple[Any, ...]] = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (0, -1), 10),
        ("LEFTPADDING", (1, 0), (1, -1), 6),
        ("RIGHTPADDING", (1, 0), (1, -1), 6),
    ]
    if len(timeline_rows) > 1:
        timeline_style.append(("LINEBELOW", (0, 0), (-1, -2), 0.25, grid_color))
    timeline_table.setStyle(TableStyle(timeline_style))
    story.append(timeline_table)
    story.append(Indenter(-section_indent, 0))

    story.append(Spacer(1, 0.4 * cm))
    story.append(_rounded_heading("Overall Summary"))
    story.append(Spacer(1, 0.18 * cm))
    story.append(Indenter(section_indent, 0))

    overall_para = Paragraph(escape(overall_text).replace("\n", "<br/><br/>"), overall_style)
    overall_wrapper = RLTable(
        [[overall_para]],
        colWidths=[content_width],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), surface_header),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        ),
    )
    story.append(overall_wrapper)
    story.append(Indenter(-section_indent, 0))

    story.append(Spacer(1, 0.5 * cm))

    footer_text = (
        "Generated with MOSAICX • DIGIT-X Lab, LMU Radiology | LMU University Hospital"
        f" • Summary timestamp: {generated_on_display}"
    )

    def _draw_footer(canvas, doc_obj) -> None:
        canvas.saveState()
        footer_margin = 1.2 * cm
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawString(doc_obj.leftMargin, footer_margin, footer_text)
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)


def save_summary_artifacts(
    ps: PatientSummary,
    *,
    artifacts: Iterable[str],
    json_path: Optional[Path],
    pdf_path: Optional[Path],
    patient_id: Optional[str],
    model_name: Optional[str] = None,
    emit_messages: bool = True,
) -> Dict[str, Path]:
    """Persist selected summary artifacts and return their filesystem paths."""

    selected = tuple(dict.fromkeys(a.lower() for a in artifacts if a))
    if not selected:
        return {}

    allowed = {"json", "pdf"}
    invalid = set(selected) - allowed
    if invalid:
        raise ValueError(f"Unsupported artifact type(s): {', '.join(sorted(invalid))}.")

    saved: Dict[str, Path] = {}
    default_stem: Optional[Path] = None

    def _default_stem() -> Path:
        nonlocal default_stem
        if default_stem is None:
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            base = (patient_id or ps.patient.patient_id or "patient").lower()
            default_stem = Path("output") / f"summary_{base}_{ts}"
        return default_stem

    derived_pdf: Optional[Path] = None

    if "json" in selected:
        target = Path(json_path) if json_path is not None else _default_stem().with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_summary_json(ps, target)
        saved["json"] = target
        if emit_messages:
            styled_message(f"Saved JSON: {target}", "accent")
        if "pdf" in selected and pdf_path is None:
            derived_pdf = target.with_suffix(".pdf")

    if "pdf" in selected:
        if pdf_path is not None:
            pdf_target = Path(pdf_path)
        elif derived_pdf is not None:
            pdf_target = derived_pdf
        else:
            pdf_target = _default_stem().with_suffix(".pdf")
        try:
            write_summary_pdf(ps, pdf_target, model_name=model_name)
        except RuntimeError as exc:
            if emit_messages:
                styled_message(str(exc), "muted")
        except Exception as exc:  # pragma: no cover - runtime safeguard
            if emit_messages:
                styled_message(f"PDF export failed: {exc}", "warning")
        else:
            saved["pdf"] = pdf_target
            if emit_messages:
                styled_message(f"Saved PDF: {pdf_target}", "accent")

    return saved


# =============================================================================
# Public API
# =============================================================================

def summarize_reports_to_terminal_and_json(
    paths: List[Path],
    *,
    patient_id: Optional[str],
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    temperature: float = 0.2,
    json_out: Optional[Path] = None,
    artifacts: Optional[Iterable[str]] = None,
    pdf_out: Optional[Path] = None,
) -> PatientSummary:
    """
    Top-level: load, summarize, render (terminal) + optionally write artifacts.
    - If json_out is None and JSON requested, auto-names into ./output/summary_<patient>_<ts>.json
    - If PDF requested without explicit path, mirrors the JSON stem or falls back to the same naming scheme.
    - Returns the PatientSummary object.
    """
    docs = load_reports(paths)
    if not docs:
        supported = ", ".join(sorted(DOC_SUFFIXES.keys()))
        raise ValueError(f"No textual content found in the provided inputs (supported: {supported}).")

    ps = summarize_with_llm(
        docs,
        patient_id=patient_id,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )

    # Terminal view
    render_summary_rich(ps)

    requested = artifacts if artifacts is not None else ("json",)
    save_summary_artifacts(
        ps,
        artifacts=requested,
        json_path=json_out,
        pdf_path=pdf_out,
        patient_id=patient_id,
        model_name=model,
        emit_messages=True,
    )

    return ps
