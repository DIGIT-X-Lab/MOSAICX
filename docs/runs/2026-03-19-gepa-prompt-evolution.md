# GEPA Prompt Evolution — ProstataPatient

> Automatic prompt optimization via DSPy GEPA on CORE (gpt-oss:120b)
> Training set: 20 synthetic German prostate pathology reports
> Total proposals: 4 | Last updated: 18:55:46

---

## Iteration 0 — Baseline

**Score:** 0.8612740943608522

```
Extract structured data matching the ProstataPatient schema from the document.
```

---

## Iteration 1 — extract_custom.predict

**Score:** 1.834625850340136

````
**Task Overview**
You must read a German‑language pathology report (usually a prostate biopsy or radical prostatectomy report) and extract every piece of information required by the **ProstataPatient** JSON schema.  
The output must contain two parts:

1. **reasoning** – a concise, human‑readable explanation of how you derived each field, including any calculations or assumptions you made.  
2. **extracted** – a literal Python‑style representation of the JSON object that conforms exactly to the schema (field names, enum values, and nesting must match the schema definitions).

**General Extraction Rules**

| Item | Source in the document | How to map / transform |
|------|------------------------|------------------------|
| `patientenid` | The **Journalnummer** (e.g. `E/2026/006123`). Use the whole string. |
| `einsendenummer` | Same as `patientenid`. |
| `ort` | City name that appears in the header (e.g. `München`). |
| `pathologisches_institut` | The institute name line that starts with “Pathologisches Institut …”. |
| `histologiedatum` | Prefer the **Eingang** date (specimen receipt). If missing, use the **Ausgang** date. Keep the format `DD.MM.YYYY`. |
| `befundausgang` | The **Ausgang** date (report finalisation). If only one date is present, use it for both `histologiedatum` and `befundausgang`. |
| `massgeblicher_befund` | Always set to enum `Ja` (the report is the definitive pathology report). |
| `tumornachweis` | `Ja` if any core/section is reported as tumor positive, otherwise `Nein`. |
| `icdo3_histologie` | Value after “ICD‑O‑3:” (e.g. `8140/3`). |
| `lokalisation_icd10` | Value after “ICD‑10‑GM‑2024:” (e.g. `C61`). |
| `who_isup_grading` | Enum value that matches the “WHO‑/ISUP‑Graduierungsgruppe …” text. Use the exact wording from the enum (e.g. `Graduierungsgruppe 1`). |
| `gleason` | The **highest** Gleason score that appears anywhere in the report (compare numeric sums; if equal, prefer the later listed). Use the exact string as written (e.g. `4 + 3 = 7b`). |
| `untersuchtes_praeparat` | `Biopsie` for needle‑core biopsies, `Resektat` for radical prostatectomy or any surgical specimen. |
| `art_der_biopsie` | If `untersuchtes_praeparat` = `Biopsie`, set to `Stanzbiopsie` when the clinical note contains “Prostatastanzbiopsien”. Otherwise leave `null`. |
| `entnahmestelle_der_biopsie` | For biopsies, default to `Primärtumor` unless the report explicitly states another site. Leave `null` for resections. |
| `makroskopie_liste` | Build a list of `ProstataPatientitemItem` objects, one per **Makroskopie** entry.  
  - `nr` = the order number shown.  
  - `gesamt_stanzen` = number of cores in that entry. Usually `1`; if the text says “Zwei …” or “2 …” set to `2`.  
  - `laenge_cm` = the measured length in centimeters (numeric, keep one decimal). If the entry says “bis X cm”, use `X`. |
| `begutachtung_liste` | One `ProstataPatientitemItem` per core (same ordering as `makroskopie_liste`).  
  - `nr` = core number.  
  - `tumor` = `True` if the microscopic description for that core mentions carcinoma (e.g. “mit herdförmigen Infiltraten”). Otherwise `False`.  
  - `tumorausdehnung_mm` = tumor length in millimetres as given after “Längsausdehnung …”. If missing, `null`.  
  - `tumor_prozent` = `(tumorausdehnung_mm / (laenge_cm * 10)) * 100`. Keep the full float; do **not** round here.  
  - `gleason` = Gleason string for that core if tumor‑positive, otherwise `null`. |
| `anzahl_entnommener_stanzen` | Sum of all `gesamt_stanzen` across `makroskopie_liste`. |
| `anzahl_befallener_stanzen` | Count of cores where `tumor` = `True`. |
| `maximaler_anteil_befallener_stanzen` | The **largest** `tumor_prozent` among all positive cores, **rounded to the nearest whole percent** (use standard rounding). If no positive cores, set `null`. |
| `calculation_details` | A list of strings, one per tumor‑positive core, formatted exactly as:  
  `"Sample {nr}: {tumorausdehnung_mm} mm / {laenge_cm*10:.0f} mm = {tumor_prozent:.2f}%"`  
  (use the core number from `nr`). |
| `tnm_nach` | If a “pTNM (… )” line exists, extract the source (e.g. `UICC`). Otherwise `null`. |
| `ptnm` | The full TNM stage string after the source (e.g. `pT2c pN0`). |
| `lymphgefaessinvasion` | Enum value from the line “Lymphgefäßinvasion: …” (`L0`, `L1`, …). If not present, `null`. |
| `veneninvasion` | Enum value from “Venöse Invasion: …” (`V0`, `V1`, …). |
| `perineurale_invasion` | Enum value from “Perineurale Invasion: …” (`Pn0`, `Pn1`, …). |
| `lk_befallen_untersucht` | If the report lists lymph‑node counts like “0/12”, store the exact string (`"0/12"`). |
| `r_klassifikation` | Enum from the resection margin line (e.g. `R0`, `R1`). |
| `resektionsrand` | If a separate comment about the resection margin is given, copy the text; otherwise `null`. |

**Special Handling Notes**

* **Multiple cores in one macro entry** – When the macro description says “Zwei …” or “2 …”, treat `gesamt_stanzen` as `2`. The length given applies to each core individually.
* **Missing values** – Any field not explicitly present in the report must be set to `null` (or omitted if the schema allows omission). Do **not** fabricate data.
* **Date formats** – Preserve the German format `DD.MM.YYYY`. Do not convert to ISO.
* **Enum values** – Use the exact enum member names defined in the schema (e.g. `ProstatapatientitemMassgeblicherBefund.Ja`). Do not quote the enum name; represent it as the enum object as shown in the examples.
* **Rounding** – Only the field `maximaler_anteil_befallener_stanzen` is rounded. All intermediate percentages (`tumor_prozent`) stay as full floats.
* **Overall Gleason** – When several Gleason scores appear, compare the numeric sum (e.g., 3+4=7, 4+3=7). If sums are equal, choose the one with the higher primary pattern (the first number). If still equal, pick the later‑listed core.
* **Biopsy vs Resection** – For radical prostatectomy reports, set `untersuchtes_praeparat = Resektat`, leave `art_der_biopsie` and `entnahmestelle_der_biopsie` as `null`. For needle biopsies, set them as described above.
* **Tumor detection** – If the report contains any phrase indicating carcinoma (e.g., “azinäres Adenokarzinom”, “Infiltrate”, “Tumor”), treat `tumornachweis = Ja`. Otherwise `Nein`.
* **Calculation details** – Include **all** tumor‑positive cores, even if the percentage is very low.

**Output Formatting**

Your final answer must be a single code block containing:

```text
### reasoning
<your step‑by‑step explanation>

### extracted
patientenid='...' befunde=[ProstatapatientItem(...), ...]
````

---

## Iteration 5 — extract_custom.predict

**Score:** 0.8612740943608522

````
**Task Overview**
You must read a German‑language pathology report (usually a prostate‑specific report) and extract every piece of information required by the **ProstataPatient** data schema. The output must contain two parts:

1. **reasoning** – a concise, human‑readable explanation of how you derived each field.
2. **extracted** – a literal Python‑style representation of the populated `ProstataPatient` object (or list of objects if the document contains multiple reports). Use the exact field names and enum values shown in the examples.

**Document Structure (Typical)**
- Header with institution, contact data, and **Journalnummer** (submission number).
- Patient block: `Name`, `Vorname`, `Geboren am`, `Eingang` (date material received), `Ausgang` (date report finalized).  
- Section titles (all German, may appear in any order, often separated by blank lines):
  - `Übersandtes Material:` – list of specimens (e.g., “1. Rechtes AFS medial”, “1. Radikale Prostatektomie …”).
  - `Klinische Angaben:` – clinical context, may contain PSA, prior Gleason, biopsy type, etc.
  - `Makroskopie:` – description of each specimen/core. For biopsies each line is “Ein X cm messender Stanzzylinder”. For resections it may give overall size, weight, number of blocks, etc.
  - `Mikroskopie:` – microscopic technique list (e.g., “12x HE”) and any immunohistochemistry.
  - `Begutachtung:` – narrative of findings, often followed by a concise list of tumor‑positive cores with Gleason, WHO‑grading, and tumor extension in **mm**.
  - `Kommentar:` – optional free text.
  - `Klassifikation:` – ICD‑10, ICD‑O‑3, WHO/ISUP grading, Gleason score, pTNM or TNM classification, resection‑margin status, etc.
- Optional footers or page breaks.

**General Extraction Rules**

| Schema Field | Source / Derivation | Notes |
|--------------|---------------------|-------|
| `patientenid` | Use the **Journalnummer** (e.g., `E/2026/011890`). If the report lacks a journal number, set to the literal string `"unknown"`. |
| `einsendenummer` | Same as `patientenid`. |
| `ort` | City name from the header (e.g., “München”). If not present, set to `None`. |
| `pathologisches_institut` | Full institution line(s) at the top of the document (e.g., “Pathologisches Institut der LMU”). If missing, `None`. |
| `histologiedatum` | The **Eingang** date (date the material arrived). Use the exact string as in the report (DD.MM.YYYY). |
| `befundausgang` | The **Ausgang** date (date the report was finalized). |
| `massgeblicher_befund` | Always `Ja` (the report is the definitive pathology result). |
| `untersuchtes_praeparat` | • `"Biopsie"` if the `Übersandtes Material` lists individual cores (e.g., “Rechtes AFS medial”). <br>• `"Resektat"` if a whole organ or surgical specimen is described (e.g., “Radikale Prostatektomie”). |
| `art_der_biopsie` | For biopsies only. Detect the biopsy type from the clinical or material description: <br>‑ `"Stanzbiopsie"` if “Stanz” or “Core” is mentioned. <br>‑ `"TUR‑P"` if TUR‑P is mentioned. <br>‑ `None` for resections. |
| `entnahmestelle_der_biopsie` | For biopsies only. Default to `Primärtumor` unless the report explicitly states a different site (e.g., “Transition zone”). |
| `makroskopie_liste` | Parse each numbered line under **Makroskopie**. For each core: <br>‑ `nr` = the line number. <br>‑ `gesamt_stanzen` = number of cores in that line (usually `1`). <br>‑ `laenge_cm` = length in centimeters (convert German decimal comma to dot). |
| `begutachtung_liste` | For each core that appears in the **Begutachtung** (or narrative) with tumor: <br>‑ `nr` = core number. <br>‑ `tumor` = `True` if any malignant description is present, otherwise `False`. <br>‑ `tumorausdehnung_mm` = explicit tumor length in mm (if given). <br>‑ `tumor_prozent` = `(tumorausdehnung_mm / (laenge_cm*10)) * 100` rounded to one decimal place. <br>‑ `gleason` = Gleason pattern as written (e.g., `"3 + 4 = 7a"`). <br>For cores without tumor, set `tumor=False` and leave the other fields `None`. |
| `anzahl_entnommener_stanzen` | Total number of cores/specimens (`len(makroskopie_liste)`). |
| `anzahl_befallener_stanzen` | Count of entries in `begutachtung_liste` where `tumor=True`. |
| `maximaler_anteil_befallener_stanzen` | Highest `tumor_prozent` among tumor‑positive cores, rounded to the nearest integer. If no tumor, set to `None`. |
| `calculation_details` | List of strings describing each percentage calculation, e.g., `"Core 5: 3.5 mm / 17 mm * 100 = 20.6 %"`. Include one entry per tumor‑positive core. If no tumor, set to `None`. |
| `icdo3_histologie` | Value after “ICD‑O‑3:” in **Klassifikation** (e.g., `8140/3`). |
| `who_isup_grading` | Enum value after “WHO/ISUP‑Graduierungsgruppe” or “WHO‑Grading‑Group”. Map the textual group to the enum (e.g., `Graduierungsgruppe 3`). |
| `gleason` | Overall Gleason score – use the highest‑grade pattern reported in **Begutachtung** or **Klassifikation** (e.g., `"4 + 3 = 7b"`). |
| `tnm_nach` | The TNM system used (e.g., `"UICC"`). |
| `ptnm` | Full pTNM string from **Klassifikation** (e.g., `"pT3b pN1 L1 V0 Pn1 R1"`). |
| `lymphgefaessinvasion` | Enum based on “L1”, “L0”, etc. If not mentioned, `None`. |
| `veneninvasion` | Enum based on “V1”, “V0”, etc. |
| `perineurale_invasion` | Enum based on “Pn1”, “Pn0”, etc. |
| `lk_befallen_untersucht` | Lymph‑node involvement expressed as “x/y” (e.g., `"2/22"`). |
| `r_klassifikation` | Enum for resection‑margin status (`R0`, `R1`, etc.). |
| `resektionsrand` | Enum for margin description (`fokal`, `diffus`, etc.). |
| `lokalisation_icd10` | ICD‑10 code for the anatomical site if present (e.g., `C61`). If the report states “keine Kodierung (kein Malignom)”, set to `None`. |
| `tumornachweis` | `Ja` if any core has `tumor=True`; otherwise `Nein`. |

**Parsing Strategy (to be followed for every document)**
1. **Normalize Text** – Replace line‑break variations, ensure consistent spacing, and convert German decimal commas to periods for numeric conversion.
2. **Section Detection** – Locate headings (`Übersandtes Material:`, `Klinische Angaben:`, `Makroskopie:`, `Mikroskopie:`, `Begutachtung:`, `Klassifikation:`). The order may vary; treat each heading as a delimiter.
3. **Extract Header Information** – Scan the top of the file for `Journalnummer`, city name, and institution lines.
4. **Parse Dates** – Capture `Eingang` and `Ausgang` values; they are always in `DD.MM.YYYY` format.
5. **Material List** – Determine whether the material list describes individual cores (biopsy) or a surgical specimen (resection). This decides `untersuchtes_praeparat`.
6. **Makroskopie** – For each numbered line, extract the core number and length. If the line contains “Stanzzylinder”, set `gesamt_stanzen = 1`. For resections, treat the whole organ as a single entry (`nr=1`).
7. **Begutachtung / Tumor Findings** –  
   - Search for patterns like `Gleason X + Y = Z`, `WHO‑Grading‑Group`, `Länge … mm`.  
   - When a core number is mentioned (e.g., “5.” or “Core 5”), associate the findings with that core.  
   - If the narrative states “ohne Nachweis eines Malignoms” for all cores, set all `tumor=False`.
8. **Compute Percentages** – Convert core length to mm (`laenge_cm * 10`). Use the explicit tumor extension (mm) to compute percentage. Round to one decimal place for the field, but keep the full value in `calculation_details`.
9. **Aggregate Statistics** – Count cores, tumor‑positive cores, and derive the maximal percentage.
10. **Classification Section** – Extract ICD‑10, ICD‑O‑3, WHO/ISUP group, Gleason, pTNM, and any invasion or margin descriptors. Map each to the appropriate enum/value.
11. **Finalize Fields** – Populate any missing optional fields with `None`. Ensure enum fields use the exact enum member names shown in the examples (e.g., `ProstatapatientitemMassgeblicherBefund.Ja`).
12. **Output Formatting** –  
    - First line: `### reasoning` followed by the explanatory paragraph(s).  
    - Second line: `### extracted` followed by the exact Python‑style object literal.  
    - Preserve the ordering of fields as in the examples.

**Edge Cases & Common Pitfalls**
- **Missing Journalnummer** – Use `"unknown"` for `patientenid` and `einsendenummer`.
- **Decimal Comma** – German numbers use commas (e.g., `1,8 cm`). Convert to float (`1.8`).
- **Multiple Gleason Scores** – Choose the highest‑grade (largest primary + secondary) for the overall `gleason` field.
- **No Tumor** – Set `tumornachweis` to `Nein`, `anzahl_befallener_stanzen` to `0`, and leave percentage‑related fields `None`.
- **Inconsistent Core Numbering** – Core numbers may appear as Arabic numerals (`1.`) or spelled out (`Core 1`). Use regex to capture any number preceding a period or after the word “Core”.
- **Redundant Information** – If the same data appears in both **Begutachtung** and **Klassifikation**, prefer the more detailed source (usually **Begutachtung** for per‑core data, **Klassifikation** for overall staging).
- **Multiple Reports in One File** – If the document clearly contains separate reports (e.g., two distinct `Journalnummer` blocks), create a list of `ProstataPatient` objects, each with its own `befunde` list containing a single `ProstatapatientItem`.

**Final Deliverable**
Your response must contain **only** the two sections (`reasoning` and `extracted`) as shown in the examples, with no additional commentary or markup. Ensure the Python‑style literals are syntactically correct and that enum members are fully qualified (e.g., `<ProstatapatientitemUntersuchtesPraeparat.Biopsie: 'Biopsie'>`).
````

---

## Iteration 9 — extract_custom.predict

**Score:** 2.233296888365879

````
**Task Overview**
You are given the full text of a German‑language prostate pathology report.  
Your job is to parse the document and produce a single JSON object that conforms exactly to the **ProstataPatient** schema (see below). The JSON must contain all required fields, use the correct enum values, and set any unavailable optional fields to `null`. Do **not** add any explanatory text, comments, or extra keys.

**ProstataPatient schema (simplified)**
```
{
  "patientenid": string,                     # journal number (e.g. "E/2026/015678")
  "befunde": [
    {
      "untersuchtes_praeparat": enum,        # "Biopsie" | "Resektat"
      "histologiedatum": string,             # date DD.MM.YYYY (prefer "Ausgang", fallback "Eingang")
      "befundausgang": string | null,        # same as histologiedatum for biopsies, otherwise report date
      "ort": string,                         # city (e.g. "München")
      "pathologisches_institut": string,     # institute name
      "einsendenummer": string,              # same as patientenid
      "massgeblicher_befund": enum,          # always "Ja"
      "tumornachweis": enum,                 # "Ja" if any core positive, else "Nein"
      "icdo3_histologie": string,            # e.g. "8140/3"
      "who_isup_grading": enum | null,       # Graduierungsgruppe 1‑5
      "gleason": string | null,              # highest Gleason (e.g. "4 + 3 = 7b")
      "makroskopie_liste": [                 # one entry per core/specimen
        {
          "nr": int,
          "gesamt_stanzen": int,             # usually 1 for biopsies
          "laenge_cm": float
        }, …
      ],
      "begutachtung_liste": [                # one entry per core/specimen
        {
          "nr": int,
          "tumor": bool,
          "tumorausdehnung_mm": float | null,
          "tumor_prozent": float | null,     # calculated, rounded to one decimal if you wish
          "gleason": string | null
        }, …
      ],
      "art_der_biopsie": enum | null,        # "Stanzbiopsie", "Kernbiopsie", etc.
      "entnahmestelle_der_biopsie": enum | null, # "Primärtumor", "Transition", …
      "lokalisation_icd10": string | null,   # e.g. "C61"
      "anzahl_entnommener_stanzen": int,     # total cores/specimens
      "anzahl_befallener_stanzen": int,      # cores with tumor=true
      "maximaler_anteil_befallener_stanzen": float | null, # highest tumor_prozent
      "calculation_details": [string] | null, # optional human‑readable formulas
      "tnm_nach": string | null,            # e.g. "UICC"
      "ptnm": string | null,                # e.g. "pT2c pN0"
      "lymphgefaessinvasion": enum | null,  # "L0", "L1", …
      "veneninvasion": enum | null,         # "V0", "V1", …
      "perineurale_invasion": enum | null,  # "Pn0", "Pn1", …
      "lk_befallen_untersucht": string | null, # e.g. "0/12"
      "r_klassifikation": enum | null,      # "R0", "R1", …
      "resektionsrand": string | null        # free text if given
    }
  ]
}
```

**Step‑by‑step extraction guide**

1. **Identify the journal number**  
   - Look for a line containing “Journalnummer” or a pattern like `E/2026/xxxxx`.  
   - Use this value for both `patientenid` and `einsendenummer`.

2. **Dates**  
   - `histologiedatum` = date after “Ausgang”.  
   - If “Ausgang” is missing, use the date after “Eingang”.  
   - `befundausgang` = same as `histologiedatum` for biopsies; for resektate use the “Ausgang” date if present, otherwise `null`.

3. **Institution & location**  
   - The first lines usually contain the institute name (e.g. “Pathologisches Institut der LMU”).  
   - Extract the city (e.g. “München”) from the address lines.

4. **Specimen type (`untersuchtes_praeparat`)**  
   - If the “Übersandtes Material” line mentions “Biopsie”, “Stanzbiopsie”, “Kernbiopsie”, etc. → `Biopsie`.  
   - If it mentions “Prostatektomie”, “Resektat”, “Radikale Prostatektomie” → `Resektat`.

5. **Biopsy‑specific fields** (only when `untersuchtes_praeparat` = `Biopsie`)  
   - `art_der_biopsie` – map the word after “Prostata‑” in the clinical note:
     - “Stanzbiopsie” → `Stanzbiopsie`
     - “Kernbiopsie” → `Kernbiopsie`
     - If not explicit, default to `Stanzbiopsie`.
   - `entnahmestelle_der_biopsie` – look for “Primärtumor”, “Transition”, “Peripherie”, etc. If absent, default to `Primärtumor`.

6. **Macroscopic list (`makroskopie_liste`)**  
   - Find the “Makroskopie:” section. Each line like “1. Ein 1,7 cm messender Stanzzylinder” gives:
     - `nr` = the leading number.
     - `gesamt_stanzen` = usually 1 (unless the line says “2 Stanzzylinder” etc.).
     - `laenge_cm` = the numeric value before “cm”. Use a dot as decimal separator (replace commas).

7. **Microscopic / assessment list (`begutachtung_liste`)**  
   - Locate the “Begutachtung:” block. Each entry follows the pattern:
     ```
     5. Gleason 3 + 4 = 7a, WHO‑Graduierungsgruppe 2, Längsausdehnung 2,0 mm.
     ```
   - For each core number:
     - `tumor` = true if a Gleason/WHO line is present, otherwise false.
     - `tumorausdehnung_mm` = the number before “mm”. Convert commas to dots.
     - `gleason` = the exact Gleason string (e.g. “3 + 4 = 7a”).  
     - `tumor_prozent` = **only for biopsies** and only when both `tumorausdehnung_mm` and the corresponding macroscopic length are known:
       ```
       tumor_prozent = (tumorausdehnung_mm / (laenge_cm * 10)) * 100
````

---

## Iteration 13 — extract_custom.predict

**Score:** 0.8612740943608522

````
# Task Overview
You must read a German‑language prostate pathology report (usually a PDF‑text dump) and produce a **single JSON object** that conforms exactly to the `ProstataPatient` schema described below.  
The report contains a header, a list of submitted specimens, a macroscopic (Makroskopie) description, a microscopic (Mikroskopie/Begutachtung) description, and a classification block. Your output must capture every piece of information required by the schema, compute derived values, and follow the conventions demonstrated in the example responses.

---

## 1. ProstataPatient Schema (exact field names & types)

```
ProstataPatient {
    patientenid: string                     # unique identifier – use the Journalnummer (e.g. "E/2026/010567")
    befunde: [ProstatapatientItem]          # list with exactly ONE item (the current report)
}
```

```
ProstatapatientItem {
    untersuchtes_praeparat: enum[Biopsie, ...]          # always "Biopsie" for these reports
    histologiedatum: string (DD.MM.YYYY)               # date from “Ausgang” (finalisation)
    befundausgang: string (DD.MM.YYYY)                 # same as histologiedatum
    ort: string                                        # city of the pathology institute (e.g. "München")
    pathologisches_institut: string                    # full institute name from the header
    einsendenummer: string                             # same as patientenid (Journalnummer)
    massgeblicher_befund: enum[Ja, Nein]               # always "Ja" for the definitive report
    tumornachweis: enum[Ja, Nein]                     # "Ja" if any core contains tumor, else "Nein"
    icdo3_histologie: string                          # value after “ICD-O-3:” (e.g. "8140/3")
    who_isup_grading: enum[
        Graduierungsgruppe 1,
        Graduierungsgruppe 2,
        Graduierungsgruppe 3,
        Graduierungsgruppe 4,
        Graduierungsgruppe 5,
        Graduierungsgruppe 6,
        Graduierungsgruppe 7,
        Graduierungsgruppe 8,
        Graduierungsgruppe 9,
        Graduierungsgruppe 10,
        Graduierungsgruppe 11,
        Graduierungsgruppe 12,
        Graduierungsgruppe 13,
        Graduierungsgruppe 14,
        Graduierungsgruppe 15,
        Graduierungsgruppe 16,
        Graduierungsgruppe 17,
        Graduierungsgruppe 18,
        Graduierungsgruppe 19,
        Graduierungsgruppe 20,
        Graduierungsgruppe 21,
        Graduierungsgruppe 22,
        Graduierungsgruppe 23,
        Graduierungsgruppe 24,
        Graduierungsgruppe 25,
        Graduierungsgruppe 26,
        Graduierungsgruppe 27,
        Graduierungsgruppe 28,
        Graduierungsgruppe 29,
        Graduierungsgruppe 30,
        Graduierungsgruppe 31,
        Graduierungsgruppe 32,
        Graduierungsgruppe 33,
        Graduierungsgruppe 34,
        Graduierungsgruppe 35,
        Graduierungsgruppe 36,
        Graduierungsgruppe 37,
        Graduierungsgruppe 38,
        Graduierungsgruppe 39,
        Graduierungsgruppe 40,
        Graduierungsgruppe 41,
        Graduierungsgruppe 42,
        Graduierungsgruppe 43,
        Graduierungsgruppe 44,
        Graduierungsgruppe 45,
        Graduierungsgruppe 46,
        Graduierungsgruppe 47,
        Graduierungsgruppe 48,
        Graduierungsgruppe 49,
        Graduierungsgruppe 50,
        Graduierungsgruppe 51,
        Graduierungsgruppe 52,
        Graduierungsgruppe 53,
        Graduierungsgruppe 54,
        Graduierungsgruppe 55,
        Graduierungsgruppe 56,
        Graduierungsgruppe 57,
        Graduierungsgruppe 58,
        Graduierungsgruppe 59,
        Graduierungsgruppe 60,
        Graduierungsgruppe 61,
        Graduierungsgruppe 62,
        Graduierungsgruppe 63,
        Graduierungsgruppe 64,
        Graduierungsgruppe 65,
        Graduierungsgruppe 66,
        Graduierungsgruppe 67,
        Graduierungsgruppe 68,
        Graduierungsgruppe 69,
        Graduierungsgruppe 70,
        Graduierungsgruppe 71,
        Graduierungsgruppe 72,
        Graduierungsgruppe 73,
        Graduierungsgruppe 74,
        Graduierungsgruppe 75,
        Graduierungsgruppe 76,
        Graduierungsgruppe 77,
        Graduierungsgruppe 78,
        Graduierungsgruppe 79,
        Graduierungsgruppe 80,
        Graduierungsgruppe 81,
        Graduierungsgruppe 82,
        Graduierungsgruppe 83,
        Graduierungsgruppe 84,
        Graduierungsgruppe 85,
        Graduierungsgruppe 86,
        Graduierungsgruppe 87,
        Graduierungsgruppe 88,
        Graduierungsgruppe 89,
        Graduierungsgruppe 90,
        Graduierungsgruppe 91,
        Graduierungsgruppe 92,
        Graduierungsgruppe 93,
        Graduierungsgruppe 94,
        Graduierungsgruppe 95,
        Graduierungsgruppe 96,
        Graduierungsgruppe 97,
        Graduierungsgruppe 98,
        Graduierungsgruppe 99,
        Graduierungsgruppe 100
    ]                                                   # highest WHO‑ISUP grade among all tumor‑positive cores
    gleason: string                                     # highest Gleason pattern (e.g. "4 + 5 = 9")
    makroskopie_liste: [MakroskopieItem]                # one entry per line in the Makroskopie section
    begutachtung_liste: [BegutachtungItem]              # one entry per core number (including benign cores)
    art_der_biopsie: enum[
        Fusionsbiopsie,
        Stanzbiopsie,
        Standardbiopsie,
        Andere
    ]                                                   # derived from the clinical text (see below)
    entnahmestelle_der_biopsie: enum[
        Primärtumor,
        Periprostatisch,
        Andere
    ]                                                   # default = Primärtumor for all prostate biopsies
    lokalisation_icd10: string | null                   # optional – not present in examples
    anzahl_entnommener_stanzen: int                     # total number of cores (sum of “Gesamt Stanzen”)
    anzahl_befallener_stanzen: int                      # number of cores with tumor
    maximaler_anteil_befallener_stanzen: int            # max tumor‑percentage among positive cores, rounded to nearest integer
    calculation_details: [string]                       # human‑readable list of the percentage calculations
    tnm_nach: string | null
    ptnm: string | null
    lymphgefaessinvasion: string | null
    veneninvasion: string | null
    perineurale_invasion: string | null
    lk_befallen_untersucht: string | null
    r_klassifikation: string | null
    resektionsrand: string | null
}
```

```
MakroskopieItem {
    nr: int                     # core number as listed in the “Übersandtes Material” section
    gesamt_stanzen: int         # number of cores represented by this line (default 1; if “(nx)” appears, n = number)
    laenge_cm: float            # length in centimeters (the value before “cm”)
}
```

```
BegutachtungItem {
    nr: int                     # core number (matches MakroskopieItem.nr)
    tumor: bool                 # true if this core contains tumor (i.e. a Gleason entry exists)
    tumorausdehnung_mm: float | null   # tumor length in mm from the “Begutachtung” line
    tumor_prozent: float | null         # (tumor length mm) / (core length mm) * 100, rounded to two decimals
    gleason: string | null              # Gleason string from the “Begutachtung” line (e.g. "3 + 4 = 7a")
}
````

---

