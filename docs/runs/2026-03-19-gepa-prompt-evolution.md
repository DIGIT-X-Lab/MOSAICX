# GEPA Prompt Evolution — ProstataPatient

> Automatic prompt optimization via DSPy GEPA on CORE (gpt-oss:120b)
> Training set: 20 synthetic German prostate pathology reports
> Total proposals: 2 | Last updated: 18:02:13

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

