# GEPA Prompt Evolution — ProstataPatient
> Automatic prompt optimization via DSPy GEPA on CORE (gpt-oss:120b)
> Training set: 20 synthetic German prostate pathology reports
> Total proposals: 7

---

## Iteration 0 — Baseline

**Score:** 0.8620964627819049

```
Extract structured data matching the ProstataPatient schema from the document.
```

---

## Iteration 1 — extract_custom.predict

**Score:** 2.5044288060051607

```
# Instruction for extracting a ProstataPatient record from a German pathology report

You are given a single pathology report (plain‑text, German) that follows the typical layout used by the Pathologisches Institut der LMU (and similar German pathology labs).  
Your job is to **extract every piece of information required by the `ProstataPatient` schema** and return a single JSON object that conforms exactly to that schema.

Below is a complete description of the schema, the meaning of each field, how to obtain it from the text, and the rules you must follow when a value is missing, ambiguous, or needs to be derived.

---

## 1. ProstataPatient schema (high‑level)

```
{
  "patientenid": str,                     # journal number, e.g. "E/2026/006123"
  "befunde": [ProstataPatientItem, ...]   # exactly ONE item per report
}
```

### 1.1 ProstataPatientItem (one per report)

| Field | Type | Required? | How to obtain / derive |
|-------|------|-----------|------------------------|
| `untersuchtes_praeparat` | enum `Biopsie` | Yes | If the report describes core‑biopsies (keywords: *Stanzbiopsie*, *Biopsie*, *Stanzzylinder*). Otherwise set to `Resektat` (surgical specimen). |
| `histologiedatum` | date string `DD.MM.YYYY` | Yes | Use the **Ausgang** date (final report date). If missing, fall back to **Eingang**. |
| `befundausgang` | date string `DD.MM.YYYY` | Yes | Same value as `histologiedatum`. |
| `ort` | str | Yes | City of the institute (e.g. “München”). Extract from the address line (`... | 80337 München`). |
| `pathologisches_institut` | str | Yes | Full name from the first line of the report (e.g. “Pathologisches Institut der LMU”). |
| `einsendenummer` | str | Yes | Same as `patientenid`. |
| `massgeblicher_befund` | enum `Ja` | Yes | Always `Ja` for the primary report. |
| `tumornachweis` | enum `Ja` / `Nein` | Yes | `Ja` if **any** core / resection piece shows carcinoma (see “Begutachtung” / “Mikroskopie”). Otherwise `Nein`. |
| `icdo3_histologie` | str | Yes | Value after “ICD‑O‑3:” (e.g. “8140/3”). |
| `who_isup_grading` | enum (Graduierungsgruppe 1‑5) | Optional | Extract the highest WHO/ISUP grade mentioned (e.g. “WHO‑Graduierungsgruppe 2”). If none, leave `null`. |
| `gleason` | str | Optional | Highest Gleason pattern string found in the report (e.g. “4 + 3 = 7b”). If none, `null`. |
| `makroskopie_liste` | list of MakroskopieItem | Optional | One entry per line in the **Makroskopie** section. See section 2. |
| `begutachtung_liste` | list of BegutachtungItem | Optional | One entry per core / resection piece that is described in the **Begutachtung** / **Mikroskopie** section. See section 3. |
| `art_der_biopsie` | enum `Stanzbiopsie` / `Kernbiopsie` / `Nadelbiopsie` | Optional | For biopsies only. If the clinical note contains “Stanzbiopsie”, set to `Stanzbiopsie`. If “Kernbiopsie” or “Nadelbiopsie” appears, use that. If none, leave `null`. |
| `entnahmestelle_der_biopsie` | enum `Primärtumor` / `Periprostatisch` / `Lymphknoten` | O

[... truncated, full prompt ~14184 chars ...]
```

---

## Iteration 5 — extract_custom.predict

**Score:** 0.8620964627819049

```
**Task Overview**

You are given the full text of a German‑language prostate pathology report.  
Your job is to extract every piece of information required by the **ProstataPatient** data model and output a single JSON‑compatible Python‑like representation that can be directly used to instantiate a `ProstataPatient` object.

**Data Model Summary (for reference)**  

```
ProstataPatient
 └─ patientenid: str                     # unique case identifier (journal number)
 └─ befunde: List[ProstataPatientItem]   # usually one item per report
```

`ProstataPatientItem` fields (most are enums; see notes below):

| Field | Meaning | Extraction Rules |
|-------|---------|------------------|
| untersuchtes_praeparat | “Biopsie” or “Resektat” | If the report mentions *Stanzbiopsie*, *Biopsie*, *Re‑Biopsie* → **Biopsie**. If it mentions *Radikale Prostatektomie*, *Prostatektomie*, *Resektat* → **Resektat**. |
| art_der_biopsie | enum `Stanzbiopsie`, `TUR‑P`, … | Only for **Biopsie** reports. Detect the specific biopsy technique (most commonly “Stanzbiopsie”). If not stated, leave `null`. |
| entnahmestelle_der_biopsie | enum `Primärtumor`, `Periprostatikum`, `Lymphknoten` … | For biopsy reports, default to **Primärtumor** unless the text explicitly says otherwise (e.g., “pelvine Lymphknoten”). |
| histologiedatum / befundausgang | Date (dd.mm.yyyy) | Use the **Ausgang** (report finalisation) date for both fields. |
| ort | City | Extract the city from the address line of the pathology institute (e.g., “München”). |
| pathologisches_institut | Full institute name | First line(s) before the address block. |
| einsendenummer | Same as `patientenid` (journal number) | The value after “Journalnummer”. |
| massgeblicher_befund | enum `Ja` / `Nein` | Set to **Ja** for the final report (the one you are processing). |
| tumornachweis | enum `Ja` / `Nein` | **Ja** if any core/section is reported as tumor‑positive; otherwise **Nein**. |
| icdo3_histologie | String | Value after “ICD‑O‑3:” in the *Klassifikation* section. |
| lokalisation_icd10 | String | Value after “ICD‑10‑GM‑2024:” in the *Klassifikation* section. May be missing (e.g., “keine Kodierung”). |
| who_isup_grading | Enum `Graduierungsgruppe 1` … `Graduierungsgruppe 5` | Extract the WHO/ISUP grade from the *Klassifikation* section (e.g., “Graduierungsgruppe 3”). |
| gleason | String (e.g., “4 + 3 = 7b”) | Overall Gleason score = the **highest** Gleason reported in any core or in the summary line. |
| makroskopie_liste | List of items `{nr, gesamt_stanzen, laenge_cm}` | For each numbered line under **Makroskopie**: <br>• `nr` = the line number. <br>• `gesamt_stanzen` = number of cores in that line (usually 1). <br>• `laenge_cm` = the measured length in centimeters (numeric, keep one decimal). |
| begutachtung_liste | List of items `{nr, tumor, tumorausdehnung_mm, tumor_prozent, gleason}` | For each core that is explicitly described in the *Begutachtung* (or *Mikroskopie* when a Gleason is given): <br>• `nr`

[... truncated, full prompt ~9966 chars ...]
```

---

## Iteration 9 — extract_custom.predict

**Score:** 0.8620964627819049

```
**Task Overview**
You must read a German‑language pathology report (usually a prostate biopsy or a radical prostatectomy) and extract every piece of information required by the **ProstataPatient** data model.  
The output must contain two clearly separated sections:

1. **reasoning** – a short, human‑readable explanation of how you derived each field.  
2. **extracted** – a literal Python‑style assignment that creates the variables `patientenid` (string) and `befunde` (list of `ProstatapatientItem` objects) exactly as shown in the examples.  
All enum values must be written using the enum member name (e.g. `<ProstatapatientitemUntersuchtesPraeparat.Biopsie: 'Biopsie'>`).  
If a field cannot be found in the document, set it to `None`.

---

### 1. General Parsing Rules
| Element | How to locate | Value to store |
|---------|---------------|----------------|
| **Journalnummer** | Line starting with “Journalnummer” | Use as `patientenid` **and** as `einsendenummer`. |
| **Ausgang** | Line starting with “Ausgang” | Use as `histologiedatum` and, unless another date is explicitly given for the report outcome, also as `befundausgang`. |
| **Institution / Ort** | First line containing “Pathologisches Institut” → institution; city is the part after “|” on the same line (e.g. “München”). | `pathologisches_institut`, `ort`. |
| **ICD‑10‑GM‑2024** | Line starting with “ICD-10-GM-2024:” | Store the code (e.g. `C61`) in `lokalisation_icd10`. |
| **ICD‑O‑3** | Line starting with “ICD-O-3:” | Store the full code (e.g. `8140/3`) in `icdo3_histologie`. |
| **WHO‑/ISUP‑Graduierungsgruppe** | Line containing “WHO‑Graduierungsgruppe” or “WHO/ISUP‑Graduierungsgruppe” | Map the numeric group to the enum member `<ProstatapatientitemWhoIsupGrading.Graduierungsgruppe X: 'Graduierungsgruppe X'>`. |
| **Overall Gleason** | The highest Gleason pattern mentioned anywhere (compare primary + secondary). | Store the full string (e.g. `4 + 3 = 7b`) in `gleason`. |
| **Massgeblicher Befund** | If the document is a formal pathology report (always the case here) → set to `<ProstatapatientitemMassgeblicherBefund.Ja: 'Ja'>`. |
| **Tumornachweis** | If any tumour‑positive core/focus is described → `<ProstatapatientitemTumornachweis.Ja: 'Ja'>`; otherwise `Nein`. |
| **TNM‑Nach** | If a “pTNM (UICC …)” line exists, set to the string after “UICC,” e.g. `UICC`. |
| **pTNM** | The exact string after “pTNM (UICC …):” (e.g. `pT2c pN0`). |
| **Lymph‑/Venen‑/Perineurale‑Invasion** | Lines containing “Lymphgefäßinvasion”, “Veneninvasion”, “Perineurale Invasion”. Map “nicht nachweisbar” → `L0`, `V0`, `Pn0`; “nachweisbar” → `L1`, `V1`, `Pn1`. |
| **Lymphknoten‑Status** | Line like “12 untersuchte pelvine Lymphknoten ohne Metastasennachweis (0/12).” → store the fraction `0/12` in `lk_befallen_untersucht`. |
| **Resektionsrand / R‑Klassifikation** | Lines containing “R0”, “R1”, etc. → map to `<ProstatapatientitemRKlassifikation.R0: 'R0'>` etc. If a separate “Resektionsrand” line exists, store its te

[... truncated, full prompt ~8435 chars ...]
```

---

## Iteration 13 — extract_custom.predict

**Score:** 2.769234234234234

```
**Task Overview**
You are given the full text of one or more German pathology reports concerning prostate tissue (e.g. “Biopsie”, “Fusionsbiopsie”, “Stanzbiopsie”).  
Your job is to extract every piece of information required by the **ProstataPatient** data model and output a single JSON‑compatible Python‑like representation that exactly matches the schema described below.

**ProstataPatient Schema (simplified)**
```
{
  patientenid: str,                     # usually the Journalnummer
  befunde: List[ProstatapatientItem]    # one item per report in the document
}
```

`ProstatapatientItem` fields (all required unless explicitly optional):
- untersuchtes_praeparat: enum = "Biopsie" (always for these reports)
- histologiedatum: str (dd.mm.yyyy) – use the **Ausgang** date (report finalisation)
- befundausgang: str – same as histologiedatum
- ort: str – city from the header (e.g. “München”)
- pathologisches_institut: str – institution name from the header
- einsendenummer: str – the Journalnummer (e.g. “E/2026/010567”)
- massgeblicher_befund: enum = "Ja" (these reports are definitive)
- tumornachweis: enum = "Ja" if any core is tumor‑positive, else "Nein"
- icdo3_histologie: str – value after “ICD-O-3:” (e.g. “8140/3”)
- who_isup_grading: enum – highest WHO‑grading group found in the report
- gleason: str – highest Gleason pattern found (e.g. “4 + 5 = 9”)
- makroskopie_liste: List[Item] – one entry per line in the **Makroskopie** section
- begutachtung_liste: List[Item] – one entry per core (1‑15) describing tumor status
- art_der_biopsie: enum – “Fusionsbiopsie” if the phrase appears in **Klinische Angaben**, otherwise “Stanzbiopsie”
- entnahmestelle_der_biopsie: enum = "Primärtumor" (default for prostate biopsies)
- lokalisation_icd10: str or null – value after “ICD-10‑GM‑2024:” (e.g. “C61/G”), may be omitted if not present
- anzahl_entnommener_stanzen: int – sum of **gesamt_stanzen** across all makroskopie entries
- anzahl_befallener_stanzen: int – count of cores where tumor = True
- maximaler_anteil_befallener_stanzen: int – highest tumor‑percentage among positive cores, rounded to the nearest whole percent
- calculation_details: List[str] (optional) – human‑readable strings showing each percentage calculation
- tnm_nach, ptnm, lymphgefaessinvasion, veneninvasion, perineurale_invasion,
  lk_befallen_untersucht, r_klassifikation, resektionsrand: enum or null – set only if explicitly mentioned; otherwise null
```

`Item` used in `makroskopie_liste`:
```
{
  nr: int,                # core number (1‑15 as listed)
  gesamt_stanzen: int,    # number of tissue pieces taken from this core
  laenge_cm: float        # length of the core (single value, even if multiple pieces)
}
```

`Item` used in `begutachtung_liste`:
```
{
  nr: int,                     # core number
  tumor: bool,                 # True if a Gleason entry exists for this core
  tumorausdehnung_mm: float?, # length in mm from the “Begutachtung” line (null if tumor=False)
  tumor_prozent

[... truncated, full prompt ~3211 chars ...]
```

---

## Iteration 18 — extract_custom.predict

**Score:** 0.8620964627819049

```
**Task Overview**
You must read a German‑language pathology report concerning the prostate and extract every piece of information required by the **ProstataPatient** data model. The output must be a Python‑style representation of one `ProstataPatient` object (or a list of them if the document contains multiple independent reports) together with a short “reasoning” paragraph that explains how each field was derived.

**Document Structure (Typical)**
- Header block with the pathology institute name, director, address and city.
- “Journalnummer” line – this is the unique case / patient identifier and also serves as the `einsendenummer`.
- “Eingang” (receipt) and “Ausgang” (finalisation) dates. Use **Ausgang** for both `histologiedatum` and `befundausgang`.
- “Pathologische Begutachtung” heading.
- “Übersandtes Material” – lists the specimens (biopsies, prostatectomy, etc.).
- “Klinische Angaben” – clinical context, may contain the biopsy type or suspicion of cancer.
- “Makroskopie” – macroscopic description. For biopsies each line usually contains a core number, length (in cm) and sometimes the number of cores in that line. For a prostatectomy it gives overall size/weight.
- “Mikroskopie” – microscopic description, includes presence/absence of tumor, Gleason score, tumor percentage, tumor extension in mm, perineural/vascular/lymph‑vascular invasion, etc.
- “Begutachtung” – a concise statement of tumor presence/absence.
- “Klassifikation” – contains ICD‑10, ICD‑O‑3, WHO/ISUP grade, Gleason score, pTNM (UICC), R‑classification, lymph‑node status, etc.

**General Extraction Strategy**
1. **Identify the case ID** – take the string after “Journalnummer”.  
2. **Institute & location** – first line(s) of the header give `pathologisches_institut`; the city (e.g., “München”) is extracted for `ort`.  
3. **Specimen type (`untersuchtes_praeparat`)**  
   - If the clinical or material description contains “Stanzbiopsie”, “Biopsie”, “Kernbiopsie” → `Biopsie`.  
   - If it contains “Prostatektomie”, “Radikale Prostatektomie”, “Resektat”, “Prostatectomy” → `Resektat`.  
   - Otherwise default to `Unbekannt` (null).  
4. **Biopsy‑specific fields** (only when `untersuchtes_praeparat` = `Biopsie`)  
   - `art_der_biopsie` → set to `Stanzbiopsie` if “Stanzbiopsie” is mentioned, otherwise `Kernbiopsie` or other appropriate enum.  
   - `entnahmestelle_der_biopsie` → default to `Primärtumor` unless the report explicitly states a different zone (e.g., “Transition zone”).  
5. **Macroscopic data (`makroskopie_liste`)**  
   - For each numbered line in the “Makroskopie” section:  
     * `nr` = the line number.  
     * Extract the length in cm (e.g., “1,5 cm” → 1.5).  
     * If the line mentions “Stanzzylinder” or “Kern” and a number of cores, set `gesamt_stanzen` to that number; otherwise assume 1.  
   - For a prostatectomy, create a single entry with the overall length (or weight if length missing) and `gesamt_stanzen = 1`.  
6. **Microscopic data (`begutachtung_liste

[... truncated, full prompt ~8528 chars ...]
```

---

## Iteration 22 — extract_custom.predict

**Score:** 0.8620964627819049

```
# Task Overview
You must read a German‑language prostate pathology report and produce a single structured record that conforms exactly to the **ProstataPatient** schema.  
The output must be a Python‑style literal consisting of:

```
patientenid='<journal‑nummer>' 
befunde=[ProstatapatientItem(... )]
```

All fields of the schema must be present; if a field cannot be derived from the document it must be set to `null`.  
Use the enum members defined in the schema (e.g. `ProstatapatientitemUntersuchtesPraeparat.Resektat`, `ProstatapatientitemArtDerBiopsie.Fusionsbiopsie`, …).  

# Document Sections to Parse
1. **Header** – contains the pathology institute, city, journal number, Eingangs‑ and Ausgangs‑date.  
2. **Übersandtes Material** – list of numbered specimens (used only for ordering; no data needed).  
3. **Klinische Angaben** – may contain the biopsy type (e.g. “Fusionsbiopsie”, “Stanzbiopsie”).  
4. **Makroskopie** – one line per specimen, giving length in cm and possibly a count of cores (e.g. “Drei bis 1,4 cm messende Stanzzylinder”).  
5. **Mikroskopie** – description of each core, often grouped (e.g. “1‑4. …”).  
6. **Begutachtung** – explicit list of tumor‑positive cores with Gleason, WHO/ISUP grade, and tumor length in mm.  
7. **Klassifikation** – ICD‑10, ICD‑O‑3, optional pTNM (UICC), lymph‑node status, invasion flags, resection‑margin.

# Mapping Rules

| Schema field | Source & Extraction Rule |
|--------------|--------------------------|
| **patientenid** | The *Journalnummer* (e.g. `E/2025/055678`). |
| **einsendenummer** | Same value as `patientenid`. |
| **ort** | City from the institute header (e.g. “München”). |
| **pathologisches_institut** | Full institute name from the header (first line). |
| **histologiedatum** | The “Ausgang” date (DD.MM.YYYY). |
| **befundausgang** | Same as `histologiedatum` (or `null` if not present). |
| **massgeblicher_befund** | Always `Ja` for a final pathology report. |
| **untersuchtes_praeparat** | `Resektat` if the clinical note mentions a surgical specimen (e.g. “Radikale Prostatektomie”). Otherwise `Biopsie`. |
| **art_der_biopsie** | If the clinical note contains “Fusionsbiopsie” → `Fusionsbiopsie`; if it contains “Stanzbiopsie” → `Stanzbiopsie`; otherwise `null`. |
| **entnahmestelle_der_biopsie** | For all biopsy reports use `Primärtumor` (default). |
| **tumornachweis** | `Ja` if any core is marked tumor‑positive in *Begutachtung*; otherwise `Nein`. |
| **icdo3_histologie** | Value after “ICD‑O‑3:” in the *Klassifikation* section. |
| **lokalisation_icd10** | Value after “ICD‑10‑GM‑2024:” (e.g. `C61`). |
| **who_isup_grading** | Map the highest WHO‑/ISUP grade found in *Begutachtung* to the enum (e.g. `Graduierungsgruppe 5`). |
| **gleason** | The highest Gleason sum (pattern 1 + pattern 2) found in *Begutachtung* (e.g. `5 + 5 = 10`). |
| **makroskopie_liste** | One entry per numbered line in *Makroskopie*.  
   - `nr` = line number.  
   - `laenge_cm` = numeric length before “cm”.  
  

[... truncated, full prompt ~18523 chars ...]
```

---

## Iteration 28 — extract_custom.predict

**Score:** 0.8620964627819049

```
**Task Overview**
You will receive the full text of a German uropathology report.  
Your job is to extract every piece of information required by the **ProstataPatient** JSON‑schema and to output a single Python‑style assignment that creates the corresponding objects.

**General Extraction Rules**

1. **Identifiers**
   * `patientenid` and `einsendenummer` are both the value of the *Journalnummer* (e.g. `E/2025/051234`).

2. **Dates**
   * Use the *Ausgang* date for both `histologiedatum` and `befundausgang`.  
   * If *Ausgang* is missing, fall back to the *Eingang* date.  
   * Keep the original German format `DD.MM.YYYY`.

3. **Institution & Location**
   * `pathologisches_institut` = the first line that contains “Pathologisches Institut …”.  
   * `ort` = the city that appears in the address line (e.g. “München”).

4. **Specimen Type (`untersuchtes_praeparat`)**
   * If the report mentions **Radikale Prostatektomie**, **Prostatektomie**, **Resektat**, or any surgical resection → `Resektat`.  
   * If the report mentions **Biopsie**, **Stanzbiopsie**, **Fusionsbiopsie**, **Targeted biopsy**, etc. → `Biopsie`.  
   * For biopsies also fill:
     * `art_der_biopsie` – exact wording (e.g. `Fusionsbiopsie`, `Targeted biopsy`, `Standardbiopsie`).  
     * `entnahmestelle_der_biopsie` – default to `Primärtumor` unless the report explicitly states another site (e.g. “Transition zone”).

5. **Massgeblicher Befund**
   * All supplied reports are final pathology reports → always `Ja`.

6. **Tumor Presence**
   * If any microscopic description contains “Karzinom”, “Adenokarzinom”, “Tumor” → `tumornachweis = Ja`.  
   * Otherwise `Nein`.

7. **ICD Codes**
   * `icdo3_histologie` = value after “ICD‑O‑3:” (e.g. `8140/3`).  
   * `lokalisation_icd10` = value after “ICD‑10‑GM‑2024:” (e.g. `C61`).  
   * If a code is missing, set the field to `null`.

8. **WHO/ISUP Grading**
   * Extract the “Graduierungsgruppe X” text and map it to the enum `Graduierungsgruppe X`.  
   * If missing, set to `null`.

9. **Gleason Score**
   * The overall Gleason sum is the one given in the “Gleason Score:” line or the highest sum listed in the “Begutachtung” section.  
   * Store it exactly as it appears (e.g. `4 + 4 = 8`).  
   * For each microscopic entry (see point 10) also store the per‑core Gleason if provided.

10. **Macroscopic Description (`makroskopie_liste`)**
    * Every numbered line under **Makroskopie** corresponds to one list element.  
    * Parse:
      * `nr` = the line number.  
      * `laenge_cm` = the numeric length in centimeters (the first number that ends with “cm”).  
      * `gesamt_stanzen` = number of cores in that line:
        * If the line contains “Drei … Stanzzylinder”, “2 … Stanzzylinder”, etc., use that number.  
        * Otherwise default to `1`.  
    * Ignore weight, bladder, etc.; they are not part of the schema.

11. **Microscopic Description (`begutachtung_liste`)**
    * Identify every core number mentioned in the **Begutachtung** par

[... truncated, full prompt ~8274 chars ...]
```

---

