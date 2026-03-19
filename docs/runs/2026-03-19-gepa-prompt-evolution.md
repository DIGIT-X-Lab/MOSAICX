# GEPA Prompt Evolution — ProstataPatient

> Automatic prompt optimization via DSPy GEPA on CORE (gpt-oss:120b)
> Training set: 20 synthetic German prostate pathology reports
> Total proposals: 1 | Last updated: 17:23:11

---

## Iteration 0 — Baseline

**Score:** 0.8612740943608522

```
Extract structured data matching the ProstataPatient schema from the document.
```

---

## Iteration 1 — extract_custom.predict

**Score:** 0.8612740943608522

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

