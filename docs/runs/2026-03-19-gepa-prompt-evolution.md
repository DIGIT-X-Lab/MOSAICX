# GEPA Prompt Evolution — ProstataPatient

> Automatic prompt optimization via DSPy GEPA on CORE (gpt-oss:120b)
> Training set: 20 synthetic German prostate pathology reports
> Total proposals: 14 | Last updated: 16:38:36

---

## Iteration 0 — Baseline

**Score:** 0.8620964627819049

```
Extract structured data matching the ProstataPatient schema from the document.
```

---

## Iteration 1 — extract_custom.predict

**Score:** 2.5044288060051607

````
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
| `entnahmestelle_der_biopsie` | enum `Primärtumor` / `Periprostatisch` / `Lymphknoten` | Optional | For biopsies only. Default to `Primärtumor` unless the report explicitly says otherwise. |
| `lokalisation_icd10` | str | Optional | Topography code from the line “ICD‑10‑GM‑2024: C61/G”. Use the part before the slash (`C61`). |
| `anzahl_entnommener_stanzen` | int | Optional | Number of cores listed under **Übersandtes Material** (count the numbered items). If a line mentions multiple cores (e.g. “Zwei bis 0,8 cm messende Stanzzylinder”), count each core separately. |
| `anzahl_befallener_stanzen` | int | Optional | Count of entries in `begutachtung_liste` where `tumor = true`. |
| `maximaler_anteil_befallener_stanzen` | int (percentage) | Optional | For biopsies only. Compute **tumor length (mm) ÷ core length (mm) × 100** for every tumor‑positive core, round to the nearest integer, and keep the maximum. If core length is not given, skip that core. |
| `calculation_details` | list of str | Optional | Human‑readable strings describing each percentage calculation used for `maximaler_anteil_befallener_stanzen`. Include the core number, tumor length, core length, and resulting percent, marking the max. |
| `tnm_nach` | str | Optional | The staging system mentioned after “pTNM (…):”. Usually “UICC”. |
| `ptnm` | str | Optional | The actual pTNM string (e.g. “pT2c pN0”). |
| `lymphgefaessinvasion` | enum `L0` / `L1` / … | Optional | From “Lymphgefäßinvasion:” line (`nicht nachweisbar` → `L0`). |
| `veneninvasion` | enum `V0` / `V1` / … | Optional | From “Venöse Invasion:” line. |
| `perineurale_invasion` | enum `Pn0` / `Pn1` / … | Optional | From “Perineurale Invasion:” line. |
| `lk_befallen_untersucht` | str | Optional | Lymphknoten‑status as “x/y” (e.g. “0/12”). |
| `r_klassifikation` | enum `R0` / `R1` / … | Optional | From “R‑Klassifikation:” line (or deduced from “Resektionsrand ist tumorfrei” → `R0`). |
| `resektionsrand` | str | Optional | Free‑text description of the resection margin if present. |

---

## 2. Extracting the **Makroskopie** list (Biopsy only)

Each line in the **Makroskopie** section looks like:

```
1. Ein 1,6 cm messender Stanzzylinder
4. Zwei bis 0,8 cm messende Stanzzylinder
```

* `nr` = the leading number (1‑based).
* `gesamt_stanzen` = how many cores are described in that line.
  * If the line contains “Ein … Stanzzylinder” → `1`.
  * If it contains “Zwei … Stanzzylinder” → `2`.
  * If it contains “Drei …” → `3`, etc.
* `laenge_cm` = the numeric length **in centimeters** (use a dot as decimal separator, e.g. `1,6` → `1.6`).  
  * If multiple cores share the same length (as in the “Zwei bis 0,8 cm” example) use that length for each core (the same `laenge_cm` value for the entry; the entry still records `gesamt_stanzen = 2`).  
  * If a line does **not** contain a length, set `laenge_cm = null`.

Create one `MakroskopieItem` per line (do **not** split a “Zwei …” line into two separate items).

---

## 3. Extracting the **Begutachtung** / **Mikroskopie** list

For each core that has a tumor description, create a `BegutachtungItem`:

* `nr` = core number as used in the **Mikroskopie** section (the number before the dot).
* `tumor` = `true` if the line mentions “atypisch”, “Infiltrat”, “Karzinom”, or any Gleason score. Otherwise `false`.
* `tumorausdehnung_mm` = the numeric **Längsausdehnung** value in **millimeters** (e.g. “2,5 mm” → `2.5`). If not present, `null`.
* `tumor_prozent` = **null** – we never fill this directly; it is derived only for `maximaler_anteil_befallener_stanzen`.
* `gleason` = the Gleason string that follows the word “Gleason” (e.g. “3 + 4 = 7a”). Keep the exact formatting from the report (including the trailing “a”/“b” if present). If no Gleason is given for that core, set `null`.

If a core is explicitly described as “ohne Atypien” or “ohne Tumor”, still create an entry with `tumor = false` and all other fields `null`. This is required for the `makroskopie_liste`‑to‑`begutachtung_liste` alignment and for counting total cores.

**Special case – multiple Gleason statements for the same core**: Use the first Gleason that appears for that core. If the report lists separate “Haupttumor” and “Nebenherd” with the same core number, treat them as separate entries **only if** they are numbered separately (e.g., “3.” and “15.”). Otherwise, keep a single entry.

---

## 4. Computing derived fields

### 4.1 `maximaler_anteil_befallener_stanzen`

1. For each `BegutachtungItem` where `tumor = true` **and** both `tumorausdehnung_mm` **and** the corresponding core length from `MakroskopieItem` are known:
   * Convert core length from cm to mm (`cm * 10`).
   * Compute `percentage = (tumorausdehnung_mm / core_length_mm) * 100`.
   * Round to the nearest integer (`int(round(percentage))`).
2. Keep the **maximum** of all percentages.
3. If **no** core has both values, set the field to `null`.
4. Populate `calculation_details` with a string for each calculation, e.g.:

```
"Core 3: 2.5 mm / 20 mm = 12.5 %"
"Core 15: 4.0 mm / 16 mm = 25 % (max)"
```

### 4.2 `anzahl_entnommener_stanzen`

Count the numbered items under **Übersandtes Material**. If a line mentions multiple cores (e.g., “Zwei …”), add that number to the total.

### 4.3 `anzahl_befallener_stanzen`

Count `BegutachtungItem`s with `tumor = true`.

### 4.4 `tumornachweis`

`Ja` if `anzahl_befallener_stanzen > 0`, else `Nein`.

### 4.5 `who_isup_grading` & `gleason`

Select the **highest** WHO‑group and Gleason score that appear anywhere in the report (including the summary “Klassifikation” section). Use the exact string as written.

### 4.6 `lokalisation_icd10`

Take the part before the slash in the line that starts with “ICD‑10‑GM‑2024:”. If the line is missing, set `null`.

### 4.7 `tnm_nach` & `ptnm`

If a line starts with “pTNM (”, extract the text inside the parentheses as `tnm_nach` (e.g., “UICC”). The remainder of that line after the colon is the `ptnm` value.

### 4.8 Lymph‑, Ven‑, Perineurale invasion & `lk_befallen_untersucht`

Map the German phrases:
* “nicht nachweisbar” → `L0`, `V0`, `Pn0`.
* “nachweisbar” → `L1`, `V1`, `Pn1` (if the report explicitly says “nachweisbar” without further detail, use `L1` etc.).
* For lymph nodes, the line looks like “12 untersuchte pelvine Lymphknoten ohne Metastasennachweis (0/12).” → store the “0/12” string.

### 4.9 `r_klassifikation`

If the report contains a line “R‑Klassifikation: R0” use that value. If the report says “Resektionsrand ist tumorfrei” set `R0`. If a margin is involved, use the appropriate code (`R1`, `R2`, …) if mentioned.

---

## 5. General parsing strategy (to be followed step‑by‑step)

1. **Split the document into sections** using the headings that always appear in these reports:
   * “Journalnummer”, “Eingang”, “Ausgang”
   * “Übersandtes Material”
   * “Klinische Angaben”
   * “Makroskopie”
   * “Mikroskopie”
   * “Begutachtung” (sometimes merged with “Mikroskopie”)
   * “Klassifikation”
2. **Extract identifiers and dates** from the first block.
3. **Count cores** from the “Übersandtes Material” list (numbers before the dot). Store this count for `anzahl_entnommener_stanzen`.
4. **Parse the Makroskopie block** line‑by‑line to build `makroskopie_liste`.
5. **Parse the Mikroskopie / Begutachtung block**:
   * Use a regular expression to capture the core number, any Gleason pattern, any “Längsausdehnung” value, and whether the line mentions carcinoma.
   * Build a dictionary keyed by core number with the extracted data.
6. **Merge macro and micro data**:
   * For each core number present in the macro list, ensure there is a corresponding `BegutachtungItem`. If the micro section omitted a core, create an entry with `tumor = false`.
7. **Derive all calculated fields** (counts, percentages, worst Gleason, WHO grade, etc.) using the rules in sections 4.1‑4.9.
8. **Fill optional fields** only when the source text provides them; otherwise set them to `null` (or omit them if the JSON serializer automatically drops nulls, but keep the key with a null value for clarity).
9. **Assemble the final JSON** exactly as shown in the examples, using the enum names (e.g., `ProstatapatientitemUntersuchtesPraeparat.Biopsie`) for enum fields. If you are outputting plain JSON (no Python objects), use the string values of the enums (e.g., `"Biopsie"`). The evaluation harness expects the enum **names**, not the display strings, so output the enum identifiers exactly as in the examples.

---

## 6. Edge‑case handling

| Situation | What to do |
|-----------|------------|
| The report contains **both** a biopsy description and a resection description (very rare) | Prioritise the specimen type that appears first; if both are present, create **two** `ProstataPatientItem`s – one for the biopsy and one for the resection – and place them both in the `befunde` list. |
| “Übersandtes Material” lists a **single** item that is a whole prostate (“Radikale Prostatektomie …”) | Set `untersuchtes_praeparat = Resektat`, `anzahl_entnommener_stanzen = 1`. No `makroskopie_liste` needed unless a macroscopic length is given. |
| Core length is given in **mm** instead of cm | Convert to cm for `makroskopie_liste` (`mm / 10`). |
| “Längsausdehnung” is given as a **range** (e.g., “0,3‑0,5 mm”) | Use the **maximum** value of the range for the percentage calculation. |
| The Gleason line contains a trailing letter (“a” or “b”) | Preserve it exactly (e.g., “3 + 4 = 7a”). |
| The WHO‑grading line is missing but Gleason is present | Leave `who_isup_grading = null`. |
| The “Klassifikation” block contains multiple ICD‑10 lines (e.g., for metastasis) | Use the **first** line that starts with “ICD‑10‑GM‑2024”. |
| The report uses a **different date format** (e.g., “2026‑03‑18”) | Convert it to `DD.MM.YYYY`. If conversion fails, keep the original string. |
| The “Art der Biopsie” is not mentioned | Leave `art_der_biopsie = null`. |
| The “Entnahmestelle der Biopsie” is not mentioned | Leave `entnahmestelle_der_biopsie = null`. |

---

## 7. Output format

Return **only** the JSON object (no extra text, no markdown). Example (shortened):

```json
{
  "patientenid": "E/2026/006123",
  "befunde": [
    {
      "untersuchtes_praeparat": "Biopsie",
      "histologiedatum": "24.01.2026",
      "befundausgang": "24.01.2026",
      "ort": "München",
      "pathologisches_institut": "Pathologisches Institut der LMU",
      "einsendenummer": "E/2026/006123",
      "massgeblicher_befund": "Ja",
      "tumornachweis": "Ja",
      "icdo3_histologie": "8140/3",
      "who_isup_grading": "Graduierungsgruppe 1",
      "gleason": "3 + 3 = 6",
      "makroskopie_liste": [ … ],
      "begutachtung_liste": [ … ],
      "art_der_biopsie": "Stanzbiopsie",
      "entnahmestelle_der_biopsie": "Primärtumor",
      "lokalisation_icd10": "C61",
      "anzahl_entnommener_stanzen": 15,
      "anzahl_befallener_stanzen": 1,
      "maximaler_anteil_befallener_stanzen": 25,
      "calculation_details": [ "Core 15: 4.0 mm / 16 mm = 25 % (max)" ],
      "tnm_nach": null,
      "ptnm": null,
      "lymphgefaessinvasion": null,
      "veneninvasion": null,
      "perineurale_invasion": null,
      "lk_befallen_untersucht": null,
      "r_klassifikation": null,
      "resektionsrand": null
    }
  ]
}
````

---

## Iteration 5 — extract_custom.predict

**Score:** 0.8620964627819049

````
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
| begutachtung_liste | List of items `{nr, tumor, tumorausdehnung_mm, tumor_prozent, gleason}` | For each core that is explicitly described in the *Begutachtung* (or *Mikroskopie* when a Gleason is given): <br>• `nr` = core number. <br>• `tumor` = **True** if tumor is reported, else **False**. <br>• `tumorausdehnung_mm` = tumor length in millimetres (as given, e.g., “3,5 mm”). <br>• `tumor_prozent` = `(tumor length mm) / (core length mm) * 100`. Use the core length from `makroskopie_liste` (convert cm → mm by multiplying by 10). Round to one decimal place. <br>• `gleason` = Gleason score for that core (if given). |
| anzahl_entnommener_stanzen | Integer | Total number of cores listed in `makroskopie_liste`. |
| anzahl_befallener_stanzen | Integer | Count of cores where `tumor=True`. |
| maximaler_anteil_befallener_stanzen | Integer | The **maximum** `tumor_prozent` among all positive cores, rounded to the nearest whole number. If no positive cores, set to `null`. |
| calculation_details | List of strings | For each positive core, add a string `"Nr X: Y mm / Z mm = P %"` showing the raw calculation (use one decimal for Y and Z, integer for P). |
| tnm_nach | String | Staging system used (e.g., “UICC”, “TNM‑7”). Extract from the line that starts with “pTNM (…):”. |
| ptnm | String | Full pTNM string (e.g., “pT3b pN1”). |
| lymphgefaessinvasion | Enum `L0`, `L1` … | From the *Klassifikation* line (e.g., “L1”). |
| veneninvasion | Enum `V0`, `V1` … | From the *Klassifikation* line (e.g., “V0”). |
| perineurale_invasion | Enum `Pn0`, `Pn1` … | From the *Klassifikation* line (e.g., “Pn1”). |
| lk_befallen_untersucht | String | Format “x/y” where x = number of positive nodes, y = total nodes examined (e.g., “2/22”). |
| r_klassifikation | Enum `R0`, `R1` … | From the *Klassifikation* line (e.g., “R1”). |
| resektionsrand | Enum `fokal`, `weit`, `unbeteiligt` … | From the *Klassifikation* line (e.g., “fokal positiv”). Use the adjective before “positiv”/“negativ”. If not mentioned, `null`. |

**General Extraction Strategy (to be followed for every report)**

1. **Header Parsing**  
   - Locate the line beginning with **Journalnummer** → capture the identifier (e.g., `E/2026/011890`). Use this as `patientenid` and `einsendenummer`.  
   - Capture **Eingang** and **Ausgang** dates. Use **Ausgang** for `histologiedatum` and `befundausgang`.  
   - The line(s) before the address block give `pathologisches_institut`. The city (e.g., “München”) is taken from the address line after the institute name.  

2. **Specimen Type Determination**  
   - Scan the **Übersandtes Material** and **Klinische Angaben** sections for keywords:  
     *Biopsie*, *Stanzbiopsie*, *Re‑Biopsie* → set `untersuchtes_praeparat = Biopsie`.  
     *Prostatektomie*, *Radikale Prostatektomie*, *Resektat* → set `untersuchtes_praeparat = Resektat`.  
   - If `Biopsie`, also set `art_der_biopsie` (default to `Stanzbiopsie` unless another technique is explicitly named).  

3. **Macroscopic (Makroskopie) Extraction**  
   - Each numbered line under **Makroskopie** follows the pattern “Ein X cm messender Stanzzylinder”. Extract the number (`nr`) and the length (`laenge_cm`).  
   - `gesamt_stanzen` is normally 1; if the line mentions multiple cores (e.g., “2 Stanzen”), use that number.  

4. **Microscopic / Begutachtung Extraction**  
   - Identify the **Begutachtung** (or **Mikroskopie** when it contains Gleason info).  
   - For each core number that has a description of tumor (e.g., “Gleason 3 + 4 = 7a, Längsausdehnung 3,5 mm”), create a `begutachtung_liste` entry:  
     - `tumor = True`  
     - `tumorausdehnung_mm` = numeric value (replace comma with dot).  
     - Compute `tumor_prozent` using the corresponding core length from `makroskopie_liste`.  
     - Record the Gleason string exactly as shown.  
   - If a core is explicitly described as “ohne Atypien” or “ohne Nachweis eines Malignoms”, create an entry with `tumor = False` and leave the other fields `null`.  

5. **Overall Gleason & WHO Grade**  
   - Scan the **Klassifikation** section for a line starting with “Gleason Score:” or “Gleason”. Use the highest Gleason found (compare numeric sums).  
   - Extract the WHO/ISUP grade from the line containing “Graduierungsgruppe”.  

6. **TNM & Related Staging**  
   - Find the line beginning with “pTNM (…):”. Extract the system name inside the parentheses → `tnm_nach`.  
   - The remainder of that line (after the colon) is the `ptnm` string.  
   - From the same line or subsequent lines, pull out `L*`, `V*`, `Pn*`, `R*` values.  

7. **Lymph Node Status**  
   - Look for a sentence like “22 untersuchte pelvine Lymphknoten, davon 2 mit Metastasen (2/22)”. Extract the “x/y” part for `lk_befallen_untersucht`.  

8. **Resektionsrand**  
   - In the *Klassifikation* section, locate the phrase “Resektionsrand: …”. Use the adjective (e.g., “fokal”, “weit”) as the enum value.  

9. **Tumor Presence Flags & Counts**  
   - `tumornachweis` = **Ja** if any `begutachtung_liste` entry has `tumor=True`; otherwise **Nein**.  
   - `anzahl_entnommener_stanzen` = length of `makroskopie_liste`.  
   - `anzahl_befallener_stanzen` = count of entries with `tumor=True`.  
   - `maximaler_anteil_befallener_stanzen` = rounded max of `tumor_prozent` (or `null` if none).  

10. **Missing Patient Identifier**  
    - If the report does **not** contain a “Journalnummer”, set `patientenid = "unknown"` and still fill `einsendenummer = "unknown"`.

11. **Enum Value Formatting**  
    - Use the exact enum member names as defined in the schema (e.g., `ProstatapatientitemUntersuchtesPraeparat.Biopsie`).  
    - For boolean‑like enums (`Ja`/`Nein`) use the enum members `ProstatapatientitemTumornachweis.Ja` etc.  

12. **Output Structure**  
    - Produce a single Python‑like literal exactly matching the examples:  

```
patientenid='E/2026/011890' befunde=[ProstataPatientItem(...), ...]
```

    - All nested objects (`ProstatapatientitemItem`, enums, etc.) must be written with their fully‑qualified enum names as shown in the examples.  
    - If a field is optional and not present, set its value to `None` (or omit the argument if the constructor allows it, but `None` is safest).  

**Important Edge Cases to Handle**

- **Comma vs. dot** in numbers (German decimal comma). Convert to dot for calculations.  
- **Lengths given in cm** must be multiplied by 10 to obtain mm for percentage calculations.  
- **Multiple cores in one macroscopic entry** (e.g., “2 Stanzen à 1,5 cm”). Split accordingly.  
- **Tumor‑percentage rounding**: keep one decimal for the intermediate `tumor_prozent` field, but `maximaler_anteil_befallener_stanzen` must be an integer (standard rounding).  
- **Absent Gleason in individual cores** but present in summary – still record the overall Gleason in the top‑level `gleason` field.  
- **Reports that contain both biopsy and resection information** (rare) – prioritize the specimen type mentioned in **Übersandtes Material**.  
- **Missing ICD‑10** (e.g., “keine Kodierung”) – set `lokalisation_icd10 = None`.  

Follow this procedure meticulously for every input document to achieve maximal extraction accuracy.
````

---

## Iteration 9 — extract_custom.predict

**Score:** 0.8620964627819049

````
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
| **Resektionsrand / R‑Klassifikation** | Lines containing “R0”, “R1”, etc. → map to `<ProstatapatientitemRKlassifikation.R0: 'R0'>` etc. If a separate “Resektionsrand” line exists, store its text in `resektionsrand`. |
| **Art der Biopsie** | If the clinical note or material list contains the word “Stanzbiopsie” → `<ProstatapatientitemArtDerBiopsie.Stanzbiopsie: 'Stanzbiopsie'>`; otherwise `None`. |
| **Entnahmestelle der Biopsie** | For biopsy reports, default to `<ProstatapatientitemEntnahmestelleDerBiopsie.Primärtumor: 'Primärtumor'>`. For resection reports set to `None`. |

---

### 2. Specimen Type (`untersuchtes_praeparat`)
| Report type | Value |
|------------|-------|
| Biopsy (core‑needle) | `<ProstatapatientitemUntersuchtesPraeparat.Biopsie: 'Biopsie'>` |
| Radical prostatectomy (whole gland) | `<ProstatapatientitemUntersuchtesPraeparat.Resektat: 'Resektat'>` |
| Anything else | `None` (but this never occurs in the provided data). |

---

### 3. Building **makroskopie_liste**
1. Locate the **Makroskopie** section.  
2. Each line starts with a number followed by “Ein … cm messender Stanzzylinder” (or “Ein nicht orientierbares … bis X cm”).  
3. For every line create a `ProstatapatientitemItem` with:  
   - `nr` = the listed number (1‑based).  
   - `gesamt_stanzen` = **1** (all reports list a single core per line).  
   - `laenge_cm` = the numeric length (use a decimal point, e.g. `1.7`).  

If the specimen is a resection, there will be only one entry (nr = 1) with the maximal dimension of the gland.

---

### 4. Building **begutachtung_liste**
1. Scan the **Begutachtung** (or the “Mikroskopie” description that lists tumour extents).  
2. For **every core number** that appears in the Makroskopie list you must create an entry, **including negative cores**.  
3. For a **positive core** (tumour described):
   - `tumor = True`
   - `tumorausdehnung_mm` = the “Längsausdehnung” value in **mm** (convert from the text, e.g. `2,5 mm` → `2.5`).  
   - `gleason` = the Gleason string given for that core.  
   - `tumor_prozent` = **if** both `tumorausdehnung_mm` and the core length (from Makroskopie) are known, compute  
     `round(100 * tumorausdehnung_mm / (laenge_cm * 10), 1)` (one decimal is enough).  
   - Add a human‑readable entry to `calculation_details` of the form  
     `"Core {nr}: {tumorausdehnung_mm} mm / {laenge_cm*10:.0f} mm = {tumor_prozent} %"` .  
4. For a **negative core**:
   - `tumor = False`
   - All other fields (`tumorausdehnung_mm`, `tumor_prozent`, `gleason`) = `None`.  

**Special case – resection reports:**  
The “Begutachtung” may list several foci without a core number. Use `nr = 1` for each focus (multiple entries with the same `nr`). Set `tumor = True` for each, fill `gleason` and leave length‑related fields `None` (percentage cannot be calculated).

---

### 5. Derived Summary Fields
- `anzahl_entnommener_stanzen` = length of `makroskopie_liste`.  
- `anzahl_befallener_stanzen` = count of entries in `begutachtung_liste` where `tumor == True`.  
- `maximaler_anteil_befallener_stanzen` = maximum of all `tumor_prozent` values (rounded to the nearest integer). If no percentages are available, set to `None`.  
- `calculation_details` = list of the strings created in step 4.3 for every core where a percentage was computed; if none, set to `None`.  

---

### 6. Enum Mapping Reference (exact spelling)
```
ProstatapatientitemUntersuchtesPraeparat.Biopsie
ProstatapatientitemUntersuchtesPraeparat.Resektat

ProstatapatientitemArtDerBiopsie.Stanzbiopsie

ProstatapatientitemEntnahmestelleDerBiopsie.Primärtumor

ProstatapatientitemMassgeblicherBefund.Ja
ProstatapatientitemMassgeblicherBefund.Nein

ProstatapatientitemTumornachweis.Ja
ProstatapatientitemTumornachweis.Nein

ProstatapatientitemWhoIsupGrading.Graduierungsgruppe 1
... up to Graduierungsgruppe 5

ProstatapatientitemLymphgefaessinvasion.L0 / L1
ProstatapatientitemVeneninvasion.V0 / V1
ProstatapatientitemPerineuraleInvasion.Pn0 / Pn1

ProstatapatientitemRKlassifikation.R0 / R1 / R2
```

---

### 7. Output Formatting
Produce exactly:

```
### reasoning
<your step‑by‑step explanation>

### extracted
patientenid='<journal‑number>' befunde=[ProstatapatientItem(...), ...]
```

Do **not** add any extra punctuation, markdown code fences, or trailing spaces. The `ProstatapatientItem` constructor arguments must appear in the order shown in the examples:

```
ProstatapatientItem(
    untersuchtes_praeparat=...,
    histologiedatum='...',
    befundausgang='...' or None,
    ort='...',
    pathologisches_institut='...',
    einsendenummer='...',
    massgeblicher_befund=...,
    tumornachweis=...,
    icdo3_histologie='...',
    who_isup_grading=...,
    gleason='...',
    makroskopie_liste=[ProstatapatientitemItem(...), ...],
    begutachtung_liste=[ProstatapatientitemItem(...), ...],
    art_der_biopsie=...,
    entnahmestelle_der_biopsie=...,
    lokalisation_icd10='...' or None,
    anzahl_entnommener_stanzen=...,
    anzahl_befallener_stanzen=...,
    maximaler_anteil_befallener_stanzen=...,
    calculation_details=[...] or None,
    tnm_nach='...' or None,
    ptnm='...' or None,
    lymphgefaessinvasion=...,
    veneninvasion=...,
    perineurale_invasion=...,
    lk_befallen_untersucht='...' or None,
    r_klassifikation=...,
    resektionsrand='...' or None
)
```

All numeric values must be plain Python numbers (int or float). Strings must be quoted with single quotes. Use `None` (capital N) for missing values.

Follow these rules precisely; the evaluator will compare your output against the expected schema and will penalize any deviation. Good luck!
````

---

## Iteration 13 — extract_custom.predict

**Score:** 2.769234234234234

````
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
  tumor_prozent: float?,       # (tumorausdehnung_mm / (laenge_cm * 10)) * 100, rounded to two decimals (null if tumor=False)
  gleason: str?                # Gleason string from the “Begutachtung” line (null if tumor=False)
}
````

---

## Iteration 18 — extract_custom.predict

**Score:** 0.8620964627819049

````
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
6. **Microscopic data (`begutachtung_liste`)**  
   - For each core (or for the whole resection) create an entry:  
     * `nr` = core number (or 1 for resection).  
     * `tumor` = **True** if any malignant description appears (e.g., “Adenokarzinom”, “Malignom”, “Krebs”). Otherwise **False**.  
     * `tumorausdehnung_mm` – if the report gives a size in mm (e.g., “12 mm”) for that core, store it.  
     * `tumor_prozent` – if a percentage of tumor involvement is given (e.g., “30 %”), store it as a float.  
     * `gleason` – copy the exact Gleason notation found (e.g., “4 + 5 = 9”).  
7. **Derived counts**  
   - `anzahl_entnommener_stanzen` = sum of all `gesamt_stanzen` across `makroskopie_liste`.  
   - `anzahl_befallener_stanzen` = count of `begutachtung_liste` items where `tumor` is True.  
   - `maximaler_anteil_befallener_stanzen` – if a per‑core tumor percentage is available, compute the maximum; otherwise leave null.  
8. **Tumor presence (`tumornachweis`)** – “Ja” if any `tumor=True`; otherwise “Nein”.  
9. **Massgeblicher Befund (`massgeblicher_befund`)** – set to `Ja` for the final report (the document you are processing). If the report is explicitly marked as a “Zwischenbefund” or “Vorläufig”, set to `Nein`.  
10. **Classification fields** (populate when present)  
    - `icdo3_histologie` – value after “ICD‑O‑3:”.  
    - `lokalisation_icd10` – value after “ICD‑10‑GM‑2024:” (strip any trailing text).  
    - `who_isup_grading` – map “WHO/ISUP‑Graduierungsgruppe X” to the corresponding enum value.  
    - `gleason` – overall Gleason score if given in the “Klassifikation” block (use the same string as in the microscopic entry).  
    - `tnm_nach` – the staging system name, e.g., “UICC”.  
    - `ptnm` – the pTNM string (e.g., “pT3b pN1”).  
    - `lymphgefaessinvasion`, `veneninvasion`, `perineurale_invasion` – map the presence of “L1”, “V1”, “Pn1” (or their textual equivalents) to the corresponding enum values; if absent, leave null.  
    - `lk_befallen_untersucht` – format “positive/total” (e.g., “5/18”) when lymph‑node data are given.  
    - `r_klassifikation` – map “R0”, “R1”, “R2” etc. to the enum.  
    - `resektionsrand` – if the margin description contains “multifokal”, “fokal”, “apikal” etc., map to the appropriate enum value; otherwise null.  
11. **Optional fields** – any field not found in the text must be set to `None` (or omitted in the enum representation).  

**Output Format**
Your answer must contain **two sections**:

1. **reasoning** – a concise paragraph (or a few bullet points) that explains how each extracted value was obtained, referencing the exact section or line of the source text when helpful.

2. **extracted** – a literal Python‑style construction of the `ProstataPatient` object, exactly as shown in the examples:
   ```
   patientenid='E/2025/055678' befunde=[ProstatapatientItem(
       untersuchtes_praeparat=<ProstatapatientitemUntersuchtesPraeparat.Resektat: 'Resektat'>,
       histologiedatum='28.10.2025',
       befundausgang='28.10.2025',
       ort='München',
       pathologisches_institut='Pathologisches Institut der LMU',
       einsendenummer='E/2025/055678',
       massgeblicher_befund=<ProstatapatientitemMassgeblicherBefund.Ja: 'Ja'>,
       tumornachweis=<ProstatapatientitemTumornachweis.Ja: 'Ja'>,
       icdo3_histologie='8140/3',
       who_isup_grading=<ProstatapatientitemWhoIsupGrading.Graduierungsgruppe 5: 'Graduierungsgruppe 5'>,
       gleason='4 + 5 = 9',
       makroskopie_liste=[ProstatapatientitemItem(nr=1, gesamt_stanzen=1, laenge_cm=5.9)],
       begutachtung_liste=[ProstatapatientitemItem(nr=1, tumor=True, tumorausdehnung_mm=None, tumor_prozent=None, gleason='4 + 5 = 9')],
       art_der_biopsie=None,
       entnahmestelle_der_biopsie=None,
       lokalisation_icd10='C61',
       anzahl_entnommener_stanzen=1,
       anzahl_befallener_stanzen=1,
       maximaler_anteil_befallener_stanzen=None,
       calculation_details=None,
       tnm_nach='UICC',
       ptnm='pT3b pN1',
       lymphgefaessinvasion=<ProstatapatientitemLymphgefaessinvasion.L1: 'L1'>,
       veneninvasion=<ProstatapatientitemVeneninvasion.V1: 'V1'>,
       perineurale_invasion=<ProstatapatientitemPerineuraleInvasion.Pn1: 'Pn1'>,
       lk_befallen_untersucht='5/18',
       r_klassifikation=<ProstatapatientitemRKlassifikation.R1: 'R1'>,
       resektionsrand=<ProstatapatientitemResektionsrand.multifokal: 'multifokal'>)]
   ```

**Special Cases & Tips**
- Numbers in German use commas as decimal separators (e.g., “1,5 cm”). Convert them to standard Python floats (1.5).  
- The same core may be described in both “Makroskopie” and “Mikroskopie”; ensure the `nr` values line up.  
- If the report lists a range of cores (e.g., “1‑15”) with a single statement, apply that statement to each core in the range.  
- When the “Klinische Angaben” mention a Gleason score that differs from the one in “Klassifikation”, prefer the value in “Klassifikation”.  
- For prostatectomy specimens, fields that are biopsy‑specific (`art_der_biopsie`, `entnahmestelle_der_biopsie`) must be left as `None`.  
- The enum names and values must match exactly the ones used in the examples (e.g., `ProstatapatientitemLymphgefaessinvasion.L1`).  

**Goal**
Produce a complete, accurate, and syntactically correct representation of the report according to the schema, with clear reasoning that demonstrates how each piece of data was derived. This instruction should enable the assistant to handle any future German prostate pathology report, even when wording or ordering varies.
````

---

## Iteration 22 — extract_custom.predict

**Score:** 0.8620964627819049

````
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
   - `gesamt_stanzen` = 1 unless the line explicitly states a count (e.g. “Drei …” → 3). |
| **begutachtung_liste** | One entry per core number (1‑N).  
   - `nr` = core number.  
   - `tumor` = `True` if the core appears in the *Begutachtung* tumor list, else `False`.  
   - `gleason` = Gleason string for tumor‑positive cores, else `null`.  
   - `tumorausdehnung_mm` = tumor length in mm from *Begutachtung* (numeric).  
   - `tumor_prozent` = **computed** (see below) for tumor‑positive cores, else `null`. |
| **anzahl_entnommener_stanzen** | Sum of `gesamt_stanzen` over all `makroskopie_liste` entries. |
| **anzahl_befallener_stanzen** | Count of entries in `begutachtung_liste` where `tumor=True`. |
| **maximaler_anteil_befallener_stanzen** | For each tumor‑positive core compute:  

```
percentage = (tumorausdehnung_mm) / (laenge_cm * 10 * gesamt_stanzen) * 100
```

Round to the nearest integer. Store the **maximum** of these percentages. If no tumor, set `null`. |
| **calculation_details** | List of strings `"Sample X: Y mm / Z mm = P%"` for every tumor‑positive core, using the same numbers as above (rounded to whole percent). |
| **tnm_nach** | If a pTNM string is present (e.g. “pTNM (UICC, 8. Auflage): pT3b pN1”) → `UICC`. If no pTNM → `null`. |
| **ptnm** | The pT and pN part of the pTNM string (e.g. `pT3b pN1`). |
| **lymphgefaessinvasion** | `L1` if the pTNM string contains “L1” **or** the text explicitly mentions “Lymphgefaessinvasion: nachweisbar”. Otherwise `L0` or `null`. |
| **veneninvasion** | `V1` if pTNM contains “V1” or text mentions “venöse Invasion: nachweisbar”. |
| **perineurale_invasion** | `Pn1` if pTNM contains “Pn1” or text mentions “Perineurale Invasion: nachweisbar”. |
| **lk_befallen_untersucht** | If the report says “X untersuchte … davon Y mit Metastasen” → `"Y/X"` (e.g. `5/18`). |
| **r_klassifikation** | `R1`/`R0` from the pTNM string (e.g. “… R1”). |
| **resektionsrand** | If the pTNM or text mentions “multifokal positiv” → `multifokal`; if “tumorfrei” → `tumorfrei`; otherwise `null`. |

# General Extraction Strategy
1. **Pre‑process** the text: split into lines, trim whitespace, keep original numbering.  
2. **Identify sections** by their headings (`Journalnummer`, `Übersandtes Material`, `Klinische Angaben`, `Makroskopie`, `Mikroskopie`, `Begutachtung`, `Klassifikation`).  
3. **Parse header** for institute, city, journal number, dates.  
4. **Parse Makroskopie**: regex `(\d+)\.\s+Ein\s+([^\s]+)\s+messende\s+Stanzzylinder` to get length; detect a leading word indicating count (`Ein`, `Drei`, `Zwei` etc.) – map German cardinal words to integers (Ein = 1, Zwei = 2, Drei = 3, …). If no explicit count, use 1.  
5. **Parse Begutachtung**: locate lines that start with a core number followed by a period and a Gleason description. Extract core number, Gleason pattern, WHO grade, tumor length (`Längsausdehnung X,0 mm`).  
6. **Build macroscopic list** first, then **microscopic list** using the core numbers from Begutachtung (tumor = True) and marking all other numbers as tumor = False.  
7. **Compute derived fields** (counts, percentages, max percentage, calculation details).  
8. **Parse Klassifikation** for ICD‑10, ICD‑O‑3, pTNM, lymph‑node status, invasion flags, resection margin. Use simple regexes:  
   - `ICD‑10‑GM‑2024:\s*([A-Z]\d{2})`  
   - `ICD‑O‑3:\s*([\d/]+)`  
   - `pTNM.*?:\s*([^\n]+)`  
   - `(\d+)\s+untersuchte.*?(\d+)\s+mit Metastasen` for lymph nodes.  
9. **Enum mapping**: convert raw strings to the exact enum member names (e.g. `"Graduierungsgruppe 5"` → `ProstatapatientitemWhoIsupGrading.Graduierungsgruppe 5`).  
10. **Finalize** the `ProstatapatientItem` object, ensuring every field appears in the order defined by the schema (the order does not affect JSON validity but aids readability).  

# Edge Cases & Defaults
- If a macroscopic entry lists a length but no explicit count, assume `gesamt_stanzen = 1`.  
- If a biopsy report does **not** specify a biopsy type, leave `art_der_biopsie = null`.  
- If the pTNM line is missing, set all TNM‑related fields (`tnm_nach`, `ptnm`, `lymphgefaessinvasion`, `veneninvasion`, `perineurale_invasion`, `r_klassifikation`, `resektionsrand`) to `null`.  
- If no tumor is found, set `tumornachweis = Nein`, `gleason = null`, `who_isup_grading = null`, `maximaler_anteil_befallener_stanzen = null`, `calculation_details = []`.  
- When multiple cores share the same macroscopic length (e.g., “Drei … 1,4 cm”), treat the denominator for each core as `1.4 cm * 10 mm * 3` when computing percentages.  

# Output Format Example
```python
patientenid='E/2025/055678' 
befunde=[ProstatapatientItem(
    untersuchtes_praeparat=ProstatapatientitemUntersuchtesPraeparat.Resektat,
    histologiedatum='28.10.2025',
    befundausgang='28.10.2025',
    ort='München',
    pathologisches_institut='Pathologisches Institut der LMU',
    einsendenummer='E/2025/055678',
    massgeblicher_befund=ProstatapatientitemMassgeblicherBefund.Ja,
    tumornachweis=ProstatapatientitemTumornachweis.Ja,
    icdo3_histologie='8140/3',
    who_isup_grading=ProstatapatientitemWhoIsupGrading.Graduierungsgruppe_5,
    gleason='4 + 5 = 9',
    makroskopie_liste=[ProstatapatientitemItem(nr=1, gesamt_stanzen=1, laenge_cm=5.9)],
    begutachtung_liste=[ProstatapatientitemItem(nr=1, tumor=True, tumorausdehnung_mm=None, tumor_prozent=None, gleason='4 + 5 = 9')],
    art_der_biopsie=None,
    entnahmestelle_der_biopsie=None,
    lokalisation_icd10='C61',
    anzahl_entnommener_stanzen=1,
    anzahl_befallener_stanzen=1,
    maximaler_anteil_befallener_stanzen=None,
    calculation_details=None,
    tnm_nach='UICC',
    ptnm='pT3b pN1',
    lymphgefaessinvasion=ProstatapatientitemLymphgefaessinvasion.L1,
    veneninvasion=ProstatapatientitemVeneninvasion.V1,
    perineurale_invasion=ProstatapatientitemPerineuraleInvasion.Pn1,
    lk_befallen_untersucht='5/18',
    r_klassifikation=ProstatapatientitemRKlassifikation.R1,
    resektionsrand=ProstatapatientitemResektionsrand.multifokal
)]
```

Follow **exactly** these rules for every new report you process. Ensure no field is omitted and that enum values match the schema definitions verbatim.```text
# Task: Extract a ProstataPatient record from a German prostate pathology report

## 1. Overall Goal
Read the supplied pathology report and output **one** Python‑style literal that conforms to the `ProstataPatient` schema:

```
patientenid='<Journalnummer>'
befunde=[ProstatapatientItem(... )]
```

All fields defined in the schema must be present. If a value cannot be derived, set the field to `null`. Use the exact enum members from the schema (e.g. `ProstatapatientitemUntersuchtesPraeparat.Resektat`, `ProstatapatientitemArtDerBiopsie.Fusionsbiopsie`, …).

## 2. Sections to Identify
The report follows a fairly fixed order. Locate each heading (case‑insensitive) and treat the text that follows until the next heading as that section:

| Heading (German)                | Purpose |
|--------------------------------|---------|
| `Journalnummer`                | Unique case identifier (patientenid & einsendenummer). |
| `Eingang` / `Ausgang`          | Dates – use `Ausgang` as `histologiedatum` and `befundausgang`. |
| `Pathologisches Institut` / `Urologische Klinik` | Institute name and city (city is the last word after “|”). |
| `Klinische Angaben`            | May contain biopsy type (`Fusionsbiopsie`, `Stanzbiopsie`). |
| `Makroskopie`                  | One line per specimen: length in cm, optional core count. |
| `Mikroskopie`                  | Usually just a list of core numbers; not needed for data extraction. |
| `Begutachtung`                 | Lists tumor‑positive cores with Gleason, WHO/ISUP grade, tumor length (mm). |
| `Klassifikation`               | ICD‑10, ICD‑O‑3, optional pTNM (UICC), lymph‑node status, invasion flags, resection margin. |

## 3. Extraction Rules

### 3.1 Header / Identifiers
| Schema field | Extraction |
|--------------|------------|
| `patientenid` / `einsendenummer` | Text after `Journalnummer` (e.g. `E/2025/055678`). |
| `ort` | City after the pipe `|` in the institute address line (e.g. `München`). |
| `pathologisches_institut` | First line of the institute header (full text). |
| `histologiedatum` | Date after `Ausgang` (format `DD.MM.YYYY`). |
| `befundausgang` | Same as `histologiedatum` (or `null` if missing). |
| `massgeblicher_befund` | Always `Ja` for a final pathology report. |
| `untersuchtes_praeparat` | `Resektat` if the clinical note mentions a surgical specimen (e.g. “Radikale Prostatektomie”). Otherwise `Biopsie`. |
| `art_der_biopsie` | From `Klinische Angaben`: `Fusionsbiopsie` → `Fusionsbiopsie`; `Stanzbiopsie` → `Stanzbiopsie`; else `null`. |
| `entnahmestelle_der_biopsie` | For all biopsy reports use `Primärtumor`. |
| `tumornachweis` | `Ja` if any core is tumor‑positive in *Begutachtung*; otherwise `Nein`. |
| `icdo3_histologie` | Regex `ICD‑O‑3:\s*([^\s]+)`. |
| `lokalisation_icd10` | Regex `ICD‑10‑GM‑2024:\s*([A-Z]\d{2})`. |
| `who_isup_grading` | Take the **highest** WHO/ISUP grade mentioned in *Begutachtung* and map to the enum (`Graduierungsgruppe 1` … `Graduierungsgruppe 5`). |
| `gleason` | Highest Gleason sum (pattern 1 + pattern 2) from *Begutachtung* (e.g. `5 + 5 = 10`). |
| `tnm_nach` | If a pTNM line exists (contains “pTNM” and “UICC”) → `UICC`; else `null`. |
| `ptnm` | The pT and pN part of the pTNM line (e.g. `pT3b pN1`). |
| `lymphgefaessinvasion` | `L1` if pTNM contains “L1” **or** the text explicitly says “Lymphgefaessinvasion: nachweisbar”; else `L0` or `null`. |
| `veneninvasion` | `V1` if pTNM contains “V1” **or** text says “venöse Invasion: nachweisbar”. |
| `perineurale_invasion` | `Pn1` if pTNM contains “Pn1” **or** text says “Perineurale Invasion: nachweisbar”. |
| `lk_befallen_untersucht` | Regex `(\d+)\s+untersuchte.*?(\d+)\s+mit Metastasen` → `"Y/X"` where X = examined, Y = positive. |
| `r_klassifikation` | `R1`/`R0` from pTNM (e.g. “… R1”). |
| `resektionsrand` | If pTNM or text mentions “multifokal positiv” → `multifokal`; if “tumorfrei” → `tumorfrei`; else `null`. |

### 3.2 Makroskopie (gross description)
For each numbered line:

1. Extract the line number `nr`.
2. Extract the length in cm (`(\d+,\d+)` → replace comma with dot, convert to float) → `laenge_cm`.
3. Detect an explicit core count:
   - If the line starts with a German cardinal word (`Ein`, `Zwei`, `Drei`, `Vier`, `Fünf`, …) followed by “… Stanzzylinder”, map the word to an integer (`Ein` = 1, `Zwei` = 2, `Drei` = 3, …).  
   - If the line contains “Drei bis …” treat it as 3 cores.  
   - If no count is mentioned, set `gesamt_stanzen = 1`.
4. Create a `ProstatapatientitemItem` with fields `nr`, `gesamt_stanzen`, `laenge_cm`.

### 3.3 Begutachtung (microscopic evaluation)
1. Locate lines that start with a core number followed by a period and a Gleason description, e.g.  
   `3. Gleason 4 + 4 = 8, WHO‑Graduierungsgruppe 4, Längsausdehnung 9,0 mm.`  
2. For each such line:
   - `nr` = core number.
   - `tumor = True`.
   - `gleason` = exact Gleason string (`4 + 4 = 8`).
   - `tumorausdehnung_mm` = numeric value before “mm” (replace comma with dot).
3. For every core number **not** listed as tumor‑positive, create an entry with `tumor = False` and all other fields `null`.
4. Assemble all entries (ordered by `nr`) into `begutachtung_liste`.

### 3.4 Derived Counts & Percentages
* `anzahl_entnommener_stanzen` = Σ `gesamt_stanzen` over **all** `makroskopie_liste` entries.  
* `anzahl_befallener_stanzen` = count of `begutachtung_liste` entries where `tumor=True`.  
* For each tumor‑positive core compute:

```
core_length_mm = laenge_cm * 10 * gesamt_stanzen   # total length of that core(s)
percentage = (tumorausdehnung_mm / core_length_mm) * 100
rounded = int(round(percentage))
```

* `maximaler_anteil_befallener_stanzen` = maximum of the rounded percentages (or `null` if no tumor).  
* `calculation_details` = list of strings  
  `"Sample X: Y mm / Z mm = P%"`  
  where X = core number, Y = tumor length, Z = `core_length_mm`, P = rounded percentage.

### 3.5 Handling Missing / Optional Data
- If a section (e.g., `Klassifikation` pTNM) is absent, set all related fields to `null`.  
- If no tumor is found, set `tumornachweis = Nein`, `gleason = null`, `who_isup_grading = null`, `maximaler_anteil_befallener_stanzen = null`, `calculation_details = []`.  
- If a macroscopic line mentions a length but no explicit count, assume `gesamt_stanzen = 1`.  
- If the biopsy type cannot be inferred, leave `art_der_biopsie = null`.  

## 4. Enum Mapping (exact names)
| Raw text | Enum member |
|----------|-------------|
| `Resektat` | `ProstatapatientitemUntersuchtesPraeparat.Resektat` |
| `Biopsie` | `ProstatapatientitemUntersuchtesPraeparat.Biopsie` |
| `Fusionsbiopsie` | `ProstatapatientitemArtDerBiopsie.Fusionsbiopsie` |
| `Stanzbiopsie` | `ProstatapatientitemArtDerBiopsie.Stanzbiopsie` |
| `Primärtumor` | `ProstatapatientitemEntnahmestelleDerBiopsie.Primärtumor` |
| `Ja` / `Nein` | `ProstatapatientitemMassgeblicherBefund.Ja` etc. |
| WHO grades | `ProstatapatientitemWhoIsupGrading.Graduierungsgruppe 1` … `Graduierungsgruppe 5` |
| Lymph‑vascular invasion | `ProstatapatientitemLymphgefaessinvasion.L0` / `L1` |
| Venous invasion | `ProstatapatientitemVeneninvasion.V0` / `V1` |
| Perineural invasion | `ProstatapatientitemPerineuraleInvasion.Pn0` / `Pn1` |
| R‑classification | `ProstatapatientitemRKlassifikation.R0` / `R1` |
| Resection margin | `ProstatapatientitemResektionsrand.multifokal`, `tumorfrei` etc. |
| Tumor detection | `ProstatapatientitemTumornachweis.Ja` / `Nein` |
| Massgeblicher Befund | `ProstatapatientitemMassgeblicherBefund.Ja` / `Nein` |

## 5. Output Formatting
Produce a **single** line (or multi‑line for readability) that exactly matches the Python literal syntax shown in the examples. Do **not** add any extra text, markdown fences, or explanations.

### Example Skeleton
```python
patientenid='E/2025/055678' 
befunde=[ProstatapatientItem(
    untersuchtes_praeparat=ProstatapatientitemUntersuchtesPraeparat.Resektat,
    histologiedatum='28.10.2025',
    befundausgang='28.10.2025',
    ort='München',
    pathologisches_institut='Pathologisches Institut der LMU',
    einsendenummer='E/2025/055678',
    massgeblicher_befund=ProstatapatientitemMassgeblicherBefund.Ja,
    tumornachweis=ProstatapatientitemTumornachweis.Ja,
    icdo3_histologie='8140/3',
    who_isup_grading=ProstatapatientitemWhoIsupGrading.Graduierungsgruppe_5,
    gleason='4 + 5 = 9',
    makroskopie_liste=[ProstatapatientitemItem(nr=1, gesamt_stanzen=1, laenge_cm=5.9)],
    begutachtung_liste=[ProstatapatientitemItem(nr=1, tumor=True, tumorausdehnung_mm=None, tumor_prozent=None, gleason='4 + 5 = 9')],
    art_der_biopsie=None,
    entnahmestelle_der_biopsie=None,
    lokalisation_icd10='C61',
    anzahl_entnommener_stanzen=1,
    anzahl_befallener_stanzen=1,
    maximaler_anteil_befallener_stanzen=None,
    calculation_details=None,
    tnm_nach='UICC',
    ptnm='pT3b pN1',
    lymphgefaessinvasion=ProstatapatientitemLymphgefaessinvasion.L1,
    veneninvasion=ProstatapatientitemVeneninvasion.V1,
    perineurale_invasion=ProstatapatientitemPerineuraleInvasion.Pn1,
    lk_befallen_untersucht='5/18',
    r_klassifikation=ProstatapatientitemRKlassifikation.R1,
    resektionsrand=ProstatapatientitemResektionsrand.multifokal
)]
```

Follow **all** the rules above for every new report. Ensure no field is omitted and that enum values match the schema verbatim.
````

---

## Iteration 28 — extract_custom.predict

**Score:** 0.8620964627819049

````
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
    * Identify every core number mentioned in the **Begutachtung** paragraph (or the numbered list in **Mikroskopie**).  
    * For each core create an entry with:
      * `nr` = core number.  
      * `tumor` = `True` if the core is listed as containing carcinoma; otherwise `False`.  
      * `gleason` = the Gleason string that follows the core (if any).  
      * `tumorausdehnung_mm` = the numeric tumor length in millimetres that follows “Längsausdehnung … mm” or “Ausdehnung … mm”.  
      * `tumor_prozent` = **null** at this stage (will be filled in point 13).  
    * If a core is not mentioned in the “Begutachtung” list, assume it is tumor‑free.

12. **Derived Counts**
    * `anzahl_entnommener_stanzen` = sum of all `gesamt_stanzen` from `makroskopie_liste`.  
    * `anzahl_befallener_stanzen` = count of entries in `begutachtung_liste` where `tumor == True`.  

13. **Maximum Tumor‑Core Percentage (`maximaler_anteil_befallener_stanzen`)**
    * For each tumor‑positive core compute  
      `percentage = (tumorausdehnung_mm) / (laenge_cm * 10) * 100`  
      (because 1 cm = 10 mm).  
    * Round the result to the nearest integer.  
    * The field stores the **largest** integer percentage among all tumor‑positive cores.  
    * Populate `calculation_details` with a list of strings of the form  
      `"Nr X: Ymm / Zcm = P%"` (use the original numbers, e.g. `"Nr 3: 9.0mm / 2.1cm = 42%"`).  
    * If no tumor‑positive core has a measurable length, set both fields to `null`.

14. **TNM Staging**
    * Locate the line that starts with “Klassifikation:” or contains “pTNM”.  
    * Extract the staging system after the colon (e.g. `UICC, 8. Auflage`) → store only the system name (`UICC`). → `tnm_nach`.  
    * Extract the pT and pN part (e.g. `pT3b pN1`) → store as `ptnm`.  
    * Do **not** include L, V, Pn, R information here (they have dedicated fields).

15. **Invasion & Margin Flags**
    * `lymphgefaessinvasion` = `L1` if “Lymphgefäßinvasion: nachweisbar” or “L1” appears; otherwise `L0` or `null`.  
    * `veneninvasion` = `V1` if “Venöse Invasion: nachweisbar” or “V1” appears; otherwise `V0` or `null`.  
    * `perineurale_invasion` = `Pn1` if “Perineurale Invasion: nachweisbar” or “Pn1” appears; otherwise `Pn0` or `null`.  
    * `lk_befallen_untersucht` = the “x/y” string from the lymph‑node summary (e.g. `2/22`).  
    * `r_klassifikation` = the R‑value from the classification line (e.g. `R1`).  
    * `resektionsrand` = the descriptive word before the R‑value (e.g. `fokal`, `multifokal`, `diffus`).  
    * If any of these items are missing, set the corresponding field to `null`.

16. **Optional Fields**
    * `art_der_biopsie`, `entnahmestelle_der_biopsie`, `lokalisation_icd10` may be `null` when not applicable.  
    * `calculation_details` may be `null` when no percentage calculation is possible.

**Output Format**
Produce **exactly one** Python‑style assignment statement that creates the variables `patientenid` and `befunde`.  
`befunde` must be a list containing a single `ProstatapatientItem` (or more if the document contains multiple independent reports – in that case create one item per report, preserving the order they appear).  

Example skeleton (fill in all fields, use the enum names exactly as shown in the schema):

```python
patientenid='E/2025/051234'
befunde=[ProstatapatientItem(
    untersuchtes_praeparat=<ProstatapatientitemUntersuchtesPraeparat.Resektat: 'Resektat'>,
    histologiedatum='19.12.2025',
    befundausgang='19.12.2025',
    ort='München',
    pathologisches_institut='Pathologisches Institut der LMU',
    einsendenummer='E/2025/051234',
    massgeblicher_befund=<ProstatapatientitemMassgeblicherBefund.Ja: 'Ja'>,
    tumornachweis=<ProstatapatientitemTumornachweis.Ja: 'Ja'>,
    icdo3_histologie='8140/3',
    who_isup_grading=<ProstatapatientitemWhoIsupGrading.Graduierungsgruppe 4: 'Graduierungsgruppe 4'>,
    gleason='4 + 4 = 8',
    makroskopie_liste=[
        ProstatapatientitemItem(nr=1, gesamt_stanzen=1, laenge_cm=7.2)
    ],
    begutachtung_liste=[
        ProstatapatientitemItem(nr=1, tumor=True, tumorausdehnung_mm=None, tumor_prozent=None, gleason='4 + 4 = 8')
    ],
    art_der_biopsie=None,
    entnahmestelle_der_biopsie=None,
    lokalisation_icd10='C61',
    anzahl_entnommener_stanzen=1,
    anzahl_befallener_stanzen=1,
    maximaler_anteil_befallener_stanzen=None,
    calculation_details=None,
    tnm_nach='UICC',
    ptnm='pT3b pN1',
    lymphgefaessinvasion=<ProstatapatientitemLymphgefaessinvasion.L1: 'L1'>,
    veneninvasion=<ProstatapatientitemVeneninvasion.V0: 'V0'>,
    perineurale_invasion=<ProstatapatientitemPerineuraleInvasion.Pn1: 'Pn1'>,
    lk_befallen_untersucht='2/22',
    r_klassifikation=<ProstatapatientitemRKlassifikation.R1: 'R1'>,
    resektionsrand=<ProstatapatientitemResektionsrand.fokal: 'fokal'>
)]
```

**Important**
- Do **not** add any explanatory text, comments, or markdown wrappers around the output.  
- Every enum field must use the exact enum reference syntax shown above.  
- Preserve the order of list items (`makroskopie_liste` and `begutachtung_liste`) by their numeric identifiers.  
- If any required field cannot be determined, set it to `null` (without quotes).  

Follow these rules precisely to ensure the extracted data validates against the `ProstataPatient` schema.
````

---

## Iteration 30 — extract_custom.predict

**Score:** 0.9421768707482994

````
# 📋  Detailed Instruction for Extracting Prostate‑Biopsy Data  

You will receive **one or more complete German pathology reports** that describe prostate tissue biopsies (e.g. “Biopsie”, “Fusionsbiopsie”, “Stanzbiopsie”).  
Your job is to **parse every report** and produce a **single JSON‑compatible Python‑like object** that follows the **ProstataPatient** schema described below.  

The output must be **exactly** as specified – field names, data types, enum values and rounding rules are strict.  
If any required information is missing in the source text, use the defaults or `null` as described.

---

## 1️⃣  Overall Output Structure  

```python
{
  "patientenid": str,                     # the Journalnummer (e.g. "E/2026/012345")
  "befunde": [                            # list – one entry per report in the input
    ProstatapatientItem,                  # see schema below
    ...
  ]
}
```

### 1.1  ProstatapatientItem (one per report)

| Field | Type | Required? | How to obtain / default |
|-------|------|-----------|------------------------|
| `untersuchtes_praeparat` | enum `"Biopsie"` | ✅ | Always the literal string `"Biopsie"` (these reports are only biopsies). |
| `histologiedatum` | `str` `"dd.mm.yyyy"` | ✅ | Value of **Ausgang** (report finalisation date). |
| `befundausgang` | `str` `"dd.mm.yyyy"` | ✅ | Same as `histologiedatum`. |
| `ort` | `str` | ✅ | City from the header line that contains a pipe `|` (e.g. `"München"`). |
| `pathologisches_institut` | `str` | ✅ | Institution name from the very first line of the header (e.g. `"Pathologisches Institut der LMU"`). |
| `einsendenummer` | `str` | ✅ | Same as `patientenid` (the Journalnummer). |
| `massgeblicher_befund` | enum `"Ja"` | ✅ | Always `"Ja"` for these definitive reports. |
| `tumornachweis` | enum `"Ja"` / `"Nein"` | ✅ | `"Ja"` if **any** core in this report is tumor‑positive, otherwise `"Nein"`. |
| `icdo3_histologie` | `str` | ✅ | Value after the literal text `ICD‑O‑3:` (e.g. `"8140/3"`). |
| `who_isup_grading` | enum `"Graduierungsgruppe X"` (X = 1‑5) | ✅ | Highest WHO‑/ISUP grading group mentioned in the **Begutachtung** section. |
| `gleason` | `str` | ✅ | Highest Gleason score string found (e.g. `"5 + 5 = 10"`). |
| `makroskopie_liste` | list of **Item** (see 1.2) | ✅ | One entry for each line in the **Makroskopie** section (normally 15). |
| `begutachtung_liste` | list of **Item** (see 1.3) | ✅ | One entry for each core (1‑15). |
| `art_der_biopsie` | enum `"Fusionsbiopsie"` or `"Stanzbiopsie"` | ✅ | If the phrase **“Fusionsbiopsie”** appears anywhere in the **Klinische Angaben** → `"Fusionsbiopsie"`, otherwise `"Stanzbiopsie"`. |
| `entnahmestelle_der_biopsie` | enum `"Primärtumor"` | ✅ | Default for all prostate biopsies. |
| `lokalisation_icd10` | `str` or `null` | ✅ | Value after `ICD‑10‑GM‑2024:` (e.g. `"C61/G"`). If the line is absent → `null`. |
| `anzahl_entnommener_stanzen` | `int` | ✅ | Sum of `gesamt_stanzen` across **all** `makroskopie_liste` entries. |
| `anzahl_befallener_stanzen` | `int` | ✅ | Count of cores where `tumor` = `True` in `begutachtung_liste`. |
| `maximaler_anteil_befallener_stanzen` | `int` | ✅ | Highest `tumor_prozent` among positive cores, **rounded to the nearest whole percent** (standard rounding, .5 → up). |
| `calculation_details` | list of `str` (optional) | ✅ | Human‑readable strings for each positive core: `"Nr X: {tumorausdehnung_mm} mm / {laenge_cm*10:.1f} mm = {percentage:.2f}%"`. Include **only** positive cores. |
| `tnm_nach` | enum or `null` | ✅ | Set only if the report explicitly contains a TNM‑stage (e.g. “TNM nach pT2c pN0 cM0”). Otherwise `null`. |
| `ptnm` | enum or `null` | ✅ | Same rule as `tnm_nach`. |
| `lymphgefaessinvasion` | enum or `null` | ✅ | Set to `"Ja"` if the phrase **“Lymphgefäßinvasion”** (or similar) appears, otherwise `null`. |
| `veneninvasion` | enum or `null` | ✅ | Set to `"Ja"` if **“Veneninvasion”** appears, otherwise `null`. |
| `perineurale_invasion` | enum or `null` | ✅ | Set to `"Ja"` if **“Perineurale Invasion”** appears anywhere (Mikroskopie, Begutachtung, …), otherwise `null`. |
| `lk_befallen_untersucht` | enum or `null` | ✅ | Set to `"Ja"` if the report states that lymph‑nodes were examined and found positive/negative, otherwise `null`. |
| `r_klassifikation` | enum or `null` | ✅ | Set only if a **R‑Klassifikation** (e.g. “R0”, “R1”) is mentioned. |
| `resektionsrand` | enum or `null` | ✅ | Set only if a resection margin status is given. |

---

## 2️⃣  Sub‑structures  

### 2.1  Item for `makroskopie_liste`

```python
{
  "nr": int,                # core number (1‑15) as listed
  "gesamt_stanzen": int,    # how many tissue pieces were taken from this core
  "laenge_cm": float        # length of the core in centimetres (single value)
}
```

**Parsing rules for the Makroskopie lines**

* The line always starts with the core number followed by a period.  
* The **quantity** is expressed in German words:  

| German word | Integer |
|-------------|---------|
| Ein         | 1 |
| Zwei        | 2 |
| Drei        | 3 |
| Vier        | 4 |
| Fünf        | 5 |
| …           | … |

* The wording can be:  

  * `"Ein 1,6 cm messender Stanzzylinder"` → `gesamt_stanzen = 1`, `laenge_cm = 1.6`  
  * `"Zwei bis 0,9 cm messende Stanzzylinder"` → `gesamt_stanzen = 2`, `laenge_cm = 0.9`  
  * `"Vier bis 1,1 cm messende Stanzzylinder"` → `gesamt_stanzen = 4`, `laenge_cm = 1.1`

* Numbers use a **comma** as decimal separator – convert to a Python `float` (replace `,` with `.`).  

* If the line contains **“bis”**, the number before “bis” is the piece count, the number after “bis” is the length.  
* If the line does **not** contain “bis”, the quantity is always 1 and the number after the first space is the length.

### 2.2  Item for `begutachtung_liste`

```python
{
  "nr": int,                     # core number
  "tumor": bool,                 # True if a Gleason entry exists for this core
  "tumorausdehnung_mm": float?,  # length in mm from the Begutachtung line (null if tumor=False)
  "tumor_prozent": float?,       # (tumorausdehnung_mm / (laenge_cm * 10)) * 100, rounded to 2 decimals (null if tumor=False)
  "gleason": str?                # Gleason string from the Begutachtung line (null if tumor=False)
}
```

**Parsing rules for the Begutachtung lines**

* Each line starts with the core number followed by a period.  
* If the line contains a Gleason pattern (e.g. `Gleason 4 + 3 = 7b`) the core is **tumor‑positive**.  
* Extract:  

  * `tumorausdehnung_mm` – the number after `Längsausdehnung` (comma → dot).  
  * `gleason` – the exact Gleason string **including** the “= …” part (e.g. `"4 + 3 = 7b"`).  
  * `who_isup_grading` – the number after `Graduierungsgruppe` (store later as `"Graduierungsgruppe X"`).  

* Compute `tumor_prozent` with the formula given above, using the **corresponding** `laenge_cm` from `makroskopie_liste`.  
  * Round to **two decimal places** (`round(value, 2)`).  
* If a core has **no** Gleason line (or the line says “ohne Atypien”), set `tumor = False` and all other fields to `null`.  

### 2.3  Determining the Highest Gleason & WHO‑grading  

* Scan all positive cores.  
* **Highest Gleason** = the one with the greatest numeric sum (first number + second number).  
  * If two scores have the same sum, pick the one with the higher **first** number.  
* **Highest WHO‑grading** = the greatest numeric value after `Graduierungsgruppe`.  
* Store them as strings exactly as they appear in the source (e.g. `"Graduierungsgruppe 5"`).

---

## 3️⃣  General Parsing Guidelines  

1. **Section detection** – The report always contains the headings (case‑insensitive):  
   * `Journalnummer`, `Eingang`, `Ausgang`  
   * `Klinische Angaben`  
   * `Makroskopie`  
   * `Mikroskopie` (optional for invasion info)  
   * `Begutachtung`  
   * `Klassifikation`  

2. **Line breaks & whitespace** – Strip leading/trailing spaces. Treat a line that ends with a period as complete.  

3. **Number conversion** – Replace commas with dots before converting to `float`.  

4. **Multiple reports** – If the input contains more than one complete set of the sections above, treat each set as a separate report and create a separate `ProstatapatientItem`. The order of items in `befunde` must follow the order they appear in the document.  

5. **Enums** – Output enum fields as **plain strings** exactly matching the values shown in the tables (e.g. `"Ja"`, `"Nein"`, `"Stanzbiopsie"`, `"Fusionsbiopsie"`, `"Graduierungsgruppe 3"`).  

6. **Null handling** – For any optional field that is not explicitly mentioned, output the literal `null` (without quotes).  

7. **Rounding** –  
   * `tumor_prozent` → two decimal places (`round(..., 2)`).  
   * `maximaler_anteil_befallener_stanzen` → nearest whole percent (`round(... )`).  

8. **Calculation details** – Build a list of strings **only for cores where `tumor = True`**. Use the exact format:  

   ```
   "Nr {nr}: {tumorausdehnung_mm} mm / {laenge_cm*10:.1f} mm = {tumor_prozent:.2f}%"
   ```

   *Do not* include entries for tumor‑negative cores.  

9. **Invasion fields** – Search the whole report (Mikroskopie, Begutachtung, Klassifikation) for the exact German terms:  
   * `Perineurale Invasion` → `perineurale_invasion = "Ja"`  
   * `Lymphgefäßinvasion` → `lymphgefaessinvasion = "Ja"`  
   * `Veneninvasion` → `veneninvasion = "Ja"`  

   If the term is absent, set the field to `null`.  

10. **TNM / pTNM** – Look for patterns like `TNM nach …` or `pTNM …`. If found, copy the whole string after the keyword as the enum value (keep the original spacing/casing). If not present, set to `null`.  

---

## 4️⃣  Expected Output Example (single report)

```python
{
  "patientenid": "E/2026/012345",
  "befunde": [
    {
      "untersuchtes_praeparat": "Biopsie",
      "histologiedatum": "19.03.2026",
      "befundausgang": "19.03.2026",
      "ort": "München",
      "pathologisches_institut": "Pathologisches Institut der LMU",
      "einsendenummer": "E/2026/012345",
      "massgeblicher_befund": "Ja",
      "tumornachweis": "Ja",
      "icdo3_histologie": "8140/3",
      "who_isup_grading": "Graduierungsgruppe 5",
      "gleason": "5 + 5 = 10",
      "makroskopie_liste": [
        {"nr": 1, "gesamt_stanzen": 2, "laenge_cm": 0.9},
        {"nr": 2, "gesamt_stanzen": 1, "laenge_cm": 1.6},
        ...
        {"nr": 15, "gesamt_stanzen": 4, "laenge_cm": 1.1}
      ],
      "begutachtung_liste": [
        {"nr": 1, "tumor": true, "tumorausdehnung_mm": 4.0, "tumor_prozent": 44.44, "gleason": "4 + 3 = 7b"},
        {"nr": 2, "tumor": true, "tumorausdehnung_mm": 6.0, "tumor_prozent": 37.5,  "gleason": "4 + 4 = 8"},
        ...
        {"nr": 15, "tumor": true, "tumorausdehnung_mm": 6.0, "tumor_prozent": 54.55, "gleason": "4 + 4 = 8"}
      ],
      "art_der_biopsie": "Stanzbiopsie",
      "entnahmestelle_der_biopsie": "Primärtumor",
      "lokalisation_icd10": "C61/G",
      "anzahl_entnommener_stanzen": 22,
      "anzahl_befallener_stanzen": 12,
      "maximaler_anteil_befallener_stanzen": 55,
      "calculation_details": [
        "Nr 1: 4.0 mm / 9.0 mm = 44.44%",
        "Nr 2: 6.0 mm / 16.0 mm = 37.50%",
        ...
        "Nr 15: 6.0 mm / 11.0 mm = 54.55%"
      ],
      "tnm_nach": null,
      "ptnm": null,
      "lymphgefaessinvasion": null,
      "veneninvasion": null,
      "perineurale_invasion": null,
      "lk_befallen_untersucht": null,
      "r_klassifikation": null,
      "resektionsrand": null
    }
  ]
}
````

---

## Iteration 33 — extract_custom.predict

**Score:** 0.8620964627819049

````
# Instruction for extracting a **ProstataPatient** data structure from German prostate pathology reports

You will be given the full plain‑text of a single pathology report (German language) that follows the typical layout used by German university pathology institutes (e.g. “Pathologisches Institut …”, “Journalnummer”, “Eingang”, “Ausgang”, “Makroskopie”, “Mikroskopie”, “Begutachtung”, “Klassifikation”).  
Your task is to **populate every field of the `ProstataPatient` schema** (see the schema definition below) with the information contained in the document, applying the rules and conventions described in this instruction.

---

## 1. Overall structure of the output

Your final answer must contain **two sections**:

1. **`reasoning`** – a short, human‑readable description of how you derived the values (optional but highly recommended for debugging).
2. **`extracted`** – a literal Python‑style representation of the populated data structure, exactly matching the schema (see examples).  
   Use the enum members shown in the schema (e.g. `ProstatapatientitemUntersuchtesPraeparat.Biopsie`).  
   If a field cannot be determined, set it to `None`.  
   Lists (`makroskopie_liste`, `begutachtung_liste`, `calculation_details`) must be present (empty list `[]` if no data).

---

## 2. Schema reference (simplified)

```python
class ProstataPatient:
    patientenid: str
    befunde: List[ProstatapatientItem]

class ProstatapatientItem:
    untersuchtes_praeparat: ProstatapatientitemUntersuchtesPraeparat   # Biopsie | Resektat
    histologiedatum: str                                            # DD.MM.YYYY (Ausgang)
    befundausgang: str                                              # same as histologiedatum
    ort: str                                                        # city from address line
    pathologisches_institut: str
    einsendenummer: str                                             # Journalnummer
    massgeblicher_befund: ProstatapatientitemMassgeblicherBefund   # Ja (default) | Nein
    tumornachweis: ProstatapatientitemTumornachweis               # Ja if any tumor, else Nein
    icdo3_histologie: str                                           # from ICD‑O‑3 line
    who_isup_grading: ProstatapatientitemWhoIsupGrading            # enum value, see below
    gleason: str                                                    # overall (worst) Gleason string
    makroskopie_liste: List[ProstatapatientitemItem]               # one entry per macro line
    begutachtung_liste: List[ProstatapatientitemItem]              # one entry per core (nr)
    art_der_biopsie: Optional[ProstatapatientitemArtDerBiopsie]    # Fusionsbiopsie, Stanzbiopsie, …
    entnahmestelle_der_biopsie: Optional[ProstatapatientitemEntnahmestelleDerBiopsie]
    lokalisation_icd10: Optional[str]                              # e.g. "C61"
    anzahl_entnommener_stanzen: int                                 # sum of gesamt_stanzen
    anzahl_befallener_stanzen: int                                  # count of cores with tumor=True
    maximaler_anteil_befallener_stanzen: Optional[int]             # rounded % of the core with highest % involvement
    calculation_details: Optional[List[str]]                        # human‑readable % calculations
    tnm_nach: Optional[str]                                        # e.g. "UICC"
    ptnm: Optional[str]                                            # e.g. "pT2c pN0"
    lymphgefaessinvasion: Optional[ProstatapatientitemLymphgefaessinvasion]
    veneninvasion: Optional[ProstatapatientitemVeneninvasion]
    perineurale_invasion: Optional[ProstatapatientitemPerineuraleInvasion]
    lk_befallen_untersucht: Optional[str]                           # e.g. "0/12"
    r_klassifikation: Optional[ProstatapatientitemRKlassifikation]  # R0, R1, …
    resektionsrand: Optional[str]                                   # free text if given
```

### Enum values you may need
| Enum | Members (use exactly as shown) |
|------|--------------------------------|
| `ProstatapatientitemUntersuchtesPraeparat` | `Biopsie`, `Resektat` |
| `ProstatapatientitemMassgeblicherBefund` | `Ja`, `Nein` |
| `ProstatapatientitemTumornachweis` | `Ja`, `Nein` |
| `ProstatapatientitemWhoIsupGrading` | `Graduierungsgruppe 1`, `Graduierungsgruppe 2`, `Graduierungsgruppe 3`, `Graduierungsgruppe 4`, `Graduierungsgruppe 5` |
| `ProstatapatientitemArtDerBiopsie` | `Fusionsbiopsie`, `Stanzbiopsie`, `Systematische Biopsie` (any exact wording from the report) |
| `ProstatapatientitemEntnahmestelleDerBiopsie` | `Primärtumor`, `Periprostatik`, `Andere` |
| `ProstatapatientitemLymphgefaessinvasion` | `L0`, `L1` |
| `ProstatapatientitemVeneninvasion` | `V0`, `V1` |
| `ProstatapatientitemPerineuraleInvasion` | `Pn0`, `Pn1` |
| `ProstatapatientitemRKlassifikation` | `R0`, `R1`, `R2` |

---

## 3. Extraction workflow (step‑by‑step)

### 3.1 Basic identifiers
| Document element | Target field | Extraction rule |
|------------------|--------------|-----------------|
| **Journalnummer** (e.g. `E/2026/010567`) | `einsendenummer` & `patientenid` | Use the exact string. |
| **Ausgang** (final date) | `histologiedatum` & `befundausgang` | Keep format `DD.MM.YYYY`. |
| **Eingang** (optional) | ignore unless you need it for other purposes. |
| **Institution header** (first line containing “Pathologisches Institut …”) | `pathologisches_institut` | Full line as appears. |
| **City** – usually part of the address line after “|” (e.g. `80337 München`) | `ort` | Take the city name (`München`). |
| **ICD‑10‑GM‑2024** line (`C61/G`) | `lokalisation_icd10` | Take the code before the slash (`C61`). |
| **ICD‑O‑3** line (`8140/3`) | `icdo3_histologie` | Take the whole code (`8140/3`). |
| **Klassifikation** section – look for `pTNM (UICC, …): …` | `tnm_nach` = “UICC”, `ptnm` = the string after the colon (e.g. `pT2c pN0`). |
| **R‑Klassifikation** (e.g. `R0`) | `r_klassifikation` | Map to enum member. |
| **Lymphknoten** line (`12 untersuchte … (0/12)`) | `lk_befallen_untersucht` | Use the fraction part (`0/12`). |
| **Lymph‑/Venen‑/Perineurale Invasion** lines (`Lymphgefäßinvasion: nicht nachweisbar`) | Map to enum: `L0`/`V0`/`Pn0` if “nicht nachweisbar”, otherwise `L1`/`V1`/`Pn1`. If the line mentions “Perineurale Invasion” without “nicht”, set `Pn1`. |
| **Resektionsrand** – if a line like “Resektionsrand frei” appears, copy the free‑text; otherwise leave `None`. |

### 3.2 Specimen type
- If the **Klinische Angaben** or **Übersandtes Material** contains the word **„Biopsie“**, set `untersuchtes_praeparat = Biopsie`.
- If it contains **„Prostatektomie“, „Resektat“, „Radikale Prostatektomie“**, set `untersuchtes_praeparat = Resektat`.
- For biopsies, determine `art_der_biopsie`:
  - Contains **„Fusionsbiopsie“** → `Fusionsbiopsie`.
  - Contains **„Stanzbiopsie“** or the word **„Stanz“** → `Stanzbiopsie`.
  - Otherwise, if only “Biopsie” → `Systematische Biopsie` (or `None` if uncertain).
- Set `entnahmestelle_der_biopsie` to `Primärtumor` for all biopsy reports unless the text explicitly mentions a different location (e.g. “Periprostatik”). If unclear, use `None`.

### 3.3 Macro‑section (Makroskopie)
Each numbered line after “Makroskopie:” describes one **group of cores**.

Parsing rules:
1. Extract the **core number** (`nr` = the leading integer).
2. Determine **`gesamt_stanzen`**:
   - If the line starts with a number followed by “bis” (e.g. “Zwei bis 0,9 cm …”) → the word before “bis” is the count (German cardinal: *Ein* =1, *Zwei* =2, *Drei* =3, *Vier* =4, *Fünf* =5, *Sechs* =6, *Sieben* =7, *Acht* =8, *Neun* =9, *Zehn* =10).  
   - If the line starts with “Ein” or a plain number without “bis”, set `gesamt_stanzen = 1`.
3. Extract the **length in cm** (`laenge_cm`):
   - The number after “bis” (or after “Ein … messender …”) is the length, using a decimal point (`.`) as separator (replace commas with dots).  
   - Example: “0,9 cm” → `0.9`.
4. Create a `ProstatapatientitemItem` with fields `nr`, `gesamt_stanzen`, `laenge_cm`.  
   - Other fields (`tumor`, `tumorausdehnung_mm`, `tumor_prozent`, `gleason`) are left `None` for now; they will be filled from the microscopic section.

Collect all macro items into `makroskopie_liste`.  
Compute **`anzahl_entnommener_stanzen`** as the sum of all `gesamt_stanzen`.

### 3.4 Micro‑section (Mikroskopie) & Begutachtung
The **Mikroskopie** section lists, for each core number, a description that includes a **tumour length** in mm (e.g. “Längsausdehnung 4,0 mm”).  
The **Begutachtung** section provides the Gleason score (and WHO‑grading) for the same core numbers.

Processing steps for each core number `nr` (1‑N):
1. **Tumor presence**: If the corresponding line in *Mikroskopie* mentions “Längsausdehnung … mm” **or** the Begutachtung line contains a Gleason score, set `tumor = True`; otherwise `False`.
2. **Tumour length** (`tumorausdehnung_mm`): Extract the numeric value (replace commas with dots). If no length is given, keep `None`.
3. **Tumour percentage** (`tumor_prozent`):
   - Find the macro entry with the same `nr`. Use its `laenge_cm` (single length) as the length of **one** core.  
   - Compute `percentage = (tumour_length_mm) / (laenge_cm * 10) * 100`.  
   - Round to two decimal places (keep as float). If either value is missing, keep `None`.
4. **Gleason**: Copy the exact Gleason string from the Begutachtung line (e.g. `"4 + 3 = 7b"`). If no Gleason, keep `None`.
5. Build a `ProstatapatientitemItem` with the fields above and add it to `begutachtung_liste`.

After processing all cores:
- **`tumornachweis`** = `Ja` if any `tumor=True`, else `Nein`.
- **`anzahl_befallener_stanzen`** = count of items with `tumor=True`.
- **Overall Gleason (`gleason`)**: Choose the *worst* (highest numeric sum) Gleason among all tumour‑positive cores. If multiple have the same sum, keep the one with the highest secondary pattern (e.g., 5+4 > 4+5). If no tumour, set `None`.
- **WHO/ISUP grading (`who_isup_grading`)**: Take the enum that matches the *worst* Graduierungsgruppe found in Begutachtung (e.g., “Graduierungsgruppe 5” → `Graduierungsgruppe 5`). If none, `None`.
- **`maximaler_anteil_befallener_stanzen`**: From all cores with a calculated `tumor_prozent`, take the maximum, round to the nearest integer (`int(round(value))`). If no percentages, `None`.
- **`calculation_details`**: For each tumour‑positive core, add a string `"Nr {nr}: {tumour_length_mm}mm / {laenge_cm*10}mm = {percentage:.2f}%"`. If no tumour cores, set `[]` or `None`.

### 3.5 Default / fallback values
- `massgeblicher_befund` is **always** `Ja` unless the report explicitly says it is a preliminary or addendum (rare).  
- If any field cannot be found, use `None` (or an empty list where a list is required).  
- Do **not** fabricate data; only use information present in the document.

### 3.6 Special cases
| Situation | Action |
|-----------|--------|
| **Multiple specimens** (e.g., both a biopsy and a resection in the same report) | Treat each specimen as a separate `ProstatapatientItem`. Use the same `patientenid` but different `einsendenummer` if separate numbers are given; otherwise keep a single item. |
| **Missing macro length** for a core that has tumour length | Skip percentage calculation for that core; set `tumor_prozent = None`. |
| **Perineurale Invasion only mentioned in one core** | Set `perineurale_invasion = Pn1`. If never mentioned, `Pn0`. |
| **Lymph‑/Venen‑invasion described as “nicht nachweisbar”** | Map to `L0`, `V0`, `Pn0`. |
| **ICD‑10 line missing** | Keep `lokalisation_icd10 = None`. |
| **Gleason line without WHO group** | Still record Gleason; WHO grading stays `None` unless found elsewhere. |

---

## 4. Example of a correctly formatted output (shortened)

```python
patientenid='E/2026/010567'
befunde=[
    ProstatapatientItem(
        untersuchtes_praeparat=ProstatapatientitemUntersuchtesPraeparat.Biopsie,
        histologiedatum='06.03.2026',
        befundausgang='06.03.2026',
        ort='München',
        pathologisches_institut='Pathologisches Institut der LMU',
        einsendenummer='E/2026/010567',
        massgeblicher_befund=ProstatapatientitemMassgeblicherBefund.Ja,
        tumornachweis=ProstatapatientitemTumornachweis.Ja,
        icdo3_histologie='8140/3',
        who_isup_grading=ProstatapatientitemWhoIsupGrading.Graduierungsgruppe_2,
        gleason='3 + 4 = 7a',
        makroskopie_liste=[ProstatapatientitemItem(nr=1, gesamt_stanzen=1, laenge_cm=1.7), …],
        begutachtung_liste=[ProstatapatientitemItem(nr=1, tumor=False, …),
                            ProstatapatientitemItem(nr=9, tumor=True, tumorausdehnung_mm=1.5,
                                                    tumor_prozent=8.33, gleason='3 + 4 = 7a'), …],
        art_der_biopsie=ProstatapatientitemArtDerBiopsie.Fusionsbiopsie,
        entnahmestelle_der_biopsie=ProstatapatientitemEntnahmestelleDerBiopsie.Primärtumor,
        lokalisation_icd10='C61',
        anzahl_entnommener_stanzen=15,
        anzahl_befallener_stanzen=2,
        maximaler_anteil_befallener_stanzen=13,
        calculation_details=['Nr 9: 1.5mm / 18.0mm = 8.33%', 'Nr 15: 2.0mm / 16.0mm = 12.5%'],
        tnm_nach=None,
        ptnm=None,
        lymphgefaessinvasion=None,
        veneninvasion=None,
        perineurale_invasion=None,
        lk_befallen_untersucht=None,
        r_klassifikation=None,
        resektionsrand=None
    )
]
````

---

## Iteration 39 — extract_custom.predict

**Score:** 0.8620964627819049

````
**Task Overview**
You are given the full text of a German prostate pathology report.  
Your job is to extract every piece of information required by the **ProstataPatient** data model and output a single Python‑style representation that can be directly used to instantiate the model (as shown in the examples).

**General Extraction Rules**
1. **Identifiers**
   * `Journalnummer` → `patientenid` **and** `einsendenummer`.
2. **Dates**
   * Use the `Ausgang` date as both `histologiedatum` and `befundausgang`.
3. **Institution & Location**
   * `Pathologisches Institut …` → `pathologisches_institut`.
   * The city (e.g., “München”) → `ort`.
4. **Specimen Type**
   * The report always concerns a prostate specimen → `untersuchtes_praeparat = Biopsie`.
   * Determine `art_der_biopsie` from the clinical note:
     * Contains “Stanzbiopsie” → `Stanzbiopsie`
     * Contains “Fusionsbiopsie” → `Fusionsbiopsie`
     * Otherwise default to `Stanzbiopsie`.
   * `entnahmestelle_der_biopsie` is always `Primärtumor` for prostate biopsies.
5. **Massgeblicher Befund**
   * All supplied reports are final pathology reports → `massgeblicher_befund = Ja`.
6. **Tumor Presence**
   * If any core is listed as tumor‑positive in the **Begutachtung** paragraph → `tumornachweis = Ja`, otherwise `Nein`.
7. **ICD / WHO / Gleason**
   * `ICD‑10‑GM‑2024:` line → extract the code (e.g., `C61`). If the line says “keine Kodierung” or is missing → `lokalisation_icd10 = None`.
   * `ICD‑O‑3:` line → `icdo3_histologie` (e.g., `8140/3`). If missing → `None`.
   * `WHO‑Graduierungsgruppe X` → map to `who_isup_grading = Graduierungsgruppe X`. If missing → `None`.
   * The highest Gleason sum appearing in the **Begutachtung** paragraph becomes the overall `gleason` (e.g., `5 + 5 = 10`). If no invasive carcinoma → `None`.
8. **Macroscopic (Makroskopie) List**
   * Each numbered line under **Makroskopie** corresponds to one entry in `makroskopie_liste`.
   * Parse the pattern:
     * “Ein 1,6 cm messender Stanzzylinder” → `gesamt_stanzen = 1`, `laenge_cm = 1.6`.
     * “Drei bis 1,4 cm messende Stanzzylinder” → `gesamt_stanzen = 3`, `laenge_cm = 1.4`.
     * “Zwei bis 0,8 cm messende Stanzzylinder” → `gesamt_stanzen = 2`, `laenge_cm = 0.8`.
   * Numbers may be written as words (Ein, Zwei, Drei, Vier, …) – map them to integers.
   * Store each as `ProstatapatientitemItem(nr=<nr>, gesamt_stanzen=<int>, laenge_cm=<float>)`.
9. **Microscopic (Begutachtung) List**
   * The **Begutachtung** paragraph enumerates which cores contain carcinoma and which are tumor‑free.
   * For every core (1 … N) create an entry:
     * `nr` = core number.
     * `tumor` = `True` if the core appears in the positive list, else `False`.
     * If `tumor = True` extract:
       * Gleason score from the corresponding line in **Begutachtung** (e.g., “Gleason 4 + 4 = 8” → `gleason='4 + 4 = 8'`).
       * Tumor length in mm from the same line (e.g., “Längsausdehnung 9,0 mm” → `tumorausdehnung_mm = 9.0`). If not given → `None`.
       * `tumor_prozent` is **not** directly reported; compute it in step 10.
     * If `tumor = False` set all three fields to `None`.
10. **Derived Quantities**
    * `anzahl_entnommener_stanzen` = sum of `gesamt_stanzen` over **all** entries in `makroskopie_liste`.
    * `anzahl_befallener_stanzen` = count of cores with `tumor = True`.
    * **Maximum tumor‑percentage per core** (`maximaler_anteil_befallener_stanzen`):
      * For each tumor‑positive core:
        * Convert the macroscopic length to millimetres: `laenge_mm = laenge_cm * 10`.
        * If the core contains multiple cylinders (`gesamt_stanzen > 1`), total length = `laenge_mm * gesamt_stanzen`.
        * Compute `percentage = (tumorausdehnung_mm / total_length_mm) * 100`.
        * Round to the nearest integer.
      * The maximum of these percentages becomes `maximaler_anteil_befallener_stanzen`.
      * Store each intermediate calculation as a string in `calculation_details`, e.g. `"Nr 3: 9.0mm / 21.0mm = 42%"`.
    * If there is no tumor, both `maximaler_anteil_befallener_stanzen` and `calculation_details` are `None`.
11. **Optional / Not‑Applicable Fields**
    * For biopsy reports, the following fields are always `None` (unless explicitly present):  
      `tnm_nach`, `ptnm`, `lymphgefaessinvasion`, `veneninvasion`, `perineurale_invasion`, `lk_befallen_untersucht`, `r_klassifikation`, `resektionsrand`, `lokalisation_icd10` (if not given), `icdo3_histologie` (if not given), `who_isup_grading` (if not given), `gleason` (if no invasive carcinoma), `tumor_prozent` (individual core percentages are not stored, only the max is).
12. **Output Formatting**
    * Produce **exactly** one Python‑like assignment:
      ```
      patientenid='<Journalnummer>' befunde=[ProstatapatientItem(...)]
      ```
    * Use the enum names exactly as shown in the examples (e.g., `ProstatapatientitemUntersuchtesPraeparat.Biopsie`, `ProstatapatientitemMassgeblicherBefund.Ja`, etc.).
    * Omit any fields that are `None` **only** if the schema permits omission; otherwise include them with a literal `None`.
    * Preserve the order of fields as in the example outputs.

**Step‑by‑Step Strategy (to be followed by the assistant)**
1. **Parse Header** – locate `Journalnummer`, `Eingang`, `Ausgang`, institution lines, and city.
2. **Detect Biopsy Type** – scan the `Klinische Angaben` line for keywords.
3. **Build Makroskopie List** – iterate over the numbered lines after “Makroskopie:”, translate German number words, extract length and count.
4. **Identify Positive Cores** – from the “Begutachtung:” paragraph, read the list of core numbers that contain carcinoma.
5. **Extract Per‑Core Details** – for each positive core, locate its detailed description line (e.g., “3. Gleason … Längsausdehnung …”) and pull Gleason and length.
6. **Create Begutachtung List** – fill entries for all cores 1…N, inserting the data from step 5 where applicable.
7. **Compute Derived Numbers** – total cores, tumor‑positive cores, percentages, max percentage, and build `calculation_details`.
8. **Assemble Final JSON‑like Object** – fill all required fields, apply defaults, and output the single assignment line.

**Important Domain Knowledge**
* German decimal separator is a comma (`,`) – convert to `.` for numeric values.
* “Ein”, “Zwei”, “Drei”, “Vier”, “Fünf”, “Sechs”, “Sieben”, “Acht”, “Neun”, “Zehn” map to 1‑10.
* “bis” in the macroscopic description indicates the number of cylinders (e.g., “Zwei bis 0,8 cm messende Stanzzylinder” → 2 cylinders of 0.8 cm each).
* Gleason scores may contain suffixes like “a”, “b” (e.g., “7a”, “7b”). Keep the full string after the `=` sign (including the suffix) as the Gleason value.
* WHO‑Graduierungsgruppe numbers correspond directly to the enum values (e.g., “Graduierungsgruppe 5” → `Graduierungsgruppe 5`).
* If the report states “ICD‑10‑GM‑2024: keine Kodierung (kein invasives Malignom)”, set `lokalisation_icd10 = None`.

Follow these rules precisely to ensure the extracted data matches the `ProstataPatient` schema and achieves the highest evaluation score.
````

---

## Iteration 44 — extract_custom.predict

**Score:** 2.770262224710713

````
# Instruction for extracting a **ProstataPatient** record from German prostate pathology reports

You will be given the full plain‑text of a single pathology report (German language) that follows the typical layout used by the Pathologisches Institut der LMU (and similar German pathology labs).  
Your job is to **produce a single JSON object that conforms exactly to the `ProstataPatient` schema** (see the schema definition below).  
All fields must be present; if a field cannot be determined from the report, set its value to `null` (or the enum value `Nein`/`Ja` where appropriate).  

Below is a complete, step‑by‑step guide that you must follow for every report.

---

## 1. Overall structure of the output

```json
{
  "patientenid": "<string>",
  "befunde": [
    {
      "untersuchtes_praeparat": "<enum>",
      "histologiedatum": "<DD.MM.YYYY>",
      "befundausgang": "<DD.MM.YYYY>",
      "ort": "<string>",
      "pathologisches_institut": "<string>",
      "einsendenummer": "<string>",
      "massgeblicher_befund": "<enum>",
      "tumornachweis": "<enum>",
      "icdo3_histologie": "<string>",
      "who_isup_grading": "<enum>",
      "gleason": "<string>",
      "makroskopie_liste": [
        {
          "nr": <int>,
          "gesamt_stanzen": <int>,
          "laenge_cm": <float>
        },
        …
      ],
      "begutachtung_liste": [
        {
          "nr": <int>,
          "tumor": <enum>,
          "tumorausdehnung_mm": <float|null>,
          "tumor_prozent": <float|null>,
          "gleason": "<string|null>"
        },
        …
      ],
      "art_der_biopsie": "<enum>",
      "entnahmestelle_der_biopsie": "<enum>",
      "lokalisation_icd10": "<string|null>",
      "anzahl_entnommener_stanzen": <int>,
      "anzahl_befallener_stanzen": <int>,
      "maximaler_anteil_befallener_stanzen": <int|null>,
      "calculation_details": ["<string>", …],
      "tnm_nach": "<string|null>",
      "ptnm": "<string|null>",
      "lymphgefaessinvasion": "<enum|null>",
      "veneninvasion": "<enum|null>",
      "perineurale_invasion": "<enum|null>",
      "lk_befallen_untersucht": "<enum|null>",
      "r_klassifikation": "<enum|null>",
      "resektionsrand": "<enum|null>"
    }
  ]
}
```

All enum values must match the exact names defined in the schema (see the **Enum reference** section).

---

## 2. Section‑wise extraction strategy

### 2.1 Header (institution & city)
* The first line that contains “Pathologisches Institut” is the **`pathologisches_institut`**.
* The city is the last part of the address line (e.g. “München”). Use it for **`ort`**.

### 2.2 Journal information
* **`patientenid`** and **`einsendenummer`** are the *Journalnummer* (e.g. `E/2026/005789`).
* **`histologiedatum`** and **`befundausgang`** are the *Ausgang* date (format `DD.MM.YYYY`).

### 2.3 Clinical information (determine biopsy type)
Search the **Klinische Angaben** paragraph:
* If the text contains **“Fusionsbiopsie”** → `art_der_biopsie = Fusionsbiopsie`.
* Else if it contains **“Stanzbiopsie”** or “Stanz” → `art_der_biopsie = Stanzbiopsie`.
* Otherwise default to `Stanzbiopsie`.

The same paragraph tells you the **`untersuchtes_praeparat`** – always `Biopsie` for these reports.

### 2.4 Sampling site
Unless the report explicitly mentions a different site (e.g. “nach vorheriger Therapie” etc.), set **`entnahmestelle_der_biopsie` = Primärtumor**.

### 2.5 Massgeblicher Befund
All supplied examples are final pathology reports → set **`massgeblicher_befund = Ja`**. If a report is clearly a *Vorbericht* (e.g., contains “Vorbericht” or “vorläufig”), use `Nein`.

### 2.6 ICD codes
* Look for a line starting with **“Klassifikation:”**.
* Extract the ICD‑10 code after `ICD-10-GM-2024:` – it appears as `C61/G`. Use the part before the slash (`C61`) for **`lokalisation_icd10`**.
* Extract the ICD‑O‑3 morphology code after `ICD-O-3:` (e.g. `8140/3`) for **`icdo3_histologie`**.
* If a line `TNM‑Nach:` or `pTNM:` is present, copy the value to **`tnm_nach`** / **`ptnm`** respectively; otherwise set them to `null`.

### 2.7 WHO‑ISUP grading & overall Gleason
* In the **Begutachtung** section each positive core lists a WHO‑Grading‑Gruppe (e.g. “WHO‑Graduierungsgruppe 4”).  
  – Keep the highest numeric group and map it to the enum `Graduierungsgruppe X` (X = 1‑5).  
* The Gleason sum is the highest “Gleason A + B = C” found. Store the full string (e.g. `5 + 5 = 10`) in **`gleason`**.

### 2.8 Makroskopie list
Parse the **Makroskopie** block line by line (numbers 1‑…):
* Typical line: `1. Ein 1,8 cm messender Stanzzylinder`
  – `nr` = the leading number.
  – `gesamt_stanzen` = **1** (default).
  – `laenge_cm` = the decimal number (comma `,` → dot `.`) as a float.
* Special case: `4. Drei bis 1,4 cm messende Stanzzylinder`
  – `gesamt_stanzen` = **3** (the word “Drei” indicates three cores).
  – `laenge_cm` = the length after the word “bis” (here `1,4` → `1.4`).
* If a line contains “Ein … Stanzzylinder” with no explicit count, assume `gesamt_stanzen = 1`.
* Preserve the order; the list must contain an entry for **every** numbered line.

### 2.9 Begutachtung list
Parse the **Begutachtung** block. For each core number that appears (whether tumor‑positive or not) create an entry:

| Field | How to fill |
|-------|-------------|
| `nr` | core number |
| `tumor` | `Ja` if the core is listed in the Begutachtung paragraph with a Gleason/WHO entry **or** the preceding Mikroskopie paragraph mentions a tumor length for that core; otherwise `Nein`. |
| `tumorausdehnung_mm` | The numeric length in **mm** that follows “Längsausdehnung … mm”. If the core is tumor‑negative, set `null`. |
| `gleason` | The Gleason string from the same line (e.g. `4 + 3 = 7b`). If `tumor = Nein`, set `null`. |
| `tumor_prozent` | **Leave null for now** – it will be calculated in step 2.10. |

If a core number is **not** mentioned anywhere in Begutachtung (e.g., only listed in Makroskopie as tumor‑free), still create an entry with `tumor = Nein` and all other fields `null`.

### 2.10 Derived numeric values

1. **`anzahl_entnommener_stanzen`**  
   Sum of `gesamt_stanzen` over the whole `makroskopie_liste`.

2. **`anzahl_befallener_stanzen`**  
   Count of entries in `begutachtung_liste` where `tumor = Ja`.

3. **`tumor_prozent`** for each tumor‑positive core  
   ```
   tumor_prozent = (tumorausdehnung_mm) / (laenge_cm * 10) * 100
````

---

## Iteration 48 — extract_custom.predict

**Score:** 2.8058558558558557

````
# --------------------------------------------------------------
#  Prostata‑Pathologie‑Extraction – Detailed Assistant Instructions
# --------------------------------------------------------------

## 1.  Goal
You will receive the full plain‑text of one or more German pathology reports that describe prostate needle biopsies (e.g. “Biopsie”, “Fusionsbiopsie”, “Stanzbiopsie”).  
Your task is to **extract every piece of information required by the `ProstataPatient` data model** and output a **single JSON‑compatible Python‑like object** that follows the schema exactly.

The output must contain **all required fields** and **only the allowed enum values**. Optional fields are set to `null` unless they appear explicitly in the text.

---

## 2.  Target Schema (Python‑like)

```python
{
    "patientenid": str,                     # = Journalnummer (E/xxxx/xxxx)
    "befunde": List[ProstatapatientItem]    # one item per report in the document
}
```

`ProstatapatientItem` (all fields required unless marked optional):

```python
{
    "untersuchtes_praeparat": "Biopsie",                     # constant for these reports
    "histologiedatum": "dd.mm.yyyy",                        # = Ausgang‑Datum
    "befundausgang": "dd.mm.yyyy",                          # same as histologiedatum
    "ort": str,                                             # city from header (e.g. "München")
    "pathologisches_institut": str,                         # first line of the header
    "einsendenummer": str,                                  # Journalnummer
    "massgeblicher_befund": "Ja",                           # always “Ja” for definitive reports
    "tumornachweis": "Ja" | "Nein",                        # “Ja” if any core is tumor‑positive
    "icdo3_histologie": str,                                # value after “ICD‑O‑3:”
    "who_isup_grading": str,                                # highest WHO‑Graduierungsgruppe, e.g. "Graduierungsgruppe 3"
    "gleason": str,                                         # highest Gleason string, e.g. "4 + 3 = 7b"
    "makroskopie_liste": List[MakroItem],                   # one entry per line in Makroskopie
    "begutachtung_liste": List[BegItem],                    # one entry per core (1‑15)
    "art_der_biopsie": "Fusionsbiopsie" | "Stanzbiopsie",   # see §4
    "entnahmestelle_der_biopsie": "Primärtumor",            # default for prostate biopsies
    "lokalisation_icd10": str | None,                      # value after “ICD‑10‑GM‑2024:”, else null
    "anzahl_entnommener_stanzen": int,                      # sum of gesamt_stanzen over all makroskopie entries
    "anzahl_befallener_stanzen": int,                       # count of cores with tumor=True
    "maximaler_anteil_befallener_stanzen": int,            # round(max tumor_prozent) to nearest whole percent
    "calculation_details": List[str] | None,               # optional, human‑readable calculations
    # The following fields are set only if explicitly mentioned, otherwise null:
    "tnm_nach": str | None,
    "ptnm": str | None,
    "lymphgefaessinvasion": str | None,
    "veneninvasion": str | None,
    "perineurale_invasion": str | None,
    "lk_befallen_untersucht": str | None,
    "r_klassifikation": str | None,
    "resektionsrand": str | None
}
```

`MakroItem` (used in `makroskopie_liste`):

```python
{
    "nr": int,                # core number (1‑15)
    "gesamt_stanzen": int,    # total tissue pieces taken from this core
    "laenge_cm": float        # length of the core in cm (single value)
}
```

`BegItem` (used in `begutachtung_liste`):

```python
{
    "nr": int,                     # core number
    "tumor": bool,                 # True if a Gleason entry exists for this core
    "tumorausdehnung_mm": float | None,   # length in mm from “Längsausdehnung … mm”
    "tumor_prozent": float | None,        # (tumorausdehnung_mm / (laenge_cm*10))*100, rounded to 2 decimals
    "gleason": str | None                 # Gleason string from the “Begutachtung” line
}
```

**Enum values must match exactly** (including capitalisation and spelling).

---

## 3.  General Extraction Strategy (to be followed for every report)

1. **Split the document into separate reports**  
   - A new report always starts with a line containing the institution name (e.g. “Pathologisches Institut …”) followed by a “Journalnummer” line.  
   - If only one report is present, treat the whole text as one report.

2. **Header parsing** (first block before “Pathologische Begutachtung”):
   - `pathologisches_institut` = first non‑empty line.
   - `ort` = city name that appears after the postal code (pattern `\d{5}\s+([A-Za-zÄÖÜäöüß]+)`).
   - `einsendenummer` = value after “Journalnummer”.
   - `histologiedatum` & `befundausgang` = value after “Ausgang” (format `dd.mm.yyyy`).

3. **Determine `art_der_biopsie`**  
   - Search the **Klinische Angaben** section.  
   - If the exact word **“Fusionsbiopsie”** appears → `art_der_biopsie = "Fusionsbiopsie"`.  
   - Otherwise → `art_der_biopsie = "Stanzbiopsie"`.

4. **Parse `makroskopie_liste`** (section titled **Makroskopie**):
   - Each line starts with a number “n.” followed by a description.  
   - Extract the core number `nr`.  
   - Find **all** occurrences of “Stanzzylinder” (or “Stanzzylinder” synonyms) and read the length in cm (`X, Y` → replace comma with dot).  
   - Count how many “Stanzzylinder” (or “Stanzstücke”) appear → this is `gesamt_stanzen`.  
   - If multiple lengths are listed for the same core, **use the first length** for `laenge_cm` (the specification expects a single value).  
   - Append a `MakroItem` with the extracted values.

5. **Parse `begutachtung_liste`** (section titled **Begutachtung**):
   - Each line begins with the core number followed by a period.  
   - If the line contains the word **“Gleason”**, set `tumor = True`; otherwise `tumor = False`.  
   - When `tumor = True` extract:
     - `gleason` = the exact Gleason string (e.g. “3 + 4 = 7a”). Preserve the suffix (a/b) if present.
     - `tumorausdehnung_mm` = number before “mm” after “Längsausdehnung”. Convert commas to dots.
   - When `tumor = False` set the three optional fields to `None`.

6. **Calculate tumor percentages** for each tumor‑positive core:
   - Retrieve the matching `laenge_cm` from `makroskopie_liste` (same `nr`).  
   - Compute `tumor_prozent = (tumorausdehnung_mm / (laenge_cm * 10)) * 100`.  
   - Round **to two decimal places** (Python `round(value, 2)`).  
   - Store the value in the corresponding `BegItem`.  
   - If `calculation_details` is requested, add a string:  
     `"Nr {nr}: {tumorausdehnung_mm} mm / {laenge_cm} cm = {tumor_prozent}%"`  
     (use the same rounding as above).

7. **Derive aggregate fields**:
   - `anzahl_entnommener_stanzen` = sum of `gesamt_stanzen` over all `MakroItem`s.
   - `anzahl_befallener_stanzen` = count of `BegItem`s where `tumor == True`.
   - `maximaler_anteil_befallener_stanzen` = `int(round(max(tumor_prozent values)))`. If no tumor, set to `0`.
   - `tumornachweis` = `"Ja"` if any `tumor == True`, else `"Nein"`.

8. **WHO‑ISUP grading & Gleason**:
   - Scan all `BegItem`s for the pattern `WHO‑Graduierungsgruppe X`.  
   - Keep the **highest numeric X** → map to string `"Graduierungsgruppe X"` for `who_isup_grading`.  
   - Scan all Gleason strings, keep the **highest overall Gleason score** (compare the numeric sum after the “=” sign; if equal, prefer the one with higher primary pattern). Store that exact string in `gleason`.

9. **ICD codes**:
   - `icdo3_histologie` = text after “ICD‑O‑3:” (strip whitespace).  
   - `lokalisation_icd10` = text after “ICD‑10‑GM‑2024:” (strip). If the line is missing → `null`.

10. **Optional TNM‑related fields**:  
    - Look for exact keywords (case‑insensitive):
      - “TNM nach …”, “pTNM …”, “Lymphgefäßinvasion …”, “Veneninvasion …”, “perineurale Invasion …”, “Lymphknoten befallen …”, “R‑Klassifikation …”, “Resektionsrand …”.
    - If found, capture the value that follows the keyword (up to line end) and assign to the corresponding field.  
    - If not found, set the field to `null`.

11. **Finalize the `ProstatapatientItem`**:
    - Populate all fields as described, ensuring correct Python types (`str`, `int`, `float`, `bool`, `list`, `None`).  
    - The order of keys does **not** matter, but the structure must be exactly as the schema.

12. **Wrap into the top‑level object**:
    ```python
    {
        "patientenid": "<einsendenummer>",
        "befunde": [ <ProstatapatientItem for report 1>, <ProstatapatientItem for report 2>, ... ]
    }
    ```

---

## 4.  Important Formatting Rules

* **Numbers**: German decimal commas (e.g. “1,8 cm”) must be converted to a dot before casting to `float`.
* **Dates**: Keep the original format `dd.mm.yyyy` (do not reformat).
* **Boolean values**: Use Python booleans `True` / `False` inside `BegItem`; the outer enum fields use the strings `"Ja"` / `"Nein"`.
* **Null values**: Represent missing optional fields with the literal `None` (Python null).
* **Lists**: Use standard Python list syntax (`[item1, item2, …]`). Items themselves are Python‑like dicts.
* **Enum strings**: Must match exactly the values listed in the schema (including capitalisation, spaces, and umlauts).
* **Rounding**:
  - `tumor_prozent` → two decimal places.
  - `maximaler_anteil_befallener_stanzen` → nearest whole percent (`int(round(...))`).

---

## 5.  Example (single report)

Given the sample text in the prompt, the correct output is:

```python
{
    "patientenid": "E/2026/007456",
    "befunde": [
        {
            "untersuchtes_praeparat": "Biopsie",
            "histologiedatum": "03.02.2026",
            "befundausgang": "03.02.2026",
            "ort": "München",
            "pathologisches_institut": "Pathologisches Institut der LMU",
            "einsendenummer": "E/2026/007456",
            "massgeblicher_befund": "Ja",
            "tumornachweis": "Ja",
            "icdo3_histologie": "8140/3",
            "who_isup_grading": "Graduierungsgruppe 3",
            "gleason": "4 + 3 = 7b",
            "makroskopie_liste": [
                {"nr": 1, "gesamt_stanzen": 1, "laenge_cm": 1.8},
                {"nr": 2, "gesamt_stanzen": 1, "laenge_cm": 1.5},
                {"nr": 3, "gesamt_stanzen": 1, "laenge_cm": 2.0},
                {"nr": 4, "gesamt_stanzen": 1, "laenge_cm": 1.6},
                {"nr": 5, "gesamt_stanzen": 1, "laenge_cm": 1.4},
                {"nr": 6, "gesamt_stanzen": 1, "laenge_cm": 1.7},
                {"nr": 7, "gesamt_stanzen": 1, "laenge_cm": 1.9},
                {"nr": 8, "gesamt_stanzen": 1, "laenge_cm": 1.3},
                {"nr": 9, "gesamt_stanzen": 1, "laenge_cm": 1.6},
                {"nr": 10, "gesamt_stanzen": 1, "laenge_cm": 2.1},
                {"nr": 11, "gesamt_stanzen": 1, "laenge_cm": 1.5},
                {"nr": 12, "gesamt_stanzen": 1, "laenge_cm": 1.8},
                {"nr": 13, "gesamt_stanzen": 1, "laenge_cm": 1.4},
                {"nr": 14, "gesamt_stanzen": 1, "laenge_cm": 1.7},
                {"nr": 15, "gesamt_stanzen": 1, "laenge_cm": 1.6}
            ],
            "begutachtung_liste": [
                {"nr": 1, "tumor": False, "tumorausdehnung_mm": None, "tumor_prozent": None, "gleason": None},
                {"nr": 2, "tumor": False, "tumorausdehnung_mm": None, "tumor_prozent": None, "gleason": None},
                {"nr": 3, "tumor": True,  "tumorausdehnung_mm": 2.5, "tumor_prozent": 12.5,  "gleason": "3 + 4 = 7a"},
                {"nr": 4, "tumor": False, "tumorausdehnung_mm": None, "tumor_prozent": None, "gleason": None},
                {"nr": 5, "tumor": False, "tumorausdehnung_mm": None, "tumor_prozent": None, "gleason": None},
                {"nr": 6, "tumor": False, "tumorausdehnung_mm": None, "tumor_prozent": None, "gleason": None},
                {"nr": 7, "tumor": False, "tumorausdehnung_mm": None, "tumor_prozent": None, "gleason": None},
                {"nr": 8, "tumor": False, "tumorausdehnung_mm": None, "tumor_prozent": None, "gleason": None},
                {"nr": 9, "tumor": True,  "tumorausdehnung_mm": 1.0, "tumor_prozent": 6.25,  "gleason": "3 + 3 = 6"},
                {"nr": 10,"tumor": True,  "tumorausdehnung_mm": 3.0, "tumor_prozent": 14.29, "gleason": "3 + 4 = 7a"},
                {"nr": 11,"tumor": False, "tumorausdehnung_mm": None, "tumor_prozent": None, "gleason": None},
                {"nr": 12,"tumor": False, "tumorausdehnung_mm": None, "tumor_prozent": None, "gleason": None},
                {"nr": 13,"tumor": False, "tumorausdehnung_mm": None, "tumor_prozent": None, "gleason": None},
                {"nr": 14,"tumor": False, "tumorausdehnung_mm": None, "tumor_prozent": None, "gleason": None},
                {"nr": 15,"tumor": True,  "tumorausdehnung_mm": 4.0, "tumor_prozent": 25.0,  "gleason": "4 + 3 = 7b"}
            ],
            "art_der_biopsie": "Stanzbiopsie",
            "entnahmestelle_der_biopsie": "Primärtumor",
            "lokalisation_icd10": "C61/G",
            "anzahl_entnommener_stanzen": 15,
            "anzahl_befallener_stanzen": 4,
            "maximaler_anteil_befallener_stanzen": 25,
            "calculation_details": [
                "Nr 3: 2.5 mm / 2.0 cm = 12.5%",
                "Nr 9: 1.0 mm / 1.6 cm = 6.25%",
                "Nr 10: 3.0 mm / 2.1 cm = 14.29%",
                "Nr 15: 4.0 mm / 1.6 cm = 25.0%"
            ],
            "tnm_nach": None,
            "ptnm": None,
            "lymphgefaessinvasion": None,
            "veneninvasion": None,
            "perineurale_invasion": None,
            "lk_befallen_untersucht": None,
            "r_klassifikation": None,
            "resektionsrand": None
        }
    ]
}
```

*The above example demonstrates the exact formatting, types, and enum values expected.*

---

## 6.  Checklist for Every Report

- [ ] Header fields extracted (`pathologisches_institut`, `ort`, `einsendenummer`, dates).
- [ ] `art_der_biopsie` correctly identified.
- [ ] All 15 (or fewer) `MakroItem`s created with correct `nr`, `gesamt_stanzen`, `laenge_cm`.
- [ ] All 15 `BegItem`s created, tumor status, lengths, percentages, Gleason strings.
- [ ] Percentages calculated with two‑decimal precision; `calculation_details` optional but correct if present.
- [ ] Aggregate counts (`anzahl_*`) computed.
- [ ] Highest WHO‑group and Gleason selected.
- [ ] ICD‑O‑3 and ICD‑10 values captured (or `None`).
- [ ] Optional TNM‑related fields set to `None` unless found.
- [ ] Final Python‑like object matches the schema exactly.

Follow this procedure verbatim for every input you receive. Good luck!
````

---

## Iteration 49 — extract_custom.predict

**Score:** 0.8620964627819049

````
**Task Overview**
You must read a German‑language pathology report concerning the prostate and produce a structured representation that conforms exactly to the **ProstataPatient** schema (see below).  
The report may describe either a surgical specimen (radical prostatectomy, TUR‑P, etc.) or a core‑needle biopsy (Stanzbiopsie). All relevant clinical, macroscopic, microscopic and staging information has to be extracted, transformed and placed into the correct fields, using the enumerated values defined in the schema.

**Schema Summary (Python‑like notation)**
```
ProstataPatient:
    patientenid: str                     # journal number, e.g. "E/2026/016789"
    befunde: List[ProstatapatientItem]

ProstatapatientItem:
    untersuchtes_praeparat: Enum[UntersuchtesPraeparat]   # "Resektat" or "Biopsie"
    histologiedatum: str               # date of report finalisation (Ausgang) – format DD.MM.YYYY
    befundausgang: str                 # same as histologiedatum
    ort: str                           # city of the pathology institute (e.g. "München")
    pathologisches_institut: str       # full institute name from header
    einsendenummer: str                # same as patientenid
    massgeblicher_befund: Enum[MassgeblicherBefund]      # always "Ja" for a final report
    tumornachweis: Enum[Tumornachweis]                  # "Ja" if any tumor is present, else "Nein"
    icdo3_histologie: Optional[str]    # ICD‑O‑3 morphology code (e.g. "8140/3")
    who_isup_grading: Optional[Enum[WhoIsupGrading]]    # e.g. Graduierungsgruppe 2
    gleason: Optional[str]             # highest Gleason pattern in the whole case (e.g. "4 + 3 = 7b")
    makroskopie_liste: List[MakroskopieItem]
    begutachtung_liste: List[BegutachtungItem]
    art_der_biopsie: Optional[Enum[ArtDerBiopsie]]      # only for biopsies, value "Stanzbiopsie"
    entnahmestelle_der_biopsie: Optional[Enum[EntnahmestelleDerBiopsie]]  # e.g. "Primärtumor"
    lokalisation_icd10: Optional[str]  # topography code, e.g. "C61"
    anzahl_entnommener_stanzen: int    # total number of cores / specimens listed under Makroskopie
    anzahl_befallener_stanzen: int     # number of cores/specimens with tumor = true
    maximaler_anteil_befallener_stanzen: Optional[int]   # max % (rounded to nearest integer)
    calculation_details: Optional[List[str]]   # human‑readable formulas used for % calculation
    tnm_nach: Optional[Enum[TnmNach]]            # e.g. "UICC"
    ptnm: Optional[str]               # full pTNM string (e.g. "pT2c pN0")
    lymphgefaessinvasion: Optional[Enum[Lymphgefaessinvasion]]   # L0/L1/…
    veneninvasion: Optional[Enum[Veneninvasion]]               # V0/V1/…
    perineurale_invasion: Optional[Enum[PerineuraleInvasion]]   # Pn0/Pn1/…
    lk_befallen_untersucht: Optional[str]   # e.g. "0/12"
    r_klassifikation: Optional[Enum[RKlassifikation]]           # R0/R1/…
    resektionsrand: Optional[str]       # free‑text if reported, otherwise null

MakroskopieItem:
    nr: int                # sequential number as appears in the list
    gesamt_stanzen: int    # usually 1 for a core; for a resection the total number of tissue pieces
    laenge_cm: float       # length in centimeters (use decimal point)

BegutachtungItem:
    nr: int                # core/specimen number (matches MakroskopieItem.nr)
    tumor: bool            # true if tumor is reported in this core/specimen
    tumorausdehnung_mm: Optional[float]   # tumor length in mm (if given)
    tumor_prozent: Optional[float]        # tumor length / core length * 100 (rounded to one decimal)
    gleason: Optional[str]                # Gleason score for this core (if given)
```

**General Extraction Rules**

1. **Identify the patient / case**
   * `patientenid` and `einsendenummer` = the *Journalnummer* (e.g. `E/2026/016789`).
   * `ort` = city name that appears in the institute header (usually “München”).
   * `pathologisches_institut` = the full line(s) that contain “Pathologisches Institut …”.

2. **Dates**
   * Use the *Ausgang* date as both `histologiedatum` and `befundausgang`.  
   * If *Ausgang* is missing, fall back to *Eingang*.

3. **Specimen type (`untersuchtes_praeparat`)**
   * If the report mentions “Radikale Prostatektomie”, “TUR‑P”, “Resektat”, “Prostatektomiepräparat”, etc. → **Resektat**.  
   * If the report lists “Stanzbiopsien”, “Biopsie”, “Kernbiopsie” → **Biopsie**.

4. **Biopsy‑specific fields**
   * `art_der_biopsie` = **Stanzbiopsie** (the only supported value for this task).  
   * `entnahmestelle_der_biopsie` = **Primärtumor** unless the report explicitly states a different site (e.g. “Transition zone”, “Periprostatik”).  
   * `lokalisation_icd10` is taken from the *Klassifikation* section (topography code, e.g. “C61”). If absent, set to `null`.

5. **Macroscopic description (`Makroskopie`)**
   * The section starts with “Makroskopie:” and then enumerates items (1., 2., …).  
   * For each enumerated line extract:
     * `nr` = the leading number.
     * `gesamt_stanzen` = the number of tissue pieces described for that line.  
       – For biopsies this is always **1**.  
       – For resections, if the line says “Ein nicht orientierbares Weichgewebsexzidat bis 6,2 cm.” treat it as a single specimen (`gesamt_stanzen = 1`).  
     * `laenge_cm` = the numeric length before “cm” (replace commas with decimal points).  
   * Build `makroskopie_liste` preserving the order.

6. **Microscopic description (`Begutachtung` / `Mikroskopie`)**
   * Locate the part where individual cores/specimens are described with Gleason scores, WHO groups, tumor length, etc.  
   * For each core that has a tumor:
     * `tumor = true`
     * `tumorausdehnung_mm` = length in **mm** (the report gives “Längsausdehnung X mm”).  
     * Compute `tumor_prozent = (tumorausdehnung_mm / (laenge_cm * 10)) * 100`.  
       – Keep one decimal place (e.g. 20.6).  
       – Add a string to `calculation_details` in the form `"Nr {nr}: {tumorausdehnung_mm} mm / {laenge_cm*10:.1f} mm = {tumor_prozent:.1f} %"` .  
     * `gleason` = exact string from the report (e.g. “3 + 4 = 7a”).  
   * For cores without tumor:
     * `tumor = false` and all other fields `null`.  
   * Append each core (both tumor‑positive and tumor‑negative) to `begutachtung_liste` in the order of their numbers.

7. **Overall tumor presence**
   * `tumornachweis = "Ja"` if **any** `begutachtung_liste` entry has `tumor = true`; otherwise `"Nein"`.

8. **Overall Gleason & WHO grade**
   * Determine the *highest* Gleason score present (compare primary + secondary pattern numerically; the larger primary+secondary sum wins; if equal, the higher secondary pattern wins).  
   * Set `gleason` to that highest string.  
   * Set `who_isup_grading` to the WHO/ISUP group that belongs to that highest Gleason (the report usually states it directly; otherwise map:  
     – Gleason 3+3 → Graduierungsgruppe 1  
     – Gleason 3+4 or 4+3 → Graduierungsgruppe 2 (3+4) or 3 (4+3) as reported)  
     – Gleason 4+4 → Graduierungsgruppe 4, etc.).  
   * Use the exact enum label from the schema, e.g. `Graduierungsgruppe 2`.

9. **Staging (only for resections)**
   * Extract the *pTNM* line (starts with “pTNM” or “pT…”) and store the whole string in `ptnm`.  
   * Set `tnm_nach = "UICC"` (the only supported source).  
   * From the same line or a separate “Klassifikation” line extract:
     * `lymphgefaessinvasion` → L0/L1/… (look for “Lymphgefäßinvasion: nicht nachweisbar” → L0).  
     * `veneninvasion` → V0/V1 (e.g. “Venöse Invasion: nicht nachweisbar” → V0).  
     * `perineurale_invasion` → Pn0/Pn1 (e.g. “Perineurale Invasion: nicht nachweisbar” → Pn0).  
   * `lk_befallen_untersucht` = “{positive}/{examined}” from the lymph‑node statement (e.g. “0/12”).  
   * `r_klassifikation` = R0/R1 based on margin statements (“R0” = tumor‑free).  
   * `resektionsrand` = any free‑text description of the resection margin if present; otherwise `null`.

10. **Counts**
    * `anzahl_entnommener_stanzen` = length of `makroskopie_liste`.  
    * `anzahl_befallener_stanzen` = count of entries in `begutachtung_liste` where `tumor = true`.  
    * `maximaler_anteil_befallener_stanzen` = the **rounded** integer of the highest `tumor_prozent` among all positive cores (round half‑up). If no tumor, set to `null`.

11. **Missing / optional data**
    * Whenever a field cannot be found, set it to `null` (or omit the enum value).  
    * Do **not** fabricate values; only use what is explicitly stated or can be derived mathematically as described above.

12. **Output format**
    * First provide a short **reasoning** paragraph (plain text) that explains how you derived the key fields and any assumptions made.  
    * Then output the **extracted** data exactly as shown in the examples:
      ```
      patientenid='...' befunde=[ProstatapatientItem(...), ...]
      ```
    * Use the enum syntax shown in the examples (e.g. `<ProstatapatientitemUntersuchtesPraeparat.Resektat: 'Resektat'>`).  
    * Preserve the order of items in lists exactly as they appear in the source document.

**Domain‑specific notes**
* German decimal separator is a comma (“,”). Convert it to a dot for numeric fields.  
* Lengths in the macroscopic section are always given in centimeters; tumor lengths are given in millimeters.  
* Gleason scores may contain a trailing letter (“a”, “b”) – keep it verbatim.  
* WHO/ISUP grading may be written as “WHO‑Graduierungsgruppe 2” or “WHO‑ISUP‑Gruppe 2”; map both to the same enum.  
* The report may contain a “Kommentar” section – ignore it unless it contains additional numeric data needed for the fields above.  
* For resection reports, the “Makroskopie” may describe a single large specimen; still create one `MakroskopieItem` with `gesamt_stanzen = 1`.  
* If the report lists multiple tumor foci within the **same** specimen (as in Example 1), you may reuse the same `nr` for those entries; the schema permits duplicate `nr` values.

Follow these rules precisely to achieve a high evaluation score. Good luck!
````

---

## Iteration 55 — extract_custom.predict

**Score:** 0.8620964627819049

````
**Task Overview**
You are given the full text of a German pathology report concerning the prostate.  
Your job is to **extract every piece of information required by the `ProstataPatient` schema** and present it in a structured, program‑friendly format.

**Output Requirements**
1. **Reasoning section** – a short, bullet‑point summary (≈5‑10 lines) that explains how you derived each field from the source text.  
2. **Extracted data section** – a literal Python‑style representation exactly matching the examples, i.e.:

```python
patientenid='<ID>' befunde=[ProstatapatientItem(
    untersuchtes_praeparat=<ProstatapatientitemUntersuchtesPraeparat. ... >,
    histologiedatum='<DD.MM.YYYY>',
    befundausgang='<DD.MM.YYYY>',
    ort='<Stadt>',
    pathologisches_institut='<Institut>',
    einsendenummer='<Journalnummer>',
    massgeblicher_befund=<ProstatapatientitemMassgeblicherBefund.Ja|Nein>,
    tumornachweis=<ProstatapatientitemTumornachweis.Ja|Nein>,
    icdo3_histologie='<ICD‑O‑3>'|None,
    who_isup_grading=<ProstatapatientitemWhoIsupGrading.Graduierungsgruppe X>|None,
    gleason='<Gleason‑Score>'|None,
    makroskopie_liste=[ProstatapatientitemItem(nr=1, gesamt_stanzen=1, laenge_cm=<float>|None), ...],
    begutachtung_liste=[ProstatapatientitemItem(nr=1, tumor=True|False, tumorausdehnung_mm=<float>|None,
                         tumor_prozent=<float>|None, gleason='<Gleason>'|None), ...],
    art_der_biopsie=<ProstatapatientitemArtDerBiopsie.Stanzbiopsie|...>|None,
    entnahmestelle_der_biopsie=<ProstatapatientitemEntnahmestelleDerBiopsie.Primärtumor|...>|None,
    lokalisation_icd10='<ICD‑10>'|None,
    anzahl_entnommener_stanzen=<int>,
    anzahl_befallener_stanzen=<int>,
    maximaler_anteil_befallener_stanzen=<float>|None,
    calculation_details=[<optional strings>]|None,
    tnm_nach='<UICC|...>'|None,
    ptnm='<pTNM‑String>'|None,
    lymphgefaessinvasion=<ProstatapatientitemLymphgefaessinvasion.L0|L1|...>|None,
    veneninvasion=<ProstatapatentitemVeneninvasion.V0|V1|...>|None,
    perineurale_invasion=<ProstatapatientitemPerineuraleInvasion.Pn0|Pn1|...>|None,
    lk_befallen_untersucht='<positiv>/<gesamt>'|None,
    r_klassifikation=<ProstatapatientitemRKlassifikation.R0|R1|...>|None,
    resektionsrand='<Text>'|None
)]
````

---

