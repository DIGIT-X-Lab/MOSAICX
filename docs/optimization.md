# Optimization and Evaluation

This guide explains how to use `mosaicx optimize` and `mosaicx eval` to improve the accuracy of MOSAICX pipelines on your specific documents. If you have never heard of DSPy, do not worry — everything is explained from first principles.

## What is Optimization?

MOSAICX uses Large Language Models (LLMs) to extract data from clinical documents. These LLMs follow prompts — instructions that tell them what to extract and how to format the output. Out of the box, MOSAICX ships with generic prompts that work reasonably well across many document types.

However, every medical institution has its own document formats, terminology, and quirks. A radiology report from Hospital A looks different from a radiology report from Hospital B. Generic prompts cannot capture these nuances.

This is where optimization comes in.

**Optimization is a process that automatically improves prompts for YOUR specific documents.** You provide labeled examples (documents with correct answers), and a DSPy optimizer finds better prompts — better instructions and better few-shot examples — that produce more accurate results on your data. Think of it like fine-tuning a model, but instead of updating model weights (which requires expensive GPUs and thousands of examples), you are updating the prompts. The same LLM, the same codebase, just better instructions.

You do not need to write any code. You do not need to change how you run `mosaicx extract`. You just add one flag: `--optimized path`. Everything else stays the same.

### The Three Optimizers

MOSAICX supports three optimization strategies, each with different trade-offs:

1. **BootstrapFewShot** — The cheapest and fastest optimizer. It runs your pipeline on the training examples, collects successful outputs, and uses them as few-shot demonstrations. Best for small datasets and quick iterations. Costs around $0.50 and takes about 5 minutes.

2. **MIPROv2** — A meta-prompting optimizer. It generates multiple candidate instructions, evaluates them, and picks the best. More expensive and slower than BootstrapFewShot, but produces better results. Costs around $3 and takes about 20 minutes.

3. **GEPA** — An evolutionary optimizer. It generates many candidate programs, scores them, and evolves the best ones over multiple generations. The most expensive and slowest, but produces the best results on large datasets. Costs around $10 and takes about 45 minutes.

Which one should you use? Start with BootstrapFewShot (`--budget light`) to verify your dataset is correct. If you need better accuracy, move to MIPROv2 (`--budget medium`). If you have a large dataset and need the absolute best results, use GEPA (`--budget heavy`).

## Step 1: Prepare Your Dataset

Optimization requires labeled data: documents paired with correct answers. You provide these as a JSONL file.

### What is JSONL?

JSONL (JSON Lines) is a text file where each line is a valid JSON object. Unlike regular JSON, which wraps everything in square brackets, JSONL has one record per line:

```jsonl
{"name": "Alice", "age": 30}
{"name": "Bob", "age": 25}
{"name": "Charlie", "age": 35}
```

Each line must be valid JSON. No commas between lines. No outer array.

### Dataset Format by Pipeline

Each pipeline expects different fields. Some fields are **inputs** (what the pipeline receives), and the rest are **labels** (the correct answer you want the pipeline to produce).

#### Radiology

Extract structured findings from radiology reports.

**Inputs:** `report_text`, `report_header`
**Labels:** `exam_type`, `findings`

Example record:

```jsonl
{"report_text": "CT Chest: 5mm nodule in RUL. No lymphadenopathy. Impression: Solitary pulmonary nodule.", "report_header": "CHEST CT", "exam_type": "CT Chest", "findings": [{"anatomy": "RUL", "observation": "nodule", "size": "5mm"}]}
```

- `report_text`: The full report text (string)
- `report_header`: Optional header/title (string)
- `exam_type`: The type of exam, e.g. "CT Chest", "MRI Brain" (string)
- `findings`: A list of findings. Each finding is a dictionary with keys like `anatomy`, `observation`, `size`, `severity`, etc. (list of dictionaries)

#### Pathology

Extract structured findings from pathology reports.

**Inputs:** `report_text`, `report_header`
**Labels:** `specimen_type`, `findings`

Example record:

```jsonl
{"report_text": "Colon biopsy shows adenocarcinoma, moderately differentiated. No lymphovascular invasion.", "report_header": "PATHOLOGY REPORT", "specimen_type": "Colon biopsy", "findings": [{"diagnosis": "Adenocarcinoma", "grade": "moderately differentiated"}]}
```

- `report_text`: The full report text (string)
- `report_header`: Optional header/title (string)
- `specimen_type`: The type of specimen, e.g. "Colon biopsy", "Breast tissue" (string)
- `findings`: A list of findings. Each finding is a dictionary with keys like `diagnosis`, `grade`, `biomarker`, `tnm_stage`, etc. (list of dictionaries)

#### Extract

Generic document extraction into a flexible schema.

**Inputs:** `document_text`
**Labels:** `extracted`

Example record:

```jsonl
{"document_text": "Patient: John Doe. Age: 45. Blood pressure: 120/80.", "extracted": {"name": "John Doe", "age": 45, "bp_systolic": 120, "bp_diastolic": 80}}
```

- `document_text`: The full document text (string)
- `extracted`: A dictionary containing the extracted fields. Keys and values are whatever you want the pipeline to extract. (dictionary)

#### Summarize

Synthesize a narrative timeline from multiple reports.

**Inputs:** `reports`, `patient_id`
**Labels:** `narrative`

Example record:

```jsonl
{"reports": ["2024-01-15 CT: Nodule stable.", "2024-03-20 PET: No uptake in nodule."], "patient_id": "P001", "narrative": "Patient P001 has a stable nodule on CT from January 2024. PET scan in March 2024 shows no FDG uptake, suggesting benign etiology."}
```

- `reports`: A list of report texts (list of strings)
- `patient_id`: Optional patient identifier (string)
- `narrative`: The expected summary narrative (string)

#### Deidentify

Remove or pseudonymize Protected Health Information (PHI).

**Inputs:** `document_text`, `mode`
**Labels:** `redacted_text`

Example record:

```jsonl
{"document_text": "Patient John Doe, SSN 123-45-6789, DOB 1980-05-12.", "mode": "remove", "redacted_text": "Patient [REDACTED], SSN [REDACTED], DOB [REDACTED]."}
```

- `document_text`: The original document text with PHI (string)
- `mode`: De-identification mode: `remove`, `pseudonymize`, or `dateshift` (string)
- `redacted_text`: The expected output with PHI removed or replaced (string)

#### Template

Generate an extraction template from a description.

**Inputs:** `description`, `example_text`, `document_text`
**Labels:** `schema_spec`

Example record:

```jsonl
{"description": "echocardiography report with LVEF and valve grades", "schema_spec": {"class_name": "EchoReport", "fields": [{"name": "lvef", "type": "float", "required": true, "description": "Left ventricular ejection fraction"}, {"name": "valve_grades", "type": "list[str]", "required": false, "description": "Valve regurgitation grades"}]}}
```

- `description`: Natural-language description of the template (string)
- `example_text`: Optional example document snippet (string)
- `document_text`: Optional full document text (string)
- `schema_spec`: The expected SchemaSpec object with `class_name` and `fields` (dictionary)

### How Many Examples Do You Need?

More examples produce better results, but you do not need thousands. Here are some guidelines:

- **Minimum:** 10 examples per pipeline. Anything less is too small for reliable optimization.
- **Good:** 50+ examples. This gives the optimizer enough variety to learn robust patterns.
- **Ideal:** 100+ examples. Diminishing returns beyond this, but still helpful for complex pipelines.

### Train vs Test Split

You should split your data into two sets:

1. **Training set** — Used by the optimizer to find better prompts.
2. **Test set** — Used to evaluate how well the optimized pipeline generalizes to unseen data.

You can split manually (save as `train.jsonl` and `test.jsonl`), or let MOSAICX auto-split 80/20 if you only provide a training set. Manual splitting is recommended because it ensures the test set is truly held-out.

### Creating Your Dataset

1. Collect representative documents. Include edge cases, unusual formats, and hard examples.
2. Manually label each document with the correct answer. Use the format shown above for your pipeline.
3. Save as a `.jsonl` file. Each line is one JSON object. No commas between lines.
4. Verify the file is valid JSON. Use a tool like `jq` or an online JSON validator.

Example command to validate:

```bash
cat train.jsonl | while read line; do echo "$line" | jq . > /dev/null || echo "Invalid JSON: $line"; done
```

> **Tip:** If you are unsure what the output format should look like, run `mosaicx extract` on a few documents and inspect the JSON output. Use that structure as a template for your labels.

## Step 2: List Available Pipelines

Before optimizing, check which pipelines are available:

```bash
mosaicx optimize --list-pipelines
```

Output:

```
> deidentify
> extract
> pathology
> radiology
> template
> summarize
```

These are the pipeline names you can pass to `--pipeline`.

## Step 3: Run Optimization

Once your dataset is ready, run optimization:

```bash
mosaicx optimize --pipeline radiology --trainset train.jsonl --budget light
```

This command:

1. Loads your training examples from `train.jsonl`
2. Runs the BootstrapFewShot optimizer (because `--budget light`)
3. Saves the optimized program to `~/.mosaicx/optimized/radiology_optimized.json`

### Understanding the Flags

| Flag | Required | Description |
|------|----------|-------------|
| `--pipeline` | Yes | Which pipeline to optimize (e.g., `radiology`, `pathology`, `extract`) |
| `--trainset` | Yes | Path to your training JSONL file |
| `--valset` | No | Path to a validation JSONL file. If omitted, MOSAICX auto-splits 80/20 from the training set. |
| `--budget` | No | Optimization budget preset: `light`, `medium`, or `heavy`. Default: `medium`. |
| `--save` | No | Where to save the optimized program. Default: `~/.mosaicx/optimized/{pipeline}_optimized.json` |

### Budget Presets Explained

| Budget | Optimizer | Cost | Time | When to Use |
|--------|-----------|------|------|-------------|
| `light` | BootstrapFewShot | ~$0.50 | ~5 min | First try, small datasets, quick iterations |
| `medium` | MIPROv2 | ~$3 | ~20 min | Better results, moderate datasets, production use |
| `heavy` | GEPA | ~$10 | ~45 min | Best results, large datasets, final tuning |

Costs assume you are using a local Ollama server (free). If you are using a commercial API like OpenAI, costs scale with your model pricing.

### Where is the Optimized Program Saved?

By default, MOSAICX saves optimized programs to:

```
~/.mosaicx/optimized/{pipeline}_optimized.json
```

For example, optimizing the radiology pipeline saves to:

```
~/.mosaicx/optimized/radiology_optimized.json
```

You can override this with `--save`:

```bash
mosaicx optimize --pipeline radiology --trainset train.jsonl --budget light --save /path/to/my_program.json
```

### What Does the Output Look Like?

After optimization completes, you will see a results table:

```
Strategy          BootstrapFewShot
Train examples    40
Val examples      10
Train score       0.872
Val score         0.845
Saved to          /Users/you/.mosaicx/optimized/radiology_optimized.json
```

- **Train score:** Average metric on the training set (0.0 to 1.0)
- **Val score:** Average metric on the validation set (0.0 to 1.0)

Higher is better. If the validation score is much lower than the training score, your optimizer may be overfitting. Try using more training examples or a simpler budget (`light` instead of `medium`).

## Step 4: Evaluate Your Pipeline

Now that you have an optimized program, evaluate it on a held-out test set to see if it generalizes:

```bash
mosaicx eval --pipeline radiology --testset test.jsonl
```

This runs the **baseline** (unoptimized) pipeline on your test set and prints statistics.

To evaluate the **optimized** pipeline, add `--optimized`:

```bash
mosaicx eval --pipeline radiology --testset test.jsonl --optimized ~/.mosaicx/optimized/radiology_optimized.json
```

### What Does the Output Look Like?

The `eval` command prints three sections:

**Section 1: Configuration**

```
Pipeline     radiology
Test set     test.jsonl
Examples     20
Optimized    /Users/you/.mosaicx/optimized/radiology_optimized.json
```

**Section 2: Statistics**

```
Count      20
Mean       0.867
Median     0.890
Std Dev    0.112
Min        0.650
Max        0.980
```

- **Mean:** Average score across all test examples (0.0 to 1.0)
- **Median:** Middle score (less affected by outliers)
- **Std Dev:** Standard deviation (how much scores vary)
- **Min/Max:** Lowest and highest scores

**Section 3: Score Distribution (Histogram)**

```
Range    Count  Distribution
0.0-0.2    0
0.2-0.4    0
0.4-0.6    1    ████
0.6-0.8    5    ████████████████████
0.8-1.0   14    ████████████████████████████████
```

This shows how many test examples fell into each score bucket. A good optimized pipeline should have most examples in the 0.8-1.0 range.

### Save Detailed Results

To save per-example scores as JSON for further analysis:

```bash
mosaicx eval --pipeline radiology --testset test.jsonl --optimized ~/.mosaicx/optimized/radiology_optimized.json --output results.json
```

The output JSON contains:

```json
{
  "pipeline": "radiology",
  "testset": "test.jsonl",
  "optimized": "/Users/you/.mosaicx/optimized/radiology_optimized.json",
  "count": 20,
  "mean": 0.867,
  "median": 0.890,
  "stdev": 0.112,
  "min": 0.650,
  "max": 0.980,
  "details": [
    {"index": 0, "score": 0.920, "inputs": {"report_text": "CT Chest: ..."}},
    {"index": 1, "score": 0.850, "inputs": {"report_text": "MRI Brain: ..."}},
    ...
  ]
}
```

### Compare Baseline vs Optimized

To see if optimization helped, run evaluation twice:

```bash
# Baseline (no --optimized flag)
mosaicx eval --pipeline radiology --testset test.jsonl --output baseline.json

# Optimized
mosaicx eval --pipeline radiology --testset test.jsonl --optimized ~/.mosaicx/optimized/radiology_optimized.json --output optimized.json
```

Then compare the mean scores:

```bash
# On macOS/Linux:
echo "Baseline: $(jq .mean baseline.json)"
echo "Optimized: $(jq .mean optimized.json)"
```

If the optimized mean is higher, optimization worked. If not, you may need more training examples, a different budget, or better labeling consistency.

## Step 5: Use the Optimized Program

Once you have verified that optimization improves accuracy, use the optimized program in production:

### Single Document Extraction

```bash
mosaicx extract --document report.pdf --mode radiology --optimized ~/.mosaicx/optimized/radiology_optimized.json
```

The `--optimized` flag tells MOSAICX to load the optimized prompts instead of the default ones. Everything else stays the same.

### Batch Processing

```bash
mosaicx extract --dir ./reports --output-dir ./structured --mode radiology --optimized ~/.mosaicx/optimized/radiology_optimized.json
```

The `--dir` batch mode also accepts `--optimized`. All documents in the batch will use the optimized prompts.

> **Note:** The `--optimized` flag works with `--mode` and `--template`. It does not work with auto mode (no flags), because auto mode uses a different pipeline that infers the schema dynamically.

## How Metrics Work

Each pipeline uses a different metric to score predictions. Metrics return a score between 0.0 (worst) and 1.0 (best). Here is how they work:

### Radiology

**Formula:** 0.6 × extraction_quality + 0.2 × exam_type_match + 0.2 × finding_count_similarity

- **extraction_quality (60%):** How well the findings and impressions match the gold standard. Measured using structured overlap.
- **exam_type_match (20%):** Did the pipeline predict the correct exam type (e.g., "CT Chest")? 1.0 if yes, 0.0 if no.
- **finding_count_similarity (20%):** How close is the predicted number of findings to the gold count? `min(predicted, gold) / max(predicted, gold)`

Example: If the pipeline extracts findings with 0.8 quality, gets the exam type correct (1.0), and predicts 4 findings when there are 5 (0.8 count similarity), the score is:

```
0.6 × 0.8 + 0.2 × 1.0 + 0.2 × 0.8 = 0.48 + 0.20 + 0.16 = 0.84
```

### Pathology

**Formula:** 0.4 × finding_quality + 0.3 × diagnosis_count_similarity + 0.3 × specimen_type_match

- **finding_quality (40%):** Fraction of predicted findings that have a non-empty description.
- **diagnosis_count_similarity (30%):** `min(predicted, gold) / max(predicted, gold)` for the number of findings.
- **specimen_type_match (30%):** Did the pipeline predict the correct specimen type? 1.0 if yes, 0.0 if no.

### Extract

**Formula:** 0.5 × key_overlap + 0.5 × value_match

- **key_overlap (50%):** What fraction of gold keys appear in the prediction? `len(gold_keys ∩ pred_keys) / len(gold_keys)`
- **value_match (50%):** For keys that appear in both, what fraction have exact-match values (case-insensitive)? `matches / len(shared_keys)`

Example: Gold has `{"name": "John", "age": 45}`. Prediction has `{"name": "John", "age": 46, "sex": "M"}`. Key overlap = 2/2 = 1.0. Value match = 1/2 = 0.5 (name matches, age does not). Score = 0.5 × 1.0 + 0.5 × 0.5 = 0.75.

### Summarize

**Formula:** 0.4 × length_adequacy + 0.6 × keyword_overlap

- **length_adequacy (40%):** Is the narrative at least 50 characters? `min(len(narrative) / 50, 1.0)`
- **keyword_overlap (60%):** What fraction of gold narrative tokens appear in the predicted narrative? `len(gold_tokens ∩ pred_tokens) / len(gold_tokens)`

### Deidentify

**Formula:** 0.6 × phi_leak_score + 0.4 × text_overlap

- **phi_leak_score (60%):** How well did the pipeline avoid leaking PHI? Measured using regex patterns for common PHI (names, SSNs, dates). Lower leak = higher score.
- **text_overlap (40%):** Token overlap between predicted and gold redacted text. `len(gold_tokens ∩ pred_tokens) / len(gold_tokens)`

### Template

**Formula:** 0.7 × field_name_overlap + 0.3 × extra_field_score

- **field_name_overlap (70%):** What fraction of gold field names appear in the prediction? `len(gold_fields ∩ pred_fields) / len(gold_fields)`
- **extra_field_score (30%):** How few extra fields were predicted beyond gold? `max(0, 1 - len(extra_fields) / len(gold_fields))`. A perfect score (1.0) means no extra fields.

Example: Gold has fields `["lvef", "valve_grade"]`. Prediction has `["lvef", "valve_grade", "wall_motion"]`. Overlap = 2/2 = 1.0. Extra score = max(0, 1 - 1/2) = 0.5. Score = 0.7 × 1.0 + 0.3 × 0.5 = 0.85.

## Tips

### Start Small

Begin with `--budget light` to verify your dataset is correct. If you get a validation score above 0.7, your labels are probably good. If not, inspect your examples for inconsistencies.

### More Examples = Better Optimization

Optimization quality scales with dataset size. 10 examples is the bare minimum. 50 is good. 100+ is ideal. If you only have 10 examples, use `--budget light` — heavier budgets need more data to avoid overfitting.

### Use Separate Train and Test Sets

Always evaluate on a held-out test set. If you only have one dataset, manually split it 80/20 before optimization. Do not evaluate on the same examples you used for training — the scores will be artificially high.

### Check Label Consistency

If optimization does not improve scores, the problem is often inconsistent labeling. Review your gold labels. Are you using the same terminology? Are you extracting the same level of detail? Run a few examples through the baseline pipeline and see if your labels match the LLM's natural output style. If not, adjust your labels to be more consistent.

### Iterate

Optimization is not a one-shot process. After evaluating, inspect low-scoring examples. Are they genuinely hard? Are the labels wrong? Add more examples covering those cases and re-optimize. Iterate until you reach your target accuracy.

## Complete Walkthrough

Here is a full end-to-end example optimizing the radiology pipeline:

### 1. Create a Training Dataset

Save this as `radiology_train.jsonl`:

```jsonl
{"report_text": "CT Chest: 5mm nodule in RUL. No lymphadenopathy. Impression: Solitary pulmonary nodule.", "report_header": "CHEST CT", "exam_type": "CT Chest", "findings": [{"anatomy": "RUL", "observation": "nodule", "size": "5mm"}]}
{"report_text": "MRI Brain: No acute infarct. Mild chronic small vessel disease. Impression: Chronic microvascular changes.", "report_header": "BRAIN MRI", "exam_type": "MRI Brain", "findings": [{"anatomy": "White matter", "observation": "small vessel disease", "severity": "mild"}]}
{"report_text": "CXR: Cardiomegaly. Clear lungs. No pneumothorax. Impression: Enlarged cardiac silhouette.", "report_header": "CHEST X-RAY", "exam_type": "Chest X-ray", "findings": [{"anatomy": "Heart", "observation": "cardiomegaly"}]}
{"report_text": "CT Abdomen: Pancreatic head mass, 3cm. Dilated pancreatic duct. Impression: Pancreatic neoplasm.", "report_header": "ABDOMEN CT", "exam_type": "CT Abdomen", "findings": [{"anatomy": "Pancreas head", "observation": "mass", "size": "3cm"}, {"anatomy": "Pancreatic duct", "observation": "dilated"}]}
{"report_text": "US Thyroid: Solid nodule RLL, 1.2cm. No calcifications. Impression: Thyroid nodule, likely benign.", "report_header": "THYROID ULTRASOUND", "exam_type": "Thyroid Ultrasound", "findings": [{"anatomy": "Right lower lobe", "observation": "nodule", "size": "1.2cm", "characteristics": "solid, no calcifications"}]}
{"report_text": "MRI Spine: L4-L5 disc herniation with nerve root compression. Impression: Lumbar disc herniation.", "report_header": "LUMBAR SPINE MRI", "exam_type": "MRI Spine", "findings": [{"anatomy": "L4-L5", "observation": "disc herniation", "associated": "nerve root compression"}]}
{"report_text": "CT Chest: RUL consolidation. Air bronchograms present. Impression: Pneumonia.", "report_header": "CHEST CT", "exam_type": "CT Chest", "findings": [{"anatomy": "RUL", "observation": "consolidation", "characteristics": "air bronchograms"}]}
{"report_text": "Mammography: 8mm spiculated mass at 2 o'clock LB. Calcifications. BI-RADS 5. Impression: Suspicious for malignancy.", "report_header": "BILATERAL MAMMOGRAM", "exam_type": "Mammography", "findings": [{"anatomy": "Left breast 2 o'clock", "observation": "spiculated mass", "size": "8mm", "characteristics": "calcifications", "birads": "5"}]}
{"report_text": "CT Angiography: Pulmonary embolism in RLL segmental artery. No RV strain. Impression: Acute PE.", "report_header": "CTA CHEST", "exam_type": "CT Angiography", "findings": [{"anatomy": "RLL segmental artery", "observation": "pulmonary embolism"}]}
{"report_text": "PET-CT: FDG-avid LUL mass SUVmax 8.5. Mediastinal nodes SUVmax 3.2. Impression: Metabolically active lung cancer with nodal involvement.", "report_header": "PET-CT CHEST", "exam_type": "PET-CT", "findings": [{"anatomy": "LUL", "observation": "mass", "suv": "8.5"}, {"anatomy": "Mediastinal lymph nodes", "observation": "FDG-avid", "suv": "3.2"}]}
```

### 2. Create a Test Dataset

Save this as `radiology_test.jsonl`:

```jsonl
{"report_text": "CT Chest: Pleural effusion right side, moderate. No pulmonary nodules. Impression: Right pleural effusion.", "report_header": "CHEST CT", "exam_type": "CT Chest", "findings": [{"anatomy": "Right pleura", "observation": "effusion", "severity": "moderate"}]}
{"report_text": "MRI Brain: Left frontal lobe glioblastoma, 4cm with edema. Mass effect. Impression: High-grade glioma.", "report_header": "BRAIN MRI", "exam_type": "MRI Brain", "findings": [{"anatomy": "Left frontal lobe", "observation": "glioblastoma", "size": "4cm", "associated": "edema, mass effect"}]}
{"report_text": "CXR: No acute cardiopulmonary abnormality. Clear lungs. Normal heart size. Impression: Normal chest radiograph.", "report_header": "CHEST X-RAY", "exam_type": "Chest X-ray", "findings": []}
```

### 3. Verify the Dataset

Check that each line is valid JSON:

```bash
cat radiology_train.jsonl | while read line; do echo "$line" | python -m json.tool > /dev/null || echo "Invalid JSON"; done
```

If no errors, proceed.

### 4. Run Optimization

```bash
mosaicx optimize --pipeline radiology --trainset radiology_train.jsonl --budget light
```

Output:

```
01 · OPTIMIZATION
Pipeline          radiology
Budget            LIGHT
Strategy          BootstrapFewShot
Max iterations    10
Num candidates    5
Training set      radiology_train.jsonl
Validation set    not specified
Save path         not specified

02 · PROGRESSIVE STRATEGY
Stage                Cost    Time    Min Examples
BootstrapFewShot     ~$0.50  ~5 min  10
MIPROv2              ~$3     ~20 min 10
GEPA                 ~$10    ~45 min 10

Loaded 10 training examples

03 · RUNNING OPTIMIZATION
⠋ Optimizing... patience you must have

04 · RESULTS
Strategy          BootstrapFewShot
Train examples    8
Val examples      2
Train score       0.891
Val score         0.872
Saved to          /Users/you/.mosaicx/optimized/radiology_optimized.json
```

### 5. Evaluate on Test Set

Baseline (no optimization):

```bash
mosaicx eval --pipeline radiology --testset radiology_test.jsonl
```

Output:

```
01 · EVALUATION
Pipeline     radiology
Test set     radiology_test.jsonl
Examples     3
Optimized    baseline

Evaluating... patience you must have

02 · STATISTICS
Count      3
Mean       0.733
Median     0.800
Std Dev    0.153
Min        0.550
Max        0.850

03 · SCORE DISTRIBUTION
Range    Count  Distribution
0.0-0.2    0
0.2-0.4    0
0.4-0.6    1    ████████████████
0.6-0.8    0
0.8-1.0    2    ████████████████████████████████
```

Optimized:

```bash
mosaicx eval --pipeline radiology --testset radiology_test.jsonl --optimized ~/.mosaicx/optimized/radiology_optimized.json
```

Output:

```
01 · EVALUATION
Pipeline     radiology
Test set     radiology_test.jsonl
Examples     3
Optimized    /Users/you/.mosaicx/optimized/radiology_optimized.json

Evaluating... patience you must have

02 · STATISTICS
Count      3
Mean       0.867
Median     0.900
Std Dev    0.091
Min        0.750
Max        0.950

03 · SCORE DISTRIBUTION
Range    Count  Distribution
0.0-0.2    0
0.2-0.4    0
0.4-0.6    0
0.6-0.8    1    ████████████████
0.8-1.0    2    ████████████████████████████████
```

The mean improved from 0.733 to 0.867 — optimization worked.

### 6. Use in Production

```bash
mosaicx extract --document new_report.pdf --mode radiology --optimized ~/.mosaicx/optimized/radiology_optimized.json
```

or in batch:

```bash
mosaicx extract --dir ./new_reports --output-dir ./structured --mode radiology --optimized ~/.mosaicx/optimized/radiology_optimized.json
```

Done. Your pipeline now uses optimized prompts tuned to your institution's radiology reports.
