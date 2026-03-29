# Source Mapping Guide for Luwak GUI Integration

How to wire MOSAICX extraction/deidentification output back to the source document for "Show Source" highlighting.

## Activate Zenta Mode

Add to `.env`:

```
MOSAICX_ZENTA=1
```

Or pass as env var:

```bash
MOSAICX_ZENTA=1 mosaicx extract --document report.pdf --template chest_ct -o result.json
```

Without this flag, the JSON only contains `extracted` + `_evidence` (excerpt and reasoning per field, no coordinates).

## JSON Structure

```json
{
  "extracted": {
    "patient_name": "Sarah Johnson",
    "patient_id": "PID-12345",
    "bmi": 23.4
  },
  "_source": {
    "_guide": {
      "version": "1.0",
      "coordinate_space": "pdf_points",
      "origin": "bottom-left",
      "bbox_format": "[x0, y0, x1, y1]",
      "render_dpi": 200,
      "page_dimensions": [[612.0, 792.0], [612.0, 792.0]],
      "to_fitz_rect": "fitz.Rect(x0, page_h - y1, x1, page_h - y0)",
      "to_image_px": "scale = render_dpi / 72; (x0 * scale, (page_h - y1) * scale, x1 * scale, (page_h - y0) * scale)"
    },
    "fields": {
      "patient_name": {
        "value": "Sarah Johnson",
        "excerpt": "Patient Name: Sarah Johnson",
        "reasoning": "The document lists the patient name as Sarah Johnson.",
        "grounded": true,
        "spans": [{"page": 1, "bbox": [273.5, 604.0, 357.5, 619.0]}]
      },
      "bmi": {
        "value": 23.4,
        "excerpt": "BMI 23.4 18.5-24.9 Normal",
        "reasoning": "The BMI value is given as 23.4 in the report.",
        "grounded": true,
        "spans": [{"page": 1, "bbox": [264.2, 221.9, 282.9, 229.2]}]
      }
    }
  }
}
```

### Key fields

| Field | Description |
|-------|-------------|
| `_guide.coordinate_space` | `"pdf_points"` for PDFs, `"image_pixels"` for images, `"none"` for text files |
| `_guide.page_dimensions` | `[[width, height], ...]` per page, in the same units as `coordinate_space` |
| `_guide.render_dpi` | DPI to use when converting PDF coordinates to image pixels |
| `fields.{key}.value` | The extracted value |
| `fields.{key}.excerpt` | Verbatim line from the source document |
| `fields.{key}.reasoning` | Why the LLM chose this value |
| `fields.{key}.grounded` | Whether the value was found in the source text |
| `fields.{key}.spans` | List of `{page, bbox}` locations in the source document |

## Show Source: Highlight on PDF

User right-clicks a cell in the Excel table, clicks "Show Source":

```python
import json
import fitz

data = json.load(open("result.json"))
guide = data["_source"]["_guide"]
page_dims = guide["page_dimensions"]

# field_key comes from the table position
field_key = "patient_name"  # or "findings[0].location" for nested
field = data["_source"]["fields"][field_key]

# Draw highlight on PDF
pdf = fitz.open("original.pdf")
for span in field["spans"]:
    page = pdf[span["page"] - 1]
    page_h = page_dims[span["page"] - 1][1]
    x0, y0, x1, y1 = span["bbox"]

    # Y-axis flip: PDF origin is bottom-left, PyMuPDF is top-left
    rect = fitz.Rect(x0, page_h - y1, x1, page_h - y0)
    page.add_highlight_annot(rect)

pdf.save("highlighted.pdf")
```

## Show Source: Highlight on Rendered Image

If the GUI renders PDF pages as images:

```python
dpi = guide["render_dpi"]  # 200
scale = dpi / 72.0

for span in field["spans"]:
    page_h = page_dims[span["page"] - 1][1]
    x0, y0, x1, y1 = span["bbox"]

    # Convert PDF points to image pixels (with Y-flip)
    img_rect = (
        x0 * scale,
        (page_h - y1) * scale,
        x1 * scale,
        (page_h - y0) * scale,
    )
    draw_rectangle(img_rect)  # your rendering code
```

For image documents (`coordinate_space == "image_pixels"`), use `bbox` directly -- no transformation needed.

## Nested Schemas (Findings Tables)

Templates with lists (e.g. radiology findings with multiple nodules) use dot-path keys:

```
"findings[0].location"   -> row 0, column "location"
"findings[0].size_mm"    -> row 0, column "size_mm"
"findings[1].location"   -> row 1, column "location"
```

Build the key from the Excel table position:

```python
field_key = f"findings[{row}].{column}"
spans = data["_source"]["fields"][field_key]["spans"]
```

## Loading into SQL

The `_source.fields` dict maps directly to a flat SQL table:

```sql
CREATE TABLE fields (
    extraction_id INTEGER,
    field_key     TEXT,      -- "patient_name" or "findings[0].location"
    value         TEXT,
    excerpt       TEXT,
    reasoning     TEXT,
    grounded      BOOLEAN,
    page          INTEGER,
    x0            REAL,
    y0            REAL,
    x1            REAL,
    y1            REAL
);
```

Loading:

```python
for field_key, info in data["_source"]["fields"].items():
    for span in info["spans"]:
        db.insert("fields", {
            "extraction_id": extraction_id,
            "field_key": field_key,
            "value": info["value"],
            "excerpt": info.get("excerpt"),
            "reasoning": info.get("reasoning"),
            "grounded": info.get("grounded"),
            "page": span["page"],
            "x0": span["bbox"][0],
            "y0": span["bbox"][1],
            "x1": span["bbox"][2],
            "y1": span["bbox"][3],
        })
```

Query for "Show Source":

```sql
SELECT page, x0, y0, x1, y1, excerpt, reasoning
FROM fields
WHERE extraction_id = ? AND field_key = ?
```

## Deidentify Output

Same structure. The only difference: deidentify fields add `replacement` and `phi_type`.

```json
"fields": {
  "name_0": {
    "value": "Sarah Johnson",
    "replacement": "[REDACTED]",
    "phi_type": "NAME",
    "excerpt": "Patient Name: Sarah Johnson",
    "reasoning": "Patient name is protected health information",
    "grounded": true,
    "spans": [{"page": 1, "bbox": [273.5, 604.0, 357.5, 619.0]}]
  }
}
```

## Coordinate Spaces

| Source format | `coordinate_space` | `origin` | What to do |
|--------------|-------------------|----------|------------|
| PDF | `pdf_points` | `bottom-left` | Flip Y with `page_h - y` for display |
| Image (JPG, PNG) | `image_pixels` | `top-left` | Use bbox directly |
| Text file | `none` | -- | No coordinates, excerpts only |

## Commands

```bash
# Extract with source mapping
MOSAICX_ZENTA=1 mosaicx extract --document scan.pdf --template chest_ct -o result.json

# Extract with force OCR (layout-aware tables)
MOSAICX_ZENTA=1 mosaicx extract --document scan.pdf --template chest_ct --force-ocr -o result.json

# Deidentify with source mapping
MOSAICX_ZENTA=1 mosaicx deidentify --document scan.pdf -o result.json

# Dump OCR text for debugging
MOSAICX_ZENTA=1 mosaicx extract --document scan.pdf --template chest_ct --dump-ocr -o result.json
```
