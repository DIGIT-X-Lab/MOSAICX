# MCP Server

This guide explains how to use MOSAICX as an MCP server so that AI assistants like Claude can call MOSAICX tools directly during a conversation.

## What is MCP?

MCP stands for **Model Context Protocol**. It is an open standard that lets AI assistants (Claude, Cursor, Windsurf, and others) call external tools directly. Instead of asking you to copy-paste commands, the AI assistant can reach out to a tool server, run the tool, and use the result -- all within the conversation.

MOSAICX exposes its core features as MCP tools:

- **Extract structured data** from medical documents with optional completeness scoring
- **De-identify clinical text** by removing or replacing Protected Health Information (PHI), with optional regex-only mode
- **Summarize clinical reports** into patient timelines with structured events
- **Generate extraction templates** from natural-language descriptions or sample documents
- **List saved templates** and available extraction modes

This means you can say something like "extract the findings from this radiology report" in a Claude conversation, and Claude will call the MOSAICX extraction tool behind the scenes, returning structured JSON data without you ever touching the command line.

## Installation

MCP support is an optional dependency. Install it with:

```bash
pip install 'mosaicx[mcp]'
```

If you already have MOSAICX installed, this adds the `mcp` package on top of your existing installation. If you prefer to install the MCP dependency directly:

```bash
pip install 'mcp[cli]>=1.0.0'
```

## Quick Start

Getting MOSAICX working as an MCP server takes three steps.

### Step 1: Install MCP support

```bash
pip install 'mosaicx[mcp]'
```

### Step 2: Add MOSAICX to your AI assistant's MCP config

For Claude Code, add the following to your MCP settings:

```json
{
  "mcpServers": {
    "mosaicx": {
      "command": "mosaicx",
      "args": ["mcp", "serve"]
    }
  }
}
```

### Step 3: Start using tools in conversations

Open a Claude conversation and ask something like:

> "Extract the findings from this radiology report: [paste report text here]"

Claude will call the `extract_document` tool, run the MOSAICX extraction pipeline, and return structured JSON data in the conversation.

## Configuring with Claude Code

Claude Code reads MCP server configurations from its settings file. There are two ways to add MOSAICX.

### Option A: Project-level configuration

Create or edit `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "mosaicx": {
      "command": "mosaicx",
      "args": ["mcp", "serve"]
    }
  }
}
```

This makes MOSAICX available whenever you open this project in Claude Code.

### Option B: Global configuration

To make MOSAICX available across all projects, add it to your global Claude Code settings. Run:

```bash
claude mcp add mosaicx -- mosaicx mcp serve
```

Or manually edit `~/.claude.json` and add the server to the `mcpServers` section:

```json
{
  "mcpServers": {
    "mosaicx": {
      "command": "mosaicx",
      "args": ["mcp", "serve"]
    }
  }
}
```

### Passing environment variables

If your LLM backend requires environment variables (API keys, custom endpoints), pass them through the MCP config:

```json
{
  "mcpServers": {
    "mosaicx": {
      "command": "mosaicx",
      "args": ["mcp", "serve"],
      "env": {
        "MOSAICX_LM": "openai/gpt-oss:120b",
        "MOSAICX_API_BASE": "http://localhost:11434/v1",
        "MOSAICX_API_KEY": "ollama"
      }
    }
  }
}
```

## Configuring with Claude Desktop

Claude Desktop reads MCP server configurations from `claude_desktop_config.json`. The file location depends on your operating system:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

Open the file (create it if it does not exist) and add MOSAICX:

```json
{
  "mcpServers": {
    "mosaicx": {
      "command": "mosaicx",
      "args": ["mcp", "serve"]
    }
  }
}
```

If you need to pass environment variables for your LLM backend:

```json
{
  "mcpServers": {
    "mosaicx": {
      "command": "mosaicx",
      "args": ["mcp", "serve"],
      "env": {
        "MOSAICX_LM": "openai/gpt-oss:120b",
        "MOSAICX_API_BASE": "http://localhost:11434/v1",
        "MOSAICX_API_KEY": "ollama"
      }
    }
  }
}
```

After saving the file, restart Claude Desktop for the changes to take effect. You should see MOSAICX listed as an available tool provider in the conversation interface.

## Available Tools

The MOSAICX MCP server exposes six tools. Each tool accepts parameters as JSON and returns a JSON string with the result.

---

### extract_document

Extract structured data from a medical document.

Supports three extraction strategies:

- **auto** (default): The LLM reads the document, infers an appropriate schema, and extracts data into it. Best for one-off extractions where you do not have a predefined schema.
- **mode-based**: Uses a specialized multi-step pipeline for a specific document domain (e.g., `radiology` or `pathology`). These pipelines include classification, section parsing, and domain-specific extraction steps.
- **template-based**: Uses a previously saved template by name for targeted extraction. The template defines exactly which fields to extract.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `document_text` | `string` | (required) | Full text of the clinical document to extract from. |
| `mode` | `string` | `"auto"` | Extraction strategy. Options: `"auto"`, `"radiology"`, `"pathology"`. Ignored if `template` is provided. |
| `template` | `string` | `null` | Name of a saved template (from `~/.mosaicx/templates/`). If provided, extraction uses this template instead of `mode`. |
| `score` | `boolean` | `false` | If true, compute completeness scoring against the template. Only effective with template or mode extraction (not auto). |

**Return value:**

A JSON string containing the extracted data. The exact structure depends on the extraction strategy:

- **auto mode:** `{"extracted": {...}, "inferred_schema": {...}}`
- **mode-based:** Domain-specific fields plus `_metrics` with per-step breakdown
- **template-based:** `{"extracted": {...}}` with fields matching the template
- **with score:** Adds `"completeness"` key with field coverage metrics

If an error occurs, returns `{"error": "description of what went wrong"}`.

**Example -- auto mode:**

The AI assistant calls:

```json
{
  "document_text": "CT CHEST WITHOUT CONTRAST\nDate: 2026-02-15\nFindings: No acute cardiopulmonary abnormality. Lungs are clear bilaterally.\nImpression: Normal chest CT.",
  "mode": "auto"
}
```

Returns:

```json
{
  "extracted": {
    "exam_type": "CT Chest",
    "exam_date": "2026-02-15",
    "contrast": "Without contrast",
    "findings": "No acute cardiopulmonary abnormality. Lungs are clear bilaterally.",
    "impression": "Normal chest CT."
  },
  "inferred_schema": {
    "class_name": "RadiologyReport",
    "description": "Schema for a radiology report",
    "fields": [
      {"name": "exam_type", "type": "str", "required": true},
      {"name": "exam_date", "type": "str", "required": true},
      {"name": "contrast", "type": "str", "required": false},
      {"name": "findings", "type": "str", "required": true},
      {"name": "impression", "type": "str", "required": true}
    ]
  }
}
```

**Example -- radiology mode:**

```json
{
  "document_text": "CT CHEST WITHOUT CONTRAST\n...",
  "mode": "radiology"
}
```

Returns domain-specific radiology fields (exam classification, parsed sections, technique details, structured findings, and impression) along with processing metrics.

**Example -- template-based:**

```json
{
  "document_text": "CT CHEST WITHOUT CONTRAST\n...",
  "template": "EchoReport"
}
```

Returns only the fields defined in the `EchoReport` template.

---

### deidentify_text

Remove Protected Health Information (PHI) from clinical text.

Uses a two-layer approach for thorough de-identification:

1. **LLM-based redaction** identifies context-dependent PHI such as patient names, doctor names, addresses, and institution names.
2. **Regex safety net** catches format-based PHI such as Social Security numbers, phone numbers, medical record numbers, and email addresses.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `string` | (required) | The clinical text to de-identify. |
| `mode` | `string` | `"remove"` | De-identification strategy. Options: `"remove"` (replace PHI with `[REDACTED]`), `"pseudonymize"` (replace PHI with realistic fake values), `"dateshift"` (shift all dates by a consistent random offset). |
| `regex_only` | `boolean` | `false` | If true, skip the LLM and use only regex-based scrubbing. Faster but less comprehensive. No API key needed. |

**Return value:**

```json
{
  "redacted_text": "The de-identified text with PHI removed or replaced",
  "mode": "remove"
}
```

If an error occurs, returns `{"error": "description of what went wrong"}`.

**Example:**

The AI assistant calls:

```json
{
  "text": "Patient John Smith, DOB 03/15/1965, MRN 12345678, was seen by Dr. Jane Doe at Springfield General Hospital on 02/10/2026 for follow-up of lung nodule.",
  "mode": "remove"
}
```

Returns:

```json
{
  "redacted_text": "Patient [REDACTED], DOB [REDACTED], MRN [REDACTED], was seen by Dr. [REDACTED] at [REDACTED] on [REDACTED] for follow-up of lung nodule.",
  "mode": "remove"
}
```

---

### generate_template

Generate an extraction template from a natural-language description.

Describe what kind of document you want to extract data from and what fields matter, and the LLM will create a structured template specification. The template is automatically saved to `~/.mosaicx/templates/` and can be used for targeted extraction with `extract_document`.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `description` | `string` | (required) | Natural-language description of the document type and desired fields (e.g., "echocardiography report with LVEF, valve grades, and clinical impression"). |
| `name` | `string` | `null` | Optional class name for the generated template. If not provided, the LLM chooses an appropriate name. |
| `mode` | `string` | `null` | Optional pipeline mode to embed in the template (e.g., `"radiology"`, `"pathology"`). |
| `document_text` | `string` | `null` | Optional sample document text. When provided, the LLM uses it to infer richer field types and structure. |

**Return value:**

```json
{
  "class_name": "EchoReport",
  "description": "Template for echocardiography reports",
  "fields": [
    {"name": "lvef", "type": "float", "required": true, "description": "Left ventricular ejection fraction"},
    {"name": "valve_grades", "type": "dict", "required": true, "description": "Valve grades"},
    {"name": "clinical_impression", "type": "str", "required": true, "description": "Clinical impression"}
  ],
  "_saved_to": "/Users/yourusername/.mosaicx/templates/EchoReport.yaml"
}
```

If an error occurs, returns `{"error": "description of what went wrong"}`.

**Example:**

The AI assistant calls:

```json
{
  "description": "chest CT radiology report with findings, impression, and measurements",
  "name": "ChestCTReport"
}
```

Returns the generated template specification with all inferred fields and the path where it was saved.

---

### summarize_reports

Summarize multiple clinical reports into a patient timeline.

Takes a list of report texts and produces a narrative summary with a structured timeline of clinical events.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `reports` | `list[string]` | (required) | List of clinical report texts to summarize. |
| `patient_id` | `string` | `"unknown"` | Patient identifier for the summary. |

**Return value:**

```json
{
  "narrative": "Patient underwent chest CT on 2026-01-10 showing...",
  "events": [
    {
      "date": "2026-01-10",
      "exam_type": "CT Chest",
      "key_finding": "6mm nodule in RUL",
      "clinical_context": "Routine screening",
      "change_from_prior": null
    }
  ]
}
```

If an error occurs, returns `{"error": "description of what went wrong"}`.

---

### list_templates

List all saved extraction templates.

Returns the templates stored in `~/.mosaicx/templates/` with their class names, field counts, descriptions, and field details. This tool requires no parameters.

**Parameters:**

None.

**Return value:**

```json
{
  "template_dir": "/Users/yourusername/.mosaicx/templates",
  "count": 2,
  "templates": [
    {
      "class_name": "EchoReport",
      "description": "Template for echocardiography reports",
      "field_count": 3,
      "fields": [
        {"name": "lvef", "type": "float", "required": true},
        {"name": "valve_grades", "type": "dict", "required": true},
        {"name": "clinical_impression", "type": "str", "required": true}
      ]
    },
    {
      "class_name": "ChestCTReport",
      "description": "Template for chest CT radiology reports",
      "field_count": 4,
      "fields": [
        {"name": "findings", "type": "str", "required": true},
        {"name": "impression", "type": "str", "required": true},
        {"name": "measurements", "type": "list", "required": false},
        {"name": "exam_date", "type": "str", "required": true}
      ]
    }
  ]
}
```

If no templates are saved, `count` will be `0` and `templates` will be an empty list.

---

### list_modes

List available extraction modes.

Extraction modes are specialized multi-step pipelines for specific document domains. Each mode contains multiple stages (classification, section parsing, domain-specific extraction) optimized for a particular document type.

**Parameters:**

None.

**Return value:**

```json
{
  "count": 2,
  "modes": [
    {"name": "radiology", "description": "Multi-step extraction pipeline for radiology reports"},
    {"name": "pathology", "description": "Multi-step extraction pipeline for pathology reports"}
  ]
}
```

## Transport Modes

The MCP server supports two transport protocols for communication between the AI assistant and the server.

### stdio (default)

Standard input/output transport. The AI assistant launches the MOSAICX MCP server as a subprocess and communicates through stdin/stdout pipes. This is the standard transport for Claude Code and Claude Desktop.

```bash
mosaicx mcp serve --transport stdio
```

Since `stdio` is the default, you can omit the flag:

```bash
mosaicx mcp serve
```

This is the recommended transport for local usage. It requires no network configuration and starts instantly.

### SSE (Server-Sent Events)

HTTP-based transport using Server-Sent Events. The MCP server starts an HTTP server on a specified port, and the AI assistant connects to it over the network. Use this for remote or network-based setups.

```bash
mosaicx mcp serve --transport sse --port 8080
```

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--transport` | `stdio` | Transport protocol: `stdio` or `sse`. |
| `--port` | `8080` | Port for the SSE HTTP server. Ignored when using `stdio` transport. |

**When to use SSE:**

- Running MOSAICX on a remote server that multiple users connect to
- Integrating with AI assistants that do not support subprocess-based MCP
- Debugging MCP tool calls via HTTP

**When to use stdio:**

- Running MOSAICX locally with Claude Code or Claude Desktop
- Single-user setups
- Most common usage

## Configuration

The MCP server uses the same `MOSAICX_*` environment variables as the CLI. No separate configuration is needed. If MOSAICX works from the command line, it will work as an MCP server.

### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MOSAICX_LM` | `openai/gpt-oss:120b` | Primary LLM model identifier |
| `MOSAICX_LM_CHEAP` | `openai/gpt-oss:20b` | Fast model for classification steps |
| `MOSAICX_API_KEY` | `ollama` | API key for the LLM provider |
| `MOSAICX_API_BASE` | `http://localhost:11434/v1` | Base URL for the LLM API endpoint |
| `MOSAICX_HOME_DIR` | `~/.mosaicx` | Directory for templates, optimized programs, and logs |

### Setting environment variables for the MCP server

There are two approaches.

**Approach 1: Pass through the MCP config (recommended)**

Add an `env` block to your MCP server configuration:

```json
{
  "mcpServers": {
    "mosaicx": {
      "command": "mosaicx",
      "args": ["mcp", "serve"],
      "env": {
        "MOSAICX_LM": "openai/gpt-oss:120b",
        "MOSAICX_API_BASE": "http://localhost:8000/v1",
        "MOSAICX_API_KEY": "dummy"
      }
    }
  }
}
```

This keeps configuration self-contained and makes it easy to switch between backends.

**Approach 2: Export in your shell profile**

Add exports to your `~/.zshrc`, `~/.bashrc`, or equivalent:

```bash
export MOSAICX_LM=openai/gpt-oss:120b
export MOSAICX_API_BASE=http://localhost:8000/v1
export MOSAICX_API_KEY=dummy
```

The MCP server inherits environment variables from the shell that launches it.

### Configuring for different backends

**Local Ollama (default, no config needed):**

```json
{
  "mcpServers": {
    "mosaicx": {
      "command": "mosaicx",
      "args": ["mcp", "serve"]
    }
  }
}
```

**Remote vLLM server (via SSH tunnel):**

```json
{
  "mcpServers": {
    "mosaicx": {
      "command": "mosaicx",
      "args": ["mcp", "serve"],
      "env": {
        "MOSAICX_LM": "openai/gpt-oss:120b",
        "MOSAICX_API_BASE": "http://localhost:8000/v1",
        "MOSAICX_API_KEY": "dummy"
      }
    }
  }
}
```

**vLLM-MLX on Apple Silicon:**

```json
{
  "mcpServers": {
    "mosaicx": {
      "command": "mosaicx",
      "args": ["mcp", "serve"],
      "env": {
        "MOSAICX_LM": "openai/mlx-community/gpt-oss-20b-MXFP4-Q8",
        "MOSAICX_API_BASE": "http://localhost:8000/v1",
        "MOSAICX_API_KEY": "dummy"
      }
    }
  }
}
```

For the full list of configuration variables and backend setup instructions, see the [Configuration guide](configuration.md).

## Examples

Below are practical scenarios showing how an AI assistant uses the MOSAICX MCP tools during a conversation.

### Scenario 1: Extract findings from a radiology report

**You say:**

> Hey Claude, extract the findings from this radiology report:
>
> CT CHEST WITH CONTRAST
> Date: 2026-02-15
> Clinical History: 65-year-old male with cough.
> Technique: Helical CT of the chest with IV contrast.
> Findings: The lungs are clear. No pleural effusion. Heart size is normal. No mediastinal lymphadenopathy. A 4mm right lower lobe pulmonary nodule is noted.
> Impression: 1. 4mm right lower lobe pulmonary nodule. Recommend follow-up CT in 12 months per Fleischner criteria. 2. No acute cardiopulmonary abnormality.

**What happens behind the scenes:**

Claude calls the `extract_document` tool with `mode: "radiology"` and passes the report text. The MOSAICX radiology pipeline runs its 5-step extraction (classify exam, parse sections, extract technique, extract findings, extract impression).

**Claude responds with structured data like:**

```json
{
  "exam_type": "CT Chest",
  "contrast": "With contrast",
  "technique": "Helical CT with IV contrast",
  "findings": {
    "lungs": "Clear",
    "pleural_effusion": "None",
    "heart_size": "Normal",
    "mediastinal_lymphadenopathy": "None",
    "nodules": [
      {
        "location": "Right lower lobe",
        "size_mm": 4
      }
    ]
  },
  "impression": [
    "4mm right lower lobe pulmonary nodule. Recommend follow-up CT in 12 months per Fleischner criteria.",
    "No acute cardiopulmonary abnormality."
  ]
}
```

### Scenario 2: De-identify a clinical note

**You say:**

> De-identify this clinical note by removing all patient information:
>
> Patient: Maria Garcia, DOB 07/22/1978, MRN 98765432. Seen by Dr. Robert Chen at University Medical Center on 02/10/2026. Patient presents with a 3-week history of persistent cough. Social Security: 123-45-6789. Phone: (555) 867-5309. Email: mgarcia@email.com.

**What happens behind the scenes:**

Claude calls the `deidentify_text` tool with `mode: "remove"`. The MOSAICX de-identifier runs LLM-based redaction first, then applies the regex safety net.

**Claude responds with:**

```json
{
  "redacted_text": "Patient: [REDACTED], DOB [REDACTED], MRN [REDACTED]. Seen by Dr. [REDACTED] at [REDACTED] on [REDACTED]. Patient presents with a 3-week history of persistent cough. Social Security: [REDACTED]. Phone: [REDACTED]. Email: [REDACTED].",
  "mode": "remove"
}
```

### Scenario 3: Create a template for echocardiography reports

**You say:**

> Create an extraction template for echocardiography reports. I need LVEF, chamber dimensions, valve regurgitation grades, and the clinical impression.

**What happens behind the scenes:**

Claude calls the `generate_template` tool with the description and optionally a name like `"EchoReport"`.

**Claude responds with:**

```json
{
  "class_name": "EchoReport",
  "description": "Template for echocardiography reports including cardiac function and valve assessment",
  "fields": [
    {"name": "lvef_percent", "type": "float", "required": true, "description": "Left ventricular ejection fraction (%)"},
    {"name": "lv_end_diastolic_diameter_cm", "type": "float", "required": false, "description": "LV end-diastolic diameter in cm"},
    {"name": "lv_end_systolic_diameter_cm", "type": "float", "required": false, "description": "LV end-systolic diameter in cm"},
    {"name": "la_diameter_cm", "type": "float", "required": false, "description": "Left atrial diameter in cm"},
    {"name": "mitral_regurgitation", "type": "str", "required": false, "description": "Mitral regurgitation grade"},
    {"name": "aortic_regurgitation", "type": "str", "required": false, "description": "Aortic regurgitation grade"},
    {"name": "tricuspid_regurgitation", "type": "str", "required": false, "description": "Tricuspid regurgitation grade"},
    {"name": "clinical_impression", "type": "str", "required": true, "description": "Overall clinical impression"}
  ],
  "_saved_to": "/Users/yourusername/.mosaicx/templates/EchoReport.yaml"
}
```

You can now use this template for extraction: "Extract data from this echo report using the EchoReport template."

### Scenario 4: Check what extraction modes are available

**You say:**

> What extraction modes are available in MOSAICX?

**What happens behind the scenes:**

Claude calls the `list_modes` tool (no parameters needed).

**Claude responds with:**

> MOSAICX currently has 2 specialized extraction modes:
>
> 1. **radiology** -- Multi-step extraction pipeline for radiology reports. Includes exam classification, section parsing, technique extraction, findings extraction, and impression extraction.
>
> 2. **pathology** -- Multi-step extraction pipeline for pathology reports. Includes specimen classification, section parsing, histology extraction, TNM staging, and biomarker extraction.
>
> You can also use `auto` mode, which lets the LLM infer an appropriate schema from any document type.

## Troubleshooting

### "The 'mcp' package is required"

**Error:**

```
ERROR: The 'mcp' package is required for the MOSAICX MCP server.
Install it with:

    pip install 'mosaicx[mcp]'
```

**Cause:** The MCP dependency is not installed. MCP is an optional extra that is not included in the base MOSAICX installation.

**Solution:**

```bash
pip install 'mosaicx[mcp]'
```

If you installed MOSAICX with `pipx`, reinstall with the extra:

```bash
pipx install 'mosaicx[mcp]'
```

### "No API key configured"

**Error:**

```
No API key configured. Set MOSAICX_API_KEY or add api_key to your config.
```

**Cause:** The MCP server cannot find an API key for the LLM backend.

**Solution:**

Add the API key to your MCP config:

```json
{
  "mcpServers": {
    "mosaicx": {
      "command": "mosaicx",
      "args": ["mcp", "serve"],
      "env": {
        "MOSAICX_API_KEY": "ollama"
      }
    }
  }
}
```

For local Ollama, use `"ollama"` as the key. For cloud APIs, use your actual API key.

### LLM server is not reachable

**Symptom:** Tool calls return `{"error": "Connection refused"}` or time out.

**Cause:** The LLM backend (Ollama, vLLM, etc.) is not running or is not reachable from the MCP server process.

**Solution:**

1. Make sure the LLM server is running:
   - Ollama: `ollama serve`
   - vLLM: `vllm serve model-name --port 8000`
   - vLLM-MLX: `vllm-mlx serve model-name --port 8000`

2. Verify the endpoint matches `MOSAICX_API_BASE`:
   - Ollama: `http://localhost:11434/v1`
   - vLLM/vLLM-MLX: `http://localhost:8000/v1`

3. If using an SSH tunnel, make sure the tunnel is active before starting Claude Code or Claude Desktop.

### MCP server not showing up in Claude

**Symptom:** Claude does not list MOSAICX tools or says it cannot find the server.

**Cause:** The MCP config file is not in the right location, or the `mosaicx` command is not on the system PATH.

**Solution:**

1. Verify that `mosaicx` is available on your PATH:
   ```bash
   which mosaicx
   ```
   If this returns nothing, MOSAICX is not installed globally. You may need to provide the full path to the `mosaicx` binary in the MCP config.

2. Find the full path and update your config:
   ```bash
   python -c "import shutil; print(shutil.which('mosaicx'))"
   ```
   Use the output as the `command` value in your MCP config:
   ```json
   {
     "mcpServers": {
       "mosaicx": {
         "command": "/full/path/to/mosaicx",
         "args": ["mcp", "serve"]
       }
     }
   }
   ```

3. For Claude Desktop, restart the application after changing the config file.

### DSPy import errors

**Error:**

```
DSPy is required for MOSAICX pipelines. Install with: pip install dspy
```

**Cause:** The DSPy library is not installed. This can happen if MOSAICX was installed in a minimal configuration.

**Solution:**

```bash
pip install dspy
```

Or reinstall MOSAICX with all dependencies:

```bash
pip install 'mosaicx[mcp]'
```

### Debugging tool calls

If tools are returning unexpected results, you can run the MCP server manually to see its log output:

```bash
mosaicx mcp serve
```

The server prints its configuration (model name, available tools) on startup. Check that the model and endpoint are correct. For more detailed logs, set the `MOSAICX_HOME_DIR` environment variable and check `~/.mosaicx/logs/mosaicx.log`.

You can also test the server directly using `python -m mosaicx.mcp_server`, which runs the FastMCP server in the same way as `mosaicx mcp serve`.

---

For more information on configuring LLM backends, OCR engines, and batch processing, see the [Configuration guide](configuration.md). For a walkthrough of MOSAICX features, see the [Getting Started guide](getting-started.md).
