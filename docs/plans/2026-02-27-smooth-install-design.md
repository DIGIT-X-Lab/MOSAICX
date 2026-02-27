# Smooth Installation & Setup — Design Document

**Issue:** [#70](https://github.com/DIGIT-X-Lab/MOSAICX/issues/70)
**Date:** 2026-02-27
**Status:** Approved
**Platforms:** macOS (Apple Silicon), NVIDIA DGX Spark, Linux (generic)

## Goal

Make MOSAICX installation dead-simple for both clinical researchers (non-developers) and developers. A user should go from zero to `mosaicx extract --document report.pdf` in under 5 minutes on Mac, and under 10 minutes on DGX Spark.

## Architecture

Three components work together:

```
curl setup.sh ──> creates venv ──> pip install mosaicx ──> mosaicx setup [--full]
                                                                │
                                                                ├── detect platform
                                                                ├── detect/install backends
                                                                ├── configure .env
                                                                ├── install Deno (for query)
                                                                └── verify connectivity

mosaicx doctor [--fix] [--json]  ← run anytime to diagnose
```

## Component 1: `scripts/setup.sh` — Bootstrap Script

**Invocation:**

```bash
curl -fsSL https://raw.githubusercontent.com/DIGIT-X-Lab/MOSAICX/master/scripts/setup.sh | bash
# or with --full to also install LLM backend + pull model:
curl -fsSL https://raw.githubusercontent.com/DIGIT-X-Lab/MOSAICX/master/scripts/setup.sh | bash -s -- --full
```

**Flow:**

1. Detect platform:
   - macOS arm64 (Apple Silicon)
   - Linux x86_64/aarch64 + `/etc/dgx-release` (DGX Spark)
   - Linux generic
2. Check Python >= 3.11:
   - If missing: print platform-specific install instructions and exit
   - Check `python3.11`, `python3.12`, `python3.13`, `python3` in order
3. Check for `uv` (fast installer):
   - If available: use `uv` for venv creation and package install
   - If not available and `--full` passed: install `uv` via `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Otherwise: fall back to `pip`
4. Create virtualenv:
   - `uv venv ~/.mosaicx-venv` or `python3 -m venv ~/.mosaicx-venv`
5. Install MOSAICX:
   - `uv pip install mosaicx` or `pip install mosaicx`
6. Add to PATH:
   - Detect shell (bash/zsh/fish)
   - Append `export PATH="$HOME/.mosaicx-venv/bin:$PATH"` to rc file
   - Source rc file in current session
7. Hand off:
   - `mosaicx setup --full` (if `--full` was passed) or `mosaicx setup`

**Size:** ~50 lines of bash. No complex logic.

## Component 2: `mosaicx setup` — Python Setup Wizard

**CLI signature:**

```
mosaicx setup [--full] [--non-interactive] [--backend TYPE]
```

- `--full`: Also install vLLM-MLX (Mac), pull model, install Deno
- `--non-interactive`: Accept all defaults, no prompts (for CI/Docker)
- `--backend`: Override auto-detection (`ollama`, `vllm`, `vllm-mlx`, `llama-cpp`, `sglang`)

**Implementation:** New command in `mosaicx/cli.py`. Heavy lifting in a new `mosaicx/setup.py` module.

### Step 1: Platform Detection

```python
def detect_platform() -> str:
    """Return 'macos-arm64', 'dgx-spark', 'linux-x86_64', or 'linux-aarch64'."""
    # Check uname for OS + arch
    # Check /etc/dgx-release for DGX Spark
```

### Step 2: System Requirements Check

| Check | Pass | Warn | Fail |
|-------|------|------|------|
| Python >= 3.11 | Version shown | — | Instructions to upgrade |
| RAM >= 16 GB | Amount shown | 8-16 GB: "small models only" | < 8 GB: "may not work" |
| Disk >= 20 GB free | Amount shown | 10-20 GB: "tight" | < 10 GB: "not enough for models" |

### Step 3: Detect Running LLM Backends

Probe these ports with a timeout of 2 seconds:

| Port | Backend | Check URL |
|------|---------|-----------|
| 8000 | vLLM / vLLM-MLX | `GET /v1/models` |
| 11434 | Ollama | `GET /api/tags` |
| 8080 | llama.cpp | `GET /v1/models` |
| 30000 | SGLang | `GET /v1/models` |

If a backend responds, extract the model list.

### Step 4: Backend Installation (--full only)

**Mac (Apple Silicon):**
- Check if `vllm-mlx` is installed
- If not: `uv tool install git+https://github.com/waybarrios/vllm-mlx.git`
- Choose model based on RAM:
  - < 32 GB: `mlx-community/gpt-oss-20b-MXFP4-Q8`
  - >= 64 GB: Ask user: 20B (fast) or 120B (best quality)?
- Start vLLM-MLX: `vllm-mlx serve <model> --port 8000`
- Wait for server to be ready (probe /v1/models)

**DGX Spark:**
- Check if `vllm` is importable (pip package installed)
- If not: print clear instructions:
  ```
  vLLM is not installed. Install it with:
    pip install vllm
  Then start the server:
    vllm serve gpt-oss:120b --port 8000 --gpu-memory-utilization 0.90
  ```
- If vLLM is installed but not running: offer to start it
- Choose model based on VRAM (128 GB unified on DGX Spark):
  - Default: `gpt-oss:120b`

### Step 5: Deno Installation (--full only, or if user wants query support)

- Check if `deno` is on PATH
- If not:
  - Mac: `curl -fsSL https://deno.land/install.sh | sh`
  - Linux: same
- Verify: `deno --version`

### Step 6: Write .env File

Create/update `.env` in the current directory (or `~/.mosaicx/.env` as global fallback):

```bash
# Generated by mosaicx setup on 2026-02-27
MOSAICX_LM=openai/<detected-model>
MOSAICX_API_BASE=http://localhost:<detected-port>/v1
MOSAICX_API_KEY=dummy
```

If `.env` already exists, show diff and ask to merge/overwrite/skip.

### Step 7: Verify Connectivity

1. Hit `MOSAICX_API_BASE + /models` to confirm backend is reachable
2. Check at least one model is loaded
3. (Optional, if `--full`) Run a tiny test: extract a one-line text to confirm end-to-end works

### Step 8: Print Summary

```
Setup complete!

  Platform:  Apple Silicon Mac
  Backend:   vLLM-MLX on :8000
  Model:     mlx-community/gpt-oss-20b-MXFP4-Q8
  .env:      /Users/you/project/.env
  Deno:      2.1.4

  Try it:
    mosaicx extract --document report.pdf --mode radiology

  Health check anytime:
    mosaicx doctor
```

## Component 3: `mosaicx doctor` — Health Check & Auto-Fix

**CLI signature:**

```
mosaicx doctor [--fix] [--json]
```

- `--fix`: Auto-resolve fixable issues (install missing packages, write .env, start servers)
- `--json`: Machine-readable output for CI pipelines

**Checks (in order):**

| # | Check | How | Fix (--fix) |
|---|-------|-----|-------------|
| 1 | Python >= 3.11 | `sys.version_info` | Not fixable — print instructions |
| 2 | MOSAICX version | `importlib.metadata` | Not fixable — print `pip install --upgrade mosaicx` |
| 3 | .env file | Check cwd and ~/.mosaicx/ | Create default .env |
| 4 | LLM backend reachable | Probe configured port | Mac: offer to start vLLM-MLX. DGX: print start command |
| 5 | Model loaded | `GET /v1/models` | Print pull/download command |
| 6 | LLM responds | Send test prompt | Not fixable — check server logs |
| 7 | OCR: surya-ocr | `import surya` | `pip install surya-ocr` |
| 8 | OCR: chandra-ocr | `import chandra` | `pip install chandra-ocr` |
| 9 | Deno (for query) | `shutil.which("deno")` | `curl -fsSL https://deno.land/install.sh \| sh` |
| 10 | Disk space | `shutil.disk_usage` | Not fixable — warn only |

**Output format (default):**

```
MOSAICX Doctor
──────────────────────────────────────────

 ok  Python 3.13.2
 ok  mosaicx 2.0.0a1
 ok  .env loaded (3 vars)
 ok  vLLM-MLX on :8000 (1 model loaded)
 ok  LLM responds (gpt-oss-20b, 1.2s)
 ok  surya-ocr 0.17.2
 ok  chandra-ocr 0.1.8
 --  deno not found (needed for query)
 ok  142 GB free disk space

──────────────────────────────────────────
 8/9 passed, 1 warning

 To fix:
   deno: curl -fsSL https://deno.land/install.sh | sh
```

**Output format (--json):**

```json
{
  "checks": [
    {"name": "python", "status": "ok", "detail": "3.13.2"},
    {"name": "deno", "status": "warn", "detail": "not found", "fix": "curl -fsSL https://deno.land/install.sh | sh"}
  ],
  "summary": {"passed": 8, "warned": 1, "failed": 0}
}
```

## Component 4: `docs/quickstart.md`

Short, opinionated, no choices. Gets the user from zero to extract in 5 minutes.

```markdown
# Quickstart

## One-Line Install

Mac or Linux:

    curl -fsSL https://raw.githubusercontent.com/DIGIT-X-Lab/MOSAICX/master/scripts/setup.sh | bash

With LLM backend + model (slower, but fully automated):

    curl -fsSL https://raw.githubusercontent.com/DIGIT-X-Lab/MOSAICX/master/scripts/setup.sh | bash -s -- --full

## Manual Install

    pip install mosaicx
    mosaicx setup

## Your First Extraction

    mosaicx extract --document report.pdf

## Check Health

    mosaicx doctor

## Next Steps

- [Getting Started](getting-started.md) — full walkthrough with explanations
- [Configuration](configuration.md) — all backend options and env vars
- [CLI Reference](cli-reference.md) — every command and flag
```

## Component 5: README Update

Update the Quick Start section in `README.md` to lead with the bootstrap script:

```markdown
## Quick Start

    curl -fsSL https://raw.githubusercontent.com/.../setup.sh | bash

Or install manually:

    pip install mosaicx
    mosaicx setup
    mosaicx extract --document report.pdf --mode radiology
```

## File Layout

```
scripts/setup.sh              # Bootstrap shell script (~50 lines)
mosaicx/setup.py              # Platform detection, backend probe, .env writer
mosaicx/cli.py                # New commands: setup, doctor
docs/quickstart.md            # 5-minute guide
README.md                     # Updated Quick Start
```

## Speed Considerations

- Bootstrap script prefers `uv` over `pip` (10-100x faster installs)
- If `--full` is passed and `uv` is not found, install `uv` first
- Model pull defaults to 20B (12 GB) unless user explicitly chooses 120B
- Backend probing uses 2-second timeouts per port
- `mosaicx doctor` runs all checks in <3 seconds (no LLM calls unless checking response)

## Testing Strategy

- Unit tests for platform detection, port probing, .env generation
- Integration test: `mosaicx setup --non-interactive` in CI (no backend running, verify graceful handling)
- Integration test: `mosaicx doctor --json` output schema validation
- Manual test matrix: Mac (Apple Silicon) + DGX Spark
