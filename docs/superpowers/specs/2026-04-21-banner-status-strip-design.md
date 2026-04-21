# Banner Status Strip Refresh

**Date:** 2026-04-21
**Status:** Approved
**Scope:** `cli_theme.py`, `cli.py`, new `runtime_env.py` helper

## Problem

The CLI banner model line is flat and uninformative — just `model gpt-oss-120b-4bit` in dim text. The `lm_cheap` parameter was passed but silently ignored. Key runtime info (context window, inference engine, OCR config) is invisible unless the user runs `mosaicx config show`.

## Design

Two-line status strip below the banner rule, using badge pills for visual hierarchy:

```
  ───────────────────────────────────────────────────────────────
   model   gpt-oss-120b-4bit  ·  ctx 131k  ·  via ollama 0.9.2
    ocr    paddleocr  ·  langs en, de
```

### Line 1: Model

| Element | Source | Style |
|---------|--------|-------|
| ` model ` badge | literal | `reverse CORAL` |
| Model name | `cfg.lm`, strip `openai/` prefix | `bold` (terminal default fg) |
| Context window | `cfg.num_ctx`, humanized | `dim` label `ctx` + greige value |
| Inference engine | Auto-detected (see below) | `dim` label `via` + greige value |

Separator: `  ·  ` in GREIGE between each element.

### Line 2: OCR

| Element | Source | Style |
|---------|--------|-------|
| `  ocr  ` badge | literal (padded to match `model` width) | `reverse GREIGE` |
| OCR engine | `cfg.ocr_engine` | `bold` (terminal default fg) |
| Languages | `cfg.ocr_langs`, joined with `, ` | `dim` label `langs` + greige value |

### Inference Engine Detection

Goal: feel magical — the user never configures this, the CLI just knows.

**Detection strategy (ordered):**

1. **Cache hit** — read `~/.mosaicx/.engine_cache`. If present and `api_base` matches, use cached engine name + version. Zero latency.
2. **API probe** — HTTP GET with 300ms timeout against the base origin (strip `/v1` path from `api_base`):
   - `{origin}/api/version` (Ollama: returns `{"version":"0.9.2"}`)
   - `{origin}/version` (vLLM: returns version info)
   - On success, cache result to `~/.mosaicx/.engine_cache` as JSON: `{"api_base": "...", "engine": "ollama", "version": "0.9.2"}`
3. **Port heuristic fallback** — if probe fails/times out:
   - Port 11434 → `ollama`
   - Port 8000 → `vllm`
   - Else → hostname from `api_base`
4. **Config override** — `MOSAICX_INFERENCE_ENGINE` env var / config field. If set, skip detection entirely.

**Cache invalidation:** Cache file stores the `api_base` it was probed against. If `api_base` changes, cache is stale — re-probe.

**Failure mode:** If everything fails, display nothing for the `via` segment — just omit it. Never show an error or delay startup.

### Context Window Formatting

`num_ctx` → human-readable:
- `131072` → `131k`
- `32768` → `32k`
- `4096` → `4k`

Formula: `f"{num_ctx // 1024}k"`

### Function Signature Change

```python
# Before
def print_banner(version: str, console: Console, lm: str = "") -> None:

# After
def print_banner(version: str, console: Console, lm: str = "",
                 num_ctx: int = 0, ocr_engine: str = "",
                 ocr_langs: list[str] | None = None,
                 inference_engine: str = "") -> None:
```

The caller in `cli.py` resolves the inference engine before calling `print_banner`. Detection logic lives in a small helper (either in `runtime_env.py` or `config.py`).

### Light/Dark Terminal Compatibility

- Badge pills use Rich `reverse` which swaps fg/bg using terminal colors — adapts automatically
- Model name uses `bold` with no explicit color — inherits terminal foreground
- Labels use `dim` — Rich halves brightness, works on both (verified in mockup)
- Greige `#B5A89A` and Coral `#E87461` are mid-tone — visible on both extremes

## Files Changed

| File | Change |
|------|--------|
| `mosaicx/cli_theme.py` | Update `print_banner()` to render two-line status strip |
| `mosaicx/cli.py` | Pass new args to `print_banner()`, call engine detection |
| `mosaicx/runtime_env.py` | New: `detect_inference_engine(api_base) -> str`, cache logic |
| `mosaicx/config.py` | Add optional `inference_engine: str = ""` field |

## Out of Scope

- Removing `lm_cheap` from config (still used elsewhere in extraction pipeline)
- Changing any other CLI display elements
- Adding engine detection to SDK/MCP paths
