# Quickstart

Get from zero to your first extraction in under 5 minutes.

## One-Line Install

Mac or Linux -- creates a virtualenv, installs MOSAICX, detects your LLM backend:

```bash
curl -fsSL https://raw.githubusercontent.com/DIGIT-X-Lab/MOSAICX/master/scripts/setup.sh | bash
```

Want the fully automated experience (also installs vLLM-MLX on Mac + Deno for query)?

```bash
curl -fsSL https://raw.githubusercontent.com/DIGIT-X-Lab/MOSAICX/master/scripts/setup.sh | bash -s -- --full
```

## Manual Install

If you prefer to manage your own environment:

```bash
pip install mosaicx
mosaicx setup
```

Or with the `--full` flag to also install the LLM backend:

```bash
pip install mosaicx
mosaicx setup --full
```

## Your First Extraction

```bash
mosaicx extract --document report.pdf --mode radiology
```

## Verify Everything Works

```bash
mosaicx doctor
```

If something is broken, doctor tells you exactly what to fix. Auto-fix what it can:

```bash
mosaicx doctor --fix
```

## Next Steps

- [Getting Started](getting-started.md) -- full walkthrough with explanations for every step
- [Configuration](configuration.md) -- all backend options, env vars, OCR settings
- [CLI Reference](cli-reference.md) -- every command and flag
