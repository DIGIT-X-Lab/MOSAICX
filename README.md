<div align="center">
  <img src="assets/mosaicx_hero.png" alt="MOSAICX" width="700"/>
</div>

<p align="center">
  <a href="https://pypi.org/project/mosaicx/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mosaicx.svg?label=PyPI&style=flat-square&logo=python&logoColor=white&color=bd93f9"></a>
  <a href="https://doi.org/10.5281/zenodo.17601890"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17601890-ff79c6?style=flat-square&labelColor=282a36&logo=zenodo&logoColor=white"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-50fa7b?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-ff79c6?style=flat-square&logo=apache&logoColor=white"></a>
  <a href="https://pepy.tech/project/mosaicx"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mosaicx?style=flat-square&color=8be9fd&label=Downloads"></a>
</p>

<p align="center">
  <strong><a href="https://www.linkedin.com/company/digitx-lmu/">DIGIT-X Lab</a> &middot; LMU Munich</strong><br>
  Turn unstructured medical documents into validated, machine-readable JSON.<br>
  Runs locally &mdash; no PHI leaves your machine.
</p>

---

## How It Works

```mermaid
flowchart LR
    A["PDF / Image / Text"] --> B["Dual-Engine OCR"]
    B --> C["DSPy Pipeline"]
    C --> D["Validated JSON"]

    style A fill:#B5A89A,stroke:#8a7e72,color:#fff
    style B fill:#E87461,stroke:#c25a49,color:#fff
    style C fill:#E87461,stroke:#c25a49,color:#fff
    style D fill:#B5A89A,stroke:#8a7e72,color:#fff
```

MOSAICX ships with specialized pipelines for **radiology** and **pathology** reports, a **generic extraction** mode that adapts to any document, plus **de-identification** and **patient timeline summarization**. Every pipeline is a DSPy module -- meaning it can be optimized with labeled data for your specific use case.

## Quick Start

```bash
# Install MOSAICX
pip install mosaicx               # or: uv add mosaicx / pipx install mosaicx

# Start a local LLM (pick one)
ollama serve && ollama pull gpt-oss:20b                              # Ollama
vllm-mlx serve mlx-community/gpt-oss-20b-MXFP4-Q8 --port 8000      # Apple Silicon (vLLM-MLX)

# Extract structured data from a report
mosaicx extract --document report.pdf --mode radiology
```

> [!TIP]
> First time? The [Getting Started guide](docs/getting-started.md) walks you through setup and your first extraction in under 10 minutes.

## What You Can Do

| Capability | Commands | Guide |
|------------|----------|-------|
| **Extract structured data** from clinical documents | `mosaicx extract`, `mosaicx batch` | [Pipelines](docs/pipelines.md) |
| **Create and manage schemas** for custom extraction targets | `mosaicx schema generate / list / refine` | [Schemas & Templates](docs/schemas-and-templates.md) |
| **De-identify** reports (LLM + regex belt-and-suspenders) | `mosaicx deidentify` | [CLI Reference](docs/cli-reference.md) |
| **Summarize patient timelines** across multiple reports | `mosaicx summarize` | [CLI Reference](docs/cli-reference.md) |
| **Optimize pipelines** with labeled data (DSPy) | `mosaicx optimize`, `mosaicx eval` | [Optimization](docs/optimization.md) |
| **Extend** with custom pipelines, MCP server, Python SDK | `mosaicx pipeline new`, `mosaicx mcp serve` | [Developer Guide](docs/developer-guide.md) |

Run any command with `--help` for full options. Complete reference: [docs/cli-reference.md](docs/cli-reference.md)

## Recipes

```bash
# Radiology report -> structured JSON
mosaicx extract --document ct_chest.pdf --mode radiology

# Schema-driven extraction (define your own fields)
mosaicx schema generate --description "echo report with LVEF, valve grades, impression"
mosaicx extract --document echo.pdf --schema EchoReport

# Batch-process a folder of reports
mosaicx batch --input-dir ./reports --output-dir ./structured --mode radiology --format jsonl

# De-identify a clinical note
mosaicx deidentify --document note.txt

# Patient timeline from multiple reports
mosaicx summarize --dir ./patient_001/ --patient P001
```

See the full [CLI Reference](docs/cli-reference.md) for every flag and option.

## Privacy

> [!IMPORTANT]
> **Data stays on your machine.** MOSAICX runs against a local inference server by default -- no external API calls, no cloud uploads. For HIPAA/GDPR compliance guidance and cloud backend caveats, see [Configuration](docs/configuration.md).

## LLM Backends

MOSAICX talks to any OpenAI-compatible endpoint via DSPy + litellm. Pick the backend that fits your hardware -- override with env vars.

| Backend | Port | Example |
|---------|------|---------|
| **Ollama** | 11434 | Works out-of-the-box, no config needed |
| **llama.cpp** | 8080 | `llama-server -m model.gguf --port 8080` |
| **vLLM** | 8000 | `vllm serve gpt-oss:120b` |
| **SGLang** | 30000 | `python -m sglang.launch_server --model-path gpt-oss:120b` |
| **vLLM-MLX** | 8000 | `vllm-mlx serve mlx-community/gpt-oss-20b-MXFP4-Q8` (Apple Silicon) |

```bash
export MOSAICX_LM=openai/gpt-oss:120b
export MOSAICX_API_BASE=http://localhost:8000/v1   # point at your server
export MOSAICX_API_KEY=dummy                       # or your real key for cloud APIs
```

<details>
<summary><strong>Remote GPU server via SSH tunnel</strong></summary>

Forward the server port to your local machine, then point MOSAICX at `localhost`:

```bash
# Pick the port that matches your backend
ssh -L 8080:localhost:8080  user@gpu-server    # llama.cpp
ssh -L 8000:localhost:8000  user@gpu-server    # vLLM
ssh -L 30000:localhost:30000 user@gpu-server   # SGLang

# Configure MOSAICX
export MOSAICX_LM=openai/gpt-oss:120b
export MOSAICX_API_BASE=http://localhost:8000/v1
export MOSAICX_API_KEY=dummy
```

</details>

<details>
<summary><strong>Apple Silicon: vLLM-MLX</strong></summary>

[vLLM-MLX](https://github.com/waybarrios/vllm-mlx) runs vLLM-compatible inference natively on Apple Silicon using MLX unified memory. No remote GPU needed -- quantized models up to ~70B on a 128 GB Mac.

```bash
# Install
uv tool install git+https://github.com/waybarrios/vllm-mlx.git

# Serve (pick a model size)
vllm-mlx serve mlx-community/gpt-oss-20b-MXFP4-Q8 --port 8000     # 12 GB
vllm-mlx serve mlx-community/gpt-oss-120b-4bit --port 8000         # ~64 GB

# Enable continuous batching for concurrent requests
vllm-mlx serve mlx-community/gpt-oss-20b-MXFP4-Q8 --port 8000 --continuous-batching

# Configure MOSAICX
export MOSAICX_LM=openai/mlx-community/gpt-oss-20b-MXFP4-Q8
export MOSAICX_API_BASE=http://localhost:8000/v1
export MOSAICX_API_KEY=dummy
```

> **Note:** Do **not** use `--reasoning-parser` -- MOSAICX strips Harmony channel tokens client-side.
>
> **Troubleshooting:** If `--continuous-batching` crashes with `broadcast_shapes` or `BatchKVCache` errors, clear the prefix cache: `rm -rf ~/.cache/vllm-mlx/prefix_cache/mlx-community--gpt-oss-*` and restart. See [Troubleshooting](docs/configuration.md#vllm-mlx-crashes-with-broadcast_shapes-or-batchkvcache-errors).

</details>

<details>
<summary><strong>Batch processing tuning (vLLM / SGLang)</strong></summary>

For batch-processing many documents, vLLM or SGLang give much higher throughput than Ollama via continuous batching.

**vLLM server-side:**

```bash
# 120B -- constrain concurrency to avoid OOM
vllm serve gpt-oss:120b --gpu-memory-utilization 0.90 --max-num-seqs 4 --port 8000

# 20B -- higher concurrency
vllm serve gpt-oss:20b --gpu-memory-utilization 0.90 --max-num-seqs 16 --port 8000
```

**SGLang server-side:**

```bash
# 120B
python -m sglang.launch_server --model-path gpt-oss:120b \
  --mem-fraction-static 0.90 --max-running-requests 4 --port 30000

# 20B
python -m sglang.launch_server --model-path gpt-oss:20b \
  --mem-fraction-static 0.90 --max-running-requests 16 --port 30000
```

**Tuning `--workers`:**

| Setup | Server concurrency | `--workers` | Notes |
|-------|-------------------|-------------|-------|
| 120B, 128 GB VRAM | 4 | 4 | Best quality, moderate speed |
| 20B, 128 GB VRAM | 16 | 16 | Faster throughput, good quality |
| 120B quantized (AWQ) | 8 | 8 | Balance of quality and speed |

> **Tip:** Setting `--workers` higher than the server's max concurrency wastes overhead -- requests queue server-side. Keep them matched.

</details>

<details>
<summary><strong>Benchmarking backends</strong></summary>

Compare backend performance on your hardware with the included benchmark script:

```bash
# Benchmark all reachable backends
python scripts/benchmark_backends.py --document report.txt

# Multiple runs for stable averages
python scripts/benchmark_backends.py --document report.txt --runs 3 --mode radiology

# Add a custom backend
python scripts/benchmark_backends.py --document report.txt \
  --backend "dgx=openai/gpt-oss:120b@http://localhost:8000/v1"

# Only test specific backends
python scripts/benchmark_backends.py --document report.txt --only vllm-mlx,ollama

# Save results as JSON
python scripts/benchmark_backends.py --document report.txt --runs 3 --output results.json
```

Default backends probed: **vllm-mlx** (:8000), **ollama** (:11434), **llama-cpp** (:8080), **sglang** (:30000). Offline backends are skipped automatically.

</details>

Full backend configuration: [docs/configuration.md](docs/configuration.md)

## OCR Engines

| Engine | Approach | Best for |
|--------|----------|----------|
| **Surya** | Layout detection + recognition | Clean printed text, fast |
| **Chandra** | Vision-Language Model (Qwen3-VL 9B) | Handwriting, complex layouts, tables |

By default both engines run in parallel, score each page, and pick the best result. Override with `MOSAICX_OCR_ENGINE=surya` or `chandra`.

## Configuration

```bash
# Essential vars -- point at your local server
export MOSAICX_LM=openai/mlx-community/gpt-oss-20b-MXFP4-Q8   # model name
export MOSAICX_API_BASE=http://localhost:8000/v1                # server URL
export MOSAICX_API_KEY=dummy                                    # or real key for cloud

# View active config
mosaicx config show
```

Full variable reference, `.env` file setup, and backend scenarios: [docs/configuration.md](docs/configuration.md)

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Install, first extraction, basics |
| [CLI Reference](docs/cli-reference.md) | Every command, every flag, examples |
| [Pipelines](docs/pipelines.md) | Pipeline inputs/outputs, JSONL formats |
| [Schemas & Templates](docs/schemas-and-templates.md) | Create and manage extraction schemas |
| [Optimization](docs/optimization.md) | Improve accuracy with DSPy optimizers |
| [Configuration](docs/configuration.md) | Env vars, backends, OCR, export formats |
| [MCP Server](docs/mcp-server.md) | AI agent integration via MCP |
| [Developer Guide](docs/developer-guide.md) | Custom pipelines, Python SDK |
| [Architecture](docs/architecture.md) | System design, key decisions |

## Development

```bash
git clone https://github.com/DIGIT-X-Lab/MOSAICX.git
cd MOSAICX
pip install -e ".[dev]"          # or: uv sync --group dev
pytest tests/ -q
```

See [Developer Guide](docs/developer-guide.md) for custom pipelines and the Python SDK.

## Citation

```bibtex
@software{mosaicx2025,
  title   = {MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction},
  author  = {Sundar, Lalith Kumar Shiyam and DIGIT-X Lab},
  year    = {2025},
  url     = {https://github.com/DIGIT-X-Lab/MOSAICX},
  doi     = {10.5281/zenodo.17601890}
}
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).

## Contact

**Research:** [lalith.shiyam@med.uni-muenchen.de](mailto:lalith.shiyam@med.uni-muenchen.de) | **Commercial:** [lalith@zenta.solutions](mailto:lalith@zenta.solutions) | **Issues:** [github.com/DIGIT-X-Lab/MOSAICX/issues](https://github.com/DIGIT-X-Lab/MOSAICX/issues)
