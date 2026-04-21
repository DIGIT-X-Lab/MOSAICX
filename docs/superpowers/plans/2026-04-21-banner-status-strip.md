# Banner Status Strip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat model line in the CLI banner with a two-line status strip showing model info (name, ctx window, inference engine) and OCR config (engine, langs), using badge pills for visual hierarchy.

**Architecture:** Add `detect_inference_engine()` to `runtime_env.py` that probes the API endpoint (Ollama/vLLM) with a 300ms timeout and caches the result to `~/.mosaicx/.engine_cache`. Update `print_banner()` in `cli_theme.py` to render two lines with badge pills. Wire it up in `cli.py` by passing config values + detected engine to `print_banner()`.

**Tech Stack:** Python stdlib `urllib`, Rich text styling, JSON file cache

**Spec:** `docs/superpowers/specs/2026-04-21-banner-status-strip-design.md`

---

### Task 1: Inference Engine Detection

**Files:**
- Modify: `mosaicx/runtime_env.py` (add `detect_inference_engine` function)
- Test: `tests/test_runtime_env.py` (add detection tests)

- [ ] **Step 1: Write failing tests for engine detection**

Add to `tests/test_runtime_env.py`:

```python
def test_detect_inference_engine_ollama(tmp_path, monkeypatch):
    """Ollama probe returns engine name + version."""
    from mosaicx.runtime_env import detect_inference_engine

    class _Resp:
        def read(self, *_a, **_k):
            return b'{"version":"0.9.2"}'
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    def _fake_urlopen(req, timeout=0):
        url = str(req.full_url)
        if "/api/version" in url:
            return _Resp()
        raise OSError("not found")

    monkeypatch.setattr("mosaicx.runtime_env.urlopen", _fake_urlopen)

    result = detect_inference_engine("http://localhost:11434/v1", cache_dir=tmp_path)
    assert result == "ollama 0.9.2"


def test_detect_inference_engine_vllm(tmp_path, monkeypatch):
    """vLLM probe returns engine name + version."""
    from mosaicx.runtime_env import detect_inference_engine

    class _Resp:
        def read(self, *_a, **_k):
            return b'{"version":"0.6.0"}'
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    def _fake_urlopen(req, timeout=0):
        url = str(req.full_url)
        if "/version" in url and "/api/" not in url:
            return _Resp()
        raise OSError("not found")

    monkeypatch.setattr("mosaicx.runtime_env.urlopen", _fake_urlopen)

    result = detect_inference_engine("http://localhost:8000/v1", cache_dir=tmp_path)
    assert result == "vllm 0.6.0"


def test_detect_inference_engine_fallback_port(tmp_path, monkeypatch):
    """When probes fail, fall back to port heuristic."""
    from mosaicx.runtime_env import detect_inference_engine

    monkeypatch.setattr(
        "mosaicx.runtime_env.urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("down")),
    )

    result = detect_inference_engine("http://localhost:11434/v1", cache_dir=tmp_path)
    assert result == "ollama"


def test_detect_inference_engine_fallback_unknown_port(tmp_path, monkeypatch):
    """Unknown port falls back to hostname."""
    from mosaicx.runtime_env import detect_inference_engine

    monkeypatch.setattr(
        "mosaicx.runtime_env.urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("down")),
    )

    result = detect_inference_engine("http://gpu-server:9999/v1", cache_dir=tmp_path)
    assert result == "gpu-server"


def test_detect_inference_engine_uses_cache(tmp_path, monkeypatch):
    """Second call reads from cache, no network."""
    from mosaicx.runtime_env import detect_inference_engine
    import json

    cache_file = tmp_path / ".engine_cache"
    cache_file.write_text(json.dumps({
        "api_base": "http://localhost:11434/v1",
        "engine": "ollama 0.9.2",
    }), encoding="utf-8")

    call_count = 0
    def _no_network(*_a, **_k):
        nonlocal call_count
        call_count += 1
        raise AssertionError("should not hit network")

    monkeypatch.setattr("mosaicx.runtime_env.urlopen", _no_network)

    result = detect_inference_engine("http://localhost:11434/v1", cache_dir=tmp_path)
    assert result == "ollama 0.9.2"
    assert call_count == 0


def test_detect_inference_engine_invalidates_stale_cache(tmp_path, monkeypatch):
    """Cache for a different api_base is ignored."""
    from mosaicx.runtime_env import detect_inference_engine
    import json

    cache_file = tmp_path / ".engine_cache"
    cache_file.write_text(json.dumps({
        "api_base": "http://old-server:11434/v1",
        "engine": "ollama 0.8.0",
    }), encoding="utf-8")

    monkeypatch.setattr(
        "mosaicx.runtime_env.urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("down")),
    )

    result = detect_inference_engine("http://localhost:11434/v1", cache_dir=tmp_path)
    assert result == "ollama"  # port fallback, not stale cache
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_runtime_env.py::test_detect_inference_engine_ollama tests/test_runtime_env.py::test_detect_inference_engine_vllm tests/test_runtime_env.py::test_detect_inference_engine_fallback_port tests/test_runtime_env.py::test_detect_inference_engine_fallback_unknown_port tests/test_runtime_env.py::test_detect_inference_engine_uses_cache tests/test_runtime_env.py::test_detect_inference_engine_invalidates_stale_cache -v`

Expected: FAIL with `ImportError: cannot import name 'detect_inference_engine'`

- [ ] **Step 3: Implement `detect_inference_engine`**

Add to `mosaicx/runtime_env.py` at the end of the file (before any trailing whitespace):

```python
def _get_origin(api_base: str) -> str:
    """Strip path from api_base to get the origin (scheme + host + port)."""
    from urllib.parse import urlparse
    parsed = urlparse(api_base)
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _probe_ollama(origin: str, timeout: float) -> str | None:
    """Try Ollama's /api/version endpoint."""
    try:
        req = Request(f"{origin}/api/version", method="GET")
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
            version = data.get("version", "")
            return f"ollama {version}".strip() if version else "ollama"
    except (URLError, OSError, TimeoutError, json.JSONDecodeError):
        return None


def _probe_vllm(origin: str, timeout: float) -> str | None:
    """Try vLLM's /version endpoint."""
    try:
        req = Request(f"{origin}/version", method="GET")
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
            version = data.get("version", "")
            return f"vllm {version}".strip() if version else "vllm"
    except (URLError, OSError, TimeoutError, json.JSONDecodeError):
        return None


def _port_heuristic(api_base: str) -> str:
    """Guess engine from port number, or return hostname."""
    from urllib.parse import urlparse
    parsed = urlparse(api_base)
    port_map = {11434: "ollama", 8000: "vllm"}
    if parsed.port in port_map:
        return port_map[parsed.port]
    return parsed.hostname or "unknown"


def detect_inference_engine(
    api_base: str,
    *,
    cache_dir: Path | None = None,
    timeout: float = 0.3,
) -> str:
    """Auto-detect the inference engine behind an OpenAI-compatible endpoint.

    1. Check cache (~/.mosaicx/.engine_cache) — if api_base matches, return cached.
    2. Probe Ollama (/api/version), then vLLM (/version) with short timeout.
    3. Fall back to port heuristic (11434=ollama, 8000=vllm, else hostname).
    4. Cache successful probe results.

    Never raises — always returns a best-effort string.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".mosaicx"
    cache_file = cache_dir / ".engine_cache"

    # 1. Check cache
    try:
        if cache_file.is_file():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("api_base") == api_base and cached.get("engine"):
                return cached["engine"]
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    # 2. Probe endpoints
    origin = _get_origin(api_base)
    engine = _probe_ollama(origin, timeout) or _probe_vllm(origin, timeout)

    if engine:
        # Cache the result
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"api_base": api_base, "engine": engine}),
                encoding="utf-8",
            )
        except OSError:
            pass
        return engine

    # 3. Port heuristic fallback
    return _port_heuristic(api_base)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_runtime_env.py -k "detect_inference_engine" -v`

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mosaicx/runtime_env.py tests/test_runtime_env.py
git commit -m "feat: add inference engine auto-detection with cache"
```

---

### Task 2: Update `print_banner` to Render Two-Line Status Strip

**Files:**
- Modify: `mosaicx/cli_theme.py:42-76` (rewrite `print_banner`)

- [ ] **Step 1: Write failing test for new banner output**

Create `tests/test_cli_theme.py`:

```python
"""Tests for CLI theme banner rendering."""
from __future__ import annotations

from io import StringIO
from rich.console import Console


def test_print_banner_renders_model_and_ocr_lines():
    """Banner status strip shows model line and OCR line with badges."""
    from mosaicx.cli_theme import print_banner

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    print_banner(
        "2.0.0",
        console,
        lm="openai/gpt-oss:120b",
        num_ctx=131072,
        inference_engine="ollama 0.9.2",
        ocr_engine="paddleocr",
        ocr_langs=["en", "de"],
    )

    output = buf.getvalue()
    assert "gpt-oss:120b" in output
    assert "131k" in output
    assert "ollama 0.9.2" in output
    assert "paddleocr" in output
    assert "en, de" in output


def test_print_banner_omits_via_when_no_engine():
    """When inference_engine is empty, skip the 'via' segment."""
    from mosaicx.cli_theme import print_banner

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    print_banner(
        "2.0.0",
        console,
        lm="openai/gpt-oss:120b",
        num_ctx=131072,
        inference_engine="",
        ocr_engine="paddleocr",
    )

    output = buf.getvalue()
    assert "gpt-oss:120b" in output
    assert "via" not in output


def test_print_banner_minimal_lm_only():
    """With only lm set, model line renders and OCR line is skipped."""
    from mosaicx.cli_theme import print_banner

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    print_banner("2.0.0", console, lm="gemma3:27b")

    output = buf.getvalue()
    assert "gemma3:27b" in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli_theme.py -v`

Expected: FAIL — `print_banner()` doesn't accept the new parameters yet

- [ ] **Step 3: Update `print_banner` in `cli_theme.py`**

Replace the `print_banner` function (lines 42-76) with:

```python
def print_banner(
    version: str,
    console: Console,
    lm: str = "",
    *,
    num_ctx: int = 0,
    inference_engine: str = "",
    ocr_engine: str = "",
    ocr_langs: list[str] | None = None,
) -> None:
    """Print the MOSAICX banner using cfonts with coral-to-greige gradient."""
    try:
        from cfonts import render

        output = render(
            "MOSAICX",
            font="block",
            gradient=[CORAL, GREIGE],
            transition=True,
            space=False,
        )
        # Leading blank line + indent each line by 1 extra space to align with subtitle
        indented = "\n".join(" " + line for line in output.split("\n"))
        console.file.write("\n" + indented)
    except ImportError:
        # Fallback without cfonts
        console.print(f"\n  [bold {CORAL}]{BRAND}[/bold {CORAL}]\n")

    # Subtitle below the banner
    console.print()
    console.print(f"  [{GREIGE}]{TAGLINE}[/{GREIGE}]")
    console.print(f"  [{MUTED}]v{version} \u00b7 {ORG}[/{MUTED}]")

    # Status strip
    if lm or ocr_engine:
        rule = "\u2500" * len(TAGLINE)
        console.print(f"  [{GREIGE}]{rule}[/{GREIGE}]")

    # Line 1: model
    if lm:
        lm_short = lm.split("/", 1)[-1] if "/" in lm else lm
        t = Text("  ")
        t.append(" model ", style=f"reverse {CORAL}")
        t.append(f"  {lm_short}", style="bold")
        if num_ctx > 0:
            ctx_human = f"{num_ctx // 1024}k"
            t.append(f"  \u00b7  ", style=GREIGE)
            t.append("ctx ", style=MUTED)
            t.append(ctx_human, style=GREIGE)
        if inference_engine:
            t.append("  \u00b7  " if num_ctx > 0 else "  \u00b7  ", style=GREIGE)
            t.append("via ", style=MUTED)
            t.append(inference_engine, style=GREIGE)
        console.print(t)

    # Line 2: OCR
    if ocr_engine:
        t = Text("  ")
        t.append("  ocr  ", style=f"reverse {GREIGE}")
        t.append(f"  {ocr_engine}", style="bold")
        if ocr_langs:
            t.append("  \u00b7  ", style=GREIGE)
            t.append("langs ", style=MUTED)
            t.append(", ".join(ocr_langs), style=GREIGE)
        console.print(t)

    console.print()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli_theme.py -v`

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mosaicx/cli_theme.py tests/test_cli_theme.py
git commit -m "feat: two-line banner status strip with model + OCR badges"
```

---

### Task 3: Wire Up `cli.py` to Pass Config + Engine to Banner

**Files:**
- Modify: `mosaicx/cli.py:301-360` (both `format_help` and `cli` function)

- [ ] **Step 1: Update `format_help` in `MosaicxGroup` (line 304-311)**

Replace lines 304-311:

```python
    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Show banner at the top of root-level help (mosaicx --help)
        if ctx.parent is None:
            cfg = get_config()
            theme.print_banner(_VERSION_NUMBER, console, lm=cfg.lm)
```

With:

```python
    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Show banner at the top of root-level help (mosaicx --help)
        if ctx.parent is None:
            cfg = get_config()
            _print_banner_with_config(cfg)
```

- [ ] **Step 2: Update `cli` function (line 354-356)**

Replace lines 354-356:

```python
    if ctx.invoked_subcommand is not None:
        # Subcommand execution — show banner before running the command
        theme.print_banner(_VERSION_NUMBER, console, lm=cfg.lm)
```

With:

```python
    if ctx.invoked_subcommand is not None:
        # Subcommand execution — show banner before running the command
        _print_banner_with_config(cfg)
```

- [ ] **Step 3: Add `_print_banner_with_config` helper above `MosaicxGroup`**

Insert before the `class MosaicxGroup` definition:

```python
def _print_banner_with_config(cfg: "MosaicxConfig") -> None:
    """Resolve runtime info and print the banner."""
    from .runtime_env import detect_inference_engine

    engine = cfg.inference_engine or detect_inference_engine(cfg.api_base)

    theme.print_banner(
        _VERSION_NUMBER,
        console,
        lm=cfg.lm,
        num_ctx=cfg.num_ctx,
        inference_engine=engine,
        ocr_engine=cfg.ocr_engine,
        ocr_langs=cfg.ocr_langs,
    )
```

- [ ] **Step 4: Add `inference_engine` field to `MosaicxConfig`**

In `mosaicx/config.py`, add to the LLM section (after `num_ctx`):

```python
    inference_engine: str = ""  # Auto-detected; override with MOSAICX_INFERENCE_ENGINE
```

- [ ] **Step 5: Verify the full banner renders**

Run: `.venv/bin/python -m mosaicx --help`

Expected: Banner shows two-line status strip:
```
  ───────────────────────────────────────────────────────────────
   model   gpt-oss:120b  ·  ctx 131k  ·  via ollama 0.9.2
    ocr    paddleocr  ·  langs en, de
```

- [ ] **Step 6: Run full test suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/test_runtime_env.py tests/test_cli_theme.py -v`

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add mosaicx/cli.py mosaicx/config.py
git commit -m "feat: wire banner status strip to config + engine detection"
```
