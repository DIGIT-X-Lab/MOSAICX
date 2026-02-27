# Smooth Installation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `mosaicx setup`, `mosaicx doctor`, a bootstrap shell script, and quickstart docs so users go from zero to extraction in under 5 minutes.

**Architecture:** A new `mosaicx/setup.py` module handles platform detection, backend probing, and .env generation. Two new CLI commands (`setup`, `doctor`) in `cli.py` expose it. A thin `scripts/setup.sh` bootstraps from scratch. Reuses existing `runtime_env.py` utilities (`check_openai_endpoint_ready`, `LLMEndpointStatus`, Deno install).

**Tech Stack:** Click (CLI), Rich (display), urllib (port probing), platform/shutil (detection), existing `cli_theme.py` for output styling.

**Design doc:** `docs/plans/2026-02-27-smooth-install-design.md`

---

### Task 1: Platform Detection & Backend Probing Module (`mosaicx/setup.py`)

The foundation module. Everything else depends on this.

**Files:**
- Create: `mosaicx/setup.py`
- Create: `tests/test_setup.py`
- Reference: `mosaicx/runtime_env.py:319-425` (reuse `check_openai_endpoint_ready`, `LLMEndpointStatus`)

**Step 1: Write failing tests for platform detection**

```python
# tests/test_setup.py
"""Tests for mosaicx.setup — platform detection, backend probing, env generation."""
from __future__ import annotations

import pytest


@pytest.mark.unit
class TestDetectPlatform:
    def test_returns_string(self):
        from mosaicx.setup import detect_platform

        result = detect_platform()
        assert isinstance(result, str)
        assert result in {
            "macos-arm64",
            "macos-x86_64",
            "dgx-spark",
            "linux-x86_64",
            "linux-aarch64",
            "unknown",
        }

    def test_macos_arm64(self, monkeypatch):
        from mosaicx import setup

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("platform.machine", lambda: "arm64")
        assert setup.detect_platform() == "macos-arm64"

    def test_dgx_spark(self, monkeypatch, tmp_path):
        from mosaicx import setup

        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr("platform.machine", lambda: "aarch64")
        dgx_file = tmp_path / "dgx-release"
        dgx_file.write_text("DGX_PLATFORM=spark\n")
        monkeypatch.setattr(setup, "_DGX_RELEASE_PATH", str(dgx_file))
        assert setup.detect_platform() == "dgx-spark"

    def test_linux_generic(self, monkeypatch):
        from mosaicx import setup

        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr("platform.machine", lambda: "x86_64")
        monkeypatch.setattr(setup, "_DGX_RELEASE_PATH", "/nonexistent/dgx-release")
        assert setup.detect_platform() == "linux-x86_64"
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_setup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mosaicx.setup'`

**Step 3: Write failing tests for backend probing**

Add to `tests/test_setup.py`:

```python
@pytest.mark.unit
class TestProbeBackends:
    def test_returns_list(self):
        from mosaicx.setup import probe_backends

        result = probe_backends(timeout=0.5)
        assert isinstance(result, list)

    def test_backend_result_shape(self):
        from mosaicx.setup import BackendInfo

        b = BackendInfo(name="test", port=8000, url="http://localhost:8000/v1", models=["m1"], reachable=True)
        assert b.name == "test"
        assert b.reachable is True
        assert b.models == ["m1"]

    def test_no_server_running_returns_empty(self):
        """All ports unreachable -> no backends found."""
        from mosaicx.setup import probe_backends

        # Use high ports that nothing listens on
        result = probe_backends(ports={"test": 59999}, timeout=0.3)
        assert len(result) == 0


@pytest.mark.unit
class TestSystemRequirements:
    def test_check_system_returns_dict(self):
        from mosaicx.setup import check_system_requirements

        result = check_system_requirements()
        assert "python_version" in result
        assert "python_ok" in result
        assert "ram_gb" in result
        assert "disk_free_gb" in result


@pytest.mark.unit
class TestGenerateEnv:
    def test_generates_env_string(self):
        from mosaicx.setup import generate_env_content, BackendInfo

        backend = BackendInfo(
            name="vllm-mlx",
            port=8000,
            url="http://localhost:8000/v1",
            models=["mlx-community/gpt-oss-20b-MXFP4-Q8"],
            reachable=True,
        )
        content = generate_env_content(backend)
        assert "MOSAICX_LM=" in content
        assert "MOSAICX_API_BASE=" in content
        assert "MOSAICX_API_KEY=" in content

    def test_model_gets_openai_prefix(self):
        from mosaicx.setup import generate_env_content, BackendInfo

        backend = BackendInfo(
            name="vllm",
            port=8000,
            url="http://localhost:8000/v1",
            models=["gpt-oss:120b"],
            reachable=True,
        )
        content = generate_env_content(backend)
        assert "MOSAICX_LM=openai/gpt-oss:120b" in content
```

**Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_setup.py -v`
Expected: FAIL (module still doesn't exist or classes missing)

**Step 5: Implement `mosaicx/setup.py`**

```python
# mosaicx/setup.py
"""Platform detection, backend probing, and .env generation for mosaicx setup."""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Overridable for testing
_DGX_RELEASE_PATH = "/etc/dgx-release"

# Well-known backend ports
DEFAULT_PORTS: dict[str, int] = {
    "vllm-mlx": 8000,
    "vllm": 8000,
    "ollama": 11434,
    "llama-cpp": 8080,
    "sglang": 30000,
}


@dataclass
class BackendInfo:
    """Detected LLM backend."""

    name: str
    port: int
    url: str
    models: list[str] = field(default_factory=list)
    reachable: bool = False


@dataclass
class SystemInfo:
    """System requirements check result."""

    python_version: str
    python_ok: bool
    ram_gb: float
    ram_ok: bool
    disk_free_gb: float
    disk_ok: bool
    platform: str
    uv_available: bool


def detect_platform() -> str:
    """Detect the current platform.

    Returns one of: 'macos-arm64', 'macos-x86_64', 'dgx-spark',
    'linux-x86_64', 'linux-aarch64', 'unknown'.
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        return f"macos-{machine}" if machine in ("arm64", "x86_64") else "macos-arm64"

    if system == "Linux":
        # Check for DGX Spark
        try:
            if Path(_DGX_RELEASE_PATH).exists():
                return "dgx-spark"
        except OSError:
            pass
        if machine in ("x86_64", "aarch64"):
            return f"linux-{machine}"
        return "linux-x86_64"

    return "unknown"


def check_system_requirements() -> SystemInfo:
    """Check Python version, RAM, disk space, and available tools."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)

    # RAM
    ram_gb = _get_ram_gb()
    ram_ok = ram_gb >= 16.0

    # Disk
    disk_free_gb = _get_disk_free_gb()
    disk_ok = disk_free_gb >= 20.0

    return SystemInfo(
        python_version=py_ver,
        python_ok=py_ok,
        ram_gb=round(ram_gb, 1),
        ram_ok=ram_ok,
        disk_free_gb=round(disk_free_gb, 1),
        disk_ok=disk_ok,
        platform=detect_platform(),
        uv_available=shutil.which("uv") is not None,
    )


def _get_ram_gb() -> float:
    """Get total system RAM in GB."""
    system = platform.system()
    if system == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return int(result.stdout.strip()) / (1024**3)
        except Exception:
            return 0.0
    # Linux: read /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except Exception:
        pass
    return 0.0


def _get_disk_free_gb() -> float:
    """Get free disk space in GB for the home directory."""
    try:
        usage = shutil.disk_usage(Path.home())
        return usage.free / (1024**3)
    except Exception:
        return 0.0


def probe_backends(
    *,
    ports: dict[str, int] | None = None,
    timeout: float = 2.0,
) -> list[BackendInfo]:
    """Probe well-known ports for running LLM backends.

    Returns a list of reachable backends with their loaded models.
    """
    if ports is None:
        ports = DEFAULT_PORTS

    found: list[BackendInfo] = []
    seen_ports: set[int] = set()

    for name, port in ports.items():
        if port in seen_ports:
            continue
        seen_ports.add(port)

        url = f"http://localhost:{port}/v1"

        # Ollama has a different models endpoint
        if name == "ollama":
            models_url = f"http://localhost:{port}/api/tags"
        else:
            models_url = f"{url}/models"

        try:
            headers = {"Authorization": "Bearer dummy"}
            req = Request(models_url, headers=headers, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError):
            continue

        # Extract model IDs
        if name == "ollama":
            # Ollama returns {"models": [{"name": "...", ...}]}
            model_list = payload.get("models", [])
            model_ids = [
                str(m.get("name", ""))
                for m in model_list
                if isinstance(m, dict) and m.get("name")
            ]
        else:
            # OpenAI-compatible: {"data": [{"id": "..."}, ...]}
            data = payload.get("data", [])
            model_ids = [
                str(item.get("id", ""))
                for item in data
                if isinstance(item, dict) and item.get("id")
            ]

        found.append(
            BackendInfo(
                name=name,
                port=port,
                url=url,
                models=model_ids,
                reachable=True,
            )
        )

    return found


def generate_env_content(backend: BackendInfo, *, model: str | None = None) -> str:
    """Generate .env file content for a detected backend."""
    chosen_model = model or (backend.models[0] if backend.models else "gpt-oss:120b")
    # Ensure openai/ prefix for litellm compatibility
    if not chosen_model.startswith(("openai/", "ollama/", "anthropic/")):
        chosen_model = f"openai/{chosen_model}"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"# Generated by mosaicx setup on {now}",
        f"# Backend: {backend.name} on port {backend.port}",
        "",
        f"MOSAICX_LM={chosen_model}",
        f"MOSAICX_API_BASE={backend.url}",
        "MOSAICX_API_KEY=dummy",
        "",
    ]
    return "\n".join(lines)


def write_env_file(content: str, path: Path | None = None) -> Path:
    """Write .env content to a file. Returns the path written to."""
    if path is None:
        path = Path.cwd() / ".env"
    path.write_text(content, encoding="utf-8")
    return path
```

**Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_setup.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add mosaicx/setup.py tests/test_setup.py
git commit -m "feat: add setup module — platform detection, backend probing, env gen (#70)"
```

---

### Task 2: `mosaicx doctor` CLI Command

The health-check command. Simpler than `setup`, so we build it first.

**Files:**
- Modify: `mosaicx/cli.py` (add `doctor` command after the `runtime` group, around line ~3340)
- Create: `tests/test_cli_doctor.py`
- Reference: `mosaicx/setup.py` (Task 1)
- Reference: `mosaicx/runtime_env.py` (Deno checks)
- Reference: `mosaicx/cli_theme.py` (ok/warn/err helpers)

**Step 1: Write failing test for `mosaicx doctor`**

```python
# tests/test_cli_doctor.py
"""Tests for the mosaicx doctor CLI command."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner


@pytest.mark.unit
class TestDoctorCommand:
    def test_doctor_runs_without_error(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        # Should exit 0 even if some checks warn (only fail on blocking issues)
        # But LLM backend won't be running in tests, so exit code may be 1
        assert result.exit_code in (0, 1)
        assert "MOSAICX Doctor" in result.output or "DOCTOR" in result.output.upper()

    def test_doctor_json_output(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])
        # Should produce valid JSON
        assert result.exit_code in (0, 1)
        # Find the JSON in the output (may have banner text before it)
        output = result.output
        # Look for the JSON object
        json_start = output.find("{")
        if json_start >= 0:
            json_text = output[json_start:]
            data = json.loads(json_text)
            assert "checks" in data
            assert "summary" in data

    def test_doctor_checks_python(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        assert "Python" in result.output or "python" in result.output

    def test_doctor_fix_flag_exists(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--fix"])
        assert result.exit_code in (0, 1)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli_doctor.py -v`
Expected: FAIL with `Error: No such command 'doctor'`

**Step 3: Implement `mosaicx doctor` in cli.py**

Add after the `runtime` group (around line 3340 in `cli.py`). The doctor command performs each check sequentially, collecting results, then prints a summary.

```python
# Add to mosaicx/cli.py — after the runtime group, before the query command

@cli.command()
@click.option("--json-output", "json_output", is_flag=True, default=False, help="Output results as JSON.")
@click.option("--fix", "auto_fix", is_flag=True, default=False, help="Automatically fix resolvable issues.")
def doctor(json_output: bool, auto_fix: bool) -> None:
    """Check system health and diagnose configuration issues."""
    import shutil
    import sys

    from .config import get_config
    from .setup import check_system_requirements, probe_backends

    checks: list[dict[str, object]] = []
    cfg = get_config()

    # ── 1. Python version ──
    sysinfo = check_system_requirements()
    checks.append({
        "name": "python",
        "status": "ok" if sysinfo.python_ok else "fail",
        "detail": f"Python {sysinfo.python_version}",
        "fix": "Install Python 3.11+: brew install python@3.11 (Mac) or apt install python3.11 (Linux)" if not sysinfo.python_ok else None,
    })

    # ── 2. MOSAICX version ──
    try:
        from importlib.metadata import version as pkg_version
        mosaicx_ver = pkg_version("mosaicx")
        checks.append({"name": "mosaicx", "status": "ok", "detail": f"mosaicx {mosaicx_ver}", "fix": None})
    except Exception:
        checks.append({"name": "mosaicx", "status": "fail", "detail": "not installed", "fix": "pip install mosaicx"})

    # ── 3. .env file ──
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        # Count MOSAICX_ vars
        count = sum(1 for line in env_path.read_text().splitlines() if line.strip().startswith("MOSAICX_"))
        checks.append({"name": "dotenv", "status": "ok", "detail": f".env loaded ({count} MOSAICX vars)", "fix": None})
    else:
        checks.append({
            "name": "dotenv",
            "status": "warn",
            "detail": "No .env file (using defaults)",
            "fix": "Run: mosaicx setup",
        })
        if auto_fix:
            # Will be handled later when backend is detected
            pass

    # ── 4. LLM backend reachable ──
    backends = probe_backends(timeout=2.0)
    if backends:
        best = backends[0]
        model_count = len(best.models)
        checks.append({
            "name": "llm_backend",
            "status": "ok",
            "detail": f"{best.name} on :{best.port} ({model_count} model{'s' if model_count != 1 else ''} loaded)",
            "fix": None,
        })
    else:
        plat = sysinfo.platform
        if plat.startswith("macos"):
            fix_msg = "Start vLLM-MLX: vllm-mlx serve mlx-community/gpt-oss-20b-MXFP4-Q8 --port 8000"
        elif plat == "dgx-spark":
            fix_msg = "Start vLLM: vllm serve gpt-oss:120b --port 8000"
        else:
            fix_msg = "Start an LLM backend (Ollama, vLLM, etc.) and re-run mosaicx doctor"
        checks.append({
            "name": "llm_backend",
            "status": "fail",
            "detail": f"No LLM backend found on common ports",
            "fix": fix_msg,
        })

    # ── 5. Model loaded ──
    if backends and backends[0].models:
        checks.append({
            "name": "model_loaded",
            "status": "ok",
            "detail": ", ".join(backends[0].models[:3]) + ("..." if len(backends[0].models) > 3 else ""),
            "fix": None,
        })
    elif backends:
        checks.append({
            "name": "model_loaded",
            "status": "fail",
            "detail": "Backend reachable but no models loaded",
            "fix": "Pull a model: ollama pull gpt-oss:20b (Ollama) or check vLLM serve args",
        })

    # ── 6. LLM responds ──
    if backends and backends[0].models:
        from .runtime_env import check_openai_endpoint_ready
        ep_status = check_openai_endpoint_ready(
            api_base=backends[0].url,
            ping_model=backends[0].models[0],
            timeout_s=15.0,
        )
        if ep_status.ok:
            checks.append({"name": "llm_responds", "status": "ok", "detail": f"{ep_status.model_id} responds", "fix": None})
        else:
            checks.append({
                "name": "llm_responds",
                "status": "fail",
                "detail": ep_status.reason or "LLM did not respond",
                "fix": "Check server logs for errors",
            })

    # ── 7. OCR: surya ──
    try:
        import surya  # noqa: F401
        checks.append({"name": "surya_ocr", "status": "ok", "detail": "surya-ocr available", "fix": None})
    except ImportError:
        fix = "pip install surya-ocr"
        checks.append({"name": "surya_ocr", "status": "warn", "detail": "surya-ocr not installed", "fix": fix})
        if auto_fix:
            _doctor_pip_install("surya-ocr", checks, "surya_ocr")

    # ── 8. OCR: chandra ──
    try:
        import chandra  # noqa: F401
        checks.append({"name": "chandra_ocr", "status": "ok", "detail": "chandra-ocr available", "fix": None})
    except ImportError:
        checks.append({"name": "chandra_ocr", "status": "warn", "detail": "chandra-ocr not installed (optional)", "fix": "pip install chandra-ocr"})

    # ── 9. Deno ──
    from .runtime_env import get_deno_runtime_status, install_deno
    deno_status = get_deno_runtime_status()
    if deno_status.ok:
        checks.append({"name": "deno", "status": "ok", "detail": deno_status.deno_version or "available", "fix": None})
    else:
        fix = "curl -fsSL https://deno.land/install.sh | sh"
        checks.append({"name": "deno", "status": "warn", "detail": "deno not found (needed for query command)", "fix": fix})
        if auto_fix:
            try:
                install_deno(non_interactive=True)
                # Update the check
                checks[-1] = {"name": "deno", "status": "ok", "detail": "deno installed (auto-fix)", "fix": None}
            except Exception:
                pass

    # ── 10. Disk space ──
    if sysinfo.disk_ok:
        checks.append({"name": "disk_space", "status": "ok", "detail": f"{sysinfo.disk_free_gb} GB free", "fix": None})
    else:
        checks.append({
            "name": "disk_space",
            "status": "warn",
            "detail": f"Only {sysinfo.disk_free_gb} GB free (20+ GB recommended)",
            "fix": "Free up disk space",
        })

    # ── 11. RAM ──
    if sysinfo.ram_ok:
        checks.append({"name": "ram", "status": "ok", "detail": f"{sysinfo.ram_gb} GB", "fix": None})
    else:
        checks.append({
            "name": "ram",
            "status": "warn",
            "detail": f"{sysinfo.ram_gb} GB (16+ GB recommended, 64+ GB for 120B models)",
            "fix": None,
        })

    # ── Output ──
    passed = sum(1 for c in checks if c["status"] == "ok")
    warned = sum(1 for c in checks if c["status"] == "warn")
    failed = sum(1 for c in checks if c["status"] == "fail")
    total = len(checks)

    if json_output:
        import json as json_mod
        payload = {
            "checks": checks,
            "summary": {"total": total, "passed": passed, "warned": warned, "failed": failed},
        }
        console.print(json_mod.dumps(payload, indent=2, ensure_ascii=False))
    else:
        theme.section("DOCTOR", console)
        console.print()
        for c in checks:
            status = c["status"]
            detail = c["detail"]
            if status == "ok":
                console.print(theme.ok(str(detail)))
            elif status == "warn":
                console.print(theme.warn(str(detail)))
            else:
                console.print(theme.err(str(detail)))

        console.print()
        summary_parts = [f"{passed}/{total} passed"]
        if warned:
            summary_parts.append(f"{warned} warning{'s' if warned != 1 else ''}")
        if failed:
            summary_parts.append(f"{failed} failed")
        console.print(f"  {', '.join(summary_parts)}")

        # Print fix suggestions for non-ok checks
        fixable = [c for c in checks if c.get("fix") and c["status"] != "ok"]
        if fixable:
            console.print()
            console.print(theme.info("To fix:"))
            for c in fixable:
                console.print(f"    {c['name']}: {c['fix']}")

    if failed > 0:
        raise click.exceptions.Exit(1)


def _doctor_pip_install(package: str, checks: list, check_name: str) -> None:
    """Attempt to pip install a package during --fix."""
    import subprocess
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            check=True,
            timeout=120,
        )
        # Update the last check for this name
        for i, c in enumerate(checks):
            if c["name"] == check_name and c["status"] != "ok":
                checks[i] = {"name": check_name, "status": "ok", "detail": f"{package} installed (auto-fix)", "fix": None}
    except Exception:
        pass
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli_doctor.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add mosaicx/cli.py tests/test_cli_doctor.py
git commit -m "feat: add mosaicx doctor health-check command with --fix and --json (#70)"
```

---

### Task 3: `mosaicx setup` CLI Command

The interactive setup wizard.

**Files:**
- Modify: `mosaicx/cli.py` (add `setup` command)
- Modify: `mosaicx/setup.py` (add `install_vllm_mlx()` helper)
- Create: `tests/test_cli_setup.py`
- Reference: `mosaicx/runtime_env.py` (Deno install)
- Reference: `mosaicx/cli_theme.py` (Rich output)

**Step 1: Write failing tests for `mosaicx setup`**

```python
# tests/test_cli_setup.py
"""Tests for the mosaicx setup CLI command."""
from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.mark.unit
class TestSetupCommand:
    def test_setup_command_exists(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0
        assert "setup" in result.output.lower()

    def test_setup_non_interactive_runs(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "--non-interactive"])
        # Should complete without prompting (may warn about no backend)
        assert result.exit_code in (0, 1)

    def test_setup_detects_platform(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "--non-interactive"])
        # Should mention the detected platform
        output_lower = result.output.lower()
        assert any(
            p in output_lower
            for p in ["apple silicon", "dgx spark", "linux", "macos"]
        )

    def test_setup_full_flag_exists(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "--help"])
        assert "--full" in result.output
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli_setup.py -v`
Expected: FAIL with `Error: No such command 'setup'`

**Step 3: Add `install_vllm_mlx()` helper to `mosaicx/setup.py`**

Append to `mosaicx/setup.py`:

```python
def install_vllm_mlx() -> bool:
    """Install vLLM-MLX on macOS. Returns True on success."""
    import subprocess

    # Prefer uv for speed
    if shutil.which("uv"):
        cmd = ["uv", "tool", "install", "git+https://github.com/waybarrios/vllm-mlx.git"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "vllm-mlx"]

    try:
        subprocess.run(cmd, check=True, timeout=300)
        return True
    except Exception:
        return False


def recommend_model(platform_name: str, ram_gb: float) -> str:
    """Recommend a model based on platform and RAM."""
    if platform_name.startswith("macos"):
        if ram_gb >= 64:
            return "mlx-community/gpt-oss-120b-4bit"
        return "mlx-community/gpt-oss-20b-MXFP4-Q8"
    # DGX Spark / Linux with GPU
    if ram_gb >= 64:
        return "gpt-oss:120b"
    return "gpt-oss:20b"


def recommended_lm_for_model(model: str) -> str:
    """Return the MOSAICX_LM value for a model ID."""
    if not model.startswith(("openai/", "ollama/", "anthropic/")):
        return f"openai/{model}"
    return model
```

**Step 4: Implement `mosaicx setup` in cli.py**

Add to `cli.py` before the `doctor` command:

```python
@cli.command()
@click.option("--full", is_flag=True, default=False, help="Also install LLM backend (vLLM-MLX on Mac) and pull model.")
@click.option("--non-interactive", is_flag=True, default=False, help="Accept defaults, no prompts (for CI/scripts).")
@click.option("--backend", type=click.Choice(["auto", "ollama", "vllm", "vllm-mlx", "llama-cpp", "sglang"]), default="auto", help="Override backend auto-detection.")
def setup(full: bool, non_interactive: bool, backend: str) -> None:
    """Interactive setup wizard — detect platform, configure backend, write .env."""
    from .setup import (
        BackendInfo,
        check_system_requirements,
        detect_platform,
        generate_env_content,
        install_vllm_mlx,
        probe_backends,
        recommend_model,
        recommended_lm_for_model,
        write_env_file,
    )

    theme.section("SETUP", console)
    console.print()

    # ── 1. Platform ──
    plat = detect_platform()
    plat_display = {
        "macos-arm64": "Apple Silicon Mac",
        "macos-x86_64": "Intel Mac",
        "dgx-spark": "NVIDIA DGX Spark",
        "linux-x86_64": "Linux (x86_64)",
        "linux-aarch64": "Linux (aarch64)",
    }.get(plat, plat)
    console.print(theme.ok(f"Platform: {plat_display}"))

    # ── 2. System requirements ──
    sysinfo = check_system_requirements()
    if sysinfo.python_ok:
        console.print(theme.ok(f"Python {sysinfo.python_version}"))
    else:
        console.print(theme.err(f"Python {sysinfo.python_version} (need 3.11+)"))
        raise click.ClickException("Python 3.11+ is required. Install it and re-run mosaicx setup.")

    console.print(theme.ok(f"RAM: {sysinfo.ram_gb} GB"))
    if not sysinfo.ram_ok:
        console.print(theme.warn("16+ GB recommended for LLM inference"))

    console.print(theme.ok(f"Disk: {sysinfo.disk_free_gb} GB free"))

    if sysinfo.uv_available:
        console.print(theme.ok("uv available (fast installs)"))

    # ── 3. Detect backends ──
    console.print()
    console.print(theme.info("Scanning for LLM backends..."))
    backends = probe_backends(timeout=2.0)

    chosen_backend: BackendInfo | None = None

    if backend != "auto":
        # User specified a backend
        for b in backends:
            if b.name == backend:
                chosen_backend = b
                break
        if chosen_backend is None and backends:
            console.print(theme.warn(f"Requested backend '{backend}' not found, using first detected"))
            chosen_backend = backends[0]
    elif backends:
        chosen_backend = backends[0]

    if chosen_backend:
        model_str = ", ".join(chosen_backend.models[:3]) if chosen_backend.models else "no models"
        console.print(theme.ok(f"Found {chosen_backend.name} on :{chosen_backend.port} ({model_str})"))
    else:
        console.print(theme.warn("No LLM backend detected on common ports"))

        if full and plat.startswith("macos"):
            # Install vLLM-MLX
            if non_interactive or click.confirm("Install vLLM-MLX for Apple Silicon?", default=True):
                console.print(theme.info("Installing vLLM-MLX..."))
                with theme.spinner("Installing vLLM-MLX...", console):
                    success = install_vllm_mlx()
                if success:
                    console.print(theme.ok("vLLM-MLX installed"))
                    model = recommend_model(plat, sysinfo.ram_gb)
                    console.print(theme.info(f"Start it with: vllm-mlx serve {model} --port 8000"))
                else:
                    console.print(theme.err("Failed to install vLLM-MLX"))
                    console.print(theme.info("Install manually: uv tool install git+https://github.com/waybarrios/vllm-mlx.git"))
        elif full and plat == "dgx-spark":
            console.print(theme.info("On DGX Spark, install vLLM:"))
            console.print(theme.info("  pip install vllm"))
            console.print(theme.info("  vllm serve gpt-oss:120b --port 8000 --gpu-memory-utilization 0.90"))
        elif not full:
            if plat.startswith("macos"):
                console.print(theme.info("Re-run with --full to auto-install vLLM-MLX, or start a backend manually"))
            else:
                console.print(theme.info("Start an LLM backend, then re-run mosaicx setup"))

    # ── 4. Deno ──
    console.print()
    from .runtime_env import get_deno_runtime_status, install_deno

    deno_status = get_deno_runtime_status()
    if deno_status.ok:
        console.print(theme.ok(f"Deno: {deno_status.deno_version or 'available'}"))
    else:
        console.print(theme.warn("Deno not found (needed for query command)"))
        if full:
            if non_interactive or click.confirm("Install Deno?", default=True):
                with theme.spinner("Installing Deno...", console):
                    try:
                        install_deno(non_interactive=True)
                        console.print(theme.ok("Deno installed"))
                    except Exception as exc:
                        console.print(theme.err(f"Failed: {exc}"))
                        console.print(theme.info("Install manually: curl -fsSL https://deno.land/install.sh | sh"))
        else:
            console.print(theme.info("Install later: curl -fsSL https://deno.land/install.sh | sh"))

    # ── 5. Write .env ──
    if chosen_backend:
        console.print()
        model = recommend_model(plat, sysinfo.ram_gb)
        # Use actually loaded model if available
        if chosen_backend.models:
            model = chosen_backend.models[0]
        env_content = generate_env_content(chosen_backend, model=model)
        env_path = Path.cwd() / ".env"

        should_write = True
        if env_path.exists():
            if non_interactive:
                should_write = True  # Overwrite in non-interactive
            else:
                console.print(theme.warn(".env already exists"))
                should_write = click.confirm("Overwrite .env?", default=False)

        if should_write:
            write_env_file(env_content, env_path)
            console.print(theme.ok(f".env written to {env_path}"))
        else:
            console.print(theme.info(".env unchanged"))

    # ── 6. Summary ──
    console.print()
    console.print(f"  [{theme.GREIGE}]{'─' * 50}[/{theme.GREIGE}]")
    console.print()
    if chosen_backend and chosen_backend.models:
        console.print(theme.ok("Setup complete!"))
        console.print()
        console.print(theme.info("Try it:"))
        console.print(f"    mosaicx extract --document report.pdf --mode radiology")
    else:
        console.print(theme.warn("Setup partially complete (no LLM backend detected)"))
        console.print()
        console.print(theme.info("Start an LLM backend, then run:"))
        console.print(f"    mosaicx doctor")
    console.print()
    console.print(theme.info("Health check anytime: mosaicx doctor"))
```

**Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli_setup.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add mosaicx/cli.py mosaicx/setup.py tests/test_cli_setup.py
git commit -m "feat: add mosaicx setup wizard with platform detection and backend config (#70)"
```

---

### Task 4: Bootstrap Shell Script (`scripts/setup.sh`)

**Files:**
- Create: `scripts/setup.sh`

**Step 1: Write the bootstrap script**

```bash
#!/usr/bin/env bash
# MOSAICX bootstrap — installs mosaicx into a venv and runs mosaicx setup.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/DIGIT-X-Lab/MOSAICX/master/scripts/setup.sh | bash
#   curl -fsSL ... | bash -s -- --full   # also install LLM backend + model
set -euo pipefail

VENV_DIR="$HOME/.mosaicx-venv"
FULL_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --full) FULL_FLAG="--full" ;;
  esac
done

echo ""
echo "  MOSAICX Setup"
echo "  ────────────────────────────────────"
echo ""

# ── 1. Find Python >= 3.11 ──
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" &>/dev/null; then
    ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "  ERROR: Python 3.11+ not found."
  if [ "$(uname)" = "Darwin" ]; then
    echo "  Install with: brew install python@3.11"
  else
    echo "  Install with: sudo apt install python3.11 python3.11-venv"
  fi
  exit 1
fi
echo "  Found $PYTHON ($($PYTHON --version 2>&1))"

# ── 2. Check for uv (fast installer) ──
USE_UV=false
if command -v uv &>/dev/null; then
  USE_UV=true
  echo "  Found uv (fast mode)"
elif [ -n "$FULL_FLAG" ]; then
  echo "  Installing uv for faster package management..."
  curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null
  export PATH="$HOME/.local/bin:$PATH"
  if command -v uv &>/dev/null; then
    USE_UV=true
    echo "  uv installed"
  fi
fi

# ── 3. Create virtualenv ──
if [ -d "$VENV_DIR" ]; then
  echo "  Reusing existing venv at $VENV_DIR"
else
  echo "  Creating virtualenv at $VENV_DIR..."
  if $USE_UV; then
    uv venv "$VENV_DIR" --python "$PYTHON"
  else
    "$PYTHON" -m venv "$VENV_DIR"
  fi
fi

# ── 4. Install MOSAICX ──
echo "  Installing mosaicx..."
if $USE_UV; then
  uv pip install --python "$VENV_DIR/bin/python" mosaicx
else
  "$VENV_DIR/bin/pip" install --upgrade pip -q
  "$VENV_DIR/bin/pip" install mosaicx
fi
echo "  mosaicx installed"

# ── 5. Add to PATH ──
VENV_BIN="$VENV_DIR/bin"
case "$SHELL" in
  */zsh)  RC_FILE="$HOME/.zshrc" ;;
  */bash) RC_FILE="$HOME/.bashrc" ;;
  */fish) RC_FILE="$HOME/.config/fish/config.fish" ;;
  *)      RC_FILE="$HOME/.profile" ;;
esac

if ! echo "$PATH" | tr ':' '\n' | grep -qx "$VENV_BIN"; then
  if [ -n "$RC_FILE" ] && ! grep -q "$VENV_BIN" "$RC_FILE" 2>/dev/null; then
    if [[ "$RC_FILE" == *"config.fish" ]]; then
      echo "set -gx PATH $VENV_BIN \$PATH" >> "$RC_FILE"
    else
      echo "export PATH=\"$VENV_BIN:\$PATH\"" >> "$RC_FILE"
    fi
    echo "  Added $VENV_BIN to PATH in $RC_FILE"
  fi
  export PATH="$VENV_BIN:$PATH"
fi

# ── 6. Run mosaicx setup ──
echo ""
"$VENV_BIN/mosaicx" setup $FULL_FLAG --non-interactive

echo ""
echo "  Done! Open a new terminal or run:"
echo "    source $RC_FILE"
echo ""
```

**Step 2: Make executable and test locally**

Run: `chmod +x scripts/setup.sh && bash scripts/setup.sh --help` (should not error on parse)

**Step 3: Commit**

```bash
git add scripts/setup.sh
git commit -m "feat: add bootstrap shell script for one-line install (#70)"
```

---

### Task 5: Quickstart Documentation

**Files:**
- Create: `docs/quickstart.md`
- Modify: `README.md` (update Quick Start section)

**Step 1: Write `docs/quickstart.md`**

```markdown
# Quickstart

Get from zero to your first extraction in under 5 minutes.

## One-Line Install

Mac or Linux — creates a virtualenv, installs MOSAICX, detects your LLM backend:

    curl -fsSL https://raw.githubusercontent.com/DIGIT-X-Lab/MOSAICX/master/scripts/setup.sh | bash

Want the fully automated experience (also installs vLLM-MLX on Mac + Deno)?

    curl -fsSL https://raw.githubusercontent.com/DIGIT-X-Lab/MOSAICX/master/scripts/setup.sh | bash -s -- --full

## Manual Install

If you prefer to manage your own environment:

    pip install mosaicx
    mosaicx setup

Or with the `--full` flag to also install the LLM backend:

    pip install mosaicx
    mosaicx setup --full

## Your First Extraction

    mosaicx extract --document report.pdf --mode radiology

## Verify Everything Works

    mosaicx doctor

If something is broken, doctor tells you exactly what to fix. Auto-fix what it can:

    mosaicx doctor --fix

## Next Steps

- [Getting Started](getting-started.md) -- full walkthrough with explanations
- [Configuration](configuration.md) -- all backend options and env vars
- [CLI Reference](cli-reference.md) -- every command and flag
```

**Step 2: Update README.md Quick Start section**

Replace the current Quick Start section in README.md with a version that leads with the bootstrap script and references quickstart.md.

**Step 3: Commit**

```bash
git add docs/quickstart.md README.md
git commit -m "docs: add quickstart guide and update README Quick Start (#70)"
```

---

### Task 6: Integration Testing & Polish

**Files:**
- Modify: `tests/test_setup.py` (add edge-case tests)
- Modify: `tests/test_cli_doctor.py` (add --fix test)
- Run full test suite

**Step 1: Add edge-case tests**

Add to `tests/test_setup.py`:

```python
@pytest.mark.unit
class TestWriteEnv:
    def test_write_creates_file(self, tmp_path):
        from mosaicx.setup import write_env_file

        path = tmp_path / ".env"
        result = write_env_file("MOSAICX_LM=openai/test\n", path)
        assert result == path
        assert path.read_text().startswith("MOSAICX_LM=")

    def test_write_overwrites_existing(self, tmp_path):
        from mosaicx.setup import write_env_file

        path = tmp_path / ".env"
        path.write_text("OLD=value\n")
        write_env_file("NEW=value\n", path)
        assert "NEW=value" in path.read_text()
        assert "OLD" not in path.read_text()


@pytest.mark.unit
class TestRecommendModel:
    def test_mac_low_ram(self):
        from mosaicx.setup import recommend_model

        assert "20b" in recommend_model("macos-arm64", 32.0).lower()

    def test_mac_high_ram(self):
        from mosaicx.setup import recommend_model

        assert "120b" in recommend_model("macos-arm64", 128.0).lower()

    def test_dgx_default(self):
        from mosaicx.setup import recommend_model

        assert "120b" in recommend_model("dgx-spark", 128.0).lower()
```

**Step 2: Run full test suite**

Run: `.venv/bin/pytest tests/test_setup.py tests/test_cli_setup.py tests/test_cli_doctor.py -v`
Expected: All PASS

**Step 3: Run existing test suite to verify no regressions**

Run: `.venv/bin/pytest tests/ -q -x`
Expected: All existing tests still pass

**Step 4: Final commit**

```bash
git add tests/
git commit -m "test: add edge-case tests for setup and doctor (#70)"
```

---

### Task 7: Final Review & Cleanup

**Files:**
- All files from Tasks 1-6

**Step 1: Run linter**

Run: `.venv/bin/ruff check --fix mosaicx/setup.py tests/test_setup.py tests/test_cli_setup.py tests/test_cli_doctor.py`

**Step 2: Run type checker**

Run: `.venv/bin/mypy mosaicx/setup.py`

**Step 3: Manual smoke test**

Run:
1. `mosaicx setup --non-interactive` — should detect platform, scan ports, print summary
2. `mosaicx doctor` — should show all checks with ok/warn/fail
3. `mosaicx doctor --json` — should output valid JSON

**Step 4: Fix any issues found, commit**

```bash
git add -A
git commit -m "chore: lint and type-check fixes for setup/doctor (#70)"
```
