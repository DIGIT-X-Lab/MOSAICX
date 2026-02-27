"""Platform detection, backend probing, and .env generation for MOSAICX setup.

This is the foundation module for ``mosaicx doctor`` and ``mosaicx setup``.
It detects the host platform (macOS ARM, DGX Spark, generic Linux, ...),
probes well-known LLM backend ports (Ollama, vLLM, llama.cpp), checks
system requirements (Python, RAM, disk), and generates ``.env`` content
for the discovered backend.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DGX_RELEASE_PATH: str = "/etc/dgx-release"
"""Path to the DGX release file. Overridden in tests via monkeypatch."""

DEFAULT_PORTS: dict[str, int] = {
    "vllm-mlx": 8000,
    "vllm": 8000,
    "ollama": 11434,
    "llama-cpp": 8080,
    "sglang": 30000,
}
"""Well-known default ports for LLM backends."""

_MIN_PYTHON: tuple[int, int] = (3, 11)
_MIN_RAM_GB: float = 8.0
_MIN_DISK_GB: float = 10.0

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BackendInfo:
    """Discovery result for a single LLM backend."""

    name: str
    port: int
    url: str
    models: list[str] = field(default_factory=list)
    reachable: bool = False


@dataclass
class SystemInfo:
    """Snapshot of system requirements checks."""

    python_version: str
    python_ok: bool
    ram_gb: float
    ram_ok: bool
    disk_free_gb: float
    disk_ok: bool
    platform: str
    uv_available: bool


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def detect_platform() -> str:
    """Return a short platform tag for the current host.

    Possible values:
    - ``"macos-arm64"``
    - ``"macos-x86_64"``
    - ``"dgx-spark"``
    - ``"linux-x86_64"``
    - ``"linux-aarch64"``
    - ``"unknown"``
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        if machine == "arm64":
            return "macos-arm64"
        return "macos-x86_64"

    if system == "Linux":
        # Check for NVIDIA DGX Spark (aarch64 with /etc/dgx-release).
        if machine == "aarch64" and _is_dgx_spark():
            return "dgx-spark"
        if machine in ("x86_64", "aarch64"):
            return f"linux-{machine}"

    return "unknown"


def _is_dgx_spark() -> bool:
    """Return True when running on an NVIDIA DGX Spark."""
    try:
        text = Path(_DGX_RELEASE_PATH).read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip().startswith("DGX_PLATFORM"):
                return True
        return False
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# System requirements
# ---------------------------------------------------------------------------


def check_system_requirements() -> SystemInfo:
    """Check Python version, RAM, disk space and other prerequisites."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info[:2] >= _MIN_PYTHON

    ram = _get_ram_gb()
    ram_ok = ram >= _MIN_RAM_GB

    disk = _get_disk_free_gb()
    disk_ok = disk >= _MIN_DISK_GB

    plat = detect_platform()
    uv = shutil.which("uv") is not None

    return SystemInfo(
        python_version=py_ver,
        python_ok=py_ok,
        ram_gb=round(ram, 1),
        ram_ok=ram_ok,
        disk_free_gb=round(disk, 1),
        disk_ok=disk_ok,
        platform=plat,
        uv_available=uv,
    )


def _get_ram_gb() -> float:
    """Return total system RAM in GiB."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                timeout=5,
            )
            return int(out.strip()) / (1024**3)
        if system == "Linux":
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        # Value is in kB.
                        kb = int(line.split()[1])
                        return kb / (1024**2)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 0.0


def _get_disk_free_gb() -> float:
    """Return free disk space (GiB) for the current working directory."""
    try:
        st = os.statvfs(os.getcwd())
        return (st.f_bavail * st.f_frsize) / (1024**3)
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# Backend probing
# ---------------------------------------------------------------------------


def probe_backends(
    *,
    ports: dict[str, int] | None = None,
    timeout: float = 2.0,
) -> list[BackendInfo]:
    """Probe well-known LLM backend ports and return reachable backends.

    Parameters
    ----------
    ports:
        Mapping of backend name to port.  Defaults to :data:`DEFAULT_PORTS`.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    list[BackendInfo]
        Only backends that responded successfully (reachable=True).
    """
    if ports is None:
        ports = DEFAULT_PORTS

    found: list[BackendInfo] = []
    for name, port in ports.items():
        info = _probe_single(name, port, timeout)
        if info is not None:
            found.append(info)
    return found


def _probe_single(
    name: str,
    port: int,
    timeout: float,
) -> BackendInfo | None:
    """Probe a single backend port. Return BackendInfo or None."""
    base_url = f"http://localhost:{port}"

    # Ollama uses /api/tags; everything else uses /v1/models.
    if name == "ollama":
        models_url = f"{base_url}/api/tags"
    else:
        models_url = f"{base_url}/v1/models"

    try:
        req = Request(models_url, method="GET")
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="ignore")
            payload = json.loads(body) if body else {}
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError):
        return None

    models = _extract_model_ids(name, payload)

    # Build the canonical OpenAI-compat URL for .env generation.
    if name == "ollama":
        api_url = f"{base_url}/v1"
    else:
        api_url = f"{base_url}/v1"

    return BackendInfo(
        name=name,
        port=port,
        url=api_url,
        models=models,
        reachable=True,
    )


def _extract_model_ids(name: str, payload: dict) -> list[str]:
    """Extract model ID strings from the backend response."""
    if name == "ollama":
        # Ollama /api/tags returns {"models": [{"name": "..."}]}
        raw = payload.get("models", [])
        return [
            str(m.get("name", ""))
            for m in raw
            if isinstance(m, dict) and str(m.get("name", "")).strip()
        ]
    # OpenAI-compatible /v1/models returns {"data": [{"id": "..."}]}
    raw = payload.get("data", [])
    return [
        str(m.get("id", ""))
        for m in raw
        if isinstance(m, dict) and str(m.get("id", "")).strip()
    ]


# ---------------------------------------------------------------------------
# .env generation
# ---------------------------------------------------------------------------


def generate_env_content(
    backend: BackendInfo,
    model: str | None = None,
) -> str:
    """Generate ``.env`` file content for the given backend.

    Parameters
    ----------
    backend:
        A reachable backend from :func:`probe_backends`.
    model:
        Explicit model ID override.  If ``None``, uses the first model
        from ``backend.models``.

    Returns
    -------
    str
        Multi-line string suitable for writing to a ``.env`` file.
    """
    model_id = model or (backend.models[0] if backend.models else "")

    # DSPy requires the ``openai/`` prefix for OpenAI-compatible endpoints.
    if model_id and not model_id.startswith("openai/"):
        lm_value = f"openai/{model_id}"
    else:
        lm_value = model_id

    api_key = "ollama" if backend.name == "ollama" else "no-key"

    lines = [
        "# MOSAICX environment -- auto-generated by mosaicx setup",
        f"MOSAICX_LM={lm_value}",
        f"MOSAICX_API_BASE={backend.url}",
        f"MOSAICX_API_KEY={api_key}",
        "",
    ]
    return "\n".join(lines)


def write_env_file(
    content: str,
    path: str | Path | None = None,
) -> Path:
    """Write ``.env`` content to disk.

    Parameters
    ----------
    content:
        The env file content (from :func:`generate_env_content`).
    path:
        Destination path.  Defaults to ``.env`` in the current directory.

    Returns
    -------
    Path
        The path that was written.
    """
    dest = Path(path) if path else Path.cwd() / ".env"
    dest.write_text(content, encoding="utf-8")
    return dest
