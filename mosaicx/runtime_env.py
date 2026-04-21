"""Runtime environment helpers for local tool execution dependencies."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class DenoRuntimeStatus:
    """Runtime health snapshot for Deno-backed RLM execution."""

    available: bool
    deno_path: str | None
    deno_version: str | None
    deno_dir: str
    deno_dir_writable: bool
    path_configured: bool
    issues: list[str]

    @property
    def ok(self) -> bool:
        return self.available and self.deno_dir_writable

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LLMEndpointStatus:
    """Readiness snapshot for OpenAI-compatible LLM endpoints."""

    ok: bool
    api_base: str
    models_ok: bool
    chat_ok: bool
    model_id: str | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _default_deno_dir(home: Path | None = None) -> Path:
    if home is None:
        home = Path.home()
    return home / ".cache" / "deno"


def _workspace_deno_dir() -> Path:
    return Path.cwd() / ".mosaicx_runtime" / "deno"


def _tmp_deno_dir() -> Path:
    return Path("/tmp") / "mosaicx" / "deno"


def _default_dspy_cache_dir(home: Path | None = None) -> Path:
    if home is None:
        home = Path.home()
    return home / ".dspy_cache"


def _workspace_dspy_cache_dir() -> Path:
    return Path.cwd() / ".mosaicx_runtime" / "dspy_cache"


def _tmp_dspy_cache_dir() -> Path:
    return Path("/tmp") / "mosaicx" / "dspy_cache"


def _normalize_api_base_for_http(api_base: str) -> str:
    value = str(api_base or "").strip()
    if not value:
        return value
    value = value.replace("://localhost", "://127.0.0.1").replace("://[::1]", "://127.0.0.1")
    return value.rstrip("/")


def _strip_provider_prefix(model: str) -> str:
    model_text = " ".join(str(model or "").split())
    if "/" not in model_text:
        return model_text
    provider = model_text.split("/", 1)[0].strip().lower()
    known = {
        "openai",
        "azure",
        "anthropic",
        "ollama",
        "gemini",
        "google",
        "vertex_ai",
        "bedrock",
        "cohere",
        "mistral",
        "huggingface",
        "together_ai",
        "groq",
        "xai",
    }
    if provider in known:
        return model_text.split("/", 1)[1]
    return model_text


def _resolve_deno_executable() -> str | None:
    return shutil.which("deno")


def _read_deno_version(deno_path: str) -> str | None:
    try:
        proc = subprocess.run(
            [deno_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.lower().startswith("deno "):
            return line
    return None


def _ensure_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".mosaicx_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def ensure_runtime_env() -> None:
    """Ensure local runtime prerequisites are visible to child processes.

    This prepares environment variables that DSPy's Python interpreter path
    discovery relies on:
    - Adds ``~/.deno/bin`` to ``PATH`` when a Deno binary exists there.
    - Sets ``DENO_DIR`` to ``~/.cache/deno`` when unset.

    The function is idempotent and safe to call repeatedly.
    """
    home = Path.home()
    deno_bin_dir = home / ".deno" / "bin"
    deno_binary = deno_bin_dir / "deno"

    path = os.environ.get("PATH", "")
    path_parts = path.split(os.pathsep) if path else []

    if deno_binary.is_file() and str(deno_bin_dir) not in path_parts:
        os.environ["PATH"] = (
            f"{deno_bin_dir}{os.pathsep}{path}" if path else str(deno_bin_dir)
        )

    env_deno_dir = os.environ.get("DENO_DIR", "").strip()
    candidates: list[Path] = []
    if env_deno_dir:
        candidates.append(Path(env_deno_dir))
    candidates.extend([
        _default_deno_dir(home),
        _workspace_deno_dir(),
        _tmp_deno_dir(),
    ])

    for candidate in candidates:
        if _ensure_writable_dir(candidate):
            os.environ["DENO_DIR"] = str(candidate)
            break
    else:
        # Surface the conventional location even if writes fail.
        os.environ["DENO_DIR"] = str(_default_deno_dir(home))


def ensure_dspy_cache_env(*, preferred_dir: Path | str | None = None) -> str:
    """Ensure DSPy cache directory points to a writable location.

    Returns the cache directory path used.
    """
    home = Path.home()
    candidates: list[Path] = []

    env_cache = os.environ.get("DSPY_CACHEDIR", "").strip()
    if env_cache:
        candidates.append(Path(env_cache))
    if preferred_dir is not None:
        candidates.append(Path(preferred_dir))
    candidates.extend([
        _default_dspy_cache_dir(home),
        _workspace_dspy_cache_dir(),
        _tmp_dspy_cache_dir(),
    ])

    for candidate in candidates:
        if _ensure_writable_dir(candidate):
            cache_path = str(candidate)
            os.environ["DSPY_CACHEDIR"] = cache_path
            return cache_path

    fallback = str(_default_dspy_cache_dir(home))
    os.environ["DSPY_CACHEDIR"] = fallback
    return fallback


def ensure_dspy_runtime_env(*, preferred_cache_dir: Path | str | None = None) -> None:
    """Prepare runtime env for DSPy modules and interpreters.

    This must run before importing ``dspy`` so cache initialization uses
    a writable location in constrained environments.
    """
    ensure_runtime_env()
    ensure_dspy_cache_env(preferred_dir=preferred_cache_dir)
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def import_dspy(*, preferred_cache_dir: Path | str | None = None):
    """Import DSPy after runtime env bootstrap."""
    ensure_dspy_runtime_env(preferred_cache_dir=preferred_cache_dir)
    import dspy

    return dspy



# NOTE: _adapter_policy_sequence and configure_dspy_lm removed.
# All entry points (CLI, SDK, MCP) now configure DSPy directly:
#   dspy.LM(...) + dspy.settings.configure(lm=lm, adapter=BAMLAdapter())


def check_openai_endpoint_ready(
    *,
    api_base: str,
    api_key: str | None = None,
    ping_model: str | None = None,
    timeout_s: float = 5.0,
) -> LLMEndpointStatus:
    """Validate `/models` and `/chat/completions` for an OpenAI-compatible endpoint."""
    base = _normalize_api_base_for_http(api_base)
    if not base:
        return LLMEndpointStatus(
            ok=False,
            api_base="",
            models_ok=False,
            chat_ok=False,
            model_id=None,
            reason="api_base is empty",
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or 'ollama'}",
    }
    models_url = f"{base}/models"
    chat_url = f"{base}/chat/completions"

    try:
        req = Request(models_url, headers=headers, method="GET")
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8", errors="ignore") or "{}")
    except HTTPError as exc:
        return LLMEndpointStatus(
            ok=False,
            api_base=base,
            models_ok=False,
            chat_ok=False,
            model_id=None,
            reason=f"/models HTTP {exc.code}",
        )
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        return LLMEndpointStatus(
            ok=False,
            api_base=base,
            models_ok=False,
            chat_ok=False,
            model_id=None,
            reason=f"/models unreachable: {type(exc).__name__}: {exc}",
        )

    data = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = [
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    ]
    if not model_ids:
        return LLMEndpointStatus(
            ok=False,
            api_base=base,
            models_ok=True,
            chat_ok=False,
            model_id=None,
            reason="/models returned no model ids",
        )

    preferred = _strip_provider_prefix(str(ping_model or ""))
    model_id = preferred if preferred in model_ids else model_ids[0]

    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "temperature": 0.0,
            "max_tokens": 4,
        }
    ).encode("utf-8")
    try:
        req = Request(chat_url, data=body, headers=headers, method="POST")
        with urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            _ = resp.read(1024)
    except HTTPError as exc:
        return LLMEndpointStatus(
            ok=False,
            api_base=base,
            models_ok=True,
            chat_ok=False,
            model_id=model_id,
            reason=f"/chat/completions HTTP {exc.code}",
        )
    except (URLError, OSError, TimeoutError) as exc:
        return LLMEndpointStatus(
            ok=False,
            api_base=base,
            models_ok=True,
            chat_ok=False,
            model_id=model_id,
            reason=f"/chat/completions unreachable: {type(exc).__name__}: {exc}",
        )

    return LLMEndpointStatus(
        ok=True,
        api_base=base,
        models_ok=True,
        chat_ok=True,
        model_id=model_id,
        reason=None,
    )


def get_deno_runtime_status() -> DenoRuntimeStatus:
    """Inspect Deno runtime readiness for RLM code sandbox execution."""
    ensure_runtime_env()
    issues: list[str] = []

    deno_path = _resolve_deno_executable()
    available = deno_path is not None
    deno_version = _read_deno_version(deno_path) if deno_path else None
    deno_dir = os.environ.get("DENO_DIR", "").strip() or str(_default_deno_dir())
    deno_dir_path = Path(deno_dir)
    deno_dir_writable = _ensure_writable_dir(deno_dir_path)

    path_configured = False
    if deno_path:
        deno_parent = str(Path(deno_path).parent)
        path_configured = deno_parent in os.environ.get("PATH", "").split(os.pathsep)

    if not available:
        issues.append("Deno executable not found on PATH.")
    if not deno_dir_writable:
        issues.append(f"DENO_DIR is not writable: {deno_dir}")
    if available and deno_version is None:
        issues.append("Unable to read Deno version output.")

    return DenoRuntimeStatus(
        available=available,
        deno_path=deno_path,
        deno_version=deno_version,
        deno_dir=deno_dir,
        deno_dir_writable=deno_dir_writable,
        path_configured=path_configured,
        issues=issues,
    )


def deno_install_instructions() -> str:
    """Human-friendly Deno installation guidance by platform."""
    system = platform.system().lower()
    if system == "darwin":
        return (
            "Install Deno with `brew install deno` "
            "or `curl -fsSL https://deno.land/install.sh | sh`."
        )
    if system == "linux":
        return "Install Deno with `curl -fsSL https://deno.land/install.sh | sh`."
    if system == "windows":
        return "Install Deno with `winget install DenoLand.Deno`."
    return "Install Deno from https://deno.com/manual/getting_started/installation"


def _deno_install_command() -> list[str]:
    system = platform.system().lower()
    if system == "darwin":
        if shutil.which("brew"):
            return ["brew", "install", "deno"]
        return ["sh", "-c", "curl -fsSL https://deno.land/install.sh | sh"]
    if system == "linux":
        return ["sh", "-c", "curl -fsSL https://deno.land/install.sh | sh"]
    if system == "windows":
        if shutil.which("winget"):
            return ["winget", "install", "DenoLand.Deno"]
        if shutil.which("choco"):
            return ["choco", "install", "deno", "-y"]
    raise RuntimeError(
        "Automatic Deno installation is not supported on this platform. "
        + deno_install_instructions()
    )


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


def install_deno(*, force: bool = False, non_interactive: bool = False) -> DenoRuntimeStatus:
    """Install Deno via platform package manager or official script.

    Parameters
    ----------
    force:
        If ``True``, run install command even when Deno is already available.
    non_interactive:
        If ``True``, set ``CI=1`` while running installer to avoid prompts.
    """
    status_before = get_deno_runtime_status()
    if status_before.available and status_before.deno_dir_writable and not force:
        return status_before

    cmd = _deno_install_command()
    env = os.environ.copy()
    if non_interactive:
        env.setdefault("CI", "1")

    try:
        subprocess.run(cmd, check=True, env=env)
    except Exception as exc:
        raise RuntimeError(
            f"Deno installation failed ({type(exc).__name__}): {exc}. "
            + deno_install_instructions()
        ) from exc

    ensure_runtime_env()
    status_after = get_deno_runtime_status()
    if not status_after.available:
        raise RuntimeError(
            "Deno installer completed but executable is still not available on PATH. "
            + deno_install_instructions()
        )
    if not status_after.deno_dir_writable:
        raise RuntimeError(
            f"Deno is installed but DENO_DIR is not writable: {status_after.deno_dir}"
        )
    return status_after
