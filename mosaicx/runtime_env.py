"""Runtime environment helpers for local tool execution dependencies."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


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


def _default_deno_dir(home: Path | None = None) -> Path:
    if home is None:
        home = Path.home()
    return home / ".cache" / "deno"


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

    deno_dir = os.environ.get("DENO_DIR", "").strip()
    if not deno_dir:
        default_deno_dir = _default_deno_dir(home)
        try:
            default_deno_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # If directory creation fails, still expose the conventional path.
            pass
        os.environ["DENO_DIR"] = str(default_deno_dir)


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
