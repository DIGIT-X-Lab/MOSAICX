"""Tests for runtime environment bootstrapping."""

from __future__ import annotations

import os
import json
from pathlib import Path
from types import SimpleNamespace


def test_ensure_runtime_env_adds_deno_bin_and_deno_dir(tmp_path, monkeypatch):
    from mosaicx.runtime_env import ensure_runtime_env

    fake_home = tmp_path / "home"
    deno_bin_dir = fake_home / ".deno" / "bin"
    deno_bin_dir.mkdir(parents=True, exist_ok=True)
    (deno_bin_dir / "deno").write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("DENO_DIR", raising=False)

    ensure_runtime_env()

    assert str(deno_bin_dir) in os.environ["PATH"].split(":")
    assert os.environ["DENO_DIR"] == str(fake_home / ".cache" / "deno")


def test_ensure_runtime_env_without_deno_bin_keeps_path(tmp_path, monkeypatch):
    from mosaicx.runtime_env import ensure_runtime_env
    import os

    fake_home = tmp_path / "home2"
    fake_home.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.delenv("DENO_DIR", raising=False)

    ensure_runtime_env()

    assert os.environ["PATH"] == "/usr/local/bin:/usr/bin"
    assert os.environ["DENO_DIR"] == str(fake_home / ".cache" / "deno")


def test_ensure_runtime_env_falls_back_to_workspace_when_default_unwritable(tmp_path, monkeypatch):
    from mosaicx.runtime_env import ensure_runtime_env

    fake_home = tmp_path / "home_ro"
    fake_home.mkdir(parents=True, exist_ok=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(Path, "cwd", lambda: workspace)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("DENO_DIR", "/root/readonly/deno")

    def fake_writable(path: Path) -> bool:
        return str(path).startswith(str(workspace))

    monkeypatch.setattr("mosaicx.runtime_env._ensure_writable_dir", fake_writable)

    ensure_runtime_env()

    assert os.environ["DENO_DIR"] == str(workspace / ".mosaicx_runtime" / "deno")


def test_get_deno_runtime_status_missing_deno(tmp_path, monkeypatch):
    from mosaicx.runtime_env import get_deno_runtime_status

    fake_home = tmp_path / "home3"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("DENO_DIR", raising=False)
    monkeypatch.setattr("mosaicx.runtime_env.shutil.which", lambda name: None)

    status = get_deno_runtime_status()

    assert status.available is False
    assert status.deno_path is None
    assert status.deno_dir == str(fake_home / ".cache" / "deno")
    assert status.deno_dir_writable is True
    assert any("not found" in issue.lower() for issue in status.issues)


def test_get_deno_runtime_status_reads_version(tmp_path, monkeypatch):
    from mosaicx.runtime_env import get_deno_runtime_status

    fake_home = tmp_path / "home4"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.setenv("DENO_DIR", str(fake_home / ".cache" / "deno"))
    monkeypatch.setattr(
        "mosaicx.runtime_env.shutil.which",
        lambda name: "/usr/local/bin/deno" if name == "deno" else None,
    )
    monkeypatch.setattr(
        "mosaicx.runtime_env._read_deno_version",
        lambda _: "deno 2.1.0",
    )

    status = get_deno_runtime_status()

    assert status.available is True
    assert status.deno_path == "/usr/local/bin/deno"
    assert status.deno_version == "deno 2.1.0"
    assert status.path_configured is True
    assert status.issues == []


def test_ensure_dspy_cache_env_falls_back_to_workspace_when_default_unwritable(tmp_path, monkeypatch):
    from mosaicx.runtime_env import ensure_dspy_cache_env

    fake_home = tmp_path / "home_dspy_ro"
    fake_home.mkdir(parents=True, exist_ok=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(Path, "cwd", lambda: workspace)
    monkeypatch.setenv("DSPY_CACHEDIR", "/root/readonly/dspy")

    def fake_writable(path: Path) -> bool:
        return str(path).startswith(str(workspace))

    monkeypatch.setattr("mosaicx.runtime_env._ensure_writable_dir", fake_writable)

    cache_dir = ensure_dspy_cache_env()

    expected = str(workspace / ".mosaicx_runtime" / "dspy_cache")
    assert cache_dir == expected
    assert os.environ["DSPY_CACHEDIR"] == expected


def test_configure_dspy_lm_falls_back_to_twostep(monkeypatch):
    from mosaicx import runtime_env

    class _FakeDSPY:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(adapter=None)
            self.last_configure_kwargs = {}

        def JSONAdapter(self):
            raise RuntimeError("json adapter unavailable")

        def TwoStepAdapter(self):
            return {"adapter": "twostep"}

        def configure(self, **kwargs):
            self.last_configure_kwargs = dict(kwargs)

    fake = _FakeDSPY()
    monkeypatch.setattr(runtime_env, "import_dspy", lambda **_: fake)

    _dspy, adapter = runtime_env.configure_dspy_lm(object(), adapter_policy="auto")
    assert _dspy is fake
    assert adapter == "twostep"
    assert "adapter" in fake.last_configure_kwargs


def test_configure_dspy_lm_falls_back_to_none_adapter(monkeypatch):
    from mosaicx import runtime_env

    class _FakeDSPY:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(adapter=None)
            self.last_configure_kwargs = {}

        def JSONAdapter(self):
            raise RuntimeError("json adapter unavailable")

        def TwoStepAdapter(self):
            raise RuntimeError("twostep unavailable")

        def configure(self, **kwargs):
            self.last_configure_kwargs = dict(kwargs)

    fake = _FakeDSPY()
    monkeypatch.setattr(runtime_env, "import_dspy", lambda **_: fake)

    _dspy, adapter = runtime_env.configure_dspy_lm(object(), adapter_policy="auto")
    assert _dspy is fake
    assert adapter == "none"
    assert set(fake.last_configure_kwargs.keys()) == {"lm"}


def test_deno_install_command_prefers_brew_on_macos(monkeypatch):
    from mosaicx.runtime_env import _deno_install_command

    monkeypatch.setattr("mosaicx.runtime_env.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "mosaicx.runtime_env.shutil.which",
        lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None,
    )

    cmd = _deno_install_command()
    assert cmd == ["brew", "install", "deno"]


def test_install_deno_skips_when_available(monkeypatch):
    from mosaicx.runtime_env import DenoRuntimeStatus, install_deno

    status = DenoRuntimeStatus(
        available=True,
        deno_path="/usr/local/bin/deno",
        deno_version="deno 2.1.0",
        deno_dir="/tmp/deno-cache",
        deno_dir_writable=True,
        path_configured=True,
        issues=[],
    )
    monkeypatch.setattr("mosaicx.runtime_env.get_deno_runtime_status", lambda: status)
    monkeypatch.setattr(
        "mosaicx.runtime_env.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("install command should not run when Deno is already available")
        ),
    )

    out = install_deno(force=False)
    assert out is status


def test_check_openai_endpoint_ready_success(monkeypatch):
    from mosaicx.runtime_env import check_openai_endpoint_ready

    class _Resp:
        def __init__(self, payload: str) -> None:
            self._payload = payload

        def read(self, *_args, **_kwargs) -> bytes:
            return self._payload.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001, ARG001
        url = str(req.full_url)
        if url.endswith("/models"):
            return _Resp(json.dumps({"data": [{"id": "mlx-community/gpt-oss-120b-4bit"}]}))
        if url.endswith("/chat/completions"):
            return _Resp(json.dumps({"choices": [{"message": {"content": "OK"}}]}))
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("mosaicx.runtime_env.urlopen", _fake_urlopen)

    status = check_openai_endpoint_ready(
        api_base="http://localhost:8000/v1",
        api_key="ollama",
        ping_model="openai/mlx-community/gpt-oss-120b-4bit",
        timeout_s=1.0,
    )
    assert status.ok is True
    assert status.models_ok is True
    assert status.chat_ok is True
    assert status.model_id == "mlx-community/gpt-oss-120b-4bit"
    assert status.api_base == "http://127.0.0.1:8000/v1"


def test_check_openai_endpoint_ready_returns_models_error(monkeypatch):
    from mosaicx.runtime_env import check_openai_endpoint_ready
    from urllib.error import URLError

    monkeypatch.setattr(
        "mosaicx.runtime_env.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("down")),
    )

    status = check_openai_endpoint_ready(
        api_base="http://127.0.0.1:8000/v1",
        api_key="ollama",
    )
    assert status.ok is False
    assert status.models_ok is False
    assert status.chat_ok is False
    assert status.model_id is None
    assert "/models unreachable" in str(status.reason)


def test_check_openai_endpoint_ready_returns_chat_error(monkeypatch):
    from mosaicx.runtime_env import check_openai_endpoint_ready
    from urllib.error import URLError

    class _Resp:
        def __init__(self, payload: str) -> None:
            self._payload = payload

        def read(self, *_args, **_kwargs) -> bytes:
            return self._payload.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001, ARG001
        url = str(req.full_url)
        if url.endswith("/models"):
            return _Resp(json.dumps({"data": [{"id": "mlx-community/gpt-oss-120b-4bit"}]}))
        raise URLError("chat down")

    monkeypatch.setattr("mosaicx.runtime_env.urlopen", _fake_urlopen)

    status = check_openai_endpoint_ready(
        api_base="http://127.0.0.1:8000/v1",
        api_key="ollama",
    )
    assert status.ok is False
    assert status.models_ok is True
    assert status.chat_ok is False
    assert status.model_id == "mlx-community/gpt-oss-120b-4bit"
    assert "/chat/completions unreachable" in str(status.reason)
