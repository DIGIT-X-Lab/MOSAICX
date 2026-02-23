"""Tests for runtime environment bootstrapping."""

from __future__ import annotations

import os
from pathlib import Path


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
