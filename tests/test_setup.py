"""Tests for mosaicx.setup -- platform detection, backend probing, env generation."""
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


@pytest.mark.unit
class TestProbeBackends:
    def test_returns_list(self):
        from mosaicx.setup import probe_backends

        result = probe_backends(timeout=0.5)
        assert isinstance(result, list)

    def test_backend_result_shape(self):
        from mosaicx.setup import BackendInfo

        b = BackendInfo(
            name="test",
            port=8000,
            url="http://localhost:8000/v1",
            models=["m1"],
            reachable=True,
        )
        assert b.name == "test"
        assert b.reachable is True
        assert b.models == ["m1"]

    def test_no_server_running_returns_empty(self):
        from mosaicx.setup import probe_backends

        result = probe_backends(ports={"test": 59999}, timeout=0.3)
        assert len(result) == 0


@pytest.mark.unit
class TestSystemRequirements:
    def test_check_system_returns_dataclass(self):
        from mosaicx.setup import check_system_requirements

        result = check_system_requirements()
        assert hasattr(result, "python_version")
        assert hasattr(result, "python_ok")
        assert hasattr(result, "ram_gb")
        assert hasattr(result, "disk_free_gb")


@pytest.mark.unit
class TestGenerateEnv:
    def test_generates_env_string(self):
        from mosaicx.setup import BackendInfo, generate_env_content

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
        from mosaicx.setup import BackendInfo, generate_env_content

        backend = BackendInfo(
            name="vllm",
            port=8000,
            url="http://localhost:8000/v1",
            models=["gpt-oss:120b"],
            reachable=True,
        )
        content = generate_env_content(backend)
        assert "MOSAICX_LM=openai/gpt-oss:120b" in content
