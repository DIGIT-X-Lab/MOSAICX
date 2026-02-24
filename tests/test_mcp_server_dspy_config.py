from __future__ import annotations

from types import SimpleNamespace

import pytest


def _cfg(tmp_path, *, api_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        api_key=api_key,
        lm="openai/mlx-community/gpt-oss-120b-4bit",
        api_base="http://127.0.0.1:8000/v1",
        lm_temperature=0.0,
        home_dir=tmp_path,
    )


def test_ensure_dspy_uses_runtime_adapter_policy(monkeypatch, tmp_path):
    import mosaicx.mcp_server as mcp_mod

    mcp_mod._dspy_configured = False
    monkeypatch.setattr("mosaicx.config.get_config", lambda: _cfg(tmp_path, api_key="ollama"))

    fake_dspy = SimpleNamespace(settings=SimpleNamespace())
    seen: dict[str, object] = {}

    def _fake_configure_dspy_lm(lm, *, preferred_cache_dir=None, adapter_policy=None):
        seen["preferred_cache_dir"] = preferred_cache_dir
        seen["adapter_policy"] = adapter_policy
        return fake_dspy, "json"

    tracker = object()
    monkeypatch.setattr("mosaicx.runtime_env.configure_dspy_lm", _fake_configure_dspy_lm)
    monkeypatch.setattr("mosaicx.metrics.make_harmony_lm", lambda *args, **kwargs: object())
    monkeypatch.setattr("mosaicx.metrics.TokenTracker", lambda: tracker)
    monkeypatch.setattr("mosaicx.metrics.set_tracker", lambda t: seen.setdefault("tracker", t))

    mcp_mod._ensure_dspy()

    assert mcp_mod._dspy_configured is True
    assert seen["preferred_cache_dir"] == tmp_path / ".dspy_cache"
    assert fake_dspy.settings.usage_tracker is tracker
    assert fake_dspy.settings.track_usage is True


def test_ensure_dspy_requires_api_key(monkeypatch, tmp_path):
    import mosaicx.mcp_server as mcp_mod

    mcp_mod._dspy_configured = False
    monkeypatch.setattr("mosaicx.config.get_config", lambda: _cfg(tmp_path, api_key=""))

    with pytest.raises(RuntimeError, match="No API key configured"):
        mcp_mod._ensure_dspy()
