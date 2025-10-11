from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mosaicx.schema import builder


def test_synthesize_requires_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builder.OpenAI
    monkeypatch.setattr(builder, "OpenAI", None)
    with pytest.raises(RuntimeError):
        builder.synthesize_pydantic_model("demo")
    monkeypatch.setattr(builder, "OpenAI", original)


def test_synthesize_uses_resolved_config(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = {}

    class FakeClient:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            self.base_url = base_url
            self.api_key = api_key

        class chat:
            class completions:
                @staticmethod
                def create(*, model: str, temperature: float, messages):
                    assert model == "demo-model"
                    assert temperature == 0.5
                    messages_dict = {m["role"]: m["content"] for m in messages}
                    assert "Description" in messages_dict["user"]
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content="```python\nfrom pydantic import BaseModel\n\nclass Foo(BaseModel):\n    ...\n```"
                                )
                            )
                        ]
                    )

    monkeypatch.setattr(builder, "OpenAI", FakeClient)
    monkeypatch.setattr(
        builder,
        "resolve_openai_config",
        lambda base_url, api_key: (base_url or "http://example", api_key or "token"),
    )

    code = builder.synthesize_pydantic_model(
        "Example schema",
        class_name="Foo",
        model="demo-model",
        base_url="http://example",
        api_key="token",
        temperature=0.5,
    )

    assert "class Foo" in code
