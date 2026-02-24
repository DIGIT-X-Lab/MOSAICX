"""Token usage tracking, per-step timing metrics, and LLM helpers.

Provides a lightweight ``TokenTracker`` compatible with
``dspy.settings.usage_tracker`` and a ``track_step`` context manager
that captures wall-clock time + token deltas for each pipeline step.

Also provides ``strip_harmony_tokens`` and ``HarmonyLM`` — a ``dspy.LM``
subclass that strips gpt-oss Harmony channel tokens from every response
before DSPy's adapter parses them.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StepMetric:
    """Metrics for a single pipeline step."""

    name: str
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class PipelineMetrics:
    """Aggregated metrics for an entire pipeline run."""

    steps: list[StepMetric] = field(default_factory=list)

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.steps)

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


# ---------------------------------------------------------------------------
# Token tracker (DSPy usage_tracker interface)
# ---------------------------------------------------------------------------


class TokenTracker:
    """Simple usage tracker compatible with ``dspy.settings.usage_tracker``.

    DSPy calls ``add_usage(model, usage_dict)`` after each LLM completion.
    The *usage_dict* contains ``prompt_tokens``, ``completion_tokens``,
    and ``total_tokens``.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add_usage(self, model: str, usage: dict) -> None:  # noqa: ARG002
        self.calls.append(usage)

    def reset(self) -> None:
        self.calls.clear()

    def snapshot(self) -> int:
        """Return an opaque marker for the current position."""
        return len(self.calls)

    def tokens_since(self, snap: int) -> tuple[int, int]:
        """Return ``(input_tokens, output_tokens)`` accumulated since *snap*."""
        recent = self.calls[snap:]
        inp = sum(c.get("prompt_tokens", 0) for c in recent)
        out = sum(c.get("completion_tokens", 0) for c in recent)
        return inp, out


# ---------------------------------------------------------------------------
# Module-level tracker accessor
# ---------------------------------------------------------------------------

_tracker: TokenTracker | None = None


def get_tracker() -> TokenTracker:
    """Return the active ``TokenTracker``, creating one if needed."""
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker


def set_tracker(tracker: TokenTracker) -> None:
    """Install *tracker* as the module-level active tracker."""
    global _tracker
    _tracker = tracker


# ---------------------------------------------------------------------------
# Step-timing context manager
# ---------------------------------------------------------------------------


@contextmanager
def track_step(
    metrics: PipelineMetrics,
    step_name: str,
    tracker: TokenTracker | None = None,
) -> Generator[None, None, None]:
    """Time a pipeline step and capture its token delta.

    Usage::

        metrics = PipelineMetrics()
        tracker = get_tracker()
        with track_step(metrics, "Classify exam type", tracker):
            result = self.classify_exam(...)
    """
    if tracker is None:
        tracker = get_tracker()
    snap = tracker.snapshot()
    t0 = time.perf_counter()
    yield
    duration = time.perf_counter() - t0
    inp, out = tracker.tokens_since(snap)
    metrics.steps.append(StepMetric(step_name, duration, inp, out))


# ---------------------------------------------------------------------------
# Harmony token stripping (gpt-oss via vLLM-MLX)
# ---------------------------------------------------------------------------

_HARMONY_FINAL = "<|channel|>final<|message|>"


def strip_harmony_tokens(text: str) -> str:
    """Extract the final-channel content from a Harmony-formatted response.

    gpt-oss models wrap output in channel tokens::

        <|channel|>analysis<|message|>...thinking...
        <|start|>assistant<|channel|>final<|message|>...answer...

    Returns only the ``final`` channel content.  If no Harmony tokens
    are present the text is returned unchanged, so this is safe to call
    on responses from any backend.
    """
    if _HARMONY_FINAL in text:
        return text.split(_HARMONY_FINAL, 1)[1]
    return text


def _normalize_local_api_base(value: object) -> object:
    """Normalize localhost API bases to IPv4 loopback for transport stability."""
    if not isinstance(value, str):
        return value
    out = value
    out = out.replace("://localhost", "://127.0.0.1")
    out = out.replace("://[::1]", "://127.0.0.1")
    return out


def _normalize_model_for_api_base(model: object, api_base: object) -> object:
    """Normalize model/provider for local OpenAI-compatible endpoints.

    If a local ``api_base`` is configured and model has no provider prefix,
    force ``openai/`` so LiteLLM routes through the expected provider.
    """
    if not isinstance(model, str):
        return model

    model_name = model.strip()
    if not model_name:
        return model

    provider_prefixes = {
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
    if "/" in model_name:
        provider = model_name.split("/", 1)[0].strip().lower()
        if provider in provider_prefixes:
            return model_name

    base_norm = _normalize_local_api_base(api_base)
    if not isinstance(base_norm, str) or not base_norm.strip():
        return model_name

    parsed = urlparse(base_norm)
    host = (parsed.hostname or "").strip().lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return model_name

    # Local OpenAI-compatible servers (vLLM / vLLM-MLX / Ollama OAI mode).
    if "openai/" not in model_name.lower():
        return f"openai/{model_name}"
    return model_name


def make_harmony_lm(model: str, **kwargs: object) -> object:
    """Create a ``dspy.LM`` that strips Harmony tokens from every response.

    Drop-in replacement for ``dspy.LM(model, ...)`` that guarantees
    DSPy's adapter sees clean text regardless of backend.
    """
    from .runtime_env import import_dspy

    dspy = import_dspy()
    lm_kwargs = dict(kwargs)
    if "api_base" in lm_kwargs:
        lm_kwargs["api_base"] = _normalize_local_api_base(lm_kwargs.get("api_base"))
    if "base_url" in lm_kwargs:
        lm_kwargs["base_url"] = _normalize_local_api_base(lm_kwargs.get("base_url"))

    effective_model = _normalize_model_for_api_base(
        model,
        lm_kwargs.get("api_base") if "api_base" in lm_kwargs else lm_kwargs.get("base_url"),
    )

    class HarmonyLM(dspy.LM):
        """``dspy.LM`` subclass that strips gpt-oss Harmony channel tokens."""

        def _process_completion(self, response, merged_kwargs):
            outputs = super()._process_completion(response, merged_kwargs)
            cleaned = []
            for o in outputs:
                if isinstance(o, str):
                    cleaned.append(strip_harmony_tokens(o))
                elif isinstance(o, dict) and isinstance(o.get("text"), str):
                    cleaned.append({**o, "text": strip_harmony_tokens(o["text"])})
                else:
                    cleaned.append(o)
            return cleaned

    return HarmonyLM(effective_model, **lm_kwargs)
