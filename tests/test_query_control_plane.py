from __future__ import annotations

import sys
from types import SimpleNamespace


def _sample_profiles() -> dict[str, dict[str, object]]:
    return {
        "cohort.csv": {
            "rows": 5,
            "columns": [
                {
                    "name": "Sex",
                    "role": "categorical",
                    "top_values": [
                        {"value": "M", "count": 3},
                        {"value": "F", "count": 2},
                    ],
                }
            ],
        }
    }


def test_programmatic_table_analyst_falls_back_to_codeact(monkeypatch):
    """If ProgramOfThought fails, CodeAct fallback should still produce a SQL plan."""
    from mosaicx.query.control_plane import ProgrammaticTableAnalyst

    class _FakeProgramOfThought:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, **_kwargs):
            raise RuntimeError("pot unavailable")

    class _FakeCodeAct:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, **_kwargs):
            return SimpleNamespace(
                source="cohort.csv",
                sql="SELECT Sex AS value, COUNT(*) AS count FROM _mosaicx_table GROUP BY 1 ORDER BY count DESC",
                rationale="distribution by sex",
            )

    fake_dspy = SimpleNamespace(
        settings=SimpleNamespace(lm=object()),
        ProgramOfThought=_FakeProgramOfThought,
        CodeAct=_FakeCodeAct,
    )
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)

    analyst = ProgrammaticTableAnalyst()
    plan = analyst.propose_sql(
        question="what is the distribution of male and female?",
        history="",
        table_profiles=_sample_profiles(),
    )
    assert plan is not None
    assert plan["source"] == "cohort.csv"
    assert "SELECT" in plan["sql"]
    assert "_mosaicx_table" in plan["sql"]


def test_programmatic_table_analyst_rejects_unsafe_codeact_sql(monkeypatch):
    """CodeAct plan must be dropped if SQL is not safe/read-only."""
    from mosaicx.query.control_plane import ProgrammaticTableAnalyst

    class _FakeCodeAct:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, **_kwargs):
            return SimpleNamespace(
                source="cohort.csv",
                sql="DROP TABLE _mosaicx_table",
                rationale="unsafe",
            )

    fake_dspy = SimpleNamespace(
        settings=SimpleNamespace(lm=object()),
        CodeAct=_FakeCodeAct,
    )
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)

    analyst = ProgrammaticTableAnalyst()
    plan = analyst.propose_sql(
        question="do something dangerous",
        history="",
        table_profiles=_sample_profiles(),
    )
    assert plan is None


def test_programmatic_table_analyst_skips_pot_for_simple_prompts(monkeypatch):
    """Simple count/schema prompts should not trigger ProgramOfThought."""
    from mosaicx.query.control_plane import ProgrammaticTableAnalyst

    calls = {"pot": 0}

    class _FakeProgramOfThought:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, **_kwargs):
            calls["pot"] += 1
            return SimpleNamespace(
                source="cohort.csv",
                sql="SELECT COUNT(*) AS count FROM _mosaicx_table",
                rationale="pot",
            )

    fake_dspy = SimpleNamespace(
        settings=SimpleNamespace(lm=object()),
        ProgramOfThought=_FakeProgramOfThought,
    )
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)
    monkeypatch.delenv("MOSAICX_PROGRAM_OF_THOUGHT", raising=False)

    analyst = ProgrammaticTableAnalyst()
    plan = analyst.propose_sql(
        question="how many rows are there?",
        history="",
        table_profiles=_sample_profiles(),
    )

    assert plan is None
    assert calls["pot"] == 0


def test_programmatic_table_analyst_uses_pot_for_complex_prompts(monkeypatch):
    """Complex analytical prompts should allow ProgramOfThought fallback."""
    from mosaicx.query.control_plane import ProgrammaticTableAnalyst

    class _FakeProgramOfThought:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, **_kwargs):
            return SimpleNamespace(
                source="cohort.csv",
                sql=(
                    "SELECT Sex AS value, AVG(Age) AS count "
                    "FROM _mosaicx_table GROUP BY 1 ORDER BY count DESC"
                ),
                rationale="complex stratified analysis",
            )

    fake_dspy = SimpleNamespace(
        settings=SimpleNamespace(lm=object()),
        ProgramOfThought=_FakeProgramOfThought,
    )
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)
    monkeypatch.delenv("MOSAICX_PROGRAM_OF_THOUGHT", raising=False)

    analyst = ProgrammaticTableAnalyst()
    plan = analyst.propose_sql(
        question=(
            "Run a multivariate regression style stratified analysis and compare "
            "average age by sex with confounder adjustment."
        ),
        history="previously analyzed demographics",
        table_profiles=_sample_profiles(),
    )

    assert plan is not None
    assert plan["source"] == "cohort.csv"
    assert "SELECT" in plan["sql"]


def test_programmatic_table_analyst_env_can_force_pot(monkeypatch):
    """Explicit policy flag should force ProgramOfThought even for simple prompts."""
    from mosaicx.query.control_plane import ProgrammaticTableAnalyst

    calls = {"pot": 0}

    class _FakeProgramOfThought:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, **_kwargs):
            calls["pot"] += 1
            return SimpleNamespace(
                source="cohort.csv",
                sql="SELECT COUNT(*) AS count FROM _mosaicx_table",
                rationale="forced pot",
            )

    fake_dspy = SimpleNamespace(
        settings=SimpleNamespace(lm=object()),
        ProgramOfThought=_FakeProgramOfThought,
    )
    monkeypatch.setitem(sys.modules, "dspy", fake_dspy)
    monkeypatch.setenv("MOSAICX_PROGRAM_OF_THOUGHT", "on")

    analyst = ProgrammaticTableAnalyst()
    plan = analyst.propose_sql(
        question="how many rows are there?",
        history="",
        table_profiles=_sample_profiles(),
    )

    assert plan is not None
    assert calls["pot"] == 1
