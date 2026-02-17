# mosaicx/pipelines/_template.py
"""Pipeline template for scaffolding new extraction pipelines.

The ``PIPELINE_TEMPLATE`` string uses Python :meth:`str.format` placeholders:

- ``{name}``             -- snake_case pipeline name  (e.g. ``cardiology``)
- ``{class_name}``       -- PascalCase module class   (e.g. ``CardiologyReportStructurer``)
- ``{class_name_short}`` -- PascalCase short name     (e.g. ``Cardiology``)
- ``{description}``      -- one-line human description
"""

from __future__ import annotations

PIPELINE_TEMPLATE = '''\
# mosaicx/pipelines/{name}.py
"""{class_name} pipeline.

{description}

A single-step DSPy chain that extracts structured data from free-text
{name} documents.  Add more steps as needed.

DSPy-dependent classes are lazily imported via module-level ``__getattr__``
so that the module can be imported even when dspy is not installed.
"""

from __future__ import annotations

from mosaicx.pipelines.modes import register_mode_info

register_mode_info("{name}", "{description}")


# ---------------------------------------------------------------------------
# DSPy Signatures & Module (lazy)
# ---------------------------------------------------------------------------

def _build_dspy_classes():
    """Lazily define and return all DSPy signatures and the pipeline module.

    Called on first access via module-level ``__getattr__``.
    """
    import dspy  # noqa: F811 -- intentional lazy import

    # -- Step 1: Extract structured data ------------------------------------

    class Extract{class_name_short}(dspy.Signature):
        """Extract structured information from a {name} document."""

        document_text: str = dspy.InputField(
            desc="Full text of the {name} document"
        )
        extracted: str = dspy.OutputField(
            desc="Structured extraction result as JSON"
        )

    # -- Pipeline module ---------------------------------------------------

    class {class_name}(dspy.Module):
        """DSPy Module implementing the {name} pipeline.

        Sub-modules:
            - ``extract`` -- ChainOfThought for structured extraction.
        """

        def __init__(self) -> None:
            super().__init__()
            self.extract = dspy.ChainOfThought(Extract{class_name_short})

        def forward(self, document_text: str) -> dspy.Prediction:
            """Run the extraction pipeline.

            Parameters
            ----------
            document_text:
                Full text of the {name} document.

            Returns
            -------
            dspy.Prediction
                Keys: extracted.
            """
            from mosaicx.metrics import PipelineMetrics, get_tracker, track_step

            metrics = PipelineMetrics()
            tracker = get_tracker()

            with track_step(metrics, "Extract structured data", tracker):
                result = self.extract(document_text=document_text)

            self._last_metrics = metrics

            return dspy.Prediction(
                extracted=result.extracted,
            )

    # Register as extraction mode
    from mosaicx.pipelines.modes import register_mode
    register_mode("{name}", "{description}")({class_name})

    return {{
        "Extract{class_name_short}": Extract{class_name_short},
        "{class_name}": {class_name},
    }}


# Cache for lazily-built DSPy classes
_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({{
    "Extract{class_name_short}",
    "{class_name}",
}})


def __getattr__(name: str):
    """Module-level __getattr__ for lazy loading of DSPy classes."""
    global _dspy_classes

    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]

    raise AttributeError(f"module {{__name__!r}} has no attribute {{name!r}}")
'''
