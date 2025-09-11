"""MOSAICX - Medical cOmputational Suite for Advanced Intelligent eXtraction.

A Python package for extracting structured data from medical reports using local LLMs.
"""

from .extractor import ReportExtractor
from .core.models import ExtractionConfig, ExtractionResult, AnalysisConfig, AnalysisResult
from .core.exceptions import MosaicxError, ExtractionError, ConfigurationError
from .schema import SchemaBuilder

__version__ = "0.1.0"

__all__ = [
    "ReportExtractor",
    "ExtractionConfig", 
    "ExtractionResult",
    "AnalysisConfig",
    "AnalysisResult", 
    "MosaicxError",
    "ExtractionError",
    "ConfigurationError",
    "SchemaBuilder"
]