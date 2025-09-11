"""Core exceptions for MOSAICX."""

class MosaicxError(Exception):
    """Base exception for all MOSAICX errors."""
    pass


class ExtractionError(MosaicxError):
    """Error during report extraction process."""
    pass


class ConfigurationError(MosaicxError):
    """Error in configuration validation or loading."""
    pass


class ModelError(MosaicxError):
    """Error related to LLM model operations."""
    pass


class ValidationError(MosaicxError):
    """Error in data validation."""
    pass


class FileProcessingError(ExtractionError):
    """Error processing input files."""
    pass


class NetworkError(ModelError):
    """Network-related errors when communicating with LLM services."""
    pass
