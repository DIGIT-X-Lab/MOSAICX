# mosaicx/__init__.py
"""
MOSAICX — Medical Document Structuring Platform.

Public API:
    - mosaicx.config.MosaicxConfig / get_config()
    - mosaicx.documents.load_document()
    - mosaicx.cli.cli (Click entry point)
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("mosaicx")
except PackageNotFoundError:
    __version__ = "2.0.0a1"

__all__ = ["__version__"]
