"""
MOSAICX Logging Utilities - Comprehensive Debug & Audit Logging

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Structure first. Insight follows.

Author: Lalith Kumar Shiyam Sundar, PhD
Lab: DIGIT-X Lab
Department: Department of Radiology
University: LMU University Hospital | LMU Munich

Overview:
---------
Centralised logging configuration for MOSAICX extraction pipelines.  Provides
file-based logging with rotation, configurable verbosity, and structured output
for debugging extraction attempts, prompt construction, and LLM responses.

Log Location:
-------------
- Default: ~/.mosaicx/logs/mosaicx.log
- Can be overridden via MOSAICX_LOG_DIR environment variable
- Logs are rotated at 10MB with 5 backup files retained

Log Levels:
-----------
- DEBUG: Full prompts, raw LLM responses, extraction attempts
- INFO: High-level extraction flow, success/failure summaries
- WARNING: Fallback attempts, retries, minor issues
- ERROR: Extraction failures, exceptions

Usage:
------
    from mosaicx.utils.logging import get_logger, setup_logging
    
    # Call once at startup (CLI or API entry point)
    setup_logging(level="DEBUG")
    
    # Get logger in any module
    logger = get_logger(__name__)
    logger.info("Starting extraction...")
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ============================================================================
# Constants
# ============================================================================

DEFAULT_LOG_DIR = Path.home() / ".mosaicx" / "logs"
DEFAULT_LOG_FILE = "mosaicx.log"
DEFAULT_LOG_LEVEL = "INFO"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# Custom log format with timestamp, level, module, and message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Detailed format for file logging (includes line numbers)
FILE_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

# Track if logging has been initialised
_logging_initialised = False
_log_file_path: Optional[Path] = None


# ============================================================================
# Setup Functions
# ============================================================================

def get_log_directory() -> Path:
    """Get the log directory, respecting MOSAICX_LOG_DIR environment variable."""
    env_log_dir = os.getenv("MOSAICX_LOG_DIR")
    if env_log_dir:
        return Path(env_log_dir)
    return DEFAULT_LOG_DIR


def get_log_file_path() -> Path:
    """Get the full path to the log file."""
    return get_log_directory() / DEFAULT_LOG_FILE


def setup_logging(
    level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    console_output: bool = False,
    quiet: bool = False,
) -> Path:
    """
    Initialise MOSAICX logging with file and optional console output.
    
    Parameters
    ----------
    level : str, optional
        Log level: DEBUG, INFO, WARNING, ERROR. Defaults to INFO.
        Can also be set via MOSAICX_LOG_LEVEL environment variable.
    log_dir : Path, optional
        Directory for log files. Defaults to ~/.mosaicx/logs/
    console_output : bool
        If True, also log to console (stderr). Default False.
    quiet : bool
        If True, suppress console output entirely. Default False.
        
    Returns
    -------
    Path
        Path to the log file being written to.
    """
    global _logging_initialised, _log_file_path
    
    # Determine log level
    if level is None:
        level = os.getenv("MOSAICX_LOG_LEVEL", DEFAULT_LOG_LEVEL)
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Determine log directory
    if log_dir is None:
        log_dir = get_log_directory()
    
    # Create log directory if needed
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Full path to log file
    log_file = log_dir / DEFAULT_LOG_FILE
    _log_file_path = log_file
    
    # Get the root mosaicx logger
    mosaicx_logger = logging.getLogger("mosaicx")
    
    # If already configured, update the log level if a new one was specified
    if _logging_initialised:
        if level is not None:
            # Update log level on existing handlers
            for handler in mosaicx_logger.handlers:
                handler.setLevel(log_level)
            mosaicx_logger.setLevel(log_level)
        return log_file
    
    mosaicx_logger.setLevel(log_level)
    
    # Clear any existing handlers
    mosaicx_logger.handlers.clear()
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT, LOG_DATE_FORMAT))
    mosaicx_logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_output and not quiet:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        mosaicx_logger.addHandler(console_handler)
    
    # Prevent propagation to root logger (avoid duplicate logs)
    mosaicx_logger.propagate = False
    
    _logging_initialised = True
    
    # Log startup message
    mosaicx_logger.info("=" * 80)
    mosaicx_logger.info("MOSAICX Logging Initialised")
    mosaicx_logger.info(f"  Log file: {log_file}")
    mosaicx_logger.info(f"  Log level: {level.upper()}")
    mosaicx_logger.info(f"  Timestamp: {datetime.now().isoformat()}")
    mosaicx_logger.info("=" * 80)
    
    return log_file


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.
    
    Parameters
    ----------
    name : str
        Module name (typically __name__)
        
    Returns
    -------
    logging.Logger
        Logger instance configured for MOSAICX
        
    Example
    -------
        logger = get_logger(__name__)
        logger.debug("Processing document...")
    """
    # Ensure logging is set up (with defaults if not explicitly configured)
    if not _logging_initialised:
        setup_logging()
    
    # Create child logger under mosaicx namespace
    if name.startswith("mosaicx"):
        return logging.getLogger(name)
    else:
        return logging.getLogger(f"mosaicx.{name}")


def get_current_log_file() -> Optional[Path]:
    """Return the path to the current log file, if logging is initialised."""
    return _log_file_path


# ============================================================================
# Logging Helper Functions - Structured Logging
# ============================================================================

def log_extraction_start(
    logger: logging.Logger,
    document_path: str,
    schema_name: str,
    model: str,
) -> None:
    """Log the start of an extraction operation."""
    logger.info("-" * 60)
    logger.info("EXTRACTION START")
    logger.info(f"  Document: {document_path}")
    logger.info(f"  Schema: {schema_name}")
    logger.info(f"  Model: {model}")
    logger.info("-" * 60)


def log_extraction_method_attempt(
    logger: logging.Logger,
    method: str,
    step: int,
    details: Optional[str] = None,
) -> None:
    """Log an extraction method attempt."""
    msg = f"[Step {step}] Attempting: {method}"
    if details:
        msg += f" | {details}"
    logger.info(msg)


def log_extraction_method_success(
    logger: logging.Logger,
    method: str,
    duration_seconds: Optional[float] = None,
) -> None:
    """Log successful extraction with a method."""
    msg = f"✓ SUCCESS: {method}"
    if duration_seconds is not None:
        msg += f" ({duration_seconds:.2f}s)"
    logger.info(msg)


def log_extraction_method_failure(
    logger: logging.Logger,
    method: str,
    error: str,
    step: int,
) -> None:
    """Log a failed extraction attempt."""
    logger.warning(f"✗ FAILED [Step {step}] {method}: {error}")


def log_prompt(
    logger: logging.Logger,
    prompt_type: str,
    prompt_content: str,
    truncate_at: int = 2000,
) -> None:
    """
    Log a prompt being sent to the LLM.
    
    Parameters
    ----------
    prompt_type : str
        Description of the prompt (e.g., "Extraction Prompt", "Repair Prompt")
    prompt_content : str
        The full prompt text
    truncate_at : int
        Maximum characters to log (to avoid enormous logs)
    """
    if len(prompt_content) > truncate_at:
        display_content = prompt_content[:truncate_at] + f"... [TRUNCATED, {len(prompt_content)} chars total]"
    else:
        display_content = prompt_content
    
    logger.debug(f"PROMPT ({prompt_type}):\n{display_content}")


def log_llm_response(
    logger: logging.Logger,
    response_type: str,
    response_content: str,
    truncate_at: int = 2000,
) -> None:
    """
    Log an LLM response.
    
    Parameters
    ----------
    response_type : str
        Description (e.g., "Raw Response", "Parsed JSON")
    response_content : str
        The response text
    truncate_at : int
        Maximum characters to log
    """
    if len(response_content) > truncate_at:
        display_content = response_content[:truncate_at] + f"... [TRUNCATED, {len(response_content)} chars total]"
    else:
        display_content = response_content
    
    logger.debug(f"LLM RESPONSE ({response_type}):\n{display_content}")


def log_extraction_complete(
    logger: logging.Logger,
    document_path: str,
    success: bool,
    method_used: Optional[str] = None,
    total_duration: Optional[float] = None,
    fields_extracted: Optional[int] = None,
) -> None:
    """Log extraction completion summary."""
    logger.info("-" * 60)
    status = "SUCCEEDED" if success else "FAILED"
    logger.info(f"EXTRACTION {status}")
    logger.info(f"  Document: {document_path}")
    if method_used:
        logger.info(f"  Method: {method_used}")
    if total_duration is not None:
        logger.info(f"  Duration: {total_duration:.2f}s")
    if fields_extracted is not None:
        logger.info(f"  Fields extracted: {fields_extracted}")
    logger.info("-" * 60)


def log_schema_info(
    logger: logging.Logger,
    schema_class_name: str,
    schema_json: str,
    truncate_at: int = 1500,
) -> None:
    """Log schema information being used for extraction."""
    if len(schema_json) > truncate_at:
        display_schema = schema_json[:truncate_at] + f"... [TRUNCATED, {len(schema_json)} chars total]"
    else:
        display_schema = schema_json
    
    logger.debug(f"SCHEMA ({schema_class_name}):\n{display_schema}")


def log_text_content(
    logger: logging.Logger,
    source: str,
    text_content: str,
    truncate_at: int = 1000,
) -> None:
    """Log extracted text content (for debugging extraction issues)."""
    if len(text_content) > truncate_at:
        display_text = text_content[:truncate_at] + f"... [TRUNCATED, {len(text_content)} chars total]"
    else:
        display_text = text_content
    
    logger.debug(f"TEXT CONTENT (from {source}, {len(text_content)} chars):\n{display_text}")
