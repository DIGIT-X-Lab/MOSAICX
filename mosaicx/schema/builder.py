"""
MOSAICX Schema Builder - Direct Pydantic Model Synthesis

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
Offer a lightweight entry point for generating a single Pydantic BaseModel from
natural-language descriptions. The builder targets scenarios where the full
SchemaSpec induction pipeline is unnecessary, favouring quick iteration via any
OpenAI-compatible endpoint such as OpenAI, LM Studio, or Ollama.

Usage Example:
--------------
    python -m mosaicx.schema.builder \
        --desc "Patient demographics with age, gender, and diagnosis" \
        --class-name PatientInfo \
        --model gpt-oss:120b \
        --outfile patient_info.py

Environment Integration:
------------------------
- Reads ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` when CLI overrides are absent.
- Defaults to Ollama-friendly settings (``http://localhost:11434/v1`` / ``ollama``)
  so air-gapped experimentation remains possible.
- Extracts clean code blocks from model responses, normalising output for direct
  persistence on disk.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Optional

from ..utils import resolve_openai_config
from ..utils.logging import get_logger

# Module-level logger
_logger = get_logger(__name__)

try:
    # The OpenAI class is available via the openai package.  We avoid
    # importing at module load time so that missing dependencies can
    # surface with a clear error message in the CLI.
    from openai import OpenAI
except ImportError as e:
    OpenAI = None  # type: ignore


def _extract_code_block(text: str) -> str:
    """Extract the first fenced code block from the LLM response.

    Many models will wrap their code in triple backticks. This helper
    returns the content inside the block, or the original text if no
    fenced block is found.

    Args:
        text: The raw assistant message text.
    Returns:
        The extracted code string without surrounding backticks.
    """
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    return (m.group(1) if m else text).strip()


def synthesize_pydantic_model(
    description: str,
    *,
    class_name: str = "GeneratedModel",
    model: str = "gpt-oss:120b",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """Use an LLM to produce a Pydantic BaseModel class from a description.

    This function constructs a strict prompt instructing the model to output
    only Python code defining a single Pydantic class. It reuses the
    provided OpenAI-compatible client if available, falling back to
    environment variables for base_url and api_key.

    Args:
        description: Natural-language description of the desired schema.
        class_name: Desired name of the generated BaseModel.
        model: Model identifier (e.g., gpt-4o-mini, llama3.1:8b-instruct).
        base_url: Optional endpoint URL; defaults to OPENAI_BASE_URL.
        api_key: Optional API key; defaults to OPENAI_API_KEY.
        temperature: Sampling temperature (0–2); lower values give more
            deterministic output.

    Returns:
        A string containing Python code for the Pydantic class.

    Raises:
        RuntimeError: If the openai package is not installed or no API key
            can be determined.
    """
    _logger.info(f"Synthesizing Pydantic model: class_name={class_name}")
    _logger.debug(f"Schema config: model={model}, temperature={temperature}")
    _logger.debug(f"Description: {description[:200]}{'...' if len(description) > 200 else ''}")
    
    if OpenAI is None:
        _logger.error("OpenAI package not installed")
        raise RuntimeError(
            "The 'openai' package is required for simple schema generation. "
            "Install it via 'pip install openai'."
        )

    # Resolve endpoint and key. Default to Ollama settings.
    resolved_base_url, resolved_api_key = resolve_openai_config(base_url, api_key)
    _logger.debug(f"Using endpoint: {resolved_base_url}")

    client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    system = (
        "You are an expert Python engineer. Given a natural language "
        "description of a data schema, output ONLY a single Python code "
        "block that defines exactly one Pydantic BaseModel class named "
        f"{class_name}. "
        "Guidelines:\n"
        "- from pydantic import BaseModel, Field, EmailStr\n"
        "- from typing import Optional, List, Dict, Literal\n"
        "- Use precise types (int, float, bool, str, EmailStr, Literal, "
        "List[T], Dict[K,V])\n"
        "- When a categorical field can be absent, model it as Optional[...] "
        "and include a 'None' Literal choice so placeholder strings validate\n"
        "- Declare shared Literal aliases outside the BaseModel (or mark them as "
        "TypeAlias/ClassVar) so they are not treated as model fields\n"
        "- Use Field for constraints (gt, ge, lt, le, min_length, max_length, "
        "pattern, default)\n"
        "- Include docstrings for the class and fields when possible\n"
        "- No explanations, no extra text—only the class code."
    )
    user = (
        "Description:\n" + description.strip() + "\n\n" + "Return only the Python class."
    )

    _logger.info("Sending schema generation request to LLM")
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    content = response.choices[0].message.content or ""
    code = _extract_code_block(content)
    _logger.info(f"Schema generated successfully: {len(code)} chars")
    _logger.debug(f"Generated code preview: {code[:200]}{'...' if len(code) > 200 else ''}")
    return code


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="schema_builder_simple",
        description=(
            "Generate a single Pydantic BaseModel class from a natural-language description "
            "using an OpenAI-compatible endpoint."
        ),
    )
    parser.add_argument(
        "--desc",
        required=True,
        help=("Natural-language description of the data structure you want to extract. "
              "You can also specify '-' to read from STDIN."),
    )
    parser.add_argument(
        "--class-name",
        default="GeneratedModel",
        help="Name for the generated Pydantic class (CamelCase recommended).",
    )
    parser.add_argument(
        "--model",
        default="gpt-oss:120b",
        help="Model identifier for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--outfile",
        required=False,
        help="Optional path to write the generated class code. If omitted, prints to STDOUT.",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible API base URL. Defaults to OPENAI_BASE_URL env var or local Ollama.",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the endpoint. Defaults to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the LLM (0.0–2.0). Lower is more deterministic.",
    )
    args = parser.parse_args()

    # Read description from file or stdin if indicated
    desc_text = args.desc
    if desc_text.strip() == "-":
        desc_text = sys.stdin.read()

    code = synthesize_pydantic_model(
        desc_text,
        class_name=args.class_name,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✅ Wrote Pydantic model to {args.outfile}")
    else:
        print(code)


if __name__ == "__main__":
    main()
