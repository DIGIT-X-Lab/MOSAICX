"""HIPAA Safe Harbor conformance for de-identification.

Defines the 18 HIPAA Safe Harbor identifier categories, regex patterns
for format-based PHI (SSNs, phone numbers, MRNs, emails, dates), and
an LLM prompt fragment specifying what to look for.
"""
from __future__ import annotations

import re

from .registry import ConformanceSpec, register_conformance

HIPAA_PHI_CATEGORIES: list[str] = [
    "NAME",
    "DATE",
    "AGE",
    "ADDRESS",
    "ZIP",
    "PHONE",
    "FAX",
    "EMAIL",
    "URL",
    "SSN",
    "MRN",
    "ID",
    "ACCOUNT",
    "LICENSE",
    "INSURANCE",
    "CERTIFICATE",
    "DEVICE_ID",
    "IP_ADDRESS",
    "BIOMETRIC",
    "PHOTO",
    "OTHER",
]

HIPAA_REGEX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # SSN: 123-45-6789
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    # Phone: (555) 123-4567 or 555-123-4567 or 555.123.4567
    (re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"), "PHONE"),
    # MRN: MRN: 12345678 (case-insensitive)
    (re.compile(r"\bMRN\s*:?\s*\d{6,}\b", re.IGNORECASE), "MRN"),
    # Email: john.doe@hospital.com
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "EMAIL"),
    # US/EU dates with slashes: 1/2/2024 or 01/02/24
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "DATE"),
    # Dot-separated dates: 27.02.2026 or 27.02.26
    (re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"), "DATE"),
    # ISO dates: 2026-02-27
    (re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"), "DATE"),
    # German/English abbreviated month dates: 27.Feb.2026, 03.Mär.2024
    (
        re.compile(
            r"\b\d{1,2}\."
            r"(?:Jan|Feb|Mär|Mar|Apr|Mai|May|Jun|Jul|Aug|Sep|Okt|Oct|Nov|Dez|Dec)\."
            r"\d{2,4}\b",
            re.IGNORECASE,
        ),
        "DATE",
    ),
]

HIPAA_PROMPT_FRAGMENT: str = (
    "You are de-identifying a medical document under HIPAA Safe Harbor rules. "
    "Detect ALL of the following 18 identifier categories: "
    "names, dates (except year), ages over 89, addresses, ZIP codes, "
    "phone numbers, fax numbers, email addresses, URLs, "
    "Social Security numbers, medical record numbers, "
    "health plan beneficiary numbers, account numbers, "
    "license/certificate numbers, vehicle/device identifiers and serial numbers, "
    "IP addresses, biometric identifiers, and full-face photographs. "
    "Also detect names of physicians, institutions, and any other "
    "unique identifying numbers or codes. "
    "Be thorough: it is better to over-detect than to miss PHI."
)

HIPAA_SPEC = ConformanceSpec(
    name="hipaa",
    description="HIPAA Safe Harbor (18 identifiers)",
    phi_categories=HIPAA_PHI_CATEGORIES,
    regex_patterns=HIPAA_REGEX_PATTERNS,
    prompt_fragment=HIPAA_PROMPT_FRAGMENT,
)

register_conformance(HIPAA_SPEC)
