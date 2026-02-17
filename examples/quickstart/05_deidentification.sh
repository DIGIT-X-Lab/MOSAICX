#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# MOSAICX Quickstart Tutorial 05 -- De-identification
# ============================================================================
#
# What you will learn:
#   - How to remove protected health information (PHI) from clinical documents
#   - The difference between LLM+regex and regex-only de-identification
#   - How to pseudonymize (replace PHI with fake but realistic values)
#   - How to save de-identified output to a file
#
# Why de-identification matters:
#   Clinical documents contain PHI (names, dates, MRNs, SSNs, addresses,
#   phone numbers, etc.) that is protected under HIPAA and similar regulations.
#   Before sharing documents for research, training, or analytics, you must
#   remove or replace this information.  MOSAICX provides two approaches:
#
#   1. LLM + regex (default) -- the LLM identifies contextual PHI that
#      pattern-matching alone would miss, then regex catches structured
#      patterns.  Most thorough, but requires an API call per document.
#
#   2. Regex-only (--regex-only) -- fast, deterministic pattern matching
#      with no LLM call.  Good for structured PHI (SSNs, MRNs, dates,
#      phone numbers) but may miss names embedded in free text.
#
# Prerequisites:
#   - MOSAICX installed  (pip install mosaicx)
#   - MOSAICX_API_KEY set (required for LLM mode; not needed for --regex-only)
#
# Sample data used:
#   ../data/reports/clinical_note_admission.txt  --  a synthetic admission
#     note containing realistic PHI (all data is fictional)
#
# Estimated time: 2-3 minutes
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_FILE="${SCRIPT_DIR}/../data/reports/clinical_note_admission.txt"

echo "============================================================================"
echo " MOSAICX Tutorial 05 -- De-identification"
echo "============================================================================"
echo ""
echo "This tutorial demonstrates how to strip protected health information (PHI)"
echo "from clinical documents using the MOSAICX de-identification pipeline."
echo ""

# ---------------------------------------------------------------------------
# Step 1: Preview the input file (contains PHI)
# ---------------------------------------------------------------------------
echo "=== Step 1: Preview the input document (contains PHI) ==="
echo ""
echo "The sample admission note below contains realistic (but fictional) PHI"
echo "including patient names, dates of birth, MRNs, SSNs, addresses, phone"
echo "numbers, and insurance details.  In a real scenario these would need to"
echo "be removed before sharing."
echo ""
echo "First 15 lines of the document:"
echo "---"

head -15 "${INPUT_FILE}"

echo "---"
echo ""
echo "Notice the patient name, DOB, MRN, SSN, insurance policy number, and"
echo "attending physician name -- all of these are PHI."
echo ""

# ---------------------------------------------------------------------------
# Step 2: De-identify with LLM + regex (default mode)
# ---------------------------------------------------------------------------
echo "=== Step 2: De-identify with LLM + regex (default) ==="
echo ""
echo "The default de-identification mode combines an LLM pass (to identify"
echo "contextual PHI like names and addresses in free text) with regex"
echo "patterns (to catch structured identifiers like SSNs and MRNs)."
echo ""
echo "The default strategy is 'remove', which replaces PHI with generic"
echo "placeholders like [NAME], [DATE], [MRN], etc."
echo ""

mosaicx deidentify \
    --document "${INPUT_FILE}"

echo ""
echo "Above you can see the de-identified text.  Patient names, dates, and"
echo "identifiers have been replaced with bracketed placeholders."
echo ""

# ---------------------------------------------------------------------------
# Step 3: De-identify with regex only (no LLM needed)
# ---------------------------------------------------------------------------
echo "=== Step 3: De-identify with regex only ==="
echo ""
echo "If you do not have an LLM API key, or need fast deterministic"
echo "processing, use the --regex-only flag.  This uses pattern matching"
echo "to catch structured PHI (SSNs, phone numbers, dates, MRNs) but"
echo "may miss unstructured PHI like names in narrative text."
echo ""
echo "Regex-only mode is useful for:"
echo "  - High-throughput pipelines where speed matters"
echo "  - Environments without LLM access"
echo "  - A first pass before manual review"
echo ""

mosaicx deidentify \
    --document "${INPUT_FILE}" \
    --regex-only

echo ""
echo "Compare the output above with Step 2.  Regex-only mode catches"
echo "structured identifiers reliably, but names and addresses in running"
echo "text may remain.  For maximum coverage, use the LLM+regex default."
echo ""

# ---------------------------------------------------------------------------
# Step 4: Pseudonymize (replace PHI with fake values)
# ---------------------------------------------------------------------------
echo "=== Step 4: De-identify in pseudonymize mode ==="
echo ""
echo "Sometimes you want the de-identified text to read naturally (e.g. for"
echo "training data or demonstrations).  Pseudonymization replaces real PHI"
echo "with realistic but fake values -- real-looking names, dates, and IDs"
echo "that cannot be linked back to the actual patient."
echo ""
echo "Use --mode pseudonymize to enable this behavior."
echo ""

mosaicx deidentify \
    --document "${INPUT_FILE}" \
    --mode pseudonymize

echo ""
echo "The text above looks like a real clinical note, but all identifying"
echo "information has been replaced with fictional values."
echo ""

# ---------------------------------------------------------------------------
# Step 5: Save de-identified output to a file
# ---------------------------------------------------------------------------
echo "=== Step 5: Save de-identified output to a file ==="
echo ""
echo "To capture the de-identified text for downstream use, redirect the"
echo "output to a file.  The deidentify command prints the scrubbed text"
echo "to the terminal, so you can pipe or redirect as needed."
echo ""
echo "Example using the regex-only mode (no LLM required):"
echo ""

# Demonstrate saving output by running the command and redirecting
mosaicx deidentify \
    --document "${INPUT_FILE}" \
    --regex-only \
    > "${SCRIPT_DIR}/deidentified_output.txt" 2>/dev/null || true

echo "Output saved to: ${SCRIPT_DIR}/deidentified_output.txt"
echo ""

# ---------------------------------------------------------------------------
# Step 6: Clean up
# ---------------------------------------------------------------------------
echo "=== Step 6: Cleanup ==="
echo ""
rm -f "${SCRIPT_DIR}/deidentified_output.txt"
echo "Removed temporary output file."
echo ""

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------
echo "============================================================================"
echo " Tutorial 05 complete!"
echo "============================================================================"
echo ""
echo "You now know how to:"
echo "  - De-identify documents using LLM+regex (most thorough)"
echo "  - Use regex-only mode for fast, deterministic scrubbing"
echo "  - Pseudonymize documents with realistic fake values"
echo "  - Save de-identified output to files"
echo ""
echo "De-identification strategies at a glance:"
echo "  remove        -- replace PHI with [PLACEHOLDER] tags (default)"
echo "  pseudonymize  -- replace PHI with fake but realistic values"
echo "  dateshift     -- shift all dates by a random offset"
echo ""
echo "For batch de-identification, use the --dir flag instead of --document:"
echo "  mosaicx deidentify --dir /path/to/documents/"
echo ""
echo "============================================================================"
echo " Congratulations -- you have completed all MOSAICX quickstart tutorials!"
echo "============================================================================"
echo ""
echo "Tutorials completed:"
echo "  01  Basic extraction"
echo "  02  Schema workflow"
echo "  03  Batch processing"
echo "  04  Optimization"
echo "  05  De-identification"
echo ""
echo "For more information, run:  mosaicx --help"
echo "Documentation:  https://github.com/your-org/mosaicx"
echo ""
