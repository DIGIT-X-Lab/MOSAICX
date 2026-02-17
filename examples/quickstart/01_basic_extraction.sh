#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# MOSAICX Quickstart Tutorial 01 -- Basic Extraction
# ============================================================================
#
# What you will learn:
#   - How to verify your MOSAICX installation
#   - How to extract structured data from a clinical document in auto mode
#   - How to use a specific extraction mode (radiology)
#   - How to save extraction output to a file
#   - How to discover all available extraction modes
#
# Prerequisites:
#   - MOSAICX installed  (pip install mosaicx)
#   - MOSAICX_API_KEY set in your environment (or configured in ~/.mosaicx/)
#
# Sample data used:
#   ../data/reports/radiology_ct_chest.txt  --  a synthetic CT chest report
#
# Estimated time: 2-3 minutes (depends on LLM provider latency)
# ============================================================================

# Resolve the directory this script lives in so that relative paths work
# regardless of where the user invokes the script from.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================================"
echo " MOSAICX Tutorial 01 -- Basic Extraction"
echo "============================================================================"
echo ""
echo "This tutorial walks you through extracting structured data from a clinical"
echo "document using the MOSAICX CLI.  Each step builds on the previous one."
echo ""

# ---------------------------------------------------------------------------
# Step 1: Verify MOSAICX is installed
# ---------------------------------------------------------------------------
echo "=== Step 1: Verify MOSAICX is installed ==="
echo ""
echo "Before doing anything else, let's make sure the mosaicx CLI is available"
echo "on your PATH and check which version you are running."
echo ""

mosaicx --version

echo ""
echo "If you see a version number above, MOSAICX is installed correctly."
echo "If not, install it with:  pip install mosaicx"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Extract in auto mode
# ---------------------------------------------------------------------------
echo "=== Step 2: Extract from a clinical document (auto mode) ==="
echo ""
echo "When you run 'mosaicx extract' without specifying a mode, MOSAICX uses"
echo "auto mode.  The LLM reads the document, infers the appropriate schema,"
echo "and returns structured data.  This is the easiest way to get started."
echo ""
echo "We will extract from a sample CT chest radiology report:"
echo "  ${SCRIPT_DIR}/../data/reports/radiology_ct_chest.txt"
echo ""

mosaicx extract \
    --document "${SCRIPT_DIR}/../data/reports/radiology_ct_chest.txt"

echo ""
echo "Above you can see the structured data extracted in auto mode."
echo "Auto mode is convenient but uses a generic schema.  For higher accuracy"
echo "on known report types, use a specific extraction mode (next step)."
echo ""

# ---------------------------------------------------------------------------
# Step 3: Extract with radiology mode
# ---------------------------------------------------------------------------
echo "=== Step 3: Extract with radiology mode ==="
echo ""
echo "MOSAICX ships with built-in extraction modes optimized for specific"
echo "clinical document types.  The 'radiology' mode, for example, knows about"
echo "findings, impressions, exam types, and anatomy -- so it can produce a"
echo "richer, more consistent output schema."
echo ""

mosaicx extract \
    --document "${SCRIPT_DIR}/../data/reports/radiology_ct_chest.txt" \
    --mode radiology

echo ""
echo "Compare the output above with the auto-mode output from Step 2."
echo "The radiology mode produces domain-specific fields such as findings"
echo "broken down by anatomy, impression items, and exam metadata."
echo ""

# ---------------------------------------------------------------------------
# Step 4: Save output to a file
# ---------------------------------------------------------------------------
echo "=== Step 4: Save extraction output to a file ==="
echo ""
echo "In a real workflow you will want to save the extracted JSON to disk."
echo "Use the --output (-o) flag to write the result to a file."
echo ""

mosaicx extract \
    --document "${SCRIPT_DIR}/../data/reports/radiology_ct_chest.txt" \
    --mode radiology \
    --output "${SCRIPT_DIR}/result.json"

echo ""
echo "The extracted data has been saved to:"
echo "  ${SCRIPT_DIR}/result.json"
echo ""
echo "You can inspect it with:  cat ${SCRIPT_DIR}/result.json | python3 -m json.tool"
echo ""

# ---------------------------------------------------------------------------
# Step 5: List available extraction modes
# ---------------------------------------------------------------------------
echo "=== Step 5: List available extraction modes ==="
echo ""
echo "To see every extraction mode MOSAICX offers, use the --list-modes flag."
echo "Each mode is optimized for a particular family of clinical documents."
echo ""

mosaicx extract --list-modes

echo ""

# ---------------------------------------------------------------------------
# Clean up
# ---------------------------------------------------------------------------
echo "=== Cleanup ==="
echo ""
echo "Removing the result.json file we created in Step 4."
rm -f "${SCRIPT_DIR}/result.json"
echo "Done."
echo ""

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------
echo "============================================================================"
echo " Tutorial 01 complete!"
echo "============================================================================"
echo ""
echo "You now know how to:"
echo "  - Run a basic extraction in auto mode"
echo "  - Use a domain-specific extraction mode"
echo "  - Save output to a JSON file"
echo "  - Discover available modes"
echo ""
echo "Next tutorial:  02_schema_workflow.sh"
echo "  Learn how to create custom schemas so you can control exactly which"
echo "  fields MOSAICX extracts from your documents."
echo ""
