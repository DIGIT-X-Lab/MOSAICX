#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# MOSAICX Quickstart Tutorial 02 -- Template Workflow
# ============================================================================
#
# What you will learn:
#   - How to create a custom YAML template from a natural-language
#     description using the LLM
#   - How to list and inspect available templates
#   - How to extract data using a custom template
#   - How to refine a template via LLM instruction
#   - How to view template version history
#
# Prerequisites:
#   - MOSAICX installed  (pip install mosaicx)
#   - MOSAICX_API_KEY set in your environment
#   - Tutorial 01 completed (not strictly required, but recommended)
#
# Sample data used:
#   ../data/reports/echo_report.txt  --  a synthetic echocardiography report
#
# Estimated time: 3-5 minutes
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================================"
echo " MOSAICX Tutorial 02 -- Template Workflow"
echo "============================================================================"
echo ""
echo "Templates let you define exactly which fields MOSAICX should extract from a"
echo "document.  You describe what you need in plain English; the LLM generates"
echo "a typed YAML template; and MOSAICX uses that template to constrain extraction."
echo ""

# ---------------------------------------------------------------------------
# Step 1: Create a template from a natural-language description
# ---------------------------------------------------------------------------
echo "=== Step 1: Create a template from a description ==="
echo ""
echo "We want to extract structured data from echocardiography reports."
echo "Instead of writing a YAML template by hand, we will describe what we"
echo "need in plain English and let the LLM create the template for us."
echo ""
echo "Note: The LLM chooses the template name automatically.  We will"
echo "use the --name flag to pin it to 'EchoReport' for reproducibility in"
echo "later steps of this tutorial."
echo ""

mosaicx template create \
    --describe "echocardiography report with LVEF, valve assessments, and impression" \
    --name EchoReport

echo ""
echo "The template has been created and saved to ~/.mosaicx/templates/."
echo "It contains typed fields for LVEF, valve findings, and impression --"
echo "exactly what we described."
echo ""

# ---------------------------------------------------------------------------
# Step 2: List available templates
# ---------------------------------------------------------------------------
echo "=== Step 2: List available templates ==="
echo ""
echo "You can create as many templates as you need.  The 'template list' command"
echo "shows every template -- both built-in and user-created."
echo ""

mosaicx template list

echo ""

# ---------------------------------------------------------------------------
# Step 3: Show template details
# ---------------------------------------------------------------------------
echo "=== Step 3: Show template details ==="
echo ""
echo "Let's inspect the EchoReport template we just created.  The 'template show'"
echo "command displays every field, its type, whether it is required, and a"
echo "short description."
echo ""

mosaicx template show EchoReport

echo ""

# ---------------------------------------------------------------------------
# Step 4: Extract using the custom template
# ---------------------------------------------------------------------------
echo "=== Step 4: Extract using the template ==="
echo ""
echo "Now we use the EchoReport template to extract structured data from a"
echo "sample echocardiography report.  Because we defined the template, MOSAICX"
echo "will return exactly the fields we care about -- nothing more, nothing less."
echo ""

mosaicx extract \
    --document "${SCRIPT_DIR}/../data/reports/echo_report.txt" \
    --template EchoReport

echo ""
echo "The output is constrained to the EchoReport template.  This is much more"
echo "predictable than auto mode, especially for downstream data pipelines."
echo ""

# ---------------------------------------------------------------------------
# Step 5: Refine the template -- LLM-driven instruction
# ---------------------------------------------------------------------------
echo "=== Step 5: Refine the template -- LLM-driven instruction ==="
echo ""
echo "After reviewing the extraction results, you may realize you need"
echo "additional fields.  Give the LLM a natural-language instruction"
echo "and let it figure out how to update the template.  This is useful"
echo "for adding enum types, changing field descriptions, or restructuring"
echo "nested objects."
echo ""

mosaicx template refine EchoReport \
    --instruction "add pericardial effusion as an enum with values: none, trace, small, moderate, large"

echo ""
echo "The LLM added a new pericardial_effusion field as an enum.  You can"
echo "verify by running 'mosaicx template show EchoReport' again."
echo ""

# ---------------------------------------------------------------------------
# Step 6: View version history
# ---------------------------------------------------------------------------
echo "=== Step 6: View template version history ==="
echo ""
echo "Every time you refine a template, MOSAICX archives the previous version."
echo "The 'template history' command shows all versions so you can track changes"
echo "over time and revert if needed."
echo ""

mosaicx template history EchoReport

echo ""

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------
echo "============================================================================"
echo " Tutorial 02 complete!"
echo "============================================================================"
echo ""
echo "You now know how to:"
echo "  - Create a custom template from a description"
echo "  - List and inspect available templates"
echo "  - Extract data constrained to a specific template"
echo "  - Refine templates with LLM instructions"
echo "  - View template version history"
echo ""
echo "Tip: You can also revert to a previous version with:"
echo "  mosaicx template revert EchoReport --version 1"
echo ""
echo "Tip: Migrate legacy JSON schemas to YAML templates with:"
echo "  mosaicx template migrate"
echo ""
echo "Next tutorial:  03_batch_processing.sh"
echo "  Learn how to process a whole directory of documents at once."
echo ""
