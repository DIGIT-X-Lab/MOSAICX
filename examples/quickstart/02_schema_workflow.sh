#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# MOSAICX Quickstart Tutorial 02 -- Schema Workflow
# ============================================================================
#
# What you will learn:
#   - How to generate a custom Pydantic schema from a natural-language
#     description using the LLM
#   - How to list and inspect saved schemas
#   - How to extract data using a custom schema
#   - How to refine a schema by adding fields manually or via LLM instruction
#   - How to view schema version history
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
echo " MOSAICX Tutorial 02 -- Schema Workflow"
echo "============================================================================"
echo ""
echo "Schemas let you define exactly which fields MOSAICX should extract from a"
echo "document.  You describe what you need in plain English; the LLM generates"
echo "a typed Pydantic model; and MOSAICX uses that model to constrain extraction."
echo ""

# ---------------------------------------------------------------------------
# Step 1: Generate a schema from a natural-language description
# ---------------------------------------------------------------------------
echo "=== Step 1: Generate a schema from a description ==="
echo ""
echo "We want to extract structured data from echocardiography reports."
echo "Instead of writing a Pydantic model by hand, we will describe what we"
echo "need in plain English and let the LLM create the schema for us."
echo ""
echo "Note: The LLM chooses the schema class name automatically.  We will"
echo "use the --name flag to pin it to 'EchoReport' for reproducibility in"
echo "later steps of this tutorial."
echo ""

mosaicx schema generate \
    --description "echocardiography report with LVEF, valve assessments, and impression" \
    --name EchoReport

echo ""
echo "The schema has been generated and saved to ~/.mosaicx/schemas/."
echo "It contains typed fields for LVEF, valve findings, and impression --"
echo "exactly what we described."
echo ""

# ---------------------------------------------------------------------------
# Step 2: List saved schemas
# ---------------------------------------------------------------------------
echo "=== Step 2: List saved schemas ==="
echo ""
echo "You can generate as many schemas as you need.  The 'schema list' command"
echo "shows every schema stored in your local registry (~/.mosaicx/schemas/)."
echo ""

mosaicx schema list

echo ""

# ---------------------------------------------------------------------------
# Step 3: Show schema details
# ---------------------------------------------------------------------------
echo "=== Step 3: Show schema details ==="
echo ""
echo "Let's inspect the EchoReport schema we just created.  The 'schema show'"
echo "command displays every field, its type, whether it is required, and a"
echo "short description."
echo ""
# NOTE: If you did not use --name in Step 1, the LLM would have chosen its
# own class name (e.g., EchocardiographyReport).  You can find the actual
# name by running 'mosaicx schema list' and adjusting the command below.

mosaicx schema show EchoReport

echo ""

# ---------------------------------------------------------------------------
# Step 4: Extract using the custom schema
# ---------------------------------------------------------------------------
echo "=== Step 4: Extract using the schema ==="
echo ""
echo "Now we use the EchoReport schema to extract structured data from a"
echo "sample echocardiography report.  Because we defined the schema, MOSAICX"
echo "will return exactly the fields we care about -- nothing more, nothing less."
echo ""

mosaicx extract \
    --document "${SCRIPT_DIR}/../data/reports/echo_report.txt" \
    --schema EchoReport

echo ""
echo "The output is constrained to the EchoReport schema.  This is much more"
echo "predictable than auto mode, especially for downstream data pipelines."
echo ""

# ---------------------------------------------------------------------------
# Step 5: Refine the schema -- add a field manually
# ---------------------------------------------------------------------------
echo "=== Step 5: Refine the schema -- add a field ==="
echo ""
echo "After reviewing the extraction results, you may realize you need an"
echo "additional field.  The 'schema refine --add' flag lets you add a"
echo "new field without regenerating the entire schema."
echo ""
echo "Here we add a float field for RVSP (right ventricular systolic pressure)."
echo ""

mosaicx schema refine \
    --schema EchoReport \
    --add "rvsp: float"

echo ""

# ---------------------------------------------------------------------------
# Step 6: Refine the schema -- LLM-driven instruction
# ---------------------------------------------------------------------------
echo "=== Step 6: Refine the schema -- LLM-driven instruction ==="
echo ""
echo "For more complex changes, you can give the LLM a natural-language"
echo "instruction and let it figure out how to update the schema.  This is"
echo "useful for adding enum types, changing field descriptions, or"
echo "restructuring nested objects."
echo ""

mosaicx schema refine \
    --schema EchoReport \
    --instruction "add pericardial effusion as an enum with values: none, trace, small, moderate, large"

echo ""
echo "The LLM added a new pericardial_effusion field as an enum.  You can"
echo "verify by running 'mosaicx schema show EchoReport' again."
echo ""

# ---------------------------------------------------------------------------
# Step 7: View version history
# ---------------------------------------------------------------------------
echo "=== Step 7: View schema version history ==="
echo ""
echo "Every time you refine a schema, MOSAICX archives the previous version."
echo "The 'schema history' command shows all versions so you can track changes"
echo "over time and revert if needed."
echo ""

mosaicx schema history EchoReport

echo ""

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------
echo "============================================================================"
echo " Tutorial 02 complete!"
echo "============================================================================"
echo ""
echo "You now know how to:"
echo "  - Generate a custom schema from a description"
echo "  - List and inspect saved schemas"
echo "  - Extract data constrained to a specific schema"
echo "  - Refine schemas manually and with LLM instructions"
echo "  - View schema version history"
echo ""
echo "Tip: You can also revert to a previous version with:"
echo "  mosaicx schema revert EchoReport --version 1"
echo ""
echo "Next tutorial:  03_batch_processing.sh"
echo "  Learn how to process a whole directory of documents at once."
echo ""
