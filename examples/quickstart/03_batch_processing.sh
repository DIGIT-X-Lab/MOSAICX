#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# MOSAICX Quickstart Tutorial 03 -- Batch Processing
# ============================================================================
#
# What you will learn:
#   - How to process a directory of clinical documents in a single command
#   - How to run batch extraction with a specific mode
#   - How to export results in multiple formats (JSON, JSONL, Parquet)
#   - How the resume capability works for interrupted jobs
#
# Prerequisites:
#   - MOSAICX installed  (pip install mosaicx)
#   - MOSAICX_API_KEY set in your environment
#   - For Parquet export: pip install pandas pyarrow
#
# Sample data used:
#   ../data/reports/   --  a directory containing synthetic clinical reports
#
# Estimated time: 3-5 minutes (scales with number of documents)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/reports"
OUTPUT_DIR="${SCRIPT_DIR}/batch_output"

echo "============================================================================"
echo " MOSAICX Tutorial 03 -- Batch Processing"
echo "============================================================================"
echo ""
echo "When you have dozens or thousands of clinical documents, extracting them"
echo "one by one is impractical.  The 'mosaicx batch' command processes an"
echo "entire directory in parallel, tracks progress, and can resume if"
echo "interrupted."
echo ""

# ---------------------------------------------------------------------------
# Step 1: Show the input directory contents
# ---------------------------------------------------------------------------
echo "=== Step 1: Review input documents ==="
echo ""
echo "Let's see which sample documents are available in the data directory."
echo ""

ls -1 "${DATA_DIR}"

echo ""
echo "Each file above is a synthetic clinical report.  In a real-world"
echo "scenario this directory might contain hundreds of files -- the batch"
echo "command handles them all."
echo ""

# ---------------------------------------------------------------------------
# Step 2: Create the output directory
# ---------------------------------------------------------------------------
echo "=== Step 2: Create output directory ==="
echo ""
echo "Batch results are written to an output directory.  Each input document"
echo "gets a corresponding JSON file."
echo ""

mkdir -p "${OUTPUT_DIR}"
echo "Created: ${OUTPUT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Run batch extraction in auto mode
# ---------------------------------------------------------------------------
echo "=== Step 3: Run batch extraction (auto mode) ==="
echo ""
echo "The simplest batch invocation processes every document in the input"
echo "directory using auto mode.  The LLM infers the schema for each document"
echo "independently."
echo ""

mosaicx batch \
    --input-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Each input document now has a corresponding JSON result in the output"
echo "directory."
echo ""

# ---------------------------------------------------------------------------
# Step 4: Run batch with radiology mode
# ---------------------------------------------------------------------------
echo "=== Step 4: Run batch with a specific mode ==="
echo ""
echo "If you know that all (or most) documents belong to a specific type,"
echo "you can pass --mode to use a domain-optimized extraction pipeline."
echo "Here we use 'radiology' mode.  Documents that do not match (e.g. the"
echo "pathology report) may produce less accurate output -- in production"
echo "you would separate reports by type first."
echo ""

# Clean the output directory so results are fresh
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

mosaicx batch \
    --input-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --mode radiology

echo ""

# ---------------------------------------------------------------------------
# Step 5: Show output files
# ---------------------------------------------------------------------------
echo "=== Step 5: Inspect output files ==="
echo ""
echo "Let's see what was produced."
echo ""

ls -lh "${OUTPUT_DIR}"

echo ""
echo "Each .json file contains the structured extraction for one document."
echo ""

# ---------------------------------------------------------------------------
# Step 6: Export in multiple formats
# ---------------------------------------------------------------------------
echo "=== Step 6: Export in multiple formats (JSONL + Parquet) ==="
echo ""
echo "For data science workflows, you often want results in JSONL or Parquet"
echo "format.  The --format flag accepts one or more format names.  MOSAICX"
echo "will write per-document JSON files as usual and also produce consolidated"
echo "results.jsonl and/or results.parquet files."
echo ""

# Clean the output directory again
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

mosaicx batch \
    --input-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --format jsonl \
    --format parquet

echo ""
echo "Output directory now contains individual JSON files plus consolidated"
echo "results.jsonl and results.parquet (if pandas/pyarrow are installed):"
echo ""

ls -lh "${OUTPUT_DIR}"

echo ""

# ---------------------------------------------------------------------------
# Step 7: Mention the resume capability
# ---------------------------------------------------------------------------
echo "=== Step 7: Resume interrupted jobs ==="
echo ""
echo "If a batch job is interrupted (network error, timeout, Ctrl-C), you"
echo "do not have to start over.  Pass the --resume flag and MOSAICX will"
echo "skip documents that have already been processed:"
echo ""
echo "  mosaicx batch \\"
echo "      --input-dir ${DATA_DIR} \\"
echo "      --output-dir ${OUTPUT_DIR} \\"
echo "      --resume"
echo ""
echo "This is especially valuable for large datasets where each document"
echo "incurs an LLM API call."
echo ""

# ---------------------------------------------------------------------------
# Step 8: Clean up
# ---------------------------------------------------------------------------
echo "=== Step 8: Cleanup ==="
echo ""
rm -rf "${OUTPUT_DIR}"
echo "Removed ${OUTPUT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------
echo "============================================================================"
echo " Tutorial 03 complete!"
echo "============================================================================"
echo ""
echo "You now know how to:"
echo "  - Batch-process a directory of documents"
echo "  - Use a specific extraction mode for batch jobs"
echo "  - Export results in JSONL and Parquet formats"
echo "  - Resume interrupted batch runs"
echo ""
echo "Next tutorial:  04_optimization.sh"
echo "  Learn how to optimize extraction accuracy with labeled training data."
echo ""
