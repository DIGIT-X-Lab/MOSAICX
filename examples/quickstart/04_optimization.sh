#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# MOSAICX Quickstart Tutorial 04 -- Optimization
# ============================================================================
#
# What you will learn:
#   - How DSPy optimization improves extraction accuracy over time
#   - How to inspect the training data format
#   - How to run optimization with a training set and budget preset
#   - How to evaluate baseline vs. optimized pipeline performance
#   - How to use the optimized program for extraction
#
# Prerequisites:
#   - MOSAICX installed  (pip install mosaicx)
#   - MOSAICX_API_KEY set in your environment
#   - Labeled training data (provided as sample data in this tutorial)
#
# Sample data used:
#   ../data/training/radiology_train.jsonl  --  training examples
#   ../data/training/radiology_test.jsonl   --  test examples for evaluation
#   ../data/reports/radiology_ct_chest.txt  --  a sample report for extraction
#
# How optimization works:
#   MOSAICX pipelines are built on DSPy, a framework that treats LLM prompts
#   as learnable programs.  Optimization reads labeled examples and searches
#   for better prompt instructions or few-shot demonstrations that maximize
#   a metric (e.g. extraction accuracy).  The result is an "optimized program"
#   -- a JSON file with tuned prompts -- that you can load at inference time.
#
# Estimated time: 5-15 minutes (depends on budget preset and LLM latency)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_FILE="${SCRIPT_DIR}/../data/training/radiology_train.jsonl"
TEST_FILE="${SCRIPT_DIR}/../data/training/radiology_test.jsonl"
REPORT_FILE="${SCRIPT_DIR}/../data/reports/radiology_ct_chest.txt"

echo "============================================================================"
echo " MOSAICX Tutorial 04 -- Optimization"
echo "============================================================================"
echo ""
echo "Out of the box, MOSAICX extraction pipelines produce good results using"
echo "carefully engineered prompts.  But every clinical domain is different."
echo "Optimization lets you teach the pipeline what 'good output' looks like"
echo "for YOUR data, resulting in higher accuracy and consistency."
echo ""

# ---------------------------------------------------------------------------
# Step 1: List available pipelines
# ---------------------------------------------------------------------------
echo "=== Step 1: List available pipelines ==="
echo ""
echo "Not every pipeline supports optimization yet.  Use --list-pipelines to"
echo "see which ones are available."
echo ""

mosaicx optimize --list-pipelines

echo ""

# ---------------------------------------------------------------------------
# Step 2: Inspect training data format
# ---------------------------------------------------------------------------
echo "=== Step 2: Inspect the training data format ==="
echo ""
echo "Training data is stored as JSONL (one JSON object per line).  Each line"
echo "contains the input fields (e.g. report_text) and the expected output"
echo "fields (e.g. findings, impressions).  The exact fields depend on the"
echo "pipeline being optimized."
echo ""
echo "Here is the first example from the radiology training set:"
echo ""

# Pretty-print the first line of the training file
head -1 "${TRAIN_FILE}" | python3 -m json.tool

echo ""
echo "The training file contains $(wc -l < "${TRAIN_FILE}" | tr -d ' ') labeled examples."
echo "More examples generally leads to better optimization, but even 5-10"
echo "high-quality examples can yield meaningful improvements."
echo ""

# ---------------------------------------------------------------------------
# Step 3: Run optimization (light budget)
# ---------------------------------------------------------------------------
echo "=== Step 3: Run optimization with light budget ==="
echo ""
echo "The --budget flag controls how much compute the optimizer uses:"
echo "  light  --  fast, low cost, good for prototyping"
echo "  medium --  balanced (default)"
echo "  heavy  --  thorough search, higher cost, best accuracy"
echo ""
echo "We use 'light' here to keep the tutorial quick.  In production you would"
echo "typically use 'medium' or 'heavy'."
echo ""

mosaicx optimize \
    --pipeline radiology \
    --trainset "${TRAIN_FILE}" \
    --budget light

echo ""
echo "The optimized program has been saved to ~/.mosaicx/optimized/."
echo "This JSON file contains the tuned prompts and/or few-shot demonstrations"
echo "that the optimizer found to maximize accuracy on the training set."
echo ""

# ---------------------------------------------------------------------------
# Step 4: Evaluate baseline (without optimization)
# ---------------------------------------------------------------------------
echo "=== Step 4: Evaluate baseline pipeline ==="
echo ""
echo "Before we can appreciate the improvement, we need a baseline."
echo "The 'mosaicx eval' command scores the pipeline on a held-out test set"
echo "using domain-specific metrics."
echo ""

mosaicx eval \
    --pipeline radiology \
    --testset "${TEST_FILE}"

echo ""
echo "Note the mean score above -- this is the unoptimized baseline."
echo ""

# ---------------------------------------------------------------------------
# Step 5: Evaluate with the optimized program
# ---------------------------------------------------------------------------
echo "=== Step 5: Evaluate optimized pipeline ==="
echo ""
echo "Now we run the same evaluation but load the optimized program.  If"
echo "optimization worked, the mean score should be higher."
echo ""

# The default save location for optimized programs
OPTIMIZED_PATH="${HOME}/.mosaicx/optimized/radiology_optimized.json"

# Check that the optimized file exists before proceeding
if [ ! -f "${OPTIMIZED_PATH}" ]; then
    echo "WARNING: Optimized program not found at ${OPTIMIZED_PATH}"
    echo "This can happen if the optimization step above did not complete."
    echo "Skipping this step."
else
    mosaicx eval \
        --pipeline radiology \
        --testset "${TEST_FILE}" \
        --optimized "${OPTIMIZED_PATH}"

    echo ""
    echo "Compare the mean score above with the baseline from Step 4."
    echo "Even with a 'light' budget, you should see an improvement."
fi

echo ""

# ---------------------------------------------------------------------------
# Step 6: Use the optimized program for extraction
# ---------------------------------------------------------------------------
echo "=== Step 6: Extract with the optimized program ==="
echo ""
echo "The optimized program is not just for evaluation -- you can use it for"
echo "actual extraction too.  Pass --optimized to the extract command and"
echo "MOSAICX will load the tuned prompts automatically."
echo ""

if [ ! -f "${OPTIMIZED_PATH}" ]; then
    echo "Skipping (optimized program not available)."
else
    # Note: --optimized works with auto mode (no --mode flag).  The optimized
    # program already encodes the pipeline behavior learned during training.
    mosaicx extract \
        --document "${REPORT_FILE}" \
        --optimized "${OPTIMIZED_PATH}"

    echo ""
    echo "The extraction above used the optimized prompts.  In production you"
    echo "would pass --optimized to every extract or batch command to benefit"
    echo "from the tuned pipeline."
fi

echo ""

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------
echo "============================================================================"
echo " Tutorial 04 complete!"
echo "============================================================================"
echo ""
echo "You now know how to:"
echo "  - Prepare labeled training data in JSONL format"
echo "  - Run DSPy optimization at different budget levels"
echo "  - Evaluate baseline vs. optimized pipeline accuracy"
echo "  - Use optimized programs for inference"
echo ""
echo "Key takeaways:"
echo "  - Even small labeled datasets (5-10 examples) can help"
echo "  - The 'light' budget is great for iteration; use 'heavy' for production"
echo "  - Optimized programs are portable JSON files you can version-control"
echo ""
echo "Next tutorial:  05_deidentification.sh"
echo "  Learn how to remove protected health information (PHI) from documents."
echo ""
