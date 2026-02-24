#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PIPELINE="${1:-query}"
PROFILE="${MOSAICX_OPT_PROFILE:-quick}"
TRAINSET="${2:-tests/datasets/evaluation/${PIPELINE}_optimizer_tiny_train.jsonl}"
VALSET="${3:-tests/datasets/evaluation/${PIPELINE}_optimizer_tiny_val.jsonl}"
STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
OUT_DIR="docs/runs/${STAMP}-${PIPELINE}-optimizer-seq"

if [[ ! -f "$TRAINSET" ]]; then
  echo "trainset not found: $TRAINSET" >&2
  exit 1
fi
if [[ ! -f "$VALSET" ]]; then
  echo "valset not found: $VALSET" >&2
  exit 1
fi

scripts/ensure_vllm_mlx_server.sh

PYTHONPATH=. .venv/bin/python scripts/run_optimizer_sequence.py \
  --pipeline "$PIPELINE" \
  --trainset "$TRAINSET" \
  --valset "$VALSET" \
  --profile "$PROFILE" \
  --out-dir "$OUT_DIR"

echo "Optimizer artifacts written to: $OUT_DIR"
