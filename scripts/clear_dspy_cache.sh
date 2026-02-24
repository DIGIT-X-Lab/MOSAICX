#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

rm -rf \
  "${HOME}/.dspy_cache" \
  "${HOME}/.mosaicx/.dspy_cache" \
  ".mosaicx_runtime/dspy_cache" \
  "/tmp/mosaicx/dspy_cache"

mkdir -p \
  "${HOME}/.dspy_cache" \
  "${HOME}/.mosaicx/.dspy_cache" \
  ".mosaicx_runtime/dspy_cache" \
  "/tmp/mosaicx/dspy_cache"

echo "[clear_dspy_cache] cleared DSPy caches"
