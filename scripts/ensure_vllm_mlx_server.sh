#!/usr/bin/env bash
set -euo pipefail

HOST="${MOSAICX_VLLM_HOST:-127.0.0.1}"
PORT="${MOSAICX_VLLM_PORT:-8000}"
MODEL="${MOSAICX_VLLM_MODEL:-mlx-community/gpt-oss-120b-4bit}"
BASE_URL="http://${HOST}:${PORT}"
LOG_PATH="${MOSAICX_VLLM_LOG:-/tmp/vllm_mlx_server.log}"
PID_PATH="${MOSAICX_VLLM_PID:-/tmp/vllm_mlx_server.pid}"
START_TIMEOUT="${MOSAICX_VLLM_START_TIMEOUT:-180}"

ready() {
  curl -sS --max-time 3 "${BASE_URL}/v1/models" >/dev/null 2>&1
}

if ready; then
  echo "[ensure_vllm] server already healthy at ${BASE_URL}"
  exit 0
fi

if command -v pgrep >/dev/null 2>&1; then
  if pgrep -f "vllm-mlx serve .*--port ${PORT}" >/dev/null 2>&1; then
    echo "[ensure_vllm] vllm-mlx process exists; waiting for readiness..."
  else
    echo "[ensure_vllm] starting vllm-mlx server (model=${MODEL}, port=${PORT})"
    nohup vllm-mlx serve "${MODEL}" --port "${PORT}" --continuous-batching >"${LOG_PATH}" 2>&1 &
    echo "$!" >"${PID_PATH}"
  fi
else
  echo "[ensure_vllm] starting vllm-mlx server (model=${MODEL}, port=${PORT})"
  nohup vllm-mlx serve "${MODEL}" --port "${PORT}" --continuous-batching >"${LOG_PATH}" 2>&1 &
  echo "$!" >"${PID_PATH}"
fi

deadline=$((SECONDS + START_TIMEOUT))
until ready; do
  if (( SECONDS >= deadline )); then
    echo "[ensure_vllm] ERROR: server not ready within ${START_TIMEOUT}s (${BASE_URL})" >&2
    echo "[ensure_vllm] tail ${LOG_PATH}:" >&2
    tail -n 40 "${LOG_PATH}" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "[ensure_vllm] server ready at ${BASE_URL}"
