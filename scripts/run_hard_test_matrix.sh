#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="$(date +%Y-%m-%d-%H%M%S)"
LOG_PATH="docs/runs/${RUN_TS}-hard-test-matrix.md"

LLM_ENABLED="${MOSAICX_HARDTEST_LLM:-1}"
INTEGRATION_ENABLED="${MOSAICX_HARDTEST_INTEGRATION:-1}"
QUERY_ENGINE_INT_ENABLED="${MOSAICX_HARDTEST_QUERY_ENGINE_INT:-0}"
LLM_STRICT="${MOSAICX_HARDTEST_STRICT_LLM:-1}"

cat > "$LOG_PATH" <<LOG
# Hard Test Matrix Run

Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Repo: MOSAICX
LLM enabled: ${LLM_ENABLED}
Integration enabled: ${INTEGRATION_ENABLED}
QueryEngine integration enabled: ${QUERY_ENGINE_INT_ENABLED}
Strict LLM preflight: ${LLM_STRICT}

## Commands
LOG

run_cmd() {
  local cmd="$1"
  echo "- \`$cmd\`" >> "$LOG_PATH"
  echo
  echo "==> $cmd"
  eval "$cmd"
}

run_cmd_timeout() {
  local seconds="$1"
  shift
  local cmd="$*"
  echo "- \`timeout ${seconds}s :: $cmd\`" >> "$LOG_PATH"
  echo
  echo "==> $cmd (timeout=${seconds}s)"

  ( eval "$cmd" ) &
  local pid=$!
  local start_ts
  start_ts="$(date +%s)"

  while kill -0 "$pid" 2>/dev/null; do
    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= seconds )); then
      echo "Command timed out after ${seconds}s: $cmd" | tee -a "$LOG_PATH"
      kill -TERM "$pid" 2>/dev/null || true
      sleep 2
      kill -KILL "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      return 124
    fi
    sleep 1
  done

  wait "$pid"
}

clear_dspy_cache() {
  run_cmd "scripts/clear_dspy_cache.sh"
}

llm_preflight() {
  local attempts=3
  local delay=2
  local i
  local models_json
  local model_id
  for ((i=1; i<=attempts; i++)); do
    models_json="$(curl -sS --max-time 5 http://127.0.0.1:8000/v1/models)" || true
    if [[ -n "${models_json}" ]]; then
      model_id="$(
        printf '%s' "${models_json}" | python3 - <<'PY'
import json
import sys
try:
    payload = json.load(sys.stdin)
except Exception:
    print("")
    raise SystemExit(0)
data = payload.get("data", []) if isinstance(payload, dict) else []
if data and isinstance(data[0], dict):
    print(str(data[0].get("id", "")))
else:
    print("")
PY
      )"
      if [[ -n "${model_id}" ]]; then
        if curl -sS --max-time 20 http://127.0.0.1:8000/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d "{\"model\":\"${model_id}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"temperature\":0,\"max_tokens\":8}" >/dev/null; then
          echo "LLM preflight OK (model=${model_id})" | tee -a "$LOG_PATH"
          return 0
        fi
      fi
    fi
    if [[ "$i" -lt "$attempts" ]]; then
      echo "LLM preflight attempt $i failed; retrying in ${delay}s..." | tee -a "$LOG_PATH"
      sleep "$delay"
    fi
  done
  return 1
}

{
  if [[ "$LLM_ENABLED" == "1" ]]; then
    run_cmd "scripts/ensure_vllm_mlx_server.sh"
  fi

  run_cmd "python3 scripts/generate_hard_test_fixtures.py"
  clear_dspy_cache
  run_cmd_timeout 180 ".venv/bin/pytest -q tests/test_schema_gen.py tests/test_report.py"
  clear_dspy_cache
  run_cmd_timeout 180 ".venv/bin/pytest -q tests/test_query_engine.py -m 'not integration'"
  clear_dspy_cache
  run_cmd_timeout 180 ".venv/bin/pytest -q tests/test_verify_engine.py tests/test_sdk_verify.py"
  clear_dspy_cache
  run_cmd_timeout 180 ".venv/bin/pytest -q tests/test_cli_query.py tests/test_mcp_verify.py"

  if [[ "$INTEGRATION_ENABLED" == "1" ]]; then
    clear_dspy_cache
    run_cmd_timeout 120 ".venv/bin/pytest -q tests/integration/test_full_pipeline.py -m integration"
  fi

  if [[ "$LLM_ENABLED" == "1" ]]; then
    echo "- \`curl -sS --max-time 5 http://127.0.0.1:8000/v1/models\`" >> "$LOG_PATH"
    echo
    echo "==> curl -sS --max-time 5 http://127.0.0.1:8000/v1/models"
    if llm_preflight; then
      clear_dspy_cache
      run_cmd "curl -sS --max-time 120 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"mlx-community/gpt-oss-120b-4bit\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"temperature\":0,\"max_tokens\":8}'"
      if [[ "$QUERY_ENGINE_INT_ENABLED" == "1" ]]; then
        clear_dspy_cache
        run_cmd ".venv/bin/pytest -q tests/test_query_engine.py -m integration -k ask_returns_string"
      fi

      clear_dspy_cache
      run_cmd_timeout 240 ".venv/bin/mosaicx verify --sources tests/datasets/extract/sample_patient_vitals.pdf --claim 'patient BP is 128/82' --level thorough"
      clear_dspy_cache
      run_cmd_timeout 240 ".venv/bin/mosaicx verify --sources tests/datasets/extract/sample_patient_vitals.pdf --claim 'patient BP is 120/82' --level thorough"
      clear_dspy_cache
      run_cmd_timeout 240 ".venv/bin/mosaicx query --sources 'tests/datasets/summarize/*.pdf' --question 'what is the lesion size change between the two reports?' --eda --trace --max-iterations 1"
      clear_dspy_cache
      run_cmd_timeout 240 ".venv/bin/mosaicx query --document tests/datasets/generated_hardcases/cohort_edge_cases.csv --question 'what is the distribution of sex?' --eda --trace --max-iterations 1"
    else
      msg="LLM preflight failed."
      if [[ "$LLM_STRICT" == "1" ]]; then
        echo "${msg} Strict mode enabled; aborting hard-test run." | tee -a "$LOG_PATH"
        exit 1
      fi
      echo "${msg} Strict mode disabled; skipping LLM-specific checks." | tee -a "$LOG_PATH"
    fi
  fi
} 2>&1 | tee -a "$LOG_PATH"

cat >> "$LOG_PATH" <<LOG

## Status
- Completed: yes
- Log file: $LOG_PATH
LOG

echo "Run log written to $LOG_PATH"
