#!/usr/bin/env bash
#
# run_benchmarks.sh — Automated LLM performance benchmarking with NVIDIA AIPerf
# against the Huawei MaaS OpenAI-compatible endpoint.
#
# Iterates over eight model/reasoning configurations, running `aiperf profile`
# with streaming, 3 warmup requests, and 100 evaluation requests each. JSON
# telemetry is written into timestamped folders under results/.
#
# Usage:  ./run_benchmarks.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Load MaaS credentials from ~/.bashrc (robust for non-interactive shells,
# where .bashrc returns early before the export lines).
# ---------------------------------------------------------------------------
eval "$(grep -E '^export MAAS_URL=' ~/.bashrc)"
eval "$(grep -E '^export MAAS_API_KEY=' ~/.bashrc)"

if [ -z "${MAAS_URL:-}" ] || [ -z "${MAAS_API_KEY:-}" ]; then
  echo "ERROR: MAAS_URL and MAAS_API_KEY must be set (expected in ~/.bashrc)." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/benchmark.yaml"
RESULTS_BASE="${SCRIPT_DIR}/results"
AIPERF="${SCRIPT_DIR}/.venv/bin/aiperf"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_BASE}/${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

# Full chat-completions URL (endpoint.url in the YAML is used verbatim).
export AIPERF_ENDPOINT_URL="${MAAS_URL}/chat/completions"

# ---------------------------------------------------------------------------
# Model / reasoning configurations: "label|model_id|extra_inputs_json"
#   label            - human-readable run name (also the output sub-folder)
#   model_id         - model id sent to the MaaS API (--model)
#   extra_inputs_json - JSON merged into every request payload (--extra-inputs)
#                      to control thinking / reasoning_effort. Empty = defaults.
# ---------------------------------------------------------------------------
MODELS=(
  'glm-5.2-reasoning-on|glm-5.2|{"thinking":{"type":"enabled"}}'
  'glm-5.2-reasoning-off|glm-5.2|{"thinking":{"type":"disabled"}}'
  'glm-5.3-reasoning-low|glm-5.3|{"reasoning_effort":"low","thinking":{"type":"enabled"}}'
  'glm-5.3-reasoning-default|glm-5.3|'
  'deepseek-v4.1-flash-reasoning-on|deepseek-v4.1-flash|{"thinking":{"type":"enabled"}}'
  'deepseek-v4.1-flash-reasoning-off|deepseek-v4.1-flash|{"thinking":{"type":"disabled"}}'
  'deepseek-v4.1-flash-1017-reasoning-on|deepseek-v4.1-flash-1017|{"thinking":{"type":"enabled"}}'
  'deepseek-v4.1-flash-1017-reasoning-off|deepseek-v4.1-flash-1017|{"thinking":{"type":"disabled"}}'
)

echo "============================================================"
echo " AIPerf benchmark run — ${TIMESTAMP}"
echo " Endpoint: ${AIPERF_ENDPOINT_URL}"
echo " Output:   ${RUN_DIR}"
echo " Variants: ${#MODELS[@]}"
echo "============================================================"

for entry in "${MODELS[@]}"; do
  # Parse "label|model|extra" (extra may be empty).
  label="${entry%%|*}"
  rest="${entry#*|}"
  model="${rest%%|*}"
  extra="${rest#*|}"
  [ "${extra}" = "${rest}" ] && extra=""

  out_dir="${RUN_DIR}/${label}"
  mkdir -p "${out_dir}"
  export AIPERF_MODEL="${model}"
  export AIPERF_ARTIFACT_DIR="${out_dir}"

  echo ""
  echo ">>> [${label}]  model=${model}  extra=${extra:-<none>}"

  cmd=(
    "${AIPERF}" profile
    --config "${CONFIG}"
    --model "${model}"
    --api-key "${MAAS_API_KEY}"
    --streaming
    --tokenizer builtin
    --use-server-token-count
    --no-server-metrics
    --warmup-request-count 3
    --request-count 100
    --concurrency 5
    --output-artifact-dir "${out_dir}"
    --profile-export-prefix "${label}"
    --export-level records
    --log-level info
  )
  if [ -n "${extra}" ]; then
    cmd+=(--extra-inputs "${extra}")
  fi

  # Run; a non-zero exit for one variant does not abort the whole sweep.
  "${cmd[@]}" || echo "WARNING: [${label}] exited non-zero; continuing to next variant."
done

echo ""
echo "============================================================"
echo " All variants benchmarked."
echo " Results directory: ${RUN_DIR}"
echo "============================================================"
