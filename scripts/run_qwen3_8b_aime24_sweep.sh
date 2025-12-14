#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/home/haojias/llama.cpp/models/Qwen3-8B-DeepSeek-BF16.gguf"
SERVER_HOST="0.0.0.0"
SERVER_PORT=8000
SERVER_URL="http://127.0.0.1:${SERVER_PORT}"

MODEL_ID="qwen3_8b"
MAX_EXAMPLES=7
NUM_SAMPLES_PER_QUESTION=2
REQUEST_TIMEOUT=44000  
OUTPUT_DIR="outputs"

echo "[INFO] Killing existing llama-server if any..."
pkill -f "llama-server" || true
sleep 3

CONFIGS=(
  "32  18384   16384   qwen3_8b_cpu_offload0_16384"
  "32  28768   27768   qwen3_8b_cpu_offload0_27768"

  "30  18384   16384   qwen3_8b_cpu_offload2_16384"
  "30  28768   27768   qwen3_8b_cpu_offload2_27768"

  "32  34768   32768   qwen3_8b_cpu_offload0_32768"
  "30  34768   32768   qwen3_8b_cpu_offload2_32768"

#   "30  6000   4096   qwen3_8b_cpu_offload2_4096"
#   "30  8000   6144   qwen3_8b_cpu_offload2_6144"

#   "28  9000   8192   qwen3_8b_cpu_offload4_8192"
#   "28  11240  10240  qwen3_8b_cpu_offload4_10240"
)

mkdir -p "${OUTPUT_DIR}"

run_one_config() {
  local GPU_LAYERS="$1"
  local CTX="$2"
  local MAX_NEW="$3"
  local DESC="$4"

  echo "============================================================"
  echo "[INFO] Config: ${DESC}"
  echo "       GPU layers:      ${GPU_LAYERS}"
  echo "       Context length:  ${CTX}"
  echo "       Max new tokens:  ${MAX_NEW}"
  echo "============================================================"

  pkill -f "llama-server" || true
  sleep 3

  echo "[INFO] Starting llama-server ..."
  ./build/bin/llama-server \
    -m "${MODEL_PATH}" \
    -ngl "${GPU_LAYERS}" \
    -c "${CTX}" \
    --port "${SERVER_PORT}" \
    --host "${SERVER_HOST}" \
    > "logs/server_${DESC}.log" 2>&1 &

  SERVER_PID=$!
  echo "[INFO] llama-server PID = ${SERVER_PID}"

  sleep 20

  echo "[INFO] Running AIME eval for ${DESC} ..."
  python scripts/qwen3_8b_eval_aime.py \
    --server-url "${SERVER_URL}" \
    --model-id "${MODEL_ID}" \
    --max-context "${CTX}" \
    --max-new-tokens "${MAX_NEW}" \
    --gpu-layers "${GPU_LAYERS}" \
    --offload-desc "${DESC}" \
    --num-samples-per-question "${NUM_SAMPLES_PER_QUESTION}" \
    --max-examples "${MAX_EXAMPLES}" \
    --request-timeout "${REQUEST_TIMEOUT}" \
    --output-dir "${OUTPUT_DIR}"

  echo "[INFO] Eval finished for ${DESC}, killing llama-server..."
  kill "${SERVER_PID}" || true
  sleep 3
}

for cfg in "${CONFIGS[@]}"; do
  # shellcheck disable=SC2086
  run_one_config $cfg
done

echo "[INFO] All configs finished."
