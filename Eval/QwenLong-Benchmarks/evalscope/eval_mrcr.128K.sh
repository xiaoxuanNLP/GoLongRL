#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

set -e

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-evalscope}"

cd "${SCRIPT_DIR}"

# ---------- Model ----------
MODEL_PATH="${MODEL_PATH:?MODEL_PATH environment variable must be set}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME environment variable must be set}"

# ---------- Paths ----------

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/MRCR_eval}"
DATA_FILE="${DATA_FILE:-${WORK_DIR}/data/mrcr_0_128K.jsonl}"

PORT="${PORT:-8000}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-4}"

mkdir -p "${WORK_DIR}/runs"
INFER_OUTPUT="${WORK_DIR}/runs/${MODEL_NAME}_mrcr_128k.jsonl"

echo "=========================================="
echo "MRCR 128K Evaluation (Official Method)"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Model path: ${MODEL_PATH}"
echo "Data : ${DATA_FILE}"
echo "Out  : ${INFER_OUTPUT}"
echo ""

CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --tensor-parallel-size ${TP} \
  --max-model-len 182272 \
  --port ${PORT} \
  > "${WORK_DIR}/vllm_mrcr_128k.log" 2>&1 &

VLLM_PID=$!
trap 'kill ${VLLM_PID:-} 2>/dev/null || true; wait ${VLLM_PID:-} 2>/dev/null || true' EXIT

echo "Waiting for vLLM..."
for i in $(seq 1 120); do
  if curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"${MODEL_NAME}"'","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}' \
    | grep -q 200; then
    echo "vLLM ready"
    break
  fi
  [ $i -eq 120 ] && echo "Startup timeout, check ${WORK_DIR}/vllm_mrcr_128k.log" && exit 1
  sleep 5
done

python "${SCRIPT_DIR}/mrcr_eval.py" \
  --data_file "${DATA_FILE}" \
  --model "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  --api_url "http://127.0.0.1:${PORT}/v1" \
  --concurrency 4 \
  --infer_output "${INFER_OUTPUT}" \
  --max_input_tokens 131072

echo ""
echo "=========================================="
echo "Done"
echo "Result : ${INFER_OUTPUT}"
echo "Summary: ${INFER_OUTPUT/.jsonl/_summary.json}"
echo "=========================================="
