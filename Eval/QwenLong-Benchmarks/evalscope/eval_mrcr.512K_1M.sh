#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-evalscope}"

cd "${SCRIPT_DIR}"

# ---------- Model ----------

MODEL_PATH="${MODEL_PATH:?MODEL_PATH environment variable must be set}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME environment variable must be set}"

# ---------- Paths ----------

WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/MRCR_eval}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/datasets/mrcr}"
DATA_FILE="${DATA_FILE:-${DATA_DIR}/mrcr_512K_1M.jsonl}"

# ---------- Hardware ----------

PORT=8002
GPUS="0,1,2,3,4,5,6,7"
TP=8

# ---------- YaRN RoPE Scaling ----------
# Qwen3-4B native max_position_embeddings = 262144 (256K)
# 1M / 256K = 4.0
ORIGINAL_MAX_POS=262144
YARN_FACTOR=4.0
MAX_INPUT_TOKENS=1048576
MAX_MODEL_LEN=1048576   # 1M = 1024 * 1024

# ---------- Prepare data ----------

mkdir -p "${WORK_DIR}/runs" "${DATA_DIR}"

if [ ! -f "${DATA_FILE}" ]; then
    echo "[1/3] Downloading MRCR 512K-1M data..."
    python "${SCRIPT_DIR}/download_mrcr_overlong.py" \
        --output_dir "${DATA_DIR}" --range 512k_1m
else
    echo "[1/3] Data exists ($(wc -l < "${DATA_FILE}") samples)"
fi

INFER_OUTPUT="${WORK_DIR}/runs/${MODEL_NAME}_mrcr_512k_1m.jsonl"

echo "=========================================="
echo "MRCR 512K-1M Evaluation"
echo "=========================================="
echo "Model     : ${MODEL_NAME}"
echo "Model path: ${MODEL_PATH}"
echo "Data      : ${DATA_FILE}"
echo "Output    : ${INFER_OUTPUT}"
echo "YaRN      : factor=${YARN_FACTOR}, orig_max_pos=${ORIGINAL_MAX_POS}"
echo "Context   : max_input=${MAX_INPUT_TOKENS}, max_model_len=${MAX_MODEL_LEN}"
echo ""

# ---------- Launch vLLM ----------

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --tensor-parallel-size ${TP} \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 65536 \
  --rope-scaling "{\"rope_type\": \"yarn\", \"factor\": ${YARN_FACTOR}, \"original_max_position_embeddings\": ${ORIGINAL_MAX_POS}}" \
  --port ${PORT} \
  > "${WORK_DIR}/vllm_mrcr_512k_1m.log" 2>&1 &

VLLM_PID=$!
trap 'kill ${VLLM_PID:-} 2>/dev/null || true; wait ${VLLM_PID:-} 2>/dev/null || true' EXIT
echo "vLLM PID: ${VLLM_PID}"


echo "Waiting for vLLM (1M context, may take a while)..."
for i in $(seq 1 300); do
  if curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"${MODEL_NAME}"'","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}' \
    | grep -q 200; then
    echo "vLLM ready (waited $((i*5))s)"
    break
  fi
  if [ $i -eq 300 ]; then
    echo "Startup timeout, check ${WORK_DIR}/vllm_mrcr_512k_1m.log"
    kill ${VLLM_PID} 2>/dev/null
    exit 1
  fi
  sleep 5
done

# ---------- Run evaluation ----------

python "${SCRIPT_DIR}/mrcr_eval.py" \
  --data_file "${DATA_FILE}" \
  --model "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  --api_url "http://127.0.0.1:${PORT}/v1" \
  --concurrency 4 \
  --infer_output "${INFER_OUTPUT}" \
  --max_input_tokens ${MAX_INPUT_TOKENS}

echo ""
echo "=========================================="
echo "Done"
echo "Result : ${INFER_OUTPUT}"
echo "Summary: ${INFER_OUTPUT/.jsonl/_summary.json}"
echo "=========================================="

kill ${VLLM_PID} && echo "vLLM stopped"
wait ${VLLM_PID} 2>/dev/null || true